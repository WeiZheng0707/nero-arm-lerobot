"""A minimal, dependency-free LoRA for the pi0.5 action expert.

Why not `peft`: the vendored pi0.5 forward path calls `layernorm_forward` /
`_gated_residual` on the expert's own Gemma blocks, and those helpers are
positional about module identity. `peft`'s module swapping changes attribute
paths in ways that are fine for plain HF models but add a moving part here for
no benefit -- this file is ~120 lines and keeps the checkpoint format explicit.

Two decisions worth knowing:

1. **The adapter is injected after the base weights are loaded.** Wrapping a
   `nn.Linear` renames its parameters (`q_proj.weight` -> `q_proj.base.weight`),
   so loading a pi0.5 checkpoint into an already-injected model would report
   every attention weight as unexpected. Load first, wrap second.

2. **Adapter weights stay float32 even when the base is bfloat16.** Training
   bf16 LoRA parameters directly loses small updates to rounding; the usual
   fix is to keep the low-rank factors in fp32 and cast at the boundary, which
   is what `forward` does below.
"""

from __future__ import annotations

import math
import re

import torch
from torch import nn

DEFAULT_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    """`y = W x + b + (alpha / rank) * B(A(x))`, with `W` frozen."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # standard init: A random, B zero, so the adapter starts as a no-op
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # fp32 adapter regardless of the base dtype, see module docstring
        self.lora_A.to(device=base.weight.device, dtype=torch.float32)
        self.lora_B.to(device=base.weight.device, dtype=torch.float32)

    # pi0.5's forward path reads `layer.self_attn.o_proj.weight.dtype` to align
    # dtypes between the expert and the backbone. A wrapper that only forwards
    # __call__ breaks that with AttributeError, so mirror the nn.Linear surface
    # callers actually touch.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        adapter_input = x.float() if x.dtype != torch.float32 else x
        delta = self.lora_B(self.lora_A(self.lora_dropout(adapter_input))) * self.scaling
        return out + delta.to(out.dtype)

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """Fold `B @ A` into `W` in place and zero the adapter."""
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base.weight.data += delta.to(self.base.weight.dtype)
        nn.init.zeros_(self.lora_B.weight)

    def extra_repr(self) -> str:
        return f"in={self.base.in_features}, out={self.base.out_features}, rank={self.rank}"


def _matches(name: str, targets: tuple[str, ...], include: str, exclude: tuple[str, ...]) -> bool:
    if not isinstance(name, str) or not name:
        return False
    if include and include not in name:
        return False
    if any(pat and re.search(pat, name) for pat in exclude):
        return False
    leaf = name.rsplit(".", 1)[-1]
    return leaf in targets


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    include: str = "gemma_expert",
    exclude: tuple[str, ...] = (),
) -> list[str]:
    """Wrap every matching `nn.Linear` in `model` with a `LoRALinear`.

    `include` restricts injection to a subtree by substring -- by default only
    the action expert, since the PaliGemma backbone is frozen and adapters
    there would just burn memory.

    Returns the wrapped module names so callers can assert the list is not
    empty (empty means the target names do not exist in this architecture,
    which would silently train only the projections).
    """
    replaced: list[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, LoRALinear) or not isinstance(child, nn.Linear):
                continue
            if not _matches(full, targets, include, exclude):
                continue
            setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(full)
    return replaced


def count_lora_modules(model: nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, LoRALinear))


def lora_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield from module.lora_A.parameters()
            yield from module.lora_B.parameters()
