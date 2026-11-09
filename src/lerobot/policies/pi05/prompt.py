"""Everything pi0.5 needs before the forward pass, without the processor framework.

Upstream LeRobot runs a `PolicyProcessorPipeline` that (a) normalizes state and
action with quantiles, (b) turns the normalized state into 256-bin integers and
glues them into the language prompt, and (c) tokenizes with the PaliGemma
tokenizer. This repo's `train.py` predates that framework and hands the policy a
raw dataset batch, so the three steps live here instead and are invoked from
`PI05Policy` itself.

The prompt format is copied verbatim from openpi's `PaligemmaTokenizer`:

    Task: pick up the cube, State: 127 143 ... 88;\nAction:

One deliberate difference from upstream LeRobot: the state is padded to
`max_state_dim` (32) **before** discretization, because the released weights
were pretrained on 32-dim padded state. Upstream LeRobot discretizes whatever
dimension the dataset happens to have (8 for this robot), producing a much
shorter state string than pretraining ever saw. Keep the padding unless you are
explicitly A/B testing it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

# openpi discretizes into 256 bins spanning [-1, 1]
NUM_BINS = 256
_BIN_EDGES = np.linspace(-1.0, 1.0, NUM_BINS + 1)[:-1]


@dataclass
class QuantileStats:
    """Per-dimension q01/q99 for the tensors pi0.5 normalizes with quantiles."""

    state_q01: torch.Tensor
    state_q99: torch.Tensor
    action_q01: torch.Tensor
    action_q99: torch.Tensor

    @classmethod
    def from_json(cls, path: str | Path) -> QuantileStats:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            state = payload["observation.state"]
            action = payload["action"]
        except KeyError as exc:
            raise KeyError(
                f"{path} 必须包含 'observation.state' 和 'action' 两项，"
                f"且各有 q01/q99；实际拿到 {sorted(payload)}"
            ) from exc
        return cls(
            state_q01=torch.as_tensor(state["q01"], dtype=torch.float32),
            state_q99=torch.as_tensor(state["q99"], dtype=torch.float32),
            action_q01=torch.as_tensor(action["q01"], dtype=torch.float32),
            action_q99=torch.as_tensor(action["q99"], dtype=torch.float32),
        )

    def to(self, device) -> QuantileStats:
        return QuantileStats(
            self.state_q01.to(device),
            self.state_q99.to(device),
            self.action_q01.to(device),
            self.action_q99.to(device),
        )


def _scaled(tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor):
    if tensor.shape[-1] != q01.numel():
        raise ValueError(
            f"最后一维应为 {q01.numel()} 才能对上分位数，实际是 {tensor.shape[-1]}"
        )
    denom = (q99 - q01).clamp_min(1e-6)
    return q01, denom


def normalize_quantiles(tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """Map [q01, q99] onto [-1, 1] (values outside are not clipped, same as openpi)."""
    q01, denom = _scaled(tensor, q01, q99)
    return 2.0 * (tensor - q01) / denom - 1.0


def unnormalize_quantiles(tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    q01, denom = _scaled(tensor, q01, q99)
    return (tensor + 1.0) * denom / 2.0 + q01


def discretize_state(state: torch.Tensor, pad_to: int = 32) -> list[str]:
    """Turn a `[B, D]` normalized state into one "127 143 ..." string per sample."""
    state = state.detach().to("cpu", torch.float32)
    if pad_to and state.shape[-1] < pad_to:
        pad = torch.zeros(*state.shape[:-1], pad_to - state.shape[-1], dtype=state.dtype)
        state = torch.cat([state, pad], dim=-1)
    elif pad_to and state.shape[-1] > pad_to:
        raise ValueError(f"state 有 {state.shape[-1]} 维，超过 max_state_dim={pad_to}")
    bins = np.digitize(state.numpy(), bins=_BIN_EDGES) - 1
    return [" ".join(map(str, row)) for row in bins]


class Pi05PromptBuilder:
    """Discretized state + task text -> PaliGemma token ids.

    The tokenizer is built lazily: `google/paligemma-3b-pt-224` is a gated repo,
    and paying for it on `__init__` would make even a weight-loading check fail
    without credentials.
    """

    def __init__(self, tokenizer_name: str, max_length: int, state_pad_dim: int = 32):
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.state_pad_dim = state_pad_dim
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            logging.info("Loading pi0.5 tokenizer from %s", self.tokenizer_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                local_files_only=True,
            )
            self._tokenizer.padding_side = "right"
        return self._tokenizer

    def build_prompts(self, tasks: list[str], state_norm: torch.Tensor) -> list[str]:
        state_strings = discretize_state(state_norm, self.state_pad_dim)
        prompts = []
        for task, state_str in zip(tasks, state_strings, strict=True):
            cleaned = str(task).strip().replace("_", " ").replace("\n", " ")
            prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")
        return prompts

    def __call__(self, tasks: list[str], state_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = self.build_prompts(tasks, state_norm)
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"].bool()


class QuantileNormalizer(nn.Module):
    """State/action normalization kept as a buffer so it follows `.to(device)`."""

    def __init__(self, stats: QuantileStats):
        super().__init__()
        # 不持久化：这是派生的常量，不是学出来的权重。
        # 持久化会让每个 checkpoint 在 strict 加载时多报四个 missing key。
        self.register_buffer("state_q01", stats.state_q01.clone(), persistent=False)
        self.register_buffer("state_q99", stats.state_q99.clone(), persistent=False)
        self.register_buffer("action_q01", stats.action_q01.clone(), persistent=False)
        self.register_buffer("action_q99", stats.action_q99.clone(), persistent=False)

    def state(self, tensor: torch.Tensor) -> torch.Tensor:
        return normalize_quantiles(tensor, self.state_q01, self.state_q99)

    def action(self, tensor: torch.Tensor) -> torch.Tensor:
        return normalize_quantiles(tensor, self.action_q01, self.action_q99)

    def inverse_action(self, tensor: torch.Tensor) -> torch.Tensor:
        return unnormalize_quantiles(tensor, self.action_q01, self.action_q99)
