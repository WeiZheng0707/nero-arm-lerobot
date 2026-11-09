"""Run one synthetic PI05 LoRA forward/backward pass on CUDA, without training or saving."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model/pi_05_base"))
    parser.add_argument("--quantile-stats", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke test")

    config = PreTrainedConfig.from_pretrained(args.model)
    config.device = "cuda"
    config.dtype = "bfloat16"
    config.lora_enabled = True
    config.lora_rank = 8
    config.lora_alpha = 16.0
    config.lora_dropout = 0.05
    config.train_expert_only = True
    config.freeze_vision_encoder = True
    config.empty_cameras = 2
    config.gradient_checkpointing = True
    config.quantile_stats_path = str(args.quantile_stats)
    config.input_features = {
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
    }
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (8,))}

    policy = PI05Policy.from_pretrained(args.model, config=config)
    policy.train()
    batch = {
        "observation.images.wrist": torch.rand(1, 3, 480, 640, device="cuda"),
        "observation.state": torch.rand(1, 8, device="cuda"),
        "action": torch.rand(1, config.chunk_size, 8, device="cuda"),
        "observation.language.tokens": torch.zeros(1, config.tokenizer_max_length, dtype=torch.long, device="cuda"),
        "observation.language.attention_mask": torch.zeros(
            1, config.tokenizer_max_length, dtype=torch.bool, device="cuda"
        ),
    }
    batch["observation.language.tokens"][0, :8] = torch.arange(1, 9, device="cuda")
    batch["observation.language.attention_mask"][0, :8] = True

    loss, _ = policy(batch)
    loss.backward()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable or any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
        raise RuntimeError("PI05 LoRA smoke test found missing or non-finite trainable gradients")
    torch.cuda.synchronize()
    print(f"PASS: synthetic PI05 loss={loss.item():.6f}, trainable_tensors={len(trainable)}")
    print(f"CUDA allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
