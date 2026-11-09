"""训练前只读自检：**不训练、不写盘**，只回答三个问题。

  1. checkpoint 能不能加载进这套代码（含视觉塔键名映射）
  2. LoRA 有没有真的只训那一小撮参数（打印显存 / 内存 / 参数分布）
  3. 适配器初始化时是不是精确 no-op

跑一次约 2 分钟，主要是读 14 GB 权重。它不会创建 checkpoint，
所以你看到的所有显存/内存数字就是正式训练时的基线。

用法：
    python tools/pi05_preflight.py ^
        --model model/pi_05_base ^
        --quantile-stats outputs/eye_in_hand_red_cube_v2/train/meta/quantile_stats.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05.lora import LoRALinear  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model/pi_05_base"))
    parser.add_argument("--quantile-stats", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    config = PreTrainedConfig.from_pretrained(args.model)
    config.device = args.device
    config.dtype = args.dtype
    config.quantile_stats_path = str(args.quantile_stats)
    config.train_expert_only = True
    config.freeze_vision_encoder = True
    config.lora_enabled = True
    config.lora_rank = args.rank
    # 模拟 make_policy 对这个数据集做的事：一路腕部相机 + 8 维 state/action
    config.input_features = {
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    config.empty_cameras = 2  # 用空相机把前缀补回 3 槽

    print(f"加载 {args.model}（{args.dtype}，{args.device}）...")
    policy = PI05Policy.from_pretrained(args.model, config=config)
    policy.eval()

    model = policy.model
    lora_layers = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"\nLoRA 模块数            : {len(lora_layers)}")
    for name in lora_layers[:3]:
        print(f"  e.g. {name}")
    print(f"总参数                : {total / 1e6:.2f} M")
    print(f"可训参数              : {trainable / 1e6:.2f} M ({100 * trainable / total:.2f}%)")

    groups: dict[str, int] = {}
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            key = "LoRA 适配器"
        elif "gemma_expert" in name:
            key = "专家本体(非 LoRA)"
        elif "paligemma" in name:
            key = "PaliGemma(不该出现)"
        else:
            key = name.split(".")[0]
        groups[key] = groups.get(key, 0) + param.numel()
    print("可训参数分布:")
    for key, count in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {key:26s} {count / 1e6:9.3f} M")

    frozen = all(not p.requires_grad for p in model.paligemma_with_expert.paligemma.parameters())
    print(f"\nPaliGemma 完全冻结     : {frozen}")

    if lora_layers:
        layer = model.get_submodule(lora_layers[0])
        x = torch.randn(2, layer.base.in_features, device=layer.base.weight.device, dtype=layer.base.weight.dtype)
        delta = layer(x) - layer.base(x)
        print(f"适配器初始为 no-op     : max|delta| = {delta.abs().max().item():.3e}")

    if args.device == "cuda":
        print(
            f"显存: 已分配 {torch.cuda.memory_allocated() / 1e9:.2f} GB / "
            f"已保留 {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )
        torch.cuda.reset_peak_memory_stats()
        print("（峰值显存要等一次 forward 才准，正式训练时看 nvidia-smi）")

    if not frozen or groups.get("PaliGemma(不该出现)"):
        raise SystemExit("失败: 冻结没生效")
    if lora_layers and groups.get("专家本体(非 LoRA)"):
        raise SystemExit("失败: 专家本体还在训练，这不是 LoRA")
    print("\nPASS")


if __name__ == "__main__":
    main()
