"""pi0.5 离线评估：按 checkpoint 出 episode 级指标。

和 ACT 那边的评估口径保持一致，方便直接对照：
  first_action_mae     每个 episode 内所有探测点首步误差的均值，再对 episode 求均值
  full_chunk_mae       整个 chunk 的平均误差
  gripper_mae          最后两维（夹爪）的误差
  gripper_oor          预测值超出真值 [min,max] 的比例
  action_jump          相邻步动作差分的 p95

与 VOC 版 eval_offline.py 的差别只有一个：pi0.5 需要 batch 里有 `task`，
提示词由策略内部构造（见 policies/pi05/prompt.py）。

用法：
    python tools/pi05_eval_offline.py ^
        --ckpt outputs/train/eye_in_hand_red_pi05_lora_v1/checkpoints/030000/pretrained_model ^
        --val-root outputs/eye_in_hand_red_cube_v2/val ^
        --repo-id local/nero_eye_in_hand_red_cube_val
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# Keep the Hugging Face datasets lock path short on Windows. The cache path is
# repeated inside the lock filename and can otherwise exceed MAX_PATH.
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(tempfile.gettempdir()) / "lerobot_hfds"))

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402


def episode_blocks(episode_index: torch.Tensor) -> list[tuple[int, int]]:
    """把连续的 episode 行号压成 (start, end) 区间。"""
    values = episode_index.tolist() if hasattr(episode_index, "tolist") else list(episode_index)
    blocks: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            blocks.append((start, i))
            start = i
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--quantile-stats", type=Path, default=None,
                        help="默认取 --val-root 的兄弟目录 train/meta/quantile_stats.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=None, help="默认用 checkpoint 里的 chunk_size")
    parser.add_argument("--max-per-episode", type=int, default=6)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    qpath = args.quantile_stats
    if qpath is None:
        qpath = args.val_root.parent / "train" / "meta" / "quantile_stats.json"
    if not qpath.exists():
        raise SystemExit(f"缺少分位数统计 {qpath}，先跑 tools/pi05_quantile_stats.py")

    config = PreTrainedConfig.from_pretrained(args.ckpt)
    config.device = args.device
    config.quantile_stats_path = str(qpath)
    policy = PI05Policy.from_pretrained(args.ckpt, config=config)
    policy.eval()
    device = config.device

    chunk = args.chunk or policy.config.chunk_size
    fps = json.loads((args.val_root / "meta" / "info.json").read_text(encoding="utf-8"))["fps"]
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.val_root,
        delta_timestamps={"action": [i / fps for i in range(chunk)]},
    )

    image_keys = [k for k, ft in policy.config.input_features.items()
                  if ft.type.name == "VISUAL" and not k.startswith("observation.images.empty_camera")]
    if not image_keys:
        raise SystemExit("策略没有声明任何图像输入")

    blocks = episode_blocks(dataset.hf_dataset["episode_index"])
    print(f"验证集 {len(blocks)} 条 episode，chunk={chunk}，每集最多 {args.max_per_episode} 个探测点")

    per_episode: dict[int, dict] = defaultdict(lambda: {"first": [], "chunk": [], "grip": [], "oor": [], "jump": []})

    for ep_idx, (start, end) in enumerate(blocks):
        length = end - start
        if length <= chunk + 2:
            continue
        lo = start + max(1, int(length * 0.35))
        picks = list(range(lo, end - chunk - 1, args.stride))[: args.max_per_episode]
        for row in picks:
            item = dataset[row]
            batch = {OBS: item[OBS].unsqueeze(0).to(device) for OBS in ("observation.state",)}
            for key in image_keys:
                batch[key] = item[key].unsqueeze(0).to(device)
            batch["task"] = [item["task"]]
            with torch.no_grad():
                pred = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
            gt = item["action"].float().cpu().numpy()

            err = np.abs(pred - gt)
            per_episode[ep_idx]["first"].append(float(err[0].mean()))
            per_episode[ep_idx]["chunk"].append(float(err.mean()))
            per_episode[ep_idx]["grip"].append(float(err[:, -2:].mean()))
            lo_gt, hi_gt = gt.min(axis=0), gt.max(axis=0)
            per_episode[ep_idx]["oor"].append(float(((pred < lo_gt) | (pred > hi_gt)).mean()))
            per_episode[ep_idx]["jump"].append(float(np.abs(np.diff(pred, axis=0)).mean()))

    if not per_episode:
        raise SystemExit("没有采到任何探测点，检查 --stride 与 chunk 设置")

    rows = []
    for ep_idx in sorted(per_episode):
        stats = per_episode[ep_idx]
        rows.append({
            "episode_index": ep_idx,
            "windows": len(stats["first"]),
            "first_action_mae": float(np.mean(stats["first"])),
            "full_chunk_mae": float(np.mean(stats["chunk"])),
            "gripper_mae": float(np.mean(stats["grip"])),
            "gripper_oor": float(np.mean(stats["oor"])),
            "action_jump": float(np.mean(stats["jump"])),
        })

    firsts = np.array([r["first_action_mae"] for r in rows])
    summary = {
        "checkpoint": str(args.ckpt),
        "episodes": len(rows),
        f"episode_mean_of_first_action_mae": float(firsts.mean()),
        f"episode_p95_of_first_action_mae": float(np.percentile(firsts, 95)),
        "full_chunk_mae": float(np.mean([r["full_chunk_mae"] for r in rows])),
        "gripper_mae": float(np.mean([r["gripper_mae"] for r in rows])),
        "gripper_oor": float(np.mean([r["gripper_oor"] for r in rows])),
        "action_jump": float(np.mean([r["action_jump"] for r in rows])),
        "per_episode": rows,
    }

    print(f"\nepisode 级首步 MAE : {summary['episode_mean_of_first_action_mae']:.5f}")
    print(f"episode 级 p95      : {summary['episode_p95_of_first_action_mae']:.5f}")
    print(f"full chunk MAE      : {summary['full_chunk_mae']:.5f}")
    print(f"夹爪 MAE            : {summary['gripper_mae']:.5f}  越界率 {summary['gripper_oor']*100:.2f}%")
    print(f"动作跳变            : {summary['action_jump']:.5f}")

    out = args.out or (args.ckpt.parent.parent.parent / f"eval_pi05_{args.ckpt.parent.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 -> {out}")


if __name__ == "__main__":
    main()
