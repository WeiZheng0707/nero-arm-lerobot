"""Offline evaluation of ACT checkpoints on a held-out LeRobot split.

Training loss cannot tell "learned the task" apart from "memorised 240
episodes". This measures first-step action error on episodes the policy never
saw, which is the metric that matters before any real-arm test.

No robot, no CAN, no camera: it only reads the dataset and the checkpoints.

Usage:
    python -X utf8 tools/eval_act_offline.py ^
        --checkpoint outputs/train/nero_red_cube_act_v1/checkpoints/030000/pretrained_model ^
        --checkpoint outputs/train/nero_red_cube_act_v1/checkpoints/050000/pretrained_model ^
        --root outputs/red_cube_300_video/val --repo-id local/nero_red_cube_bag_val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]


def evaluate_checkpoint(checkpoint: Path, dataset: LeRobotDataset, device: torch.device,
                        stride: int, max_samples: int):
    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()

    image_keys = list(policy.config.image_features)
    errors = []
    per_joint = []
    per_episode: dict[int, list[float]] = {}
    gripper_out_of_range = 0
    nonfinite = 0

    indices = list(range(0, len(dataset), stride))[:max_samples]
    for count, index in enumerate(indices, start=1):
        sample = dataset[index]
        batch = {}
        target = None
        for key, value in sample.items():
            if not torch.is_tensor(value):
                continue
            if key == "action":
                # Kept aside, never fed to the policy: ACT.forward runs its VAE
                # branch whenever "action" is present in the batch, and a single
                # frame is not an action chunk, so passing it here raises
                # "Tensors must have same number of dimensions: got 3 and 2".
                target = value.unsqueeze(0).to(device)
                continue
            if key.startswith("observation."):
                batch[key] = value.unsqueeze(0).to(device)
        if target is None:
            raise RuntimeError(f"sample {index} has no action to compare against")
        if not all(key in batch for key in image_keys):
            raise RuntimeError(f"sample {index} missing an image key: {sorted(batch)}")

        with torch.no_grad():
            predicted = policy.predict_action_chunk(batch)[:, 0, :]  # first step only

        if not torch.isfinite(predicted).all():
            nonfinite += 1
            continue
        diff = (predicted - target).abs().squeeze(0).cpu().numpy()
        errors.append(float(diff.mean()))
        per_joint.append(diff)
        episode = int(sample["episode_index"].item()) if torch.is_tensor(sample["episode_index"]) else int(sample["episode_index"])
        per_episode.setdefault(episode, []).append(float(diff.mean()))

        # The gripper is normalized to 0..1; anything outside would clip on hardware.
        if predicted[0, -1].item() < -0.05 or predicted[0, -1].item() > 1.05:
            gripper_out_of_range += 1
        if count % 200 == 0:
            print(f"    evaluated {count}/{len(indices)} samples", flush=True)

    if not errors:
        raise RuntimeError("no finite predictions; the checkpoint or dataset is wrong")

    errors_np = np.asarray(errors)
    joint_mae = np.mean(np.stack(per_joint), axis=0)
    episode_means = {ep: float(np.mean(values)) for ep, values in per_episode.items()}
    worst = sorted(episode_means.items(), key=lambda item: item[1], reverse=True)[:5]

    return {
        "checkpoint": str(checkpoint),
        "samples": len(errors),
        "first_step_mae": float(errors_np.mean()),
        "first_step_p50": float(np.percentile(errors_np, 50)),
        "first_step_p95": float(np.percentile(errors_np, 95)),
        "first_step_max": float(errors_np.max()),
        "per_joint_mae": {name: float(value) for name, value in zip(JOINT_NAMES, joint_mae)},
        "episodes_seen": len(episode_means),
        "worst_episodes": [{"episode": ep, "mae": mae} for ep, mae in worst],
        "gripper_out_of_range": gripper_out_of_range,
        "nonfinite_predictions": nonfinite,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="Path to a .../pretrained_model directory; repeat to compare")
    parser.add_argument("--root", type=Path, required=True, help="Dataset split root, e.g. .../val")
    parser.add_argument("--repo-id", default="local/nero_red_cube_bag_val")
    parser.add_argument("--video-backend", default="pyav",
                        help="torchcodec needs a shared FFmpeg build; pyav works without one")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stride", type=int, default=30,
                        help="Evaluate every Nth frame; 30 means one frame per second")
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional path to write the JSON report")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device={device} root={args.root} stride={args.stride}", flush=True)
    dataset = LeRobotDataset(args.repo_id, root=args.root, video_backend=args.video_backend)
    print(f"dataset: {dataset.num_episodes} episodes, {len(dataset)} frames", flush=True)

    reports = []
    for checkpoint in args.checkpoint:
        print(f"\n=== {checkpoint}", flush=True)
        report = evaluate_checkpoint(checkpoint, dataset, device, args.stride, args.max_samples)
        reports.append(report)
        print(f"  first-step MAE : {report['first_step_mae']:.4f}", flush=True)
        print(f"  p50 / p95 / max: {report['first_step_p50']:.4f} / "
              f"{report['first_step_p95']:.4f} / {report['first_step_max']:.4f}", flush=True)
        print("  per joint      : " + ", ".join(
            f"{name}={value:.4f}" for name, value in report["per_joint_mae"].items()), flush=True)
        print(f"  gripper out of range: {report['gripper_out_of_range']}   "
              f"non-finite: {report['nonfinite_predictions']}", flush=True)
        print("  worst episodes : " + ", ".join(
            f"{item['episode']}={item['mae']:.3f}" for item in report["worst_episodes"]), flush=True)

    print("\nSUMMARY (lower is better; only comparable within the same split)", flush=True)
    print(f"  {'checkpoint':40s} {'MAE':>8s} {'p95':>8s} {'episodes':>9s}", flush=True)
    for report in reports:
        print(f"  {Path(report['checkpoint']).parent.name:40s} "
              f"{report['first_step_mae']:>8.4f} {report['first_step_p95']:>8.4f} "
              f"{report['episodes_seen']:>9d}", flush=True)

    best = min(reports, key=lambda item: item["first_step_mae"])
    print(f"\nBest first-step MAE: {best['checkpoint']}", flush=True)
    print("Pick by this number, not by training loss or step count: a later "
          "checkpoint with a higher MAE has started to memorise the training "
          "episodes.", flush=True)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"Report written to {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
