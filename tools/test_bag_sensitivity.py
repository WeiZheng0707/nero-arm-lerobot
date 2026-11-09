r"""Does the policy actually steer by the bag position?

The dataset is fine: the bag spans 329 px across episodes and the operator's
joint_1 correlates 0.85 with the bag's x. The robot, however, always drops at the
bag's average spot, so the question is whether the network reads the bag fields at
all.

This keeps the images and the joint state fixed and only rewrites the bag box in
`observation.environment_state`. If moving the bag from the left edge to the right
edge leaves the predicted action unchanged, the policy ignores those four numbers,
and no amount of data collection will help - the fix is training or features, not
recordings.

Read-only: no hardware.

Usage:
    python -X utf8 tools\test_bag_sensitivity.py ^
        --checkpoint D:\1104\outputs\nero_act_v1\checkpoints\030000\pretrained_model ^
        --root D:\1104\nero_cube_300\val --repo-id local/nero_cube_val ^
        --episodes 8 --offset 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

# 14-dim env layout: 0..4 cube, 5..8 bag x/y/w/h, 9..13 ball
BAG_X, BAG_Y, BAG_W, BAG_H = 5, 6, 7, 8
# 22-dim merged layout: 0..7 joints+gripper, then the same 14 env values shifted by 8.
MERGED_OFFSET = 8


def to_batch(sample, device):
    batch = {}
    for key, value in sample.items():
        if key == "action" or not torch.is_tensor(value) or not key.startswith("observation."):
            continue
        batch[key] = value.unsqueeze(0).to(device)
    return batch


def predict(policy, batch):
    with torch.no_grad():
        return policy.predict_action_chunk(batch)


def bag_indices(layout: str):
    if layout == "merged22":
        return MERGED_OFFSET + BAG_X, MERGED_OFFSET + BAG_Y, MERGED_OFFSET + BAG_W, MERGED_OFFSET + BAG_H
    return BAG_X, BAG_Y, BAG_W, BAG_H


def vector_key(layout: str):
    return "observation.state" if layout == "merged22" else "observation.environment_state"


def moved_vector(vector: np.ndarray, x: float, y: float, layout: str) -> np.ndarray:
    out = vector.copy()
    ix, iy, iw, ih = bag_indices(layout)
    out[ix], out[iy] = x, y
    if out[iw] <= 0.005:
        out[iw], out[ih] = 0.195, 0.281
    return out


def zeroed_vector(vector: np.ndarray, layout: str) -> np.ndarray:
    """Blank the environment part only.

    With the merged layout the vector also holds joint angles; zeroing all of it
    would remove proprioception too and the measurement would no longer isolate
    the environment fields.
    """
    out = vector.copy()
    if layout == "merged22":
        out[MERGED_OFFSET:] = 0.0
    else:
        out[:] = 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/nero_cube_val")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--offset", type=int, default=400,
                        help="Frame offset inside the episode; ~400 is the transport phase")
    parser.add_argument("--layout", choices=("env14", "merged22"), default="env14",
                        help="env14: bag lives in observation.environment_state (8+14 split). "
                             "merged22: the 14 env values are appended to observation.state.")
    args = parser.parse_args()

    device = torch.device(args.device)
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    dataset = LeRobotDataset(args.repo_id, root=args.root, video_backend=args.video_backend)

    starts = list(dataset.episode_data_index["from"])
    ends = list(dataset.episode_data_index["to"])
    step = max(1, len(starts) // args.episodes)
    picked = []
    for i in range(0, len(starts), step):
        index = int(starts[i]) + args.offset
        if index < int(ends[i]):
            picked.append(index)
        if len(picked) >= args.episodes:
            break

    print(f"checkpoint={args.checkpoint}")
    print(f"sampling {len(picked)} transport-phase frames (offset {args.offset})\n", flush=True)

    left, right, zeroed = [], [], []
    key = vector_key(args.layout)
    for index in picked:
        sample = dataset[index]
        batch = to_batch(sample, device)
        if key not in sample:
            raise SystemExit(f"{key} not in this dataset; use --layout env14 for the "
                             f"split dataset or merged22 for the 22-dim one")
        env = sample[key].numpy().astype(np.float32)
        base = predict(policy, batch)
        scale = float(base.abs().mean())

        variants = {
            "left": moved_vector(env, 0.18, 0.30, args.layout),
            "right": moved_vector(env, 0.70, 0.60, args.layout),
            "zero": zeroed_vector(env, args.layout),
        }
        results = {}
        for name, vector in variants.items():
            variant = dict(batch)
            variant[key] = torch.from_numpy(vector).unsqueeze(0).to(device)
            changed = predict(policy, variant)
            delta = float((changed - base).abs().mean())
            results[name] = delta / max(scale, 1e-9)
        left.append(results["left"])
        right.append(results["right"])
        zeroed.append(results["zero"])
        ix, iy, _, _ = bag_indices(args.layout)
        print(f"  frame {index:6d}  bag->left {100 * results['left']:5.2f}%   "
              f"bag->right {100 * results['right']:5.2f}%   "
              f"env->zero {100 * results['zero']:5.2f}%   "
              f"(bag was x={env[ix]:.2f} y={env[iy]:.2f}, layout={args.layout})",
              flush=True)

    if not left:
        raise SystemExit("no usable frames")

    print("\n=== summary")
    print(f"  bag moved left  : {100 * np.mean(left):.2f}% of action scale")
    print(f"  bag moved right : {100 * np.mean(right):.2f}% of action scale")
    print(f"  env zeroed      : {100 * np.mean(zeroed):.2f}% of action scale")

    asymmetry = abs(np.mean(left) - np.mean(right))
    print(f"  left/right asymmetry: {100 * asymmetry:.2f} percentage points")

    if np.mean(left) < 0.02 and np.mean(right) < 0.02:
        print("\n>> The policy ignores the bag fields. It cannot place at a moved bag, "
              "and more episodes will not fix it: the network never learned to use "
              "those four numbers.")
    elif asymmetry < 0.01:
        print("\n>> The prediction changes when the bag moves, but by the same amount "
              "in both directions - it is reacting to the vector changing, not to "
              "where the bag is.")
    else:
        print("\n>> The policy does steer by the bag position. A failure to place at "
              "the real bag then points at the live bag estimate differing from the "
              "stored one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
