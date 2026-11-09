r"""Test whether the policy actually reacts to the cube's position in the image.

The earlier ablation zeroed an entire camera, which changes brightness, the table,
the bag and the gripper all at once. That proves the image matters; it does not
prove the policy looks at *where the cube is*.

This keeps everything fixed except the cube: the cube patch is cut out, the hole
is filled with the local table colour, and the patch is pasted at a shifted
position. If an 80 px move (roughly three cube widths) leaves the predicted
action unchanged, the policy is not localising the cube, and no camera or
deployment fix will make it reach for the object.

The environment-state vector is held constant on purpose, so this isolates
image-based localisation.

Read-only: no hardware.

Usage:
    python -X utf8 tools\test_cube_sensitivity.py ^
        --checkpoint D:\NERO\outputs\train\nero_relation_v2_act\checkpoints\050000\pretrained_model ^
        --root D:\NERO\outputs\red_cube_relation_v2\val ^
        --repo-id local/nero_red_cube_relation_v2_val --samples 12 --shift 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


def wrist_cube_center(env_vector: np.ndarray):
    """Wrist cube centre in pixels from the relation_v2 vector, or None."""
    if len(env_vector) < 18 or env_vector[16] < 0.5:
        return None
    dx, dy = float(env_vector[14]), float(env_vector[15])
    return (dx + 1.0) / 2.0, (dy + 1.0) / 2.0


def move_cube(image: np.ndarray, center, shift_x: int, patch: int = 72) -> np.ndarray:
    """Cut the cube out, fill the hole, paste it shifted horizontally."""
    height, width = image.shape[:2]
    cx = int(center[0] * width)
    cy = int(center[1] * height)
    half = patch // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(width, cx + half), min(height, cy + half)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return image
    patch_img = image[y0:y1, x0:x1].copy()
    out = image.copy()
    # Fill the hole with the median colour of a ring just outside the patch: the
    # table is flat white, so this reads as "cube is not here" without inventing
    # structure the policy could latch onto.
    ring = []
    pad = 8
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(width, x1 + pad), min(height, y1 + pad)
    region = image[ry0:ry1, rx0:rx1].reshape(-1, 3)
    ring.append(region)
    fill = np.median(np.concatenate(ring), axis=0).astype(np.uint8)
    out[y0:y1, x0:x1] = fill
    nx0 = int(np.clip(x0 + shift_x, 0, width - (x1 - x0)))
    ny0 = y0
    out[ny0:ny0 + (y1 - y0), nx0:nx0 + (x1 - x0)] = patch_img
    return out


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/nero_red_cube_relation_v2_val")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--stride", type=int, default=400)
    parser.add_argument("--shift", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("outputs/cube_sensitivity"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    dataset = LeRobotDataset(args.repo_id, root=args.root, video_backend=args.video_backend)

    indexes = list(range(0, len(dataset), args.stride))[:args.samples]
    print(f"checkpoint={args.checkpoint}")
    print(f"shift={args.shift} px, samples={len(indexes)}\n", flush=True)

    used = 0
    ratios = []
    for index in indexes:
        sample = dataset[index]
        env = sample["observation.environment_state"].numpy()
        center = wrist_cube_center(env)
        if center is None:
            continue
        batch = to_batch(sample, device)
        key = "observation.images.wrist"
        tensor = batch[key]
        image = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        base_action = predict(policy, batch)
        for name, shift in (("left", -args.shift), ("right", args.shift)):
            moved = move_cube(image, center, shift)
            variant = dict(batch)
            rgb = cv2.cvtColor(moved, cv2.COLOR_BGR2RGB)
            variant[key] = torch.from_numpy(rgb.astype(np.float32) / 255.0) \
                .permute(2, 0, 1).unsqueeze(0).to(device)
            moved_action = predict(policy, variant)
            delta = float((moved_action - base_action).abs().mean())
            scale = float(base_action.abs().mean())
            ratios.append(delta / max(scale, 1e-9))
            print(f"  sample {index:6d} cube->{name:5s} |Δaction|={delta:.5f} "
                  f"({100 * delta / max(scale, 1e-9):5.2f}% of action scale)", flush=True)

        if used < 3:
            cv2.imwrite(str(args.out / f"shifted_sample_{index}.png"),
                        np.hstack([image, move_cube(image, center, args.shift)]))
        used += 1

    if not ratios:
        raise SystemExit("no samples had a visible wrist cube; nothing to compare")

    mean_ratio = float(np.mean(ratios))
    print(f"\ncube moved {args.shift} px out of 640 ({100 * args.shift / 640:.0f}% of the frame)")
    print(f"mean |Δaction| = {100 * mean_ratio:.2f}% of action scale")
    if mean_ratio < 0.02:
        print(">> The prediction barely moves when the cube moves. The policy does not "
              "localise the cube; it uses the image as a scene descriptor. Retraining "
              "the same way will not fix this.")
    elif mean_ratio < 0.08:
        print(">> Weak response. The policy notices the cube but does not steer by it.")
    else:
        print(">> The policy does steer by the cube's position. A deployment failure "
              "then points at the live camera view differing from training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
