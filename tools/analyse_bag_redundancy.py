r"""Is the bag position already recoverable from what the policy does see?

The policy ignores the bag fields (0.07% sensitivity) but does use the images and
the joint state. If, during collection, the bag and the cube were placed in a
correlated way, then the bag position is implied by the cube position and the
model has no reason to read those four numbers. That would explain the ablation
result without any bug, and it would change the fix: the answer is to break the
correlation when recording, not to train longer.

This reads the parquet files only - no video, no hardware.

Usage:
    python -X utf8 tools\analyse_bag_redundancy.py --split-root D:\1104\nero_cube_300\train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# 14-dim env layout
CUBE_X, CUBE_Y, CUBE_W, CUBE_VIS = 0, 1, 2, 4
BAG_X, BAG_Y, BAG_W, BAG_H = 5, 6, 7, 8


def episode_env_state(path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["observation.environment_state", "observation.state"])
    env = np.asarray([np.asarray(v, dtype=np.float32)
                      for v in table.column("observation.environment_state").to_pylist()])
    state = np.asarray([np.asarray(v, dtype=np.float32)
                        for v in table.column("observation.state").to_pylist()])
    return env, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=Path(r"D:\1104\nero_cube_300\train"))
    parser.add_argument("--pre-grasp-frames", type=int, default=150,
                        help="Frames used for the cube position, before it is grasped")
    args = parser.parse_args()

    files = sorted((args.split_root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.split_root}")

    bag_x, bag_y, cube_x, cube_y, j1_start = [], [], [], [], []
    for path in files:
        env, state = episode_env_state(path)
        if float((env[:, BAG_W] > 0.005).mean()) < 0.5:
            continue
        bag_x.append(float(np.median(env[:, BAG_X] + env[:, BAG_W] / 2.0)))
        bag_y.append(float(np.median(env[:, BAG_Y] + env[:, BAG_H] / 2.0)))
        window = slice(0, min(args.pre_grasp_frames, len(env)))
        visible = env[window, CUBE_VIS] > 0.5
        if visible.sum() < 10:
            continue
        cube_x.append(float(np.median(env[window, CUBE_X][visible])))
        cube_y.append(float(np.median(env[window, CUBE_Y][visible])))
        j1_start.append(float(state[0, 0]))

    if len(bag_x) < 20:
        raise SystemExit(f"only {len(bag_x)} usable episodes")

    bag_x = np.asarray(bag_x)
    bag_y = np.asarray(bag_y)
    print(f"episodes analysed: {len(bag_x)}")
    print(f"bag_x  range {bag_x.min():.3f}..{bag_x.max():.3f}   "
          f"bag_y range {bag_y.min():.3f}..{bag_y.max():.3f}\n")

    def corr(name, values):
        values = np.asarray(values)
        if len(values) != len(bag_x):
            print(f"  {name:32s} skipped (missing data)")
            return None
        value = float(np.corrcoef(bag_x, values)[0, 1])
        print(f"  corr(bag_x, {name:28s}) = {value:+.3f}")
        return value

    print("=== is the bag position implied by something the policy already sees?")
    corr("cube_x (pre-grasp)", cube_x)
    corr("cube_y (pre-grasp)", cube_y)
    corr("joint_1 at episode start", j1_start)
    corr("episode index (time order)", np.arange(len(bag_x), dtype=float))

    # How much of bag_x is explained by cube position alone?
    if len(cube_x) == len(bag_x):
        design = np.stack([np.asarray(cube_x), np.ones(len(bag_x))], axis=1)
        coef, *_ = np.linalg.lstsq(design, bag_x, rcond=None)
        residual = bag_x - design @ coef
        r2 = 1.0 - float((residual ** 2).sum() / ((bag_x - bag_x.mean()) ** 2).sum())
        print(f"\n  R^2 of predicting bag_x from cube_x alone: {r2:.3f}")
        if r2 > 0.5:
            print("  >> The bag is largely predictable from the cube. A model that reads")
            print("     the cube already knows where the bag will be, which is exactly why")
            print("     the env bag fields get ignored. Collect with the bag placed")
            print("     independently of the cube, or the mapping cannot be learned.")
        elif r2 > 0.2:
            print("  >> Partially related. Enough correlation to weaken the training "
                  "signal, not enough to explain it away.")
        else:
            print("  >> The bag is NOT predictable from the cube, so the env fields are")
            print("     the only source. The failure is then on the model side and "
                  "bag-in-state is the fix.")


if __name__ == "__main__":
    raise SystemExit(main())
