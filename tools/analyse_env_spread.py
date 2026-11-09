r"""Measure how much the bag moved, and whether the demos reacted to it.

The complaint is that after grasping, the arm always heads for the bag's usual
spot. Two explanations look identical on the robot but are fixed differently:

  a) the bag barely moved during collection, so the demos are all the same task
     and the policy is right to aim for the average;
  b) the bag moved but the demos did not change their transport motion, so there
     is nothing to learn;
  c) both changed, and the policy simply failed to learn the mapping.

This reads only the parquet files (no video, no hardware) and reports:

  * per-episode bag and cube position spread, in normalised units and pixels
  * the correlation between bag position and the transport-phase joints, which
    is the signal a policy would need to copy

Usage:
    python -X utf8 tools\analyse_env_spread.py --split-root D:\1104\nero_cube_300\train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def episode_arrays(path: Path):
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
    parser.add_argument("--transport-window", default="300,460",
                        help="Frame range used as the transport phase")
    parser.add_argument("--min-visible", type=float, default=0.5)
    args = parser.parse_args()

    start, end = (int(v) for v in args.transport_window.split(","))
    files = sorted((args.split_root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.split_root}")

    bag_x, bag_y, cube_x, cube_y = [], [], [], []
    transport_j1, transport_j2, transport_j4 = [], [], []
    used = 0
    for path in files:
        env, state = episode_arrays(path)
        # Columns follow the 14-dim layout:
        # 0..4 cube x/y/w/h/visible, 5..8 bag x/y/w/h, 9..13 ball x/y/w/h/visible.
        # Only the cube and the ball carry a visible flag; the bag is considered
        # present when its width is non-zero, because undetected objects are written
        # as all-zero boxes rather than carrying their own flag.
        bag_width = env[:, 7]
        if float((bag_width > 0.005).mean()) < args.min_visible:
            continue
        bag_cx = env[:, 5] + env[:, 7] / 2.0
        bag_cy = env[:, 6] + env[:, 8] / 2.0
        cube_visible = env[:, 4]
        cube_cx = env[:, 0] + env[:, 2] / 2.0
        cube_cy = env[:, 1] + env[:, 3] / 2.0

        bag_x.append(float(np.median(bag_cx)))
        bag_y.append(float(np.median(bag_cy)))
        if cube_visible.mean() > args.min_visible:
            cube_x.append(float(np.median(cube_cx[:start or None])))
            cube_y.append(float(np.median(cube_cy[:start or None])))

        window = state[start:end]
        if len(window) > 10:
            transport_j1.append(float(window[:, 0].mean()))
            transport_j2.append(float(window[:, 1].mean()))
            transport_j4.append(float(window[:, 3].mean()))
        used += 1

    if used == 0:
        raise SystemExit("no episode had a visible bag")

    def report(name, values):
        arr = np.asarray(values)
        print(f"\n{name}: {len(arr)} episodes")
        print(f"  mean {arr.mean():.4f}  std {arr.std():.4f}  "
              f"range {arr.min():.4f}..{arr.max():.4f}")
        print(f"  span {(arr.max() - arr.min()) * 640:.0f} px (of 640)")

    print(f"episodes analysed: {used}")
    report("bag centre x", bag_x)
    report("bag centre y", bag_y)
    if cube_x:
        report("cube centre x (before transport)", cube_x)
        report("cube centre y (before transport)", cube_y)

    bag = np.asarray(bag_x)
    if len(transport_j1) == len(bag) and len(bag) > 10:
        print("\n=== does the demo motion follow the bag?")
        for name, values in (("joint_1", transport_j1), ("joint_2", transport_j2),
                             ("joint_4", transport_j4)):
            corr = float(np.corrcoef(bag, np.asarray(values))[0, 1])
            print(f"  corr(bag_x, {name} mean over frames {start}-{end}) = {corr:+.3f}")
        print("\nHow to read this:")
        print("  |corr| > 0.6  -> the operator did move differently depending on the bag,")
        print("                  so the mapping exists in the data and the policy failed")
        print("                  to learn it.")
        print("  |corr| < 0.3  -> the transport motion barely changed with the bag, so")
        print("                  there is little for the policy to learn; collect more")
        print("                  episodes with clearly different bag placements.")

    spread_px = (max(bag_x) - min(bag_x)) * 640
    if spread_px < 40:
        print(f"\n>> The bag only moved {spread_px:.0f} px across {used} episodes "
              f"(under 6% of frame width). Aiming at the average position is the "
              f"optimal policy for this data; new recordings are needed.")
    elif spread_px < 120:
        print(f"\n>> The bag moved {spread_px:.0f} px across {used} episodes. There is "
              f"variation but it is narrow; expect the policy to be conservative.")
    else:
        print(f"\n>> The bag moved {spread_px:.0f} px across {used} episodes - enough "
              f"variation to learn from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
