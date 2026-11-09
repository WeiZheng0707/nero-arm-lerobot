"""Measure how much the cube and bag actually moved between episodes.

Everything downstream assumes the demonstrations vary: the policy is expected to
look at the cube and choose a direction. If the cube was in nearly the same place
every time, the recorded trajectories are near-identical, the model can fit them
from proprioception alone, and no input-design change will make it react to
position.

This reads the relation_v2 labels (no video, no hardware) and reports the spread
of cube and bag positions across episodes, plus a scatter plot in base-image
pixels.

Usage:
    python -X utf8 tools\analyse_placement_spread.py ^
        --source outputs/nero_episodes ^
        --label env_state_relation_v2.jsonl ^
        --out outputs/placement_spread
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FRAME_W, FRAME_H = 640, 480


def first_stable(rows, key_prefix, window=120):
    """Median of the first `window` frames where the object is visible.

    The first frame alone is fragile: a detection can drop for a frame or two.
    """
    xs, ys = [], []
    for row in rows[:window]:
        if str(row.get(f"{key_prefix}_visible", "0")) not in ("1", "1.0", "True"):
            continue
        x = row.get(f"{key_prefix}_cx")
        y = row.get(f"{key_prefix}_cy")
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if not xs:
        return None
    return float(np.median(xs)), float(np.median(ys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/nero_episodes"))
    parser.add_argument("--label", default="env_state_relation_v2.jsonl")
    parser.add_argument("--out", type=Path, default=Path("outputs/placement_spread"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cube_pts, bag_pts = [], []
    missing = []
    for episode in sorted(p for p in args.source.glob("episode_*") if p.is_dir()):
        path = episode / args.label
        if not path.exists():
            missing.append(episode.name)
            continue
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(payload.get("relation_v2") or payload)
        cube = first_stable(rows, "cube_base")
        bag = first_stable(rows, "bag_mouth_base")
        if cube:
            cube_pts.append((episode.name, cube[0], cube[1]))
        if bag:
            bag_pts.append((episode.name, bag[0], bag[1]))

    if not cube_pts:
        raise SystemExit("no cube positions found; check --label")

    def report(name, points):
        arr = np.asarray([(x, y) for _, x, y in points], dtype=np.float64)
        px = arr * np.array([FRAME_W, FRAME_H])
        print(f"\n{name}: {len(points)} episodes")
        print(f"  normalised  x mean={arr[:,0].mean():.3f} std={arr[:,0].std():.3f} "
              f"range={arr[:,0].min():.3f}..{arr[:,0].max():.3f}")
        print(f"              y mean={arr[:,1].mean():.3f} std={arr[:,1].std():.3f} "
              f"range={arr[:,1].min():.3f}..{arr[:,1].max():.3f}")
        print(f"  in pixels   x span={px[:,0].max()-px[:,0].min():.0f} px "
              f"(std {px[:,0].std():.0f})")
        print(f"              y span={px[:,1].max()-px[:,1].min():.0f} px "
              f"(std {px[:,1].std():.0f})")
        # Spread relative to the object's own size is the decisive number: if the
        # whole range is a couple of object widths, every demonstration is the
        # same task with noise, and a policy cannot learn to move to the object.
        print(f"  diagonal spread: {np.hypot(px[:,0].max()-px[:,0].min(), px[:,1].max()-px[:,1].min()):.0f} px")
        return arr

    cube_arr = report("cube first position", cube_pts)
    if bag_pts:
        report("bag mouth first position", bag_pts)

    size = 560
    canvas = np.full((size, size, 3), 255, np.uint8)

    def to_px(x, y):
        # Base image coordinates, scaled into the square canvas with a margin.
        return (int(20 + x * (size - 40)), int(20 + y * (size - 40)))

    for x, y in cube_arr:
        cv2.circle(canvas, to_px(x, y), 5, (0, 0, 255), -1)
    if bag_pts:
        bag_arr = np.asarray([(x, y) for _, x, y in bag_pts])
        for x, y in bag_arr:
            cv2.circle(canvas, to_px(x, y), 5, (255, 0, 0), -1)
    cv2.putText(canvas, "red = cube, blue = bag (base image coords)", (10, size - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for frac in (0.25, 0.5, 0.75):
        p = int(20 + frac * (size - 40))
        cv2.line(canvas, (p, 20), (p, size - 20), (220, 220, 220), 1)
        cv2.line(canvas, (20, p), (size - 20, p), (220, 220, 220), 1)
    out_png = args.out / "placement_scatter.png"
    cv2.imwrite(str(out_png), canvas)
    print(f"\nscatter written to {out_png}")
    if missing:
        print(f"episodes without {args.label}: {len(missing)} ({missing[:5]} ...)")
    print("\nHow to read this: a diagonal spread of only a few dozen pixels means the "
          "cube barely moved between episodes, so every demonstration is the same "
          "trajectory and the policy has no reason to look at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
