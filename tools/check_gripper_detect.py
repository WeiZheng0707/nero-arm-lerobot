"""Check whether the gripper can be detected in the base view.

The relative-goal features (cube_minus_gripper, bag_minus_gripper) only help if
the gripper's position in the static base view can be measured reliably. This
runs the existing black-region detector over base frames from several episodes
and reports the detection rate, saving overlays so the boxes can be eyeballed.

Read-only: reads PNG frames, writes only the review images.

Usage:
    python -X utf8 tools\check_gripper_detect.py ^
        --source outputs/nero_episodes --episodes 6 --stride 60 ^
        --out outputs/gripper_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from lerobot.nero_tools.color_detector import detect_gripper_in_base


def parse_rect(value):
    parts = [int(p) for p in value.replace(" ", "").split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError(f"rect must be x,y,w,h: {value!r}")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/nero_episodes"))
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--stride", type=int, default=60,
                        help="Sample every Nth frame (30 means one per second)")
    parser.add_argument("--out", type=Path, default=Path("outputs/gripper_check"))
    parser.add_argument("--save-every", type=int, default=3,
                        help="Save an overlay for every Nth sampled frame")
    parser.add_argument("--min-area-ratio", type=float, default=0.0,
                        help="Optional sanity print only; the detector has its own "
                             "area bounds between 150 px and 20%% of the frame")
    parser.add_argument("--exclude-rect", action="append", type=parse_rect, default=None,
                        metavar="X,Y,W,H",
                        help="Static region to ignore, repeatable. The robot's black "
                             "mounting base sits at roughly 400,0,240,210 and is far "
                             "larger than the gripper, so it must be excluded.")
    parser.add_argument("--min-area", type=int, default=600)
    parser.add_argument("--max-area", type=int, default=9000)
    parser.add_argument("--max-value", type=int, default=70,
                        help="Upper HSV V for 'near black'. 105 merged the bag shadow "
                             "with the fingers; 70 keeps only the darker parts.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    episodes = sorted(p for p in args.source.glob("episode_*") if p.is_dir())
    if not episodes:
        raise SystemExit(f"no episode_* under {args.source}")
    episodes = episodes[:args.episodes]

    total = 0
    found = 0
    per_episode = []
    for episode in episodes:
        frames = sorted((episode / "frames" / "base").glob("frame_*.png"))
        if not frames:
            print(f"{episode.name}: no base frames", flush=True)
            continue
        sampled = frames[::args.stride]
        ep_total = ep_found = 0
        for order, path in enumerate(sampled):
            image = cv2.imread(str(path))
            if image is None:
                continue
            box = detect_gripper_in_base(image, exclude_rects=args.exclude_rect or (),
                                         min_area=args.min_area, max_area=args.max_area,
                                         max_value=args.max_value)
            ep_total += 1
            total += 1
            if box is not None:
                ep_found += 1
                found += 1
                if args.save_every and order % args.save_every == 0:
                    overlay = image.copy()
                    cv2.rectangle(overlay, (box["x"], box["y"]),
                                  (box["x"] + box["w"], box["y"] + box["h"]),
                                  (0, 255, 0), 2)
                    cv2.putText(overlay, f"gripper area={box['w'] * box['h']}",
                                (box["x"], max(14, box["y"] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imwrite(str(args.out / f"{episode.name}_{path.stem}.jpg"), overlay)
        rate = 100.0 * ep_found / max(1, ep_total)
        per_episode.append((episode.name, ep_found, ep_total, rate))
        print(f"{episode.name}: gripper detected on {ep_found}/{ep_total} "
              f"({rate:.0f}%)", flush=True)

    if not total:
        raise SystemExit("no frames sampled")

    print("\nSUMMARY", flush=True)
    print(f"  overall detection rate: {found}/{total} ({100.0 * found / total:.1f}%)",
          flush=True)
    rates = sorted(rate for _, _, _, rate in per_episode)
    print(f"  per-episode min/median/max: {rates[0]:.0f}% / "
          f"{rates[len(rates) // 2]:.0f}% / {rates[-1]:.0f}%", flush=True)
    print(f"  overlays written to {args.out.resolve()}", flush=True)
    print("\nRead this as: a relative-goal feature is only usable if the gripper "
          "is seen on most of the approach frames. A low rate means black-region "
          "detection is mixing the gripper with shadows or the arm body.",
          flush=True)
    print("Open a few overlays and check the box really sits on the gripper "
          "fingers, not on the arm or a shadow.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
