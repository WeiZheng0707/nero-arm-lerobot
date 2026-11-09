"""Grab one frame from each declared camera and save it as a PNG.

Used to settle "which physical camera is this?" before deploying. Camera
serial -> role (wrist / base) has to be verified on the bench, never assumed:
getting it wrong feeds the policy a wide static scene where it expects a
close-up (or the reverse) without raising any error.

Read-only: opens the cameras, saves images, never touches the robot.

Usage:
    python -X utf8 tools/snapshot_cameras.py ^
        --camera-spec wrist=serial:261622078506 ^
        --camera-spec base=serial:261622071323 ^
        --out outputs/camera_roles
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from lerobot.nero_tools.episode_capture import (
    camera_specs_from_args,
    open_reader,
    parse_camera_spec,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-spec", action="append", required=True,
                        metavar="NAME=serial:...,size:640x480,fps:30")
    parser.add_argument("--camera-type", choices=("realsense", "opencv"), default="realsense")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--out", type=Path, default=Path("outputs/camera_roles"))
    parser.add_argument("--warmup", type=float, default=3.0,
                        help="Seconds to wait for a usable frame before giving up")
    args = parser.parse_args()

    for value in args.camera_spec:
        parse_camera_spec(value)  # fail fast on a malformed spec
    args.camera_width = 640
    args.camera_height = 480
    args.camera_fps = 30

    args.out.mkdir(parents=True, exist_ok=True)
    specs = camera_specs_from_args(args)
    print(f"saving snapshots to {args.out.resolve()}", flush=True)

    failures = 0
    for spec in specs:
        print(f"\n--- opening {spec!r}", flush=True)
        try:
            reader = open_reader(spec, args)
        except Exception as exc:
            print(f"    FAILED to open: {type(exc).__name__}: {exc}", flush=True)
            failures += 1
            continue
        try:
            reader.start()
            deadline = time.monotonic() + args.warmup
            image = None
            while time.monotonic() < deadline:
                latest, _, _ = reader.snapshot()
                if latest is not None:
                    image = latest
                time.sleep(0.05)
            if image is None:
                print(f"    opened but produced no frame in {args.warmup:.0f}s", flush=True)
                failures += 1
                continue
            rgb = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            path = args.out / f"{spec.name}.png"
            # Readers hand back RGB; imwrite expects BGR.
            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            height, width = rgb.shape[:2]
            print(f"    saved {path}  {width}x{height}  "
                  f"mean_brightness={rgb.mean():.1f}", flush=True)
            # Focus on the middle of the frame to avoid floor/table edges. A
            # generous channel-spread threshold lets a green cast be measured
            # instead of filtering all affected pixels out as non-neutral.
            y0, y1 = height // 4, 3 * height // 4
            x0, x1 = width // 4, 3 * width // 4
            pixels = rgb[y0:y1, x0:x1].reshape(-1, 3)
            spread = pixels.max(axis=1).astype('int16') - pixels.min(axis=1).astype('int16')
            neutral = pixels[(pixels.mean(axis=1) >= 150) & (spread <= 100)]
            if len(neutral) >= 1000:
                median = __import__('numpy').median(neutral, axis=0).round().astype(int).tolist()
                delta = int(median[1] - (median[0] + median[2]) / 2)
                print(f"    centre bright-patch RGB median={median}; green-minus-RB-mean={delta} "
                      f"(near 0 is neutral; this is a diagnostic, not a colour correction)",
                      flush=True)
            else:
                print("    centre neutral-patch check unavailable (too few bright table pixels); "
                      "inspect the saved PNG before tuning white balance.", flush=True)
        finally:
            try:
                reader.stop()
            except Exception:
                pass

    print(f"\nDone. failures={failures}", flush=True)
    print("Check camera roles in the PNGs: wrist should be the close gripper view; "
          "base should show the table, bag and full work area.", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
