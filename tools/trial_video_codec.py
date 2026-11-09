"""Encode one episode's frames with several codecs and report real sizes.

The dataset's storage choice is a trade between disk, encode time and decode
compatibility, and the right answer depends on the footage. Rather than guessing
from bitrate tables, this encodes the same frames with each candidate and prints
measured sizes plus an extrapolation to the whole capture.

Read-only with respect to the capture: only new files under --out are written.

Usage:
    python -X utf8 tools/trial_video_codec.py --episode outputs/nero_episodes/episode_0000 ^
        --camera base --codecs h264:20 h264:23 libsvtav1:30 libsvtav1:35 ^
        --out outputs/video_trial --episodes-total 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lerobot.datasets.video_utils import encode_video_frames


def parse_codec(value: str):
    name, _, crf = value.partition(":")
    if not name:
        raise argparse.ArgumentTypeError(f"expected CODEC:CRF, got {value!r}")
    if not crf:
        raise argparse.ArgumentTypeError(f"expected CODEC:CRF, got {value!r}")
    return name, int(crf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--camera", default="base", help="wrist or base")
    parser.add_argument("--codecs", nargs="+", type=parse_codec,
                        default=[("h264", 20), ("libsvtav1", 30)])
    parser.add_argument("--out", type=Path, default=Path("outputs/video_trial"))
    parser.add_argument("--episodes-total", type=int, default=300)
    parser.add_argument("--cameras-per-episode", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    frames_dir = args.episode / "frames" / args.camera
    if not frames_dir.is_dir():
        raise SystemExit(f"no frames under {frames_dir}")
    frame_count = len(list(frames_dir.glob("frame_*.png")))
    if frame_count == 0:
        raise SystemExit(f"no PNG frames under {frames_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"episode={args.episode.name} camera={args.camera} frames={frame_count} "
          f"duration={frame_count / args.fps:.1f}s", flush=True)
    print(f"extrapolating to {args.episodes_total} episodes x "
          f"{args.cameras_per_episode} cameras\n", flush=True)

    rows = []
    for name, crf in args.codecs:
        target = args.out / f"{args.episode.name}_{args.camera}_{name}_crf{crf}.mp4"
        started = time.monotonic()
        try:
            encode_video_frames(frames_dir, target, args.fps, vcodec=name, crf=crf,
                                overwrite=True)
        except Exception as exc:
            elapsed = time.monotonic() - started
            print(f"{name:12s} crf={crf:<3d} FAILED after {elapsed:5.1f}s: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows.append((name, crf, None, elapsed))
            continue
        elapsed = time.monotonic() - started
        size_mb = target.stat().st_size / 1e6
        rows.append((name, crf, size_mb, elapsed))
        print(f"{name:12s} crf={crf:<3d} size={size_mb:7.2f} MB  "
              f"encode={elapsed:5.1f}s  "
              f"-> {size_mb * args.episodes_total * args.cameras_per_episode / 1000:.2f} GB "
              f"for the whole capture", flush=True)

    good = [row for row in rows if row[2] is not None]
    if not good:
        print("\nNo codec succeeded. Check which encoders this PyAV build ships "
              "with:\n  python -c \"import av; print([k for k in av.codecs_available "
              "if '264' in k or 'svt' in k or '265' in k])\"", flush=True)
        return 1

    print("\nSUMMARY", flush=True)
    print(f"  {'codec':12s} {'crf':>4s} {'MB/ep':>8s} {'total GB':>9s} {'enc s':>7s} "
          f"{'all-episodes':>13s}", flush=True)
    for name, crf, size_mb, elapsed in good:
        total_gb = size_mb * args.episodes_total * args.cameras_per_episode / 1000
        total_seconds = elapsed * args.episodes_total * args.cameras_per_episode
        print(f"  {name:12s} {crf:>4d} {size_mb:>8.2f} {total_gb:>9.2f} "
              f"{elapsed:>7.1f} {total_seconds / 60:>10.1f} min", flush=True)
    print("\nEncode time is single-stream; the converter runs one stream at a time. "
          "Add decode-compatibility as a separate consideration: h264 plays "
          "everywhere, AV1 needs a recent decoder.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
