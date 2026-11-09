r"""Report GOP structure and seek cost of the dataset videos.

Random access to a video frame requires decoding from the previous keyframe.
With keyint=2 the cost is at most two frames; with the encoder default (often
250+) every sampled frame can cost a hundred decodes, which shows up as training
that sits at 0% GPU while the log advances every few minutes.

This reads a handful of files and prints: frame count, keyframe count, average
GOP length, and the decoder's nominal frame rate.

Usage:
    python -X utf8 tools\inspect_video_gop.py --root D:\1104\nero_cube_300\train --count 3
"""

from __future__ import annotations

import argparse
from pathlib import Path


def inspect(path: Path) -> dict:
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = 0
    keyframes = 0
    first_key_pts = None
    last_key_pts = None
    key_pts = []
    for packet in container.demux(stream):
        if packet.is_keyframe:
            keyframes += 1
            if packet.pts is not None:
                key_pts.append(packet.pts)
        for _ in packet.decode():
            frames += 1
    container.close()
    if key_pts:
        first_key_pts = key_pts[0]
        last_key_pts = key_pts[-1]
    gop = (frames / keyframes) if keyframes else float("inf")
    return {"frames": frames, "keyframes": keyframes, "avg_gop": gop,
            "fps": float(stream.average_rate) if stream.average_rate else 0.0,
            "codec": stream.codec_context.name,
            "pix_fmt": stream.codec_context.pix_fmt,
            "first_key_pts": first_key_pts, "last_key_pts": last_key_pts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Dataset split folder containing videos/")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--camera", default="observation.images.wrist")
    args = parser.parse_args()

    videos = sorted((args.root / "videos").glob(f"chunk-*/{args.camera}/*.mp4"))
    if not videos:
        raise SystemExit(f"no videos under {args.root / 'videos'}")
    print(f"inspecting {min(args.count, len(videos))} of {len(videos)} files\n")

    worst = 0.0
    for path in videos[:args.count]:
        info = inspect(path)
        worst = max(worst, info["avg_gop"])
        print(f"{path.name}")
        print(f"  codec={info['codec']} pix_fmt={info['pix_fmt']} fps={info['fps']:.1f}")
        print(f"  frames={info['frames']} keyframes={info['keyframes']} "
              f"avg_gop={info['avg_gop']:.1f} frames")
        print(f"  first_key_pts={info['first_key_pts']} last_key_pts={info['last_key_pts']}",
              flush=True)

    print(f"\nworst average GOP: {worst:.1f} frames")
    if worst > 30:
        print(">> Large GOP. Every random sample can require decoding tens of frames "
              "per camera, which starves the GPU. Re-encoding with keyint=2 fixes the "
              "stall at the cost of a bigger file.")
    elif worst > 5:
        print(">> Moderate GOP. Seeking is not free but should not stall for minutes.")
    else:
        print(">> Short GOP. Video seeking is not the bottleneck here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
