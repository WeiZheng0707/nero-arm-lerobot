"""Record RGB frames and timestamps from a RealSense camera only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="261622078506")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output", default="outputs/realsense_rgb_episode")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    output = Path(args.output)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    timestamps_path = output / "timestamps.jsonl"
    summary_path = output / "summary.json"

    context = rs.context()
    devices = context.query_devices()
    available = [device.get_info(rs.camera_info.serial_number) for device in devices]
    if args.serial not in available:
        raise RuntimeError(f"RealSense serial {args.serial} not found; available={available}")

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    pipeline.start(config)

    started_host = time.monotonic()
    host_times: list[float] = []
    camera_times: list[float] = []
    frame_count = 0
    dropped = 0
    print(f"RGB recording only: serial={args.serial} output={output}", flush=True)
    print(f"requested={args.width}x{args.height}@{args.fps}; depth=False", flush=True)
    print("No robot connection, enable, disable, mode or motion command.", flush=True)
    try:
        with timestamps_path.open("w", encoding="utf-8") as timestamps_file:
            while time.monotonic() - started_host < args.seconds:
                try:
                    frames = pipeline.wait_for_frames(5000)
                except RuntimeError as exc:
                    dropped += 1
                    print(f"FRAME_WAIT_FAILED: {exc}", flush=True)
                    continue
                color = frames.get_color_frame()
                if not color:
                    dropped += 1
                    continue

                host_time = time.time()
                camera_time = float(color.get_timestamp())
                image_rgb = np.asanyarray(color.get_data())
                frame_name = f"frame_{frame_count:06d}.png"
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(frames_dir / frame_name), image_bgr):
                    raise RuntimeError(f"Failed to write {frames_dir / frame_name}")
                record = {
                    "frame_index": frame_count,
                    "image": f"frames/{frame_name}",
                    "host_time_s": host_time,
                    "camera_timestamp_ms": camera_time,
                    "width": int(color.get_width()),
                    "height": int(color.get_height()),
                }
                timestamps_file.write(json.dumps(record) + "\n")
                timestamps_file.flush()
                host_times.append(host_time)
                camera_times.append(camera_time)
                frame_count += 1
                if frame_count == 1 or frame_count % max(1, args.fps) == 0:
                    elapsed = time.monotonic() - started_host
                    print(f"frames={frame_count} fps={frame_count / elapsed:.2f} "
                          f"camera_timestamp_ms={camera_time:.3f}", flush=True)
    finally:
        pipeline.stop()

    elapsed = time.monotonic() - started_host
    host_intervals = np.diff(host_times) if len(host_times) > 1 else np.array([])
    camera_intervals = np.diff(camera_times) if len(camera_times) > 1 else np.array([])
    summary = {
        "serial": args.serial,
        "requested": {"width": args.width, "height": args.height, "fps": args.fps},
        "depth": False,
        "frame_count": frame_count,
        "wait_or_empty_frame_failures": dropped,
        "elapsed_s": elapsed,
        "host_fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "host_interval_min_s": float(host_intervals.min()) if host_intervals.size else None,
        "host_interval_max_s": float(host_intervals.max()) if host_intervals.size else None,
        "camera_interval_min_ms": float(camera_intervals.min()) if camera_intervals.size else None,
        "camera_interval_max_ms": float(camera_intervals.max()) if camera_intervals.size else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {frame_count} RGB frames to {frames_dir}", flush=True)
    print(f"Saved timestamps to {timestamps_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
