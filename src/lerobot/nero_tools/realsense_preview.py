"""Read-only Intel RealSense RGB preview; never connects to a robot."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="261622078506")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=None)
    args = parser.parse_args()

    context = rs.context()
    devices = context.query_devices()
    if devices.size() == 0:
        raise RuntimeError("No RealSense device detected")

    serial = args.serial
    if not any(device.get_info(rs.camera_info.serial_number) == serial for device in devices):
        available = [device.get_info(rs.camera_info.serial_number) for device in devices]
        raise RuntimeError(f"RealSense serial {serial} not found; available={available}")

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    pipeline.start(config)

    started = time.monotonic()
    frames = 0
    last_report = started
    try:
        print(f"RGB preview only: serial={serial} requested={args.width}x{args.height}@{args.fps}", flush=True)
        print("No robot connection, enable, disable, mode or motion command.", flush=True)
        while args.seconds is None or time.monotonic() - started < args.seconds:
            frame_set = pipeline.wait_for_frames(5000)
            color = frame_set.get_color_frame()
            if not color:
                raise RuntimeError("RealSense color frame unavailable")
            image = np.asanyarray(color.get_data())
            frames += 1
            now = time.monotonic()
            if now - last_report >= 1.0:
                elapsed = now - started
                print(f"frames={frames} fps={frames / elapsed:.2f} size={image.shape[1]}x{image.shape[0]} "
                      f"timestamp_ms={color.get_timestamp():.3f}", flush=True)
                last_report = now
            cv2.imshow("NERO D435i RGB preview - press Q to quit", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Preview stopped; frames={frames}; elapsed_s={time.monotonic() - started:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
