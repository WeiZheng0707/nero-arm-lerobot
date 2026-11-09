"""Manual-step eye-in-hand camera scan using follower joint_3 only."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig
from lerobot.nero_tools.start_pose import check_return_span, load as load_pose


JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
def normalize_key(value):
    return value & 0xFF


def open_camera(index, backend):
    api = cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_MSMF
    candidates = list(dict.fromkeys([index, 1, 0, 2, 3]))
    for candidate in candidates:
        camera = cv2.VideoCapture(candidate, api)
        if not camera.isOpened():
            camera.release()
            continue
        for _ in range(5):
            ok, frame = camera.read()
            if ok and frame is not None:
                print(f"Camera opened: requested={index}, selected={candidate}", flush=True)
                return camera, candidate
            time.sleep(0.02)
        camera.release()
    raise RuntimeError(f"No usable camera found; requested={index}, tried={candidates}")


def read_joints(bus):
    values = bus.read()
    return np.asarray([values[name] for name in JOINT_NAMES], dtype=np.float64)


def show(camera, window, status):
    ok, frame = camera.read()
    if not ok or frame is None:
        raise RuntimeError("Camera frame read failed")
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (0, 255, 255), 2)
    cv2.putText(frame, "S=next scan pose  Q/Esc=stop  gripper is untouched",
                (10, 466), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    cv2.imshow(window, frame)
    return normalize_key(cv2.waitKeyEx(1))


def wait_for_s(camera, window, status):
    while True:
        key = show(camera, window, status)
        if key in (ord("q"), 27):
            return False
        if key == ord("s"):
            return True
        time.sleep(0.01)


def move_with_preview(follower, camera, window, start, target, duration, status):
    steps = max(1, int(duration * 50))
    period = duration / steps
    for index in range(1, steps + 1):
        alpha = 0.5 - 0.5 * math.cos(math.pi * index / steps)
        joints = start + (target - start) * alpha
        follower.send_action({f"{name}.pos": float(value)
                              for name, value in zip(JOINT_NAMES, joints, strict=True)})
        key = show(camera, window, f"MOVING {status} {index}/{steps}")
        if key in (ord("q"), 27):
            return False
        time.sleep(period)
    return follower.bus.wait_stable_pose(timeout_s=5.0) is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--channel", default="0")
    parser.add_argument("--firmware", default="v121")
    parser.add_argument("--pose-file", type=Path, default=Path(
        "examples/nero/saved_start_pose.json"))
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--return-speed-deg-s", type=float, default=2.0)
    parser.add_argument("--scan-deg", type=float, default=30.0,
                        help="Joint_3 scan angle on each side")
    parser.add_argument("--scan-speed-deg-s", type=float, default=4.0)
    parser.add_argument("--movement-confirmed", action="store_true")
    parser.add_argument("--large-return-confirmed", action="store_true",
                        help="Explicitly allow a saved-pose return over 60 degrees")
    args = parser.parse_args()
    if not args.movement_confirmed:
        parser.error("--movement-confirmed is required")
    if not 1 <= args.speed_percent <= 10:
        parser.error("--speed-percent must be in [1, 10]")
    if not 0 < args.return_speed_deg_s <= 5:
        parser.error("--return-speed-deg-s must be in (0, 5]")
    if not 1 <= args.scan_deg <= 45:
        parser.error("--scan-deg must be in [1, 45]")
    if not 0 < args.scan_speed_deg_s <= 5:
        parser.error("--scan-speed-deg-s must be in (0, 5]")

    recovery = np.asarray(load_pose(args.pose_file)["follower"]["joints"], dtype=np.float64)
    backend = cv2.CAP_DSHOW if args.camera_backend == "dshow" else cv2.CAP_MSMF
    camera, selected = open_camera(args.camera_index, backend)
    window = "NERO eye-in-hand joint_3 scan"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    follower = None
    try:
        follower = NeroFollower(NeroFollowerConfig(
            id="nero_joint3_camera_scan", interface="agx_cando",
            channel=args.channel, firmware=args.firmware, use_gripper=False,
            speed_percent=args.speed_percent, max_relative_target=None,
            feedback_timeout_s=0.5,
        ))
        print(f"CONNECTING FOLLOWER: channel={args.channel}; firmware={args.firmware}; "
              f"camera={selected}; speed={args.speed_percent}%", flush=True)
        follower.connect()
        current = read_joints(follower.bus)
        recovery_delta = np.degrees(recovery - current)
        max_delta = float(np.max(np.abs(recovery_delta)))
        print(f"CURRENT TO RECOVERY delta_deg={np.round(recovery_delta, 3).tolist()} "
              f"max={max_delta:.3f}", flush=True)
        if max_delta > 60.0 and not args.large_return_confirmed:
            raise RuntimeError(
                "Follower is more than 60 degrees from the saved recovery pose. "
                "Move it manually closer or rerun with --large-return-confirmed "
                "only after checking the full path is clear."
            )
        check_return_span(current.tolist(), recovery.tolist(), "follower") if max_delta <= 60.0 else None
        if not wait_for_s(camera, window, "PREFLIGHT: S=return to recovery  Q/Esc=stop"):
            return 0
        scan_order = [(f"joint3_minus_{args.scan_deg:g}deg", -args.scan_deg),
                      ("recovery_center", 0.0),
                      (f"joint3_plus_{args.scan_deg:g}deg", args.scan_deg),
                      ("recovery_center", 0.0)]
        span_deg = float(np.max(np.abs(np.degrees(recovery - current))))
        duration = max(2.0, span_deg / args.return_speed_deg_s * 1.6)
        print(f"RETURN TO RECOVERY: max_delta={span_deg:.2f}deg duration={duration:.1f}s",
              flush=True)
        if not move_with_preview(follower, camera, window, current, recovery,
                                 duration, "RETURN TO RECOVERY"):
            raise RuntimeError("Follower did not stabilize at recovery pose")
        print(f"RECOVERY REACHED. Press S to scan joint_3 -{args.scan_deg:g} degrees.", flush=True)

        for step_index, (name, offset_deg) in enumerate(scan_order, start=1):
            if not wait_for_s(camera, window,
                              f"READY: S=move {name} ({step_index}/{len(scan_order)})"):
                return 0
            current = read_joints(follower.bus)
            target = recovery.copy()
            target[2] += math.radians(offset_deg)
            check_return_span(current.tolist(), target.tolist(), name)
            delta_deg = float(np.max(np.abs(np.degrees(target - current))))
            duration = max(1.5, delta_deg / args.scan_speed_deg_s * 1.6)
            print(f"MOVE {name}: joint_3 offset={offset_deg:+.1f}deg; "
                  f"max_delta={delta_deg:.2f}deg duration={duration:.1f}s", flush=True)
            if not move_with_preview(follower, camera, window, current, target,
                                     duration, name):
                raise RuntimeError(f"No stable feedback after {name}")
            stable = follower.bus.present_positions()
            if stable is None:
                raise RuntimeError(f"No stable feedback after {name}")
            stable = np.asarray(stable, dtype=np.float64)
            error_deg = float(np.max(np.abs(np.degrees(stable - target))))
            print(f"REACHED {name}: joint_3={math.degrees(stable[2]):+.2f}deg "
                  f"max_error={error_deg:.3f}deg", flush=True)
        print("SCAN COMPLETE. Press Q/Esc to exit.", flush=True)
        while True:
            if show(camera, window, "DONE: Q/Esc=finish") in (ord("q"), 27):
                return 0
    finally:
        camera.release()
        if follower is not None:
            follower.disconnect()
            print("Follower disconnected; joints disabled. Gripper was never initialized.", flush=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by operator.", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", flush=True)
        raise SystemExit(1)
