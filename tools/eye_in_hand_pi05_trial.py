"""Short, supervised PI05 eye-in-hand NERO trial.

This script is intentionally finite-duration and low-speed. It is not a
continuous deployment runner and must be supervised with a physical E-stop.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig
from lerobot.nero_tools.start_pose import check_return_span, load as load_pose

JOINTS = [f"joint_{i}" for i in range(1, 8)]
ACTIONS = [*JOINTS, "gripper"]


class LatestCamera:
    def __init__(self, index: int, backend: str, width: int = 640, height: int = 480, fps: int = 30):
        backend_order = [backend, "msmf" if backend == "dshow" else "dshow", "any"]
        api_by_name = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
        self.capture = None
        candidates = list(dict.fromkeys([index, 1, 0, 2, 3, 4]))
        for backend_name in backend_order:
            api = api_by_name[backend_name]
            for candidate in candidates:
                capture = cv2.VideoCapture(candidate, api)
                if not capture.isOpened():
                    capture.release()
                    continue
                # Keep the camera's native DirectShow format. In particular,
                # do not force MJPG: this device can return colored noise when
                # MJPG is forced. Explicit RGB conversion matches the working
                # episode-capture path.
                capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                capture.set(cv2.CAP_PROP_FPS, fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                frame = None
                for _ in range(12):
                    ok, candidate_frame = capture.read()
                    if ok and candidate_frame is not None:
                        frame = candidate_frame
                        break
                    time.sleep(0.05)
                if frame is not None:
                    self.capture = capture
                    print(
                        f"Camera opened: requested={index}, selected={candidate}, backend={backend_name}",
                        flush=True,
                    )
                    break
                capture.release()
            if self.capture is not None:
                break
        if self.capture is None:
            raise RuntimeError(
                f"No usable eye-in-hand camera; requested={index}, "
                f"backends={backend_order}, tried={candidates}"
            )
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = None
        self.updated = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.updated = time.monotonic()
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            age = time.monotonic() - self.updated
            if self.frame is None or age > 0.5:
                return False, None, age
            return True, self.frame.copy(), age

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.capture.release()


def read_joints(bus) -> np.ndarray:
    values = bus.read()
    result = np.asarray([values[name] for name in JOINTS] + [values["gripper"]], dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite robot feedback")
    return result


def move_to_start(follower, camera, window, start, goal, duration, gripper):
    steps = max(1, int(duration * 50))
    for index in range(1, steps + 1):
        alpha = 0.5 - 0.5 * math.cos(math.pi * index / steps)
        pose = start + (goal - start) * alpha
        follower.send_action(
            {f"{name}.pos": float(value) for name, value in zip(JOINTS, pose, strict=True)}
            | {"gripper.pos": float(gripper)}
        )
        ok, frame, age = camera.read()
        if not ok or age > 0.5:
            raise RuntimeError("Camera became stale during recovery")
        cv2.putText(frame, "RECOVERY - Q/Esc=stop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False
        time.sleep(duration / steps)
    return follower.bus.wait_stable_pose(timeout_s=5.0) is not None


def frame_tensor(frame_bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0)
    return image.unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", choices=("red", "blue"), default="red")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--channel", default="0")
    parser.add_argument("--firmware", default="v121")
    parser.add_argument("--pose-file", type=Path, default=Path("examples/nero/saved_start_pose.json"))
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=0.02)
    parser.add_argument("--max-gripper-step", type=float, default=0.03)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--continuous", action="store_true", help="Run until Q/Esc or Ctrl+C")
    parser.add_argument("--max-frame-age", type=float, default=0.15)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--movement-confirmed", action="store_true", required=True)
    parser.add_argument("--restore-to-start", action="store_true")
    args = parser.parse_args()
    if (not args.continuous and args.max_seconds <= 0) or args.max_frame_age <= 0:
        parser.error("max-seconds and max-frame-age must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    if args.checkpoint is None:
        args.checkpoint = Path(f"outputs/train/eye_in_hand_{args.color}_pi05_lora_full/checkpoints/030000/pretrained_model")
    if args.train_root is None:
        args.train_root = Path(f"outputs/eye_in_hand_{args.color}_cube_{'v2' if args.color == 'red' else 'v1'}/train")
    qpath = args.train_root / "meta" / "quantile_stats.json"
    if not qpath.is_file():
        raise FileNotFoundError(f"Missing PI05 quantile stats: {qpath}")

    device = args.device
    repo_id = f"local/nero_eye_in_hand_{args.color}_cube_train"
    metadata = LeRobotDatasetMetadata(repo_id, root=str(args.train_root))
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = device
    config.pretrained_path = str(args.checkpoint)
    config.quantile_stats_path = str(qpath)
    config.push_to_hub = False
    policy = make_policy(config, ds_meta=metadata)
    policy.reset()
    policy.eval()

    low = np.asarray(metadata.stats["action"]["min"], dtype=np.float64)
    high = np.asarray(metadata.stats["action"]["max"], dtype=np.float64)
    camera = LatestCamera(args.camera_index, args.camera_backend)
    window = f"NERO PI05 {args.color} 20s trial"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)
    follower = None
    stream = None
    try:
        print("PREVIEW: press S to start, Q/Esc to cancel.", flush=True)
        while True:
            ok, frame, age = camera.read()
            if not ok:
                continue
            cv2.putText(frame, "S=start  Q/Esc=cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"PI05 {args.color} supervised 20s trial", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(window, frame)
            # Keep the preview UI at a bounded rate; an unthrottled waitKey(1)
            # loop can starve camera/window event handling on Windows.
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("s"):
                break

        follower = NeroFollower(NeroFollowerConfig(
            id=f"eye_in_hand_{args.color}_pi05_trial", interface="agx_cando", channel=args.channel,
            firmware=args.firmware, use_gripper=True, speed_percent=args.speed_percent,
            # Recovery uses its own smooth interpolation. Applying the per-step
            # policy clamp here as well makes the recovery target lag forever.
            max_relative_target=None, gripper_force_n=1.0,
            gripper_send_threshold=0.01, disable_gripper_on_disconnect=True,
        ))
        connection = {"done": False, "error": None, "stable": None}

        def connect_and_stabilize():
            try:
                follower.connect()
                connection["stable"] = follower.bus.wait_stable_pose(timeout_s=5.0)
                if connection["stable"] is None:
                    raise RuntimeError("Follower feedback did not stabilize")
            except BaseException as exc:
                connection["error"] = exc
            finally:
                connection["done"] = True

        print("CONNECTING: enabling follower and checking feedback; keep clear of the arm.", flush=True)
        connect_thread = threading.Thread(target=connect_and_stabilize, daemon=True)
        connect_thread.start()
        while not connection["done"]:
            ok, frame, age = camera.read()
            if ok:
                cv2.putText(frame, "CONNECTING / ENABLE - Q/Esc does not interrupt CAN", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(frame, "Physical E-stop is the emergency stop", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow(window, frame)
                cv2.waitKey(1)
            time.sleep(0.02)
        connect_thread.join()
        if connection["error"] is not None:
            raise connection["error"]
        current = read_joints(follower.bus)
        goal = np.asarray(load_pose(args.pose_file)["follower"]["joints"], dtype=np.float64)
        if args.restore_to_start:
            check_return_span(current[:7].tolist(), goal.tolist(), "follower")
            distance = float(np.max(np.abs(current[:7] - goal)))
            if distance > math.radians(0.5):
                duration = max(2.0, np.max(np.abs(np.degrees(goal - current[:7]))) / 5.0 * 1.6)
                if not move_to_start(follower, camera, window, current[:7], goal, duration, current[-1]):
                    return 0

        output = Path(f"outputs/eye_in_hand_{args.color}_pi05_trial")
        output.mkdir(parents=True, exist_ok=True)
        path = output / ("trial_" + time.strftime("%Y%m%d_%H%M%S") + ".csv")
        stream = path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(stream)
        writer.writerow(["elapsed_s", "frame_age_s", *[f"target_{x}" for x in ACTIONS], *[f"sent_{x}" for x in ACTIONS], *[f"actual_{x}" for x in ACTIONS]])

        previous = read_joints(follower.bus)
        started = time.perf_counter()
        next_tick = started
        duration_text = "continuous" if args.continuous else f"max {args.max_seconds}s"
        print(f"TRIAL ACTIVE: PI05 {args.color}, 5% speed, {duration_text}; no camera scan.", flush=True)
        while args.continuous or time.perf_counter() - started < args.max_seconds:
            ok, frame, age = camera.read()
            if not ok or age > args.max_frame_age:
                raise RuntimeError(f"Camera frame too old: {age:.3f}s")
            actual_before = read_joints(follower.bus)
            batch = {
                "observation.state": torch.from_numpy(actual_before).float().unsqueeze(0).to(device),
                "observation.images.wrist": frame_tensor(frame, device),
                "task": [f"Pick up the {args.color} block and put it into the brown bag."],
            }
            with torch.inference_mode():
                target = policy.select_action(batch)[0].float().cpu().numpy().astype(np.float64)
            if target.shape != (8,) or not np.isfinite(target).all():
                raise RuntimeError("PI05 returned an invalid 8D action")
            target = np.clip(target, low, high)
            target[:7] = previous[:7] + np.clip(target[:7] - previous[:7], -args.max_relative_target, args.max_relative_target)
            target[7] = previous[7] + float(np.clip(target[7] - previous[7], -args.max_gripper_step, args.max_gripper_step))
            sent = follower.send_action({f"{name}.pos": float(value) for name, value in zip(ACTIONS, target, strict=True)})
            actual = read_joints(follower.bus)
            sent_values = np.asarray([sent[f"{name}.pos"] for name in ACTIONS], dtype=np.float64)
            writer.writerow([time.perf_counter() - started, age, *target, *sent_values, *actual])
            stream.flush()
            previous = sent_values
            view = frame.copy()
            cv2.putText(view, f"ACTIVE PI05 {args.color.upper()} 5% Q/Esc=stop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return 0
            next_tick += 1.0 / 30.0
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
        return 0
    finally:
        if stream:
            stream.close()
        camera.close()
        if follower:
            follower.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
