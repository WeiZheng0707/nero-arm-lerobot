"""Low-speed ACT trial for the eye-in-hand red-block policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy
from lerobot.configs.types import NormalizationMode
from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig
from lerobot.nero_tools.start_pose import check_return_span, load as load_pose


JOINTS = [f"joint_{i}" for i in range(1, 8)]
ACTIONS = [*JOINTS, "gripper"]


def load_policy(checkpoint, device, dataset):
    raw = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    fields = ACTConfig.__dataclass_fields__
    config = ACTConfig(**{key: value for key, value in raw.items() if key in fields})
    config.device = device
    config.push_to_hub = False
    config.pretrained_path = str(checkpoint)
    config.normalization_mapping = {
        key: NormalizationMode(value)
        for key, value in raw["normalization_mapping"].items()
    }
    checkpoint_inputs = set(raw.get("input_features", {}))
    dataset_inputs = {key for key in dataset.meta.features if key.startswith("observation.")}
    if checkpoint_inputs != dataset_inputs:
        raise RuntimeError(f"Checkpoint/dataset input mismatch: {checkpoint_inputs} != {dataset_inputs}")
    policy = make_policy(cfg=config, ds_meta=dataset.meta)
    policy.to(device)
    policy.eval()
    return policy


class LatestCamera:
    def __init__(self, index, backend, width=640, height=480, fps=30):
        api = cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_MSMF
        self.capture = None
        for candidate in dict.fromkeys([index, 1, 0, 2, 3, 4]):
            capture = cv2.VideoCapture(candidate, api)
            if not capture.isOpened():
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(10):
                ok, frame = capture.read()
                if ok and frame is not None:
                    self.capture = capture
                    print(f"Camera opened: requested={index}, selected={candidate}", flush=True)
                    break
                time.sleep(0.05)
            if self.capture is not None:
                break
            capture.release()
        if self.capture is None:
            raise RuntimeError("No usable eye-in-hand camera")
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
        ok, frame, _ = self.read_with_age()
        return ok, frame

    def read_with_age(self):
        with self.lock:
            age = time.monotonic() - self.updated
            if self.frame is None or age > 0.5:
                return False, None, age
            return True, self.frame.copy(), age

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.capture.release()


def joints(bus):
    values = bus.read()
    result = np.asarray([values[name] for name in JOINTS], dtype=np.float64)
    grip = float(values["gripper"])
    if not np.isfinite(result).all() or not np.isfinite(grip):
        raise RuntimeError("Non-finite robot feedback")
    return np.r_[result, grip]


def move_preview(follower, camera, window, start, target, duration, gripper):
    steps = max(1, int(duration * 50))
    for index in range(1, steps + 1):
        alpha = 0.5 - 0.5 * math.cos(math.pi * index / steps)
        pose = start + (target - start) * alpha
        follower.send_action({f"{name}.pos": float(value)
                              for name, value in zip(JOINTS, pose, strict=True)} |
                             {"gripper.pos": float(gripper)})
        ok, frame = camera.read()
        if ok:
            cv2.putText(frame, "CAMERA SCAN / RECOVERY", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            cv2.putText(frame, "Q/Esc=stop", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                return False
        time.sleep(duration / steps)
    return follower.bus.wait_stable_pose(timeout_s=5.0) is not None


def tensor(frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    return image.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/train/eye_in_hand_red_cube_act_v1/checkpoints/020000/pretrained_model"))
    parser.add_argument("--train-root", type=Path, default=Path(
        "outputs/eye_in_hand_red_cube_v1/train"))
    parser.add_argument("--color", choices=("red", "blue"), default="red")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--channel", default="0")
    parser.add_argument("--firmware", default="v121")
    parser.add_argument("--pose-file", type=Path, default=Path("examples/nero/saved_start_pose.json"))
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=0.02,
                        help="Maximum per-command joint change in radians")
    parser.add_argument("--control-fps", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--continuous", action="store_true",
                        help="Run until Q/Esc or Ctrl+C")
    parser.add_argument("--return-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--scan-deg", type=float, default=30.0)
    parser.add_argument("--scan-speed-deg-s", type=float, default=4.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--movement-confirmed", action="store_true")
    parser.add_argument("--restore-to-start", action="store_true")
    args = parser.parse_args()
    if not args.movement_confirmed:
        parser.error("--movement-confirmed is required")
    if args.control_fps <= 0:
        parser.error("--control-fps must be positive")
    max_relative = args.max_relative_target
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    repo_id = f"local/nero_eye_in_hand_{args.color}_cube_train"
    metadata = LeRobotDatasetMetadata(repo_id, root=str(args.train_root))
    policy = load_policy(args.checkpoint, device, type("Dataset", (), {"meta": metadata})())
    policy.reset()
    low = np.asarray(metadata.stats["action"]["min"], dtype=np.float64)
    high = np.asarray(metadata.stats["action"]["max"], dtype=np.float64)
    camera = LatestCamera(args.camera_index, args.camera_backend)
    window = f"NERO eye-in-hand {args.color} ACT trial"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    follower = None
    stream = None
    try:
        print("PREVIEW: camera is live; press S to enable and start, Q/Esc to cancel.", flush=True)
        while True:
            ok, frame = camera.read()
            if not ok:
                continue
            view = frame.copy()
            cv2.putText(view, "S=start  Q/Esc=cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,255,255), 2)
            cv2.putText(view, f"Eye-in-hand {args.color} policy; low-speed trial", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27): return 0
            if key == ord('s'): break

        follower = NeroFollower(NeroFollowerConfig(
            id=f"eye_in_hand_{args.color}_act_trial", interface="agx_cando", channel=args.channel,
            firmware=args.firmware, use_gripper=True, speed_percent=args.speed_percent,
            max_relative_target=max_relative, gripper_force_n=1.0,
            gripper_send_threshold=0.001, disable_gripper_on_disconnect=True,
        ))
        follower.connect()
        stable = follower.bus.wait_stable_pose(timeout_s=5.0)
        if stable is None: raise RuntimeError("Follower feedback did not stabilize")
        current = np.asarray(stable, dtype=np.float64)
        goal = np.asarray(load_pose(args.pose_file)["follower"]["joints"], dtype=np.float64)
        if args.restore_to_start:
            check_return_span(current.tolist(), goal.tolist(), "follower")
            if np.max(np.abs(current - goal)) > math.radians(.5):
                duration = max(2.0, np.max(np.abs(np.degrees(goal-current))) / args.return_speed_deg_s * 1.6)
                if not move_preview(follower, camera, window, current, goal, duration,
                                    joints(follower.bus)[-1]):
                    raise RuntimeError("Follower did not stabilize after recovery")

        current = joints(follower.bus)[:7]
        scan_offsets = (-args.scan_deg, 0.0, args.scan_deg, 0.0)
        print(f"CAMERA SCAN: joint_3 offsets={scan_offsets}deg", flush=True)
        for offset in scan_offsets:
            target = goal.copy()
            target[2] += math.radians(offset)
            duration = max(1.5, abs(offset) / args.scan_speed_deg_s * 1.6)
            if not move_preview(follower, camera, window, current, target, duration, joints(follower.bus)[-1]):
                return 0
            current = target.copy()
            time.sleep(0.5)
        print("CAMERA SCAN COMPLETE: ACT phase starting.", flush=True)

        output = Path(f"outputs/eye_in_hand_{args.color}_act_trial")
        output.mkdir(parents=True, exist_ok=True)
        path = output / ("trial_" + time.strftime("%Y%m%d_%H%M%S") + ".csv")
        stream = path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(stream)
        writer.writerow([
            "elapsed_s", "loop_period_s", "processing_s", "inference_s", "frame_age_s",
            *[f"target_{x}" for x in ACTIONS],
            *[f"sent_{x}" for x in ACTIONS],
            *[f"actual_{x}" for x in ACTIONS],
        ])
        started_ns = time.perf_counter_ns()
        next_tick_ns = started_ns
        previous_cycle_ns = None
        period_ns = int(1_000_000_000 / args.control_fps)
        print(f"TRIAL ACTIVE: 5% speed, eye-in-hand {args.color} ACT, Q/Esc=stop.", flush=True)
        while args.continuous or (time.perf_counter_ns() - started_ns) / 1e9 < args.max_seconds:
            cycle_started_ns = time.perf_counter_ns()
            loop_period_s = (
                (cycle_started_ns - previous_cycle_ns) / 1e9
                if previous_cycle_ns is not None else 0.0
            )
            previous_cycle_ns = cycle_started_ns
            ok, frame, frame_age = camera.read_with_age()
            if not ok: raise RuntimeError("Camera frame stale")
            state = joints(follower.bus)
            batch = {"observation.state": torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                     "observation.images.wrist": tensor(frame, device)}
            inference_started_ns = time.perf_counter_ns()
            with torch.inference_mode():
                target = policy.select_action(batch)[0].float().cpu().numpy().astype(np.float64)
            inference_s = (time.perf_counter_ns() - inference_started_ns) / 1e9
            if target.shape != (8,) or not np.isfinite(target).all(): raise RuntimeError("Invalid policy action")
            target = np.clip(target, low, high); target[-1] = np.clip(target[-1], 0, 1)
            sent = follower.send_action({f"{name}.pos": float(value) for name, value in zip(ACTIONS, target, strict=True)})
            actual = joints(follower.bus)
            sent_values = np.asarray([sent[f"{name}.pos"] for name in ACTIONS], dtype=np.float64)
            processing_s = (time.perf_counter_ns() - cycle_started_ns) / 1e9
            writer.writerow([
                (time.perf_counter_ns() - started_ns) / 1e9,
                loop_period_s, processing_s, inference_s, frame_age,
                *target, *sent_values, *actual,
            ])
            stream.flush()
            view = frame.copy(); cv2.putText(view, f"ACTIVE {args.color.upper()} ACT 5%  Q/Esc=stop", (10,30), cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,255),2)
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xff in (ord('q'),27): return 0
            next_tick_ns += period_ns
            delay_ns = next_tick_ns - time.perf_counter_ns()
            if delay_ns > 0:
                time.sleep(delay_ns / 1e9)
            else:
                next_tick_ns = time.perf_counter_ns()
        return 0
    finally:
        if stream: stream.close()
        camera.close()
        if follower: follower.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt: raise SystemExit(130)
    except Exception as exc: print(f"ERROR: {exc}"); raise SystemExit(1)
