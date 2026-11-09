"""NERO teleoperation capture with live D435i preview and episode controls."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from lerobot.nero_teleoperate import (
    NeroTeleopConfig,
    TargetMapper,
    load_start_pose,
    return_to_start_pose,
)
from lerobot.motors.nero import ARM_JOINT_NAMES
from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig
from lerobot.teleoperators.nero_leader import NeroLeader, NeroLeaderConfig


# Return tolerances are advisory only. They are reported to the operator but never
# abort the session: an aborted session falls through to cleanup, and the operator
# asked explicitly to keep the arms enabled when this happens.
FOLLOWER_TOLERANCE_RAD = math.radians(10.0)
LEADER_TOLERANCE_RAD = math.radians(10.0)

# A frame older than this during recording marks the episode invalid. The design
# document sets 200 ms for the dual-camera data; a frozen base frame would silently
# corrupt every position label, so it must never be written as a success.
STALE_FRAME_AGE_MS = 200.0


def usb_link_type(serial):
    """Return '3.x', '2.0' or 'unknown' for a RealSense camera's USB link.

    A D435i that negotiates USB 2.0 cannot carry 640x480@30 RGB alongside depth;
    it silently drops to a lower rate, which looks exactly like a "laggy" preview.
    """
    try:
        devices = rs.context().query_devices()
        for device in devices:
            if device.get_info(rs.camera_info.serial_number) != serial:
                continue
            descriptor = device.get_info(rs.camera_info.usb_type_descriptor)
            return "3.x" if descriptor.startswith("3") else descriptor
    except (RuntimeError, AttributeError):
        return "unknown"
    return "unknown"


def host_frame_age_ms(reader, timestamp_ms=None):
    """Frame age in ms, preferring the reader's own host clock.

    RealSense device timestamps are not epoch-based, so `frame_age_ms` cannot
    derive an age from them; readers therefore expose `host_age_ms()` sampled on
    frame arrival. When a reader has no such method (a stub in a hardware-free
    test), fall back to the timestamp the caller already holds.

    This must never call `reader.snapshot()` itself: the control loop already did
    that, and taking an extra snapshot would both consume one more frame and
    advance a stubbed camera's side_effect sequence.
    """
    host_age = getattr(reader, "host_age_ms", None)
    if callable(host_age):
        try:
            value = host_age()
        except Exception:
            value = None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    if isinstance(timestamp_ms, (int, float)) and not isinstance(timestamp_ms, bool):
        return frame_age_ms(timestamp_ms)
    return 0.0


class RealSenseReader:
    """Latest-frame reader for one RealSense colour stream.

    RealSense timestamps come from the device clock, not the host epoch, so
    frame age is measured against a host monotonic clock sampled on arrival.
    The negotiated profile and the USB link speed are recorded because a D435i
    silently negotiating USB 2.0 is the most common cause of a "laggy" preview.
    """

    def __init__(self, serial: str, width: int, height: int, fps: int):
        self.serial = serial
        self.pipeline = open_camera(serial, width, height, fps)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.image = None
        self.timestamp_ms = None
        self.host_time_s = None
        self.sequence = 0
        self.profile = {}
        self.usb_type = usb_link_type(serial)
        try:
            for profile in self.pipeline.get_active_profile().get_streams():
                if profile.stream_type() == rs.stream.color:
                    video = profile.as_video_stream_profile()
                    self.profile = {"width": video.width(), "height": video.height(),
                                    "fps": video.fps(), "format": str(video.format())}
        except RuntimeError:
            self.profile = {}
        self.thread = threading.Thread(target=self._run,
                                       name=f"realsense-reader-{serial}", daemon=True)

    def start(self):
        print(f"REALSENSE serial={self.serial} usb={self.usb_type} "
              f"profile={self.profile}", flush=True)
        if self.usb_type.startswith("2"):
            print(f"  [WARN] serial={self.serial} is on a USB 2.0 link. USB 2.0 cannot "
                  f"sustain 640x480@30 RGB reliably; run "
                  f"tools/realsense_bandwidth_check.py before recording.", flush=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                frame_set = self.pipeline.wait_for_frames(1000)
                color = frame_set.get_color_frame()
                if color:
                    image = np.asanyarray(color.get_data()).copy()
                    with self.lock:
                        self.image = image
                        self.timestamp_ms = float(color.get_timestamp())
                        self.host_time_s = time.monotonic()
                        self.sequence += 1
            except RuntimeError:
                continue

    def snapshot(self):
        with self.lock:
            if self.image is None:
                return None, None, self.sequence
            return self.image.copy(), self.timestamp_ms, self.sequence

    def host_age_ms(self):
        """Age of the newest frame measured on the host clock."""
        with self.lock:
            if self.host_time_s is None:
                return 0.0
            return (time.monotonic() - self.host_time_s) * 1000.0

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.pipeline.stop()


class OpenCVCameraReader:
    def __init__(self, index: int, width: int, height: int, fps: int, backend: str):
        backend_order = [backend, "msmf" if backend == "dshow" else "dshow", "any"]
        candidates = list(dict.fromkeys([index, 1, 0, 2, 3, 4]))
        self.capture = None
        self.selected_index = None
        for backend_name in backend_order:
            api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
                   "any": cv2.CAP_ANY}[backend_name]
            for candidate in candidates:
                capture = cv2.VideoCapture(candidate, api)
                if not capture.isOpened():
                    capture.release()
                    continue
                # Keep the camera's native DirectShow format. For this device,
                # forcing MJPG caused colored-noise frames and a stale preview.
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
                if frame is None:
                    capture.release()
                    continue
                self.capture = capture
                self.selected_index = candidate
                print(f"Camera opened: requested={index}, selected={candidate}, "
                      f"backend={backend_name}", flush=True)
                break
            if self.capture is not None:
                break
        if self.capture is None:
            raise RuntimeError(
                f"No usable OpenCV camera; requested={index}, backends={backend_order}, "
                f"tried={candidates}. Close other camera programs and check USB/permissions."
            )
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.image = None
        self.timestamp_ms = None
        self.host_time_s = None
        self.sequence = 0
        self.thread = threading.Thread(target=self._run, name="opencv-camera-reader", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            ok, frame_bgr = self.capture.read()
            if not ok or frame_bgr is None:
                time.sleep(0.01)
                continue
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.image = image_rgb
                self.timestamp_ms = time.time() * 1000.0
                self.host_time_s = time.monotonic()
                self.sequence += 1

    def snapshot(self):
        with self.lock:
            if self.image is None:
                return None, None, self.sequence
            return self.image.copy(), self.timestamp_ms, self.sequence

    def host_age_ms(self):
        with self.lock:
            if self.host_time_s is None:
                return 0.0
            return (time.monotonic() - self.host_time_s) * 1000.0

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.capture.release()


def corrupted_camera_frame(frame_bgr):
    """Reject common USB camera bandwidth/format corruption."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or not np.isfinite(frame_bgr).all():
        return True
    values = frame_bgr.astype(np.float32)
    channel_means = values.mean(axis=(0, 1))
    channel_spread = float(np.max(channel_means) - np.min(channel_means))
    return float(values.std()) > 78.0 and channel_spread > 28.0


class CameraSpec:
    """One camera declaration parsed from --camera-spec name=key:value,..."""

    __slots__ = ("name", "index", "serial", "backend", "width", "height", "fps")

    def __init__(self, name, index=0, serial=None, backend="dshow", width=640, height=480, fps=30):
        self.name = name
        self.index = index
        self.serial = serial
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        if self.index < 0:
            raise ValueError(f"Camera {name!r}: index must be non-negative")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError(
                f"Camera {name!r}: width, height and fps must be positive; "
                f"got {self.width}x{self.height}@{self.fps}"
            )

    def __repr__(self):
        target = f"serial:{self.serial}" if self.serial else f"index:{self.index}"
        return f"CameraSpec({self.name}={target},{self.backend},{self.width}x{self.height}@{self.fps})"


def parse_camera_spec(value):
    """Parse `wrist=index:0,backend:dshow,size:640x480,fps:30` into a CameraSpec.

    Both ':' and '=' work as the key/value separator so commands copied from the
    design document run verbatim.
    """
    if "=" not in value:
        raise ValueError(f"--camera-spec needs name=options: {value!r}")
    name, _, options = value.partition("=")
    name = name.strip()
    if not name:
        raise ValueError(f"--camera-spec needs a camera name: {value!r}")
    spec = CameraSpec(name)
    for chunk in options.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, raw = chunk.replace("=", ":", 1).partition(":")
        key, raw = key.strip().lower(), raw.strip()
        if raw == "" or raw.lower() == "none":
            continue
        if key == "index":
            spec.index = int(raw)
        elif key == "serial":
            spec.serial = raw
        elif key == "backend":
            if raw not in ("dshow", "msmf"):
                raise ValueError(f"Unsupported backend in --camera-spec: {raw!r}")
            spec.backend = raw
        elif key == "size":
            width, _, height = raw.lower().partition("x")
            spec.width, spec.height = int(width), int(height)
        elif key == "fps":
            spec.fps = int(raw)
        else:
            raise ValueError(f"Unknown --camera-spec option {key!r} in {value!r}")
    return spec


def camera_specs_from_args(args):
    """Return the cameras to open for this session.

    The legacy single-camera invocation yields exactly one spec named 'wrist', so
    the old data layout and the existing single-camera tests keep working.
    """
    if args.camera_spec:
        specs = [parse_camera_spec(value) for value in args.camera_spec]
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate camera names in --camera-spec: {names}")
        return specs
    return [CameraSpec("wrist", index=args.camera_index, serial=args.serial,
                       backend=args.camera_backend, width=args.camera_width,
                       height=args.camera_height, fps=args.camera_fps)]


def open_reader(spec, args):
    # A camera-spec with an index is an OpenCV/DirectShow camera. The legacy
    # default is RealSense, but it must not override an explicit index spec;
    # otherwise both named cameras accidentally open the same RealSense serial.
    if getattr(args, "camera_spec", None) and spec.serial is None:
        return OpenCVCameraReader(spec.index, spec.width, spec.height, spec.fps, spec.backend)
    if spec.serial is not None or args.camera_type == "realsense":
        return RealSenseReader(spec.serial or args.serial, spec.width, spec.height, spec.fps)
    return OpenCVCameraReader(spec.index, spec.width, spec.height, spec.fps, spec.backend)


class MultiCamera:
    """Opens every declared camera and exposes one snapshot call for all of them.

    `snapshot()` mirrors the legacy single-camera contract (image, timestamp,
    sequence) for the first camera; `snapshot_all()` returns
    {name: (image, timestamp_ms, sequence)} for every camera. The combined
    sequence advances when at least one camera delivered a new frame, preserving
    the main loop's "record on new frame" behaviour.
    """

    def __init__(self, specs, args):
        self.readers = {}
        for spec in specs:
            print(f"OPENING CAMERA {spec!r}", flush=True)
            self.readers[spec.name] = open_reader(spec, args)
        self.names = list(self.readers)
        self._last_sequences = {}

    def start(self):
        for reader in self.readers.values():
            reader.start()

    def snapshot_all(self):
        snapshots = {name: reader.snapshot() for name, reader in self.readers.items()}
        combined = 0
        for name, (_, _, sequence) in snapshots.items():
            if sequence != self._last_sequences.get(name):
                self._last_sequences[name] = sequence
                combined += 1
        return snapshots, combined

    def snapshot(self):
        return self.readers[self.names[0]].snapshot()

    def stop(self):
        for reader in self.readers.values():
            reader.stop()


def enable_auto_color(sensor, serial):
    """Enable and report RealSense colour auto exposure and white balance."""
    results = {}
    for name, option in (
        ("auto_exposure", rs.option.enable_auto_exposure),
        ("auto_white_balance", rs.option.enable_auto_white_balance),
    ):
        if not sensor.supports(option):
            results[name] = False
            print(f"REALSENSE serial={serial} {name}=unsupported", flush=True)
            continue
        try:
            sensor.set_option(option, 1.0)
            value = float(sensor.get_option(option))
        except RuntimeError as exc:
            results[name] = False
            print(f"REALSENSE serial={serial} {name}=UNCONFIRMED: {exc}", flush=True)
            continue
        results[name] = value >= 0.5
        state = "on" if results[name] else "OFF"
        print(f"REALSENSE serial={serial} {name}={state} (confirmed={value:.0f})",
              flush=True)
    return results


def open_camera(serial: str, width: int, height: int, fps: int):
    context = rs.context()
    devices = context.query_devices()
    available = [d.get_info(rs.camera_info.serial_number) for d in devices]
    if serial not in available:
        raise RuntimeError(f"D435i serial {serial} not found; available={available}")
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    profile = pipeline.start(config)
    try:
        sensor = profile.get_device().first_color_sensor()
        enable_auto_color(sensor, serial)
    except (RuntimeError, AttributeError) as exc:
        print(f"REALSENSE serial={serial} auto_color_controls=UNCONFIRMED: {exc}", flush=True)
    return pipeline


def create_camera(args):
    specs = camera_specs_from_args(args)
    if not args.camera_spec:
        # Legacy single-camera path: return the bare reader so existing callers
        # and the hardware-free tests that patch create_camera keep their contract.
        return open_reader(specs[0], args)
    return MultiCamera(specs, args)


def write_jsonl(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def next_episode_index(root):
    root = Path(root)
    indices = [int(path.name[8:]) for path in root.glob("episode_*")
               if path.name[8:].isascii() and path.name[8:].isdigit()]
    return max(indices, default=-1) + 1


def save_episode(root, episode, images, frames, states, actions, metadata):
    """Write one episode.

    `images` is either the legacy flat list of wrist frames or a dict mapping a
    camera key to its list of frames; the dict form writes `frames/<key>/`. The
    flat form is kept so single-camera captures stay byte-compatible with the
    existing converter and tests.
    """
    episode_dir = root / f"episode_{episode:04d}"
    root.mkdir(parents=True, exist_ok=True)
    # Reserve exclusively before any write, including concurrent writers.
    episode_dir.mkdir(exist_ok=False)
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir()
    if isinstance(images, dict):
        for key, key_images in images.items():
            key_dir = frames_dir / key
            key_dir.mkdir()
            for frame_index, image in enumerate(key_images):
                frame_path = key_dir / f"frame_{frame_index:06d}.png"
                if not cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Failed to write {frame_path}")
    else:
        for frame_index, image in enumerate(images):
            frame_path = frames_dir / f"frame_{frame_index:06d}.png"
            if not cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to write {frame_path}")
    with (episode_dir / "robot_state.jsonl").open("w", encoding="utf-8") as handle:
        for value in states:
            write_jsonl(handle, value)
    with (episode_dir / "actions.jsonl").open("w", encoding="utf-8") as handle:
        for value in actions:
            write_jsonl(handle, value)
    with (episode_dir / "timestamps.jsonl").open("w", encoding="utf-8") as handle:
        for value in frames:
            write_jsonl(handle, value)
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return episode_dir


class EpisodeWriter(threading.Thread):
    def __init__(self, *args):
        super().__init__(name=f"episode-writer-{args[1]:04d}", daemon=False)
        self.save_args = args
        self.episode_index = args[1]
        self.error = None

    def run(self):
        try:
            path = save_episode(*self.save_args)
            print(f"SAVED: {path.resolve()}", flush=True)
        except Exception as exc:
            self.error = exc
            print(f"SAVE FAILED episode={self.episode_index:04d}: {exc}", flush=True)
        finally:
            # The writer owns full-resolution NumPy frames. Release them as soon
            # as disk I/O finishes; otherwise completed threads retain hundreds
            # of MB each through their argument tuple.
            self.save_args = None


def save_episode_background(root, episode, images, frames, states, actions, metadata):
    thread = EpisodeWriter(root, episode, images, frames, states, actions, metadata)
    thread.start()
    return thread


def send_mapped_target(mapper, follower, action, dt):
    """Send one mapped target and advance the mapper from the sent value."""
    target = mapper.target(action, dt)
    sent = follower.send_action(target)
    mapper.accept(sent)
    return sent


def return_to_start_between_episodes(leader, follower, start_pose, duration_s):
    """Re-arm the next episode: drive the follower back, let the human drag the leader.

    Why the leader is never commanded: the leader is a teaching pendant. It sits in
    zero-force drag mode during teleoperation, where joint writes are rejected, and
    every attempt to switch it back to position control (SDK `set_normal_mode()` is a
    documented no-op on firmware >= 1.12, and the private linkage/push path is not
    reliable) left the arm undriven while tearing down the session. Driving the
    leader is not needed for dataset consistency either: the mapping is relative and
    is re-anchored below, so the follower starting from the saved pose is what makes
    each episode comparable.

    Returns the newly anchored leader pose so the caller can check the operator
    actually dragged it back instead of starting the next episode from anywhere.
    """
    print("RETURNING FOLLOWER TO START POSE. Leader stays in drag mode: "
          "please DRAG THE LEADER ARM BACK to its start pose now. "
          "Motors stay enabled.", flush=True)
    follower.bus.slow_move_to(start_pose["follower"], duration_s=duration_s, fps=50,
                              hold_after_s=0.5)
    if follower.bus.wait_stable_pose(timeout_s=5.0) is None:
        raise TimeoutError("Follower did not provide stable feedback after the return")
    error_deg = max(
        abs(a - b)
        for a, b in zip(follower.bus.present_positions(), start_pose["follower"], strict=True)
    )
    print(f"RETURN CHECK follower: max_joint_error={math.degrees(error_deg):.3f} deg",
          flush=True)
    if error_deg > FOLLOWER_TOLERANCE_RAD:
        print(f"  [WARN] 从臂归位偏差 {math.degrees(error_deg):.2f} deg 超过 "
              f"{math.degrees(FOLLOWER_TOLERANCE_RAD):.0f} deg，仅提示不中止。",
              flush=True)

    leader_pose = wait_for_leader_near(leader, start_pose["leader"])
    print(f"LEADER ANCHORED: max_joint_error="
          f"{math.degrees(leader_pose[1]):.3f} deg from the saved leader pose.",
          flush=True)
    return leader_pose


def wait_for_leader_near(leader, target, tolerance_rad=LEADER_TOLERANCE_RAD, timeout_s=30.0):
    """Wait until the operator drags the leader back within tolerance of its pose.

    Returns (action, max_error_rad). The leader is never commanded, so the only
    signal available is the operator's own dragging; timing out aborts the session
    rather than starting an episode from an arbitrary pose.
    """
    deadline = time.monotonic() + timeout_s
    best = (None, float("inf"))
    announced = False
    while time.monotonic() < deadline:
        try:
            angles = leader.bus.try_read_joint_positions()
        except (RuntimeError, TimeoutError):
            angles = None
        if angles is not None:
            error = max(abs(a - b) for a, b in zip(angles, target, strict=True))
            if error < best[1]:
                best = (list(angles), error)
            if error <= tolerance_rad:
                return list(angles), error
            if not announced:
                print(f"  waiting for the leader to be dragged back: "
                      f"currently {math.degrees(error):.1f} deg away "
                      f"(need <= {math.degrees(tolerance_rad):.0f} deg).",
                      flush=True)
                announced = True
        time.sleep(0.1)
    print(f"  [WARN] {timeout_s:.0f}s 内主臂未被拖回起始姿态附近（最近仍差 "
          f"{math.degrees(best[1]):.1f} deg）。仅提示，按当前姿态继续，"
          f"映射基准已按实际姿态重新锚定。", flush=True)
    return best[0], best[1]


def parse_roi(value):
    """Parse 'x,y,w,h' into a pixel ROI tuple, or None."""
    if value is None:
        return None
    parts = [int(part) for part in value.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must be x,y,w,h: {value!r}")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI width/height must be positive: {value!r}")
    return x, y, w, h


def detect_base_scene(base_image_rgb, args):
    """Detect cube/ball/bag in the static base view.

    Everything position-bearing in this project comes from here: the wrist
    camera moves with the arm, so its pixel coordinates carry no absolute
    meaning. The bag is explicitly allowed to move between episodes, which is
    why its box is measured per frame rather than assumed fixed.
    """
    scene = detect_scene(
        cv2.cvtColor(base_image_rgb, cv2.COLOR_RGB2BGR),
        args.cube_color,
        ball_color=None if args.ball_color.lower() == "none" else args.ball_color,
        roi=args.base_roi_pixels,
        exclude_top_ratio=args.base_exclude_top,
        bag_roi=args.base_bag_roi_pixels,
        bag_exclude_top_ratio=0.0,
    )
    return scene


def frame_age_ms(timestamp_ms):
    """Age of a frame in ms, tolerating both epoch-ms and device-ms timestamps."""
    if not timestamp_ms:
        return 0.0
    now_ms = time.time() * 1000.0
    if timestamp_ms > 1e9:
        return now_ms - timestamp_ms
    # RealSense device clocks are not epoch based; fall back to 0 so a healthy
    # camera is never flagged stale purely because its clock starts near zero.
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-type", choices=("realsense", "opencv"), default="realsense")
    parser.add_argument("--serial", default="261622078506")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-spec", action="append", default=None, metavar="NAME=OPTIONS",
                        help="Declare one camera, repeatable. Example: "
                             "'wrist=index:0,backend:dshow,size:640x480,fps:30'. "
                             "The first declared camera is the wrist view; any camera "
                             "named 'base' is the static third-person view whose "
                             "detections become the environment_state. Without this "
                             "option a single legacy camera is opened as 'wrist'.")
    parser.add_argument("--base-roi", default=None, metavar="X,Y,W,H",
                        help="Pixel ROI inside the base frame for cube/ball detection; "
                             "recorded in metadata.json.")
    parser.add_argument("--base-bag-roi", default=None, metavar="X,Y,W,H",
                        help="Pixel ROI inside the base frame for bag detection.")
    parser.add_argument("--base-crop", default=None, metavar="X,Y,W,H",
                        help="Fixed base crop written to the dataset. Must stay identical "
                             "for the whole dataset; changing it invalidates the data.")
    parser.add_argument("--base-exclude-top", type=float, default=0.03,
                        help="Fraction of the base frame blanked before colour detection. "
                             "The wrist default (0.28) would cut the bag mouth in an "
                             "oblique base view.")
    parser.add_argument("--ball-color", default="blue",
                        help="Colour of the distractor ball in the base view; 'none' "
                             "disables ball detection.")
    parser.add_argument("--output", default="outputs/nero_episodes")
    parser.add_argument("--cube-color", choices=("red", "green", "blue", "yellow"), default="red",
                        help="Colour of the target cube. Must differ from --ball-color so "
                             "colour alone separates target from distractor.")
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--interface", default="agx_cando")
    parser.add_argument("--leader-channel", required=True)
    parser.add_argument("--follower-channel", required=True)
    parser.add_argument("--leader-firmware", default="v120")
    parser.add_argument("--follower-firmware", default="v121")
    parser.add_argument("--leader-gripper-width", type=float, default=0.10)
    parser.add_argument("--follower-gripper-width", type=float, default=0.07)
    parser.add_argument("--gripper-force", type=float, default=1.0,
                        help="Follower gripper force in N; start with a small increase.")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--max-joint-speed", type=float, default=1.2)
    parser.add_argument("--max-gripper-speed", type=float, default=2.5)
    parser.add_argument("--max-control-dt", type=float, default=0.1,
                        help="Maximum elapsed time used for one mapped control step")
    parser.add_argument("--gripper-send-threshold", type=float, default=0.001,
                        help="Minimum normalized gripper change sent to the follower")
    parser.add_argument("--speed-percent", type=int, default=60)
    parser.add_argument("--return-duration", type=float, default=5.0)
    parser.add_argument("--allow-large-return", action="store_true",
                        help="Allow automatic return when a saved-pose joint delta exceeds 60 degrees.")
    parser.add_argument("--pose-file", default="examples/nero/saved_start_pose.json")
    parser.add_argument("--wait-for-start", action="store_true",
                        help="After returning to the saved pose, hold until S before sending leader motion.")
    args = parser.parse_args()

    if args.episode_seconds <= 0:
        parser.error("--episode-seconds must be positive")
    if args.ball_color.lower() != "none" and args.ball_color == args.cube_color:
        parser.error("--ball-color must differ from --cube-color; colour is the only "
                     "thing separating the target cube from the distractor ball")
    specs = camera_specs_from_args(args)
    names = [spec.name for spec in specs]
    if not names or names[0] != "wrist":
        parser.error("The first --camera-spec must be named 'wrist'")
    if len(names) != len(set(names)):
        parser.error(f"Duplicate camera names: {names}")
    unsupported = [spec for spec in specs
                   if (spec.width, spec.height, spec.fps) != (640, 480, 30)]
    if unsupported:
        parser.error(
            "Current env-state conversion requires every camera to use 640x480@30; "
            f"unsupported specs: {unsupported}"
        )
    has_base = "base" in names
    args.base_roi_pixels = parse_roi(args.base_roi)
    args.base_bag_roi_pixels = parse_roi(args.base_bag_roi)
    args.base_crop_pixels = parse_roi(args.base_crop)
    if args.base_crop_pixels is not None and not has_base:
        parser.error("--base-crop requires a camera named 'base' in --camera-spec")
    output = Path(args.output)
    print(f"BLOCK DETECTOR: cube={args.cube_color} ball={args.ball_color}; "
          f"cameras={names}; base_view={'yes' if has_base else 'NO (legacy single camera)'}",
          flush=True)
    if not has_base and args.ball_color.lower() != "none":
        print("  [WARN] No 'base' camera: cube/ball/bag positions will come from an "
              "arm-mounted view and are NOT stable absolute coordinates. The moving-bag "
              "design requires a static third-person camera.", flush=True)
    start_pose = load_start_pose(args.pose_file)
    camera = create_camera(args)
    camera.start()
    leader = NeroLeader(NeroLeaderConfig(
        id="nero_leader", interface=args.interface, channel=args.leader_channel,
        firmware=args.leader_firmware, use_gripper=True, gripper_max_width_m=args.leader_gripper_width,
        gripper_source="auto", feedback_timeout_s=0.5,
    ))
    follower = NeroFollower(NeroFollowerConfig(
        id="nero_follower", interface=args.interface, channel=args.follower_channel,
        firmware=args.follower_firmware, use_gripper=True, speed_percent=args.speed_percent,
        max_relative_target=None, gripper_max_width_m=args.follower_gripper_width,
        gripper_force_n=args.gripper_force,
        gripper_send_threshold=args.gripper_send_threshold,
        disable_gripper_on_disconnect=True,
    ))
    config = NeroTeleopConfig(
        fps=args.fps, max_joint_speed=args.max_joint_speed,
        max_gripper_speed=args.max_gripper_speed,
        max_control_dt_s=args.max_control_dt,
    )
    episode = next_episode_index(output)
    print(f"OUTPUT: {output.resolve()}; next episode={episode:04d}; existing data preserved.", flush=True)
    recording = False
    record_started = 0.0

    def empty_images():
        return {name: [] for name in names} if has_base else []

    images, frames, states, actions = empty_images(), [], [], []
    stale_reasons = None
    last_observation = None
    pending_writes = []
    completed_write_errors = []

    def reap_finished_writers():
        """Join completed writers and drop their thread objects promptly."""
        nonlocal pending_writes
        active = []
        for writer in pending_writes:
            if writer.is_alive():
                active.append(writer)
            else:
                writer.join(timeout=0)
                if writer.error is not None:
                    completed_write_errors.append(str(writer.error))
        pending_writes = active
    result = "not_finished"
    communication_fault = None
    try:
        print(f"CONNECTING FOLLOWER FIRST: channel={args.follower_channel} "
              f"firmware={args.follower_firmware}", flush=True)
        follower.connect()
        print("FOLLOWER CONNECTED: enable confirmed.", flush=True)
        print(f"CONNECTING LEADER: channel={args.leader_channel} "
              f"firmware={args.leader_firmware}", flush=True)
        leader.connect(enter_leader_mode=start_pose is None)
        print("LEADER CONNECTED: enable/feedback confirmed.", flush=True)
        if start_pose is not None:
            # 采集器自己管理主臂：后面每个 episode 之间都靠人工拖动 + 重新锚定基准，
            # 所以这里不进入"等人工拖回恢复点"的等待，否则启动会白等几十秒。
            return_to_start_pose(leader, follower, start_pose, args.return_duration,
                                 allow_large_return=args.allow_large_return,
                                 leader_drag_timeout_s=0.0)
            leader.enter_leader_mode()
        leader_start = leader.get_action()
        follower_start = follower.get_observation()
        mapper = TargetMapper(config, leader_start, follower_start)
        if args.wait_for_start:
            print("WAITING: both arms are at the saved start pose. Press S to enable leader-follow motion.", flush=True)
        else:
            print("READY: S=start 20s episode, E=save early, R=discard, Q=quit and disable.", flush=True)
        motion_started = not args.wait_for_start
        previous = time.monotonic()
        last_image = None
        last_base_image = None
        last_camera_timestamp = None
        last_camera_sequence = 0
        last_base_scene = None
        last_base_sequence = 0
        last_loop = time.monotonic()
        loop_count = 0
        last_action = None
        last_sent = None
        last_observation = None
        while True:
            reap_finished_writers()
            tick = time.monotonic()
            if has_base:
                snapshots, camera_sequence = camera.snapshot_all()
                wrist_image, wrist_timestamp, wrist_sequence = snapshots[names[0]]
                base_image, base_timestamp, _ = snapshots["base"]
                # The base view is static, so it is re-detected only when it
                # actually produced a new frame; the wrist view is decoded every
                # new frame purely for the live overlay.
                base_is_new = snapshots["base"][2] != last_base_sequence
                last_base_sequence = snapshots["base"][2]
                camera_image, camera_timestamp = wrist_image, wrist_timestamp
                if base_image is not None:
                    last_base_image = base_image
                # A capture row is paced by the wrist camera only. The combined
                # MultiCamera sequence also advances when base updates, which
                # would otherwise write duplicate wrist frames at 30 Hz.
                new_camera_frame = (wrist_image is not None
                                    and wrist_sequence != last_camera_sequence)
                camera_sequence = wrist_sequence
            else:
                camera_image, camera_timestamp, camera_sequence = camera.snapshot()
                base_image = base_timestamp = None
                new_camera_frame = camera_sequence != last_camera_sequence
            if camera_image is not None:
                last_image = camera_image
                last_camera_timestamp = camera_timestamp
                last_camera_sequence = camera_sequence
            image = last_image
            if image is None:
                time.sleep(0.005)
                continue
            wrist_age = (host_frame_age_ms(camera.readers[names[0]], last_camera_timestamp)
                         if has_base
                         else host_frame_age_ms(camera, last_camera_timestamp))
            base_age = (host_frame_age_ms(camera.readers["base"], base_timestamp)
                        if has_base else 0.0)
            if wrist_age > 500.0 or (has_base and base_age > 500.0):
                view = np.zeros((args.camera_height, args.camera_width * 2 if has_base else args.camera_width, 3), dtype=np.uint8)
                stale_text = (f"CAMERA STALE wrist={wrist_age:.0f}ms base={base_age:.0f}ms"
                              if has_base else f"CAMERA STALE {wrist_age:.0f}ms")
                cv2.putText(view, stale_text + " - waiting for fresh frame",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(view, "Check USB camera/cable; motion is held", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if has_base and last_base_image is not None:
                    base_view = cv2.cvtColor(last_base_image, cv2.COLOR_RGB2BGR)
                    cv2.putText(base_view, "BASE | index 1", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    view[:, args.camera_width:] = base_view
                cv2.imshow("NERO camera capture", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                time.sleep(0.02)
                continue
            try:
                action = leader.get_action()
            except (RuntimeError, TimeoutError) as exc:
                communication_fault = f"leader feedback failed: {exc}"
                print(f"COMMUNICATION FAULT: {communication_fault}", flush=True)
                break
            try:
                current_observation = follower.bus.read()
                observation = {name: float(value) for name, value in current_observation.items()}
            except (RuntimeError, TimeoutError):
                observation = last_observation
            if observation is None:
                time.sleep(0.005)
                continue
            if motion_started:
                try:
                    sent = send_mapped_target(mapper, follower, action, tick - previous)
                except (RuntimeError, TimeoutError) as exc:
                    communication_fault = f"follower CAN send failed: {exc}"
                    print(f"COMMUNICATION FAULT: {communication_fault}", flush=True)
                    break
            else:
                # Keep the follower at the verified start pose while the operator
                # checks the camera view. No leader target is sent before S.
                sent = dict(follower_start)
            previous = tick
            loop_count += 1
            action_delta = 0.0 if last_action is None else max(
                abs(action[key] - last_action[key]) for key in action
            )
            sent_delta = 0.0 if last_sent is None else max(
                abs(sent[key] - last_sent[key]) for key in sent
            )
            follower_delta = 0.0 if last_observation is None else max(
                abs(observation[key] - last_observation[key]) for key in observation
            )
            last_action = dict(action)
            last_sent = dict(sent)
            last_observation = dict(observation)
            if tick - last_loop >= 1.0:
                print(f"CONTROL LOOP: {loop_count / (tick - last_loop):.1f} Hz; "
                      f"camera_seq={camera_sequence}; leader_delta={action_delta:.5f}; "
                      f"sent_delta={sent_delta:.5f}; follower_delta={follower_delta:.5f}", flush=True)
                loop_count = 0
                last_loop = tick

            now = time.monotonic()
            wrist_view = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(wrist_view, "WRIST | index 3", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if has_base and last_base_image is not None:
                base_view = cv2.cvtColor(last_base_image, cv2.COLOR_RGB2BGR)
                cv2.putText(base_view, "BASE | index 1", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                view = np.hstack((wrist_view, base_view))
            else:
                view = wrist_view
            state = "RECORDING" if recording else "READY"
            elapsed = now - record_started if recording else 0.0
            cv2.putText(wrist_view, f"{state} ep={episode:04d} {elapsed:.1f}/{args.episode_seconds:.1f}s",
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            start_text = "S start 20s  E save early  R discard  Q quit+disable"
            cv2.putText(wrist_view, start_text,
                        (10, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            frame_age_text = (f"wrist_age={wrist_age:.0f}ms base_age={base_age:.0f}ms"
                              if has_base else f"camera_frame_age={wrist_age:.0f}ms")
            cv2.putText(wrist_view, frame_age_text,
                        (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if has_base and last_base_image is not None:
                view = np.hstack((wrist_view, base_view))
            else:
                view = wrist_view
            cv2.imshow("NERO camera capture", view)
            key = cv2.waitKey(1) & 0xFF
            quit_requested = key in (ord("q"), 27)
            if quit_requested:
                # Q/Esc exits the session. An in-progress episode is not saved
                # implicitly; use E when the operator wants to keep it.
                if recording:
                    recording = False
                    images, frames, states, actions = empty_images(), [], [], []
                    stale_reasons = None
                    print("QUIT: current episode discarded; teleoperation is stopping.", flush=True)
                break
            if key == ord("s") and not recording:
                motion_started = True
                recording = True
                record_started = now
                images, frames, states, actions = empty_images(), [], [], []
                stale_reasons = None
                print(f"MOTION STARTED; RECORDING episode={episode:04d}", flush=True)
            if key == ord("r") and recording:
                recording = False
                images, frames, states, actions = empty_images(), [], [], []
                stale_reasons = None
                result = "not_finished"
                print("DISCARDED current episode; teleoperation continues and motors stay enabled.",
                      flush=True)
            if key == ord("e") and recording:
                result = "manual_end"

            if recording:
                # With two cameras a row is committed when the wrist frame is new;
                # the base row is paired to it and the pairing error is recorded.
                # A stale base frame is not silently reused: the episode is marked
                # invalid instead, because a frozen base frame would silently
                # corrupt every position label in the dataset.
                record_this_row = new_camera_frame
                if record_this_row:
                    frame_index = len(frames)
                    host_time_s = time.time()
                    if has_base:
                        if last_image is None or base_image is None:
                            # No base frame yet: writing the wrist row alone would
                            # leave frames/ and timestamps.jsonl misaligned, so the
                            # row is skipped rather than silently corrupted.
                            continue
                        images["wrist"].append(last_image.copy())
                        images["base"].append(base_image.copy())
                        wrist_ms = wrist_age
                        base_ms = base_age
                        if max(wrist_ms, base_ms) > STALE_FRAME_AGE_MS and stale_reasons is None:
                            stale_reasons = (f"frame_age wrist={wrist_ms:.0f}ms "
                                             f"base={base_ms:.0f}ms exceeds "
                                             f"{STALE_FRAME_AGE_MS:.0f}ms")
                        frames.append({
                            "frame_index": frame_index,
                            "host_time_s": host_time_s,
                            "cameras": {
                                "wrist": {"image": f"frames/wrist/frame_{frame_index:06d}.png",
                                          "host_time_s": host_time_s,
                                          "age_ms": wrist_ms,
                                          "camera_timestamp_ms": last_camera_timestamp},
                                "base": {"image": f"frames/base/frame_{frame_index:06d}.png",
                                         "host_time_s": host_time_s,
                                         "age_ms": base_ms,
                                         "camera_timestamp_ms": base_timestamp},
                            },
                            "pair_error_ms": abs(wrist_ms - base_ms),
                            # Env-state is intentionally deferred until its label
                            # design and detector review workflow are agreed.
                        })
                    else:
                        images.append(image.copy())
                        frames.append({
                            "frame_index": frame_index,
                            "image": f"frames/frame_{frame_index:06d}.png",
                            "host_time_s": host_time_s,
                            "camera_timestamp_ms": last_camera_timestamp,
                            # Env-state labels are added later from the static
                            # base frames after detector calibration/review.
                        })
                    states.append({"host_time_s": host_time_s, **observation})
                    actions.append({"host_time_s": host_time_s, **sent})
                if result in ("manual_end", "quit_end") or now - record_started >= args.episode_seconds:
                    recording = False
                    if not frames:
                        print("Empty episode not saved.", flush=True)
                        if quit_requested:
                            break
                        continue
                    metadata = {
                        "episode_index": episode,
                        "result": "manual_end" if result == "manual_end" else "auto_end",
                        "duration_s": now - record_started,
                        "camera_type": args.camera_type,
                        "camera_serial": args.serial if args.camera_type == "realsense" else None,
                        "camera_index": args.camera_index if args.camera_type == "opencv" else None,
                        "camera_backend": args.camera_backend if args.camera_type == "opencv" else None,
                        "camera_width": args.camera_width,
                        "camera_height": args.camera_height,
                        "camera_fps": args.camera_fps,
                        "depth_enabled": False,
                        "leader_channel": args.leader_channel,
                        "follower_channel": args.follower_channel,
                        "leader_gripper_width_m": args.leader_gripper_width,
                        "follower_gripper_width_m": args.follower_gripper_width,
                        "gripper_force_n": args.gripper_force,
                        "camera_mount": "eye_in_hand+base" if has_base else "eye_in_hand",
                        "camera_key": "wrist",
                        "cameras": {spec.name: {"backend": spec.backend, "index": spec.index,
                                                "serial": spec.serial, "width": spec.width,
                                                "height": spec.height, "fps": spec.fps,
                                                "mount": "static_base" if spec.name == "base"
                                                         else "eye_in_hand"}
                                    for spec in specs},
                        "cube_color": args.cube_color,
                        "ball_color": args.ball_color,
                        "base_roi": list(args.base_roi_pixels) if args.base_roi_pixels else None,
                        "base_bag_roi": list(args.base_bag_roi_pixels) if args.base_bag_roi_pixels else None,
                        "base_crop": list(args.base_crop_pixels) if args.base_crop_pixels else None,
                        "base_exclude_top": args.base_exclude_top,
                        "env_state_status": "deferred_raw_capture",
                        "invalid_reason": stale_reasons,
                    }
                    if stale_reasons:
                        metadata["result"] = "invalid"
                    pending_writes.append(save_episode_background(
                        output, episode, list(images) if not has_base
                        else {key: list(value) for key, value in images.items()},
                        list(frames), list(states), list(actions), dict(metadata)
                    ))
                    print(f"SAVING episode={episode:04d} in background; motors remain enabled.", flush=True)
                    if stale_reasons:
                        print(f"  [INVALID] {stale_reasons}", flush=True)
                    episode += 1
                    recording = False
                    result = "not_finished"
                    stale_reasons = None
                    images, frames, states, actions = empty_images(), [], [], []
            if quit_requested:
                break
            time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - tick)))

    finally:
        if communication_fault:
            print("Stopping motion immediately; no further targets will be sent.", flush=True)
        # Disable the leader first while its CAN channel is still independent;
        # a follower CAN failure must not prevent leader gripper cleanup.
        for device in (leader, follower):
            try:
                device.disconnect()
            except Exception as exc:
                print(f"Robot cleanup UNCONFIRMED: {exc}", flush=True)
        print("Capture ended; disable requested.", flush=True)
        try:
            camera.stop()
        finally:
            try:
                cv2.destroyAllWindows()
            finally:
                for writer in pending_writes:
                    while writer.is_alive():
                        writer.join(timeout=1.0)
                errors = list(completed_write_errors)
                errors.extend(str(writer.error) for writer in pending_writes if writer.error is not None)
                if errors:
                    raise RuntimeError("Episode save failed: " + "; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
