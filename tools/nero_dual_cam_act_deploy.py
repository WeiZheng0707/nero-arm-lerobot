"""Deploy a dual-camera ACT policy on the NERO follower arm.

This drives the follower from a trained checkpoint. It does NOT need the leader
arm: the policy replaces the operator.

Two modes:

  --dry-run   Opens the cameras, runs the detector and the policy, prints the
              predicted actions, and never connects to the follower. Use this
              first: it validates camera framing, env-state output and the
              checkpoint without any motion.

  (default)   Drives the follower. Every safety gate below applies.

Optional --return-to-start walks the follower to a real training first-frame pose
(the one closest to the centre of all episodes) before the policy loop begins, at
--return-speed. The pose is derived from the data because the shipped
saved_start_pose.json does not match the real robot. If the move does not
converge within --return-timeout the run aborts instead of letting the policy act
from an unseen pose.

Safety gates, all fail-closed (stop and disable rather than continue):
  * a camera frame older than --max-frame-age-ms holds motion and eventually aborts
  * a non-finite or over-limit prediction aborts before sending
  * per-step joint change is clamped to --max-joint-speed * dt
  * gripper is clamped to 0..1
  * follower feedback errors abort
Exit always requests joint and gripper disable; that request is NOT proof of
disable, so verify with advancing feedback timestamps before leaving the cell.

Usage:
  python -X utf8 tools/nero_dual_cam_act_deploy.py ^
    --checkpoint outputs/train/nero_red_cube_act_v1/checkpoints/050000/pretrained_model ^
    --metadata-root outputs/red_cube_300_video/train ^
    --repo-id local/nero_red_cube_bag_train ^
    --camera-spec wrist=serial:261622078506,size:640x480,fps:30 ^
    --camera-spec base=serial:261622071323,size:640x480,fps:30 ^
    --follower-channel 0 --follower-firmware v121 --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.nero_tools.color_detector import detect_color, detect_scene, normalize_box
from lerobot.nero_tools.convert_episodes import ENVIRONMENT_STATE_NAMES, environment_state_vector
from lerobot.nero_tools.env_state_schema import environment_state_relation_v2
from lerobot.nero_tools.env_state_schema import environment_state_base14, ENVIRONMENT_STATE_BASE14_NAMES
from lerobot.nero_tools.base14 import Base14Config, Base14Detector
from lerobot.nero_tools.episode_capture import (
    MultiCamera,
    camera_specs_from_args,
    parse_camera_spec,
    parse_roi,
)
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig

JOINTS = [f"joint_{i}" for i in range(1, 8)]


def build_camera_args(args):
    specs = args.camera_spec
    return SimpleNamespace(
        camera_spec=specs,
        camera_type=args.camera_type,
        camera_index=0,
        serial=args.serial,
        camera_backend=args.camera_backend,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
    )


def state_vector(observation):
    """Follower observation -> the 8-dim state vector the policy was trained on."""
    values = [float(observation[f"{name}.pos"]) for name in JOINTS]
    values.append(float(observation["gripper.pos"]))
    return np.asarray(values, dtype=np.float32)


def scene_status(scene, cube_vetoed=False):
    """Status text shared by logging and preview.

    Every field is read defensively. A base14 tracker that has lost its target
    returns only {'visible': 0} with no geometry keys at all, so indexing 'x'
    directly raises KeyError the moment an object is occluded -- which is exactly
    the phase this run is meant to survive, not abort on. base14 also has no
    ball field, so that part stays conditional.
    """
    cube = scene.get('cube') or {}
    bag = scene.get('bag') or {}
    cube_x = cube.get('x')
    cube_text = (f"cube=({cube_x:.2f},{cube['y']:.2f})"
                 if cube.get('visible') and cube_x is not None
                 else "cube=LOST")
    if cube_vetoed:
        cube_text += "(vetoed)"
    bag_x = bag.get('x')
    bag_text = (f"bag=({bag_x:.2f},{bag['y']:.2f})"
                if bag.get('visible') and bag_x is not None
                else "bag=LOST")
    text = f"{cube_text} {bag_text}"
    ball = scene.get('ball') or {}
    if 'ball' in scene:
        text += f" ball_visible={ball.get('visible')}"
    return text


def plausible_cube_box(box, min_cy, max_cy):
    """Reject a cube detection whose centre sits outside the table band.

    Red and brown are not separable in HSV at this exposure: a cardboard box near
    the top of the base image was repeatedly accepted as the cube (observed in
    outputs/deploy_b_30k_40s at tick 399, where the cube jumped to y=0.08 and the
    bag simultaneously vanished). The training data never has the cube up there,
    so a detection there is a false positive, not a new situation.

    This only vetoes the live detection; it does not change the trained schema or
    the checkpoint's detector fingerprint.
    """
    if not box or not box.get('visible'):
        return box, False
    if not min_cy <= float(box['cy']) <= max_cy:
        return {'visible': 0}, True
    return box, False


def env_state_vector(scene, wrist_cube=None, schema="legacy14"):
    if schema == 'base14':
        return environment_state_base14(scene)
    """Build the environment-state vector the checkpoint was trained on.

    The schema must match the dataset exactly: feeding 14 dims to a model trained
    on 18 raises a shape error, but feeding the wrong *meaning* in the right shape
    fails silently, so the two layouts are kept clearly separate here.
    """
    if schema == "relation_v2":
        return environment_state_relation_v2(scene, wrist_cube)
    frame = {"cube": scene.get("cube") or {}, "ball": scene.get("ball") or {},
             "bag": scene.get("bag") or {}}
    return environment_state_vector(frame, "base")


def image_tensor(rgb):
    """HWC uint8 RGB -> (1, 3, H, W) float in [0, 1], matching LeRobot's decode."""
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def limit_step(target, previous, max_speed, dt, max_step_rad=None):
    """Clamp the per-step joint change; returns (limited, was_clamped).

    `dt` MUST be the measured interval since the previous command, not the
    nominal 1/fps. When inference runs slower than the target rate (observed:
    ~10 Hz against a nominal 30 Hz), a nominal dt silently caps the effective
    joint speed to a fraction of --max-joint-speed and the arm crawls. The
    limiter is meant to be a safety net, not the thing that decides the motion.

    `max_step_rad` is an optional absolute per-step ceiling. Speed alone does
    not bound a single step, so a long stall followed by a large dt would allow
    a large jump; this keeps one command bounded regardless of dt.
    """
    bound = max_speed * dt
    if max_step_rad is not None:
        bound = min(bound, max_step_rad)
    limited = {}
    clamped = False
    for key, value in target.items():
        if key == "gripper.pos":
            limited[key] = value
            continue
        delta = value - previous[key]
        if abs(delta) > bound:
            limited[key] = previous[key] + bound * (1.0 if delta > 0 else -1.0)
            clamped = True
        else:
            limited[key] = value
    return limited, clamped


def representative_start_state(root: Path, limit: int | None = None):
    """Pick a real first-frame pose close to the centre of the training data.

    The saved_start_pose.json shipped with the project does not match the real
    robot, so the start pose is derived from the data instead.

    Why the medoid and not the per-joint median: averaging each joint
    independently can produce a configuration that no episode ever had, and a
    7-DOF arm has no guarantee that such a mix is even reachable or collision
    free. Returning one actual recorded first frame guarantees a coherent pose
    the policy has seen.

    Reads only `observation.state` from each episode parquet, so no video is
    decoded.
    """
    import pyarrow.parquet as pq

    files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        raise RuntimeError(f"no episode parquet under {root}")
    if limit:
        files = files[:limit]
    states = []
    names = []
    for path in files:
        table = pq.read_table(path, columns=["observation.state"])
        column = table.column("observation.state")
        if len(column) == 0:
            continue
        states.append(np.asarray(column[0].as_py(), dtype=np.float32))
        names.append(path.stem)
    if not states:
        raise RuntimeError(f"no observation.state rows under {root}")
    matrix = np.stack(states)
    joints = matrix[:, :7]
    centre = np.median(joints, axis=0)
    distances = np.linalg.norm(joints - centre, axis=1)
    best = int(np.argmin(distances))
    return matrix[best], {
        "episodes": len(states),
        "source": names[best],
        "offset_from_centre_rad": float(distances[best]),
    }


def move_to_start(follower, target, max_speed, fps, timeout, tolerance=0.02):
    """Walk the follower to `target` at a bounded joint speed.

    Deliberately speed-limited rather than a single big command: the arm may be
    far away, and a jump would move it through whatever is in between.
    The gripper is left where it is; the policy takes over once the loop starts.

    All dict keys carry the ".pos" suffix, matching `send_action` and
    `get_observation`; using bare joint names here raised KeyError after the
    connection was already open.
    """
    period = 1.0 / fps
    joint_keys = [f"{name}.pos" for name in JOINTS]
    observation = follower.get_observation()
    current = {key: float(observation[key]) for key in joint_keys}
    start = dict(current)
    travel = max(abs(target[key] - start[key]) for key in joint_keys)
    print("return-to-start: target " + ", ".join(f"{k}={v:+.3f}" for k, v in target.items()),
          flush=True)
    print("                 current " + ", ".join(f"{k}={v:+.3f}" for k, v in start.items()),
          flush=True)
    print(f"                 largest joint travel {travel * 57.2958:.1f} deg; "
          f"at {max_speed:.2f} rad/s this takes about "
          f"{travel / max(max_speed, 1e-6):.0f}s", flush=True)
    deadline = time.monotonic() + timeout
    steps = 0
    while time.monotonic() < deadline:
        error = max(abs(target[key] - current[key]) for key in joint_keys)
        if error <= tolerance:
            print(f"return-to-start: reached within {error * 57.2958:.2f} deg "
                  f"after {steps} steps", flush=True)
            return True
        step = max_speed * period
        step_target = {key: current[key] + float(np.clip(target[key] - current[key],
                                                         -step, step))
                       for key in joint_keys}
        observation = follower.get_observation()
        step_target["gripper.pos"] = float(observation["gripper.pos"])
        follower.send_action(step_target)
        current = {key: step_target[key] for key in joint_keys}
        steps += 1
        time.sleep(period)
    error = max(abs(target[key] - current[key]) for key in joint_keys)
    print(f"return-to-start: NOT reached, still {error * 57.2958:.2f} deg away "
          f"after {steps} steps", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True,
                        help="Dataset split root used for feature metadata (train split)")
    parser.add_argument("--repo-id", default="local/nero_red_cube_bag_train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video-backend", default="pyav")

    parser.add_argument("--camera-type", choices=("realsense", "opencv"), default="realsense")
    parser.add_argument("--camera-spec", action="append", required=True)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)

    parser.add_argument("--base-roi", default="0,100,520,380")
    parser.add_argument("--crop", action="append", default=None, metavar="CAM=X,Y,W,H",
                        help="Apply the same crop the dataset was built with. Must match "
                             "conversion exactly, otherwise the policy sees a different "
                             "framing than it trained on. Repeatable.")
    parser.add_argument("--scale", action="append", default=None, metavar="CAM=FACTOR",
                        help="Apply the same resize the dataset was built with. Must match "
                             "conversion exactly. Repeatable.")
    parser.add_argument("--cube-color", default="red")
    parser.add_argument('--base14-config', type=Path, help='Same detector JSON used to generate training labels')
    parser.add_argument("--env-schema", choices=("legacy14", "relation_v2", "base14"),
                        default="legacy14",
                        help="Environment-state layout the checkpoint was trained on. "
                             "Must match the dataset conversion exactly.")
    parser.add_argument("--ball-roi", default="0,100,520,380",
                        help="Ball search rectangle in base pixels. The base view has a "
                             "blue chair leg in the top-right corner that the detector "
                             "otherwise locks onto.")
    parser.add_argument("--wrist-roi", default=None,
                        help="Optional wrist-camera ROI for the cube used by relation_v2")
    parser.add_argument("--ball-color", default="blue")
    parser.add_argument("--ball-min-sat", type=int, default=110)
    parser.add_argument("--morphology-kernel", type=int, default=5)
    parser.add_argument("--cube-cy-band", default=None,
                        help="base14 only: CUBE_CY_MIN,CUBE_CY_MAX as normalized image "
                             "coordinates. A cube detection whose centre falls outside "
                             "this vertical band is vetoed before it reaches the policy. "
                             "The cardboard box near the top of the base view is otherwise "
                             "accepted as the red cube. Off by default so existing runs are "
                             "unchanged; try 0.35,0.95 once the table area is confirmed.")

    parser.add_argument("--follower-channel", default=None)
    parser.add_argument("--follower-firmware", default="v121")
    parser.add_argument("--interface", default="agx_cando")
    parser.add_argument("--follower-gripper-width", type=float, default=0.07)
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-joint-speed", type=float, default=0.3)
    parser.add_argument("--max-step-rad", type=float, default=0.05,
                        help="Absolute ceiling on a single commanded joint step, in radians, "
                             "applied on top of --max-joint-speed. Speed alone does not bound "
                             "one step: after a stall the measured dt grows and a pure "
                             "speed limit would permit a jump. Default 0.05 rad (~2.9 deg) is "
                             "roughly 3x the 1.2 rad/s training rate at 30 Hz. Use 0 to disable.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action-steps", type=int, default=None,
                        help="How many actions to execute before re-predicting. Defaults to "
                             "the checkpoint's n_action_steps. A larger value means fewer "
                             "chunk switches, which removes the jump that happens every "
                             "time a new chunk starts, at the cost of slower reaction.")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        help="Exponential smoothing factor in (0,1] applied to the commanded "
                             "joint targets; 0 disables it. 0.2-0.4 removes high-frequency "
                             "jitter without adding noticeable lag.")
    parser.add_argument("--save-frames", type=Path, default=None,
                        help="Record every Nth frame to this folder as JPG plus a JSONL of "
                             "state and predicted action. Without recorded frames there is no "
                             "way to tell whether the live camera view still matches training.")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save one frame every N ticks when --save-frames is set")
    parser.add_argument("--gripper-override", type=float, default=None,
                        help="Replace the policy's gripper command with this constant "
                             "(0 = closed, 1 = open). The gripper is the policy's weakest "
                             "action dimension and can sit at zero for an entire run, which "
                             "makes a grasp impossible no matter how well the arm moves. "
                             "An override separates 'arm cannot reach the cube' from "
                             "'gripper never opens'.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--max-frame-age-ms", type=float, default=200.0)
    parser.add_argument("--show-preview", action="store_true")

    parser.add_argument("--dry-run", action="store_true",
                        help="Do not connect to the follower; only print predictions")
    parser.add_argument("--return-to-start", action="store_true",
                        help="Before the policy loop, walk the follower to the median "
                             "first-frame pose of the training episodes. Uses the data "
                             "because saved_start_pose.json does not match the robot.")
    parser.add_argument("--start-pose-file", type=Path, default=None,
                        help="Optional JSON overriding the data-derived start pose")
    parser.add_argument("--return-speed", type=float, default=0.15,
                        help="Joint speed cap in rad/s during the return move")
    parser.add_argument("--return-timeout", type=float, default=60.0,
                        help="Give up (and stop) if the start pose is not reached in time")
    parser.add_argument("--movement-confirmed", action="store_true",
                        help="Required for real motion. Confirm: arms supported, gripper empty, "
                             "workspace clear, physical emergency stop reachable.")
    args = parser.parse_args()

    roi = parse_roi(args.base_roi)
    cube_cy_band = None
    if args.cube_cy_band:
        parts = [float(part) for part in args.cube_cy_band.replace(" ", "").split(",")]
        if len(parts) != 2 or not 0.0 <= parts[0] < parts[1] <= 1.0:
            raise SystemExit("--cube-cy-band needs CY_MIN,CY_MAX within 0..1")
        cube_cy_band = (parts[0], parts[1])
        if args.env_schema != 'base14':
            print("note: --cube-cy-band only affects base14", flush=True)
    base14_detector = None
    base14_sequence = None
    base14_scene = None
    if args.env_schema == 'base14':
        if args.base14_config is None:
            parser.error('base14 requires --base14-config')
        base14_detector = Base14Detector(Base14Config(**json.loads(args.base14_config.read_text(encoding='utf-8'))))
    crop_by_camera = {}
    for value in (args.crop or []):
        cam, _, rect = value.partition("=")
        parts = [int(part) for part in rect.replace(" ", "").split(",")]
        if not cam or len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
            raise SystemExit(f"--crop needs CAM=X,Y,W,H: {value!r}")
        crop_by_camera[cam.strip()] = tuple(parts)
    scale_by_camera = {}
    for value in (args.scale or []):
        cam, _, factor = value.partition("=")
        try:
            factor_value = float(factor)
        except ValueError:
            raise SystemExit(f"--scale needs CAM=FACTOR: {value!r}")
        if not cam or not 0.1 <= factor_value <= 4.0:
            raise SystemExit(f"--scale needs a factor between 0.1 and 4.0: {value!r}")
        scale_by_camera[cam.strip()] = factor_value
    for value in args.camera_spec:
        parse_camera_spec(value)  # fail fast on a malformed spec

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.metadata_root)
    print(f"dataset metadata: {sorted(meta.features)}", flush=True)
    missing = [key for key in ("observation.images.wrist", "observation.images.base",
                               "observation.environment_state")
               if key not in meta.features]
    if missing:
        raise SystemExit(f"checkpoint/dataset mismatch, missing features: {missing}")

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    if args.env_schema == 'base14':
        if meta.features['observation.environment_state'].get('names') != ENVIRONMENT_STATE_BASE14_NAMES:
            raise SystemExit('Dataset field names do not match base14')
        if getattr(policy.config, 'environment_state_schema', None) != 'base14':
            raise SystemExit('Checkpoint is not explicitly tagged base14')
        if policy.config.environment_state_detector_hash != base14_detector.config.fingerprint():
            raise SystemExit('Live detector config differs from training config')
        if set(policy.config.input_features) != {'observation.images.wrist', 'observation.state', 'observation.environment_state'}:
            raise SystemExit('base14 checkpoint must use wrist + state + env only')
    env_feature = policy.config.env_state_feature
    expected_dim = int(env_feature.shape[0]) if env_feature else 0
    schema_dim = 18 if args.env_schema == "relation_v2" else 14
    if expected_dim and expected_dim != schema_dim:
        raise SystemExit(
            f"--env-schema {args.env_schema} produces {schema_dim} values but the "
            f"checkpoint expects {expected_dim}; the two layouts are not interchangeable")
    print(f"env schema: {args.env_schema} ({schema_dim} values, checkpoint agrees)",
          flush=True)
    save_log = None
    if args.save_frames is not None:
        args.save_frames.mkdir(parents=True, exist_ok=True)
        save_log = (args.save_frames / "frames.jsonl").open("w", encoding="utf-8")
        print(f"recording frames to {args.save_frames.resolve()} every "
              f"{args.save_every} ticks", flush=True)
    chunk = policy.config.chunk_size
    steps_per_chunk = args.action_steps or policy.config.n_action_steps
    if steps_per_chunk > chunk:
        raise SystemExit(f"--action-steps {steps_per_chunk} exceeds chunk_size {chunk}")
    print(f"policy ready: chunk={chunk} action_steps={steps_per_chunk} "
          f"dropout={getattr(policy.config, 'env_state_dropout', None)} (inactive in eval)",
          flush=True)
    if args.smoothing:
        print(f"smoothing: alpha={args.smoothing:.2f} "
              f"(higher means a stronger filter and more lag)", flush=True)

    if not args.dry_run and not args.movement_confirmed:
        raise SystemExit("refusing to move the arm without --movement-confirmed")

    camera_args = build_camera_args(args)
    cameras = MultiCamera(camera_specs_from_args(camera_args), camera_args)
    if "base" not in cameras.names:
        raise SystemExit("a camera named 'base' is required for env-state")
    cameras.start()

    follower = None
    previous = None
    period = 1.0 / args.fps
    started = time.monotonic()
    sent_count = 0
    clamped_count = 0
    pending: list[dict] = []
    last_sent = None
    step_peak = 0.0
    last_command_time = None

    try:
        if not args.dry_run:
            follower = NeroFollower(NeroFollowerConfig(
                id="nero_follower", interface=args.interface, channel=args.follower_channel,
                firmware=args.follower_firmware, use_gripper=True,
                speed_percent=args.speed_percent, max_relative_target=None,
                gripper_max_width_m=args.follower_gripper_width,
                gripper_force_n=args.gripper_force,
                gripper_send_threshold=0.001, disable_gripper_on_disconnect=True,
            ))
            print(f"connecting follower: channel={args.follower_channel} "
                  f"firmware={args.follower_firmware}", flush=True)
            follower.connect()
            obs = follower.get_observation()
            previous = {f"{name}.pos": float(obs[f"{name}.pos"]) for name in JOINTS}
            previous["gripper.pos"] = float(obs["gripper.pos"])
            print("follower connected", flush=True)

            if args.return_to_start:
                if args.start_pose_file is not None:
                    import json as _json
                    raw = _json.loads(args.start_pose_file.read_text(encoding="utf-8"))
                    values = [float(raw[f"{name}.pos"] if f"{name}.pos" in raw else raw[name])
                              for name in JOINTS]
                    print(f"start pose from {args.start_pose_file}", flush=True)
                else:
                    values, info = representative_start_state(args.metadata_root)
                    print(f"start pose = first frame of {info['source']} "
                          f"(closest of {info['episodes']} training episodes to the centre, "
                          f"{info['offset_from_centre_rad']:.3f} rad away)", flush=True)
                target = {f"{name}.pos": float(values[i]) for i, name in enumerate(JOINTS)}
                if not move_to_start(follower, target, args.return_speed, args.fps,
                                     args.return_timeout):
                    raise RuntimeError(
                        "return-to-start did not converge; refusing to start the policy "
                        "from a pose the model has never seen")
                previous = dict(target)
                previous["gripper.pos"] = float(follower.get_observation()["gripper.pos"])
            else:
                print("no return-to-start requested; starting from the current pose",
                      flush=True)
        else:
            print("DRY RUN: no follower connection, no motion.", flush=True)
            print("  Camera framing, detector output and checkpoint shapes are being "
                  "validated. The joint part of the state vector is zeros because no "
                  "arm is read, so the predicted actions are NOT meaningful yet.", flush=True)

        while time.monotonic() - started < args.duration:
            tick = time.monotonic()

            snapshots, _ = cameras.snapshot_all()
            wrist_image, _, _ = snapshots["wrist"]
            base_image, _, _ = snapshots["base"]
            if wrist_image is None or base_image is None:
                print("waiting for camera frames ...", flush=True)
                time.sleep(0.02)
                continue
            ages = {name: cameras.readers[name].host_age_ms()
                    for name in ("wrist", "base")}
            if max(ages.values()) > args.max_frame_age_ms:
                print(f"[HOLD] stale frame {ages}; not sending", flush=True)
                time.sleep(0.02)
                continue

            # Detection always runs on the full frame: --base-roi is expressed in
            # original pixel coordinates. Cropping for the policy input happens
            # further down, so the two coordinate systems never mix.

            scene = detect_scene(
                cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR), args.cube_color,
                ball_color=None if args.ball_color.lower() == "none" else args.ball_color,
                roi=roi, exclude_top_ratio=0.0,
                morphology_kernel=args.morphology_kernel,
                ball_min_sat=args.ball_min_sat,
                ball_roi=parse_roi(args.ball_roi) if args.ball_roi else None,
            )
            wrist_cube = None
            cube_vetoed = False
            if base14_detector is not None:
                sequence = snapshots['base'][2]
                if sequence != base14_sequence:
                    base14_scene, _, _ = base14_detector.update(
                        cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR), time.monotonic())
                    base14_sequence = sequence
                scene = base14_scene
                if cube_cy_band is not None:
                    # Copy so the veto does not leak into the cached detector scene.
                    scene = dict(base14_scene)
                    scene['cube'], cube_vetoed = plausible_cube_box(
                        base14_scene.get('cube'), *cube_cy_band)
            if args.env_schema == "relation_v2":
                wrist_cube = detect_color(
                    cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR), args.cube_color,
                    roi=parse_roi(args.wrist_roi) if args.wrist_roi else None,
                    exclude_top_ratio=0.28, morphology_kernel=args.morphology_kernel)
                height, width = wrist_image.shape[:2]
                wrist_cube = normalize_box(wrist_cube, width, height)

            if not pending:
                if follower is not None:
                    observation = follower.get_observation()
                else:
                    observation = {**{f"{name}.pos": 0.0 for name in JOINTS},
                                   "gripper.pos": 0.0}
                policy_wrist = wrist_image
                policy_base = base_image
                if "wrist" in crop_by_camera:
                    x, y, w, h = crop_by_camera["wrist"]
                    policy_wrist = wrist_image[y:y + h, x:x + w]
                if "base" in crop_by_camera:
                    x, y, w, h = crop_by_camera["base"]
                    policy_base = base_image[y:y + h, x:x + w]
                if "wrist" in scale_by_camera:
                    factor = scale_by_camera["wrist"]
                    policy_wrist = cv2.resize(policy_wrist, None, fx=factor, fy=factor,
                                              interpolation=cv2.INTER_CUBIC)
                if "base" in scale_by_camera:
                    factor = scale_by_camera["base"]
                    policy_base = cv2.resize(policy_base, None, fx=factor, fy=factor,
                                             interpolation=cv2.INTER_CUBIC)
                batch = {
                    "observation.state": torch.from_numpy(
                        state_vector(observation)).unsqueeze(0).to(device),
                    "observation.environment_state": torch.from_numpy(
                        env_state_vector(scene, wrist_cube,
                                         args.env_schema)).unsqueeze(0).to(device),
                    "observation.images.wrist": image_tensor(policy_wrist).to(device),
                    "observation.images.base": image_tensor(policy_base).to(device),
                }
                with torch.no_grad():
                    predicted = policy.predict_action_chunk(batch)
                if not torch.isfinite(predicted).all():
                    raise RuntimeError("policy produced non-finite actions; aborting")
                actions = predicted[0, :steps_per_chunk].cpu().numpy()
                pending = [{**{f"{name}.pos": float(row[i]) for i, name in enumerate(JOINTS)},
                            "gripper.pos": float(row[-1])} for row in actions]
                print(f"plan: {scene_status(scene, cube_vetoed)} "
                      f"| a0={np.round(actions[0], 3).tolist()}", flush=True)

            target = pending.pop(0)
            target["gripper.pos"] = min(1.0, max(0.0, target["gripper.pos"]))
            if args.gripper_override is not None:
                target["gripper.pos"] = min(1.0, max(0.0, args.gripper_override))
            if follower is None:
                print(f"[DRY] step {sent_count}: "
                      f"{np.round([target[k] for k in target], 3).tolist()}", flush=True)
                sent_count += 1
            else:
                if args.smoothing > 0.0:
                    # EMA toward the raw prediction. The limiter below only bounds
                    # speed; it does not remove high-frequency oscillation, which
                    # is what reads as jitter.
                    alpha = args.smoothing
                    target = {key: previous[key] + alpha * (target[key] - previous[key])
                              for key in target}
                raw_step = max(abs(target[f"{name}.pos"] - previous[f"{name}.pos"])
                               for name in JOINTS)
                # Use the interval actually elapsed since the last command. Inference
                # can run well below --fps (observed ~10 Hz), and charging the nominal
                # period to the limiter caps real joint speed far below
                # --max-joint-speed. First command falls back to the nominal period.
                now_cmd = time.monotonic()
                measured_dt = (period if last_command_time is None
                               else now_cmd - last_command_time)
                measured_dt = min(max(measured_dt, 1e-3), 0.25)
                limited, was_clamped = limit_step(target, previous, args.max_joint_speed,
                                                  measured_dt,
                                                  args.max_step_rad if args.max_step_rad > 0 else None)
                last_command_time = now_cmd
                clamped_count += int(was_clamped)
                sent = follower.send_action(limited)
                previous = {key: float(value) for key, value in limited.items()}
                last_sent = sent
                sent_count += 1
                step_peak = max(step_peak, raw_step)
                if sent_count % max(1, args.fps) == 0:
                    # Once a second: is the arm actually following the command, and
                    # how large are the commanded steps? A large tracking error means
                    # the controller is lagging; a large raw step means the policy
                    # output itself is jumpy.
                    measured = follower.get_observation()
                    tracking = max(abs(limited[f"{name}.pos"] - float(measured[f"{name}.pos"]))
                                   for name in JOINTS)
                    # Report the achieved rate and the dt actually charged to the
                    # limiter. If `hz` is far below --fps, the loop is inference-bound
                    # and the old nominal-dt limiter would have been the bottleneck.
                    print(f"  [jitter] max_cmd_step={np.degrees(step_peak):.2f} deg "
                          f"tracking_err={np.degrees(tracking):.2f} deg "
                          f"clamped={clamped_count} "
                          f"hz={1.0 / max(measured_dt, 1e-6):.1f} "
                          f"dt={measured_dt * 1000:.0f}ms", flush=True)
                    step_peak = 0.0

            if save_log is not None and sent_count % max(1, args.save_every) == 0:
                # Keep the raw view the policy actually consumed, plus the state, so
                # a failing run can be compared frame-for-frame with training data
                # instead of being argued about from memory.
                stem = f"{sent_count:05d}"
                cv2.imwrite(str(args.save_frames / f"{stem}_wrist.jpg"),
                            cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(args.save_frames / f"{stem}_base.jpg"),
                            cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR))
                save_log.write(json.dumps({
                    "tick": sent_count,
                    "state": {key: float(value) for key, value in previous.items()},
                    "scene": {name: {k: box.get(k) for k in ("x", "y", "w", "h", "visible")}
                              for name, box in scene.items() if isinstance(box, dict)},
                    "env_state": [float(v) for v in env_state_vector(scene, wrist_cube,
                                                                     args.env_schema)],
                }) + "\n")
                save_log.flush()

            if args.show_preview:
                # Show both streams side by side: the operator needs to see exactly
                # what the policy sees, and a single window avoids the focus
                # problems of two cv2 windows.
                half_w, half_h = 480, 360
                wrist_view = cv2.resize(cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR),
                                        (half_w, half_h))
                base_view = cv2.resize(cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR),
                                       (half_w, half_h))
                for name, box in scene.items():
                    if not isinstance(box, dict) or not box.get("visible"):
                        continue
                    if not all(key in box for key in ("x", "y", "w", "h")):
                        continue
                    cv2.rectangle(
                        base_view,
                        (int(box["x"] * half_w), int(box["y"] * half_h)),
                        (int((box["x"] + box["w"]) * half_w),
                         int((box["y"] + box["h"]) * half_h)),
                        (0, 255, 0), 2)
                    cv2.putText(base_view, name,
                                (int(box["x"] * half_w), max(12, int(box["y"] * half_h) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                canvas = np.hstack([wrist_view, base_view])
                cv2.putText(canvas, "WRIST (policy input)", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(canvas, "BASE + detections", (half_w + 8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(canvas,
                            f"steps={sent_count} clamped={clamped_count} "
                            f"{scene_status(scene, cube_vetoed)} "
                            f"age w/b={ages['wrist']:.0f}/{ages['base']:.0f}ms",
                            (8, canvas.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("NERO deploy (wrist | base)", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("stopped by operator", flush=True)
                    break

            elapsed = time.monotonic() - tick
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\ninterrupted by operator", flush=True)
    except Exception as exc:
        print(f"\n[ABORT] {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if save_log is not None:
            save_log.close()
        cameras.stop()
        if follower is not None:
            try:
                follower.disconnect()
            except Exception as exc:
                print(f"follower cleanup UNCONFIRMED: {exc}", flush=True)
            print("Disable requested for joints and gripper. This is a REQUEST: "
                  "confirm with advancing feedback timestamps before leaving the cell. "
                  "Keep the arms supported.", flush=True)
        if args.show_preview:
            cv2.destroyAllWindows()
        print(f"steps sent={sent_count} clamped_steps={clamped_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
