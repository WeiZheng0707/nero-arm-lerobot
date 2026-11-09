r"""Deploy the 14-dim ACT model that ships with D:\1104.

This is deliberately a separate script from nero_dual_cam_act_deploy.py, because
the two models disagree in three ways that fail silently if mixed:

    env layout   14 numbers in bbox form (x, y, w, h, visible) per object,
                 not the 18-dim centre-relative layout of the older script
    env source   EnvStateEstimator from D:\1104\env_v3: a stateful detector with
                 a Kalman track and a gripper-phase machine, not a per-frame HSV
                 threshold. It must be created once per episode and reset.
    colour order that detector expects BGR (OpenCV order); the policy images
                 stay RGB because that is what the videos decode to

Feeding RGB to the estimator swaps red and blue, so the cube is simply never
found - no exception, just a policy that ignores its environment state.

Usage:
    python -X utf8 tools\nero_deploy_env14.py ^
        --checkpoint D:\1104\outputs\nero_act_v1\checkpoints\030000\pretrained_model ^
        --metadata-root D:\1104\nero_cube_300\train ^
        --repo-id local/nero_cube_train ^
        --camera-spec wrist=serial:261622071323,size:640x480,fps:30 ^
        --camera-spec base=serial:261622078506,size:640x480,fps:30 ^
        --follower-channel 0 --follower-firmware v121 ^
        --speed-percent 60 --max-joint-speed 1.2 --duration 30 --movement-confirmed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from lerobot.nero_tools.episode_capture import (
    MultiCamera,
    camera_specs_from_args,
    parse_camera_spec,
)
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig
from nero_dual_cam_act_deploy import (
    JOINTS,
    image_tensor,
    limit_step,
    move_to_start,
    representative_start_state,
    state_vector,
)

EXPECTED_ENV_DIM = 14


def load_estimator(env_v3_dir: Path):
    """Import EnvStateEstimator from the handover bundle."""
    if not (env_v3_dir / "env_state_live.py").exists():
        raise SystemExit(f"env_state_live.py not found under {env_v3_dir}")
    sys.path.insert(0, str(env_v3_dir))
    from env_state_live import EnvStateEstimator  # noqa: PLC0415

    return EnvStateEstimator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/nero_cube_train")
    parser.add_argument("--env-v3-dir", type=Path, default=Path(r"D:\1104\env_v3"))
    parser.add_argument("--state-mode", choices=("split", "merged22"), default="split",
                        help="split: 8-d state plus a 14-d environment_state feature (the "
                             "old layout). merged22: the 14 env values are appended to "
                             "observation.state, which is what the model actually reads - "
                             "measured bag sensitivity went from 0.07%% to 2.96%% after "
                             "this switch.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--camera-type", choices=("realsense", "opencv"), default="realsense")
    parser.add_argument("--camera-spec", action="append", required=True)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)

    parser.add_argument("--follower-channel", default=None)
    parser.add_argument("--follower-firmware", default="v121")
    parser.add_argument("--interface", default="agx_cando")
    parser.add_argument("--follower-gripper-width", type=float, default=0.07)
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--speed-percent", type=int, default=60)
    parser.add_argument("--max-joint-speed", type=float, default=1.2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--action-steps", type=int, default=None)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--gripper-override", type=float, default=None,
                        help="Force a constant gripper command (0 closed, 1 open). "
                             "The gripper is the weakest action dimension; forcing it "
                             "separates 'arm cannot reach' from 'gripper never opens'.")
    parser.add_argument("--return-to-start", action="store_true")
    parser.add_argument("--return-speed", type=float, default=0.3)
    parser.add_argument("--return-timeout", type=float, default=60.0)
    parser.add_argument("--max-frame-age-ms", type=float, default=200.0)

    parser.add_argument("--save-frames", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--bag-correct", type=Path, default=None,
                        help="JSON from tools/fit_bag_to_joint.py. When set, the base "
                             "rotation (joint_1) is nudged toward the measured bag once "
                             "the cube is held, because the policy ignores the bag fields "
                             "(measured: moving the bag changes the prediction by 0.07%).")
    parser.add_argument("--bag-correct-limit", type=float, default=0.7,
                        help="Clamp on the joint_1 correction in radians")
    parser.add_argument("--bag-correct-ramp", type=float, default=1.5,
                        help="Seconds over which the correction ramps in")
    parser.add_argument("--show-preview", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--movement-confirmed", action="store_true")
    args = parser.parse_args()

    for value in args.camera_spec:
        parse_camera_spec(value)
    EnvStateEstimator = load_estimator(args.env_v3_dir)

    device = torch.device(args.device)
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    env_feature = policy.config.env_state_feature
    state_feature = policy.config.robot_state_feature
    state_dim = int(state_feature.shape[0]) if state_feature else 0
    if args.state_mode == "merged22":
        # In this layout there is no ENV token at all: the 14 numbers are the tail of
        # observation.state. Check both directions so a mismatched dataset cannot run.
        if env_feature is not None:
            raise SystemExit("--state-mode merged22 but the checkpoint still expects a "
                             f"separate {int(env_feature.shape[0])}-dim env feature")
        if state_dim != 8 + EXPECTED_ENV_DIM:
            raise SystemExit(f"merged22 needs a {8 + EXPECTED_ENV_DIM}-dim state, "
                             f"checkpoint has {state_dim}")
        env_dim = EXPECTED_ENV_DIM
    else:
        env_dim = int(env_feature.shape[0]) if env_feature else 0
        if env_dim != EXPECTED_ENV_DIM:
            raise SystemExit(f"this script expects a {EXPECTED_ENV_DIM}-dim env feature, "
                             f"checkpoint has {env_dim}")
    chunk = policy.config.chunk_size
    steps_per_chunk = args.action_steps or policy.config.n_action_steps
    if steps_per_chunk > chunk:
        raise SystemExit(f"--action-steps {steps_per_chunk} exceeds chunk_size {chunk}")
    print(f"policy: chunk={chunk} action_steps={steps_per_chunk} state={state_dim} "
          f"state_mode={args.state_mode} env_dim={env_dim} "
          f"ENV norm={policy.config.normalization_mapping.get('ENV')}", flush=True)

    if not args.dry_run and not args.movement_confirmed:
        raise SystemExit("refusing to move the arm without --movement-confirmed")

    camera_args = SimpleNamespace(
        camera_spec=args.camera_spec, camera_type=args.camera_type, camera_index=0,
        serial=None, camera_backend=args.camera_backend,
        camera_width=args.camera_width, camera_height=args.camera_height,
        camera_fps=args.camera_fps)
    cameras = MultiCamera(camera_specs_from_args(camera_args), camera_args)
    if "base" not in cameras.names or "wrist" not in cameras.names:
        raise SystemExit("both a 'base' and a 'wrist' camera are required")
    cameras.start()

    save_log = None
    if args.save_frames is not None:
        args.save_frames.mkdir(parents=True, exist_ok=True)
        save_log = (args.save_frames / "frames.jsonl").open("w", encoding="utf-8")

    estimator = EnvStateEstimator()
    estimator.reset()

    # Optional scripted correction of the base rotation toward the measured bag.
    bag_fit = None
    if args.bag_correct is not None:
        bag_fit = json.loads(args.bag_correct.read_text(encoding="utf-8"))
        coef = bag_fit["coefficients"]["joint_1.pos"]
        print(f"bag correction: joint_1 <- {coef[0]:+.3f}*bag_x + {coef[1]:+.3f}*bag_y "
              f"{coef[2]:+.3f}  (reference bag at {bag_fit['bag_x_mean']:.3f}, "
              f"{bag_fit['bag_y_mean']:.3f})", flush=True)
    holding_since = None
    applied_correction = 0.0
    seen_cube = False
    opened_gripper = False

    follower = None
    previous = None
    pending: list[dict] = []
    period = 1.0 / args.fps
    sent_count = 0
    clamped_count = 0
    step_peak = 0.0
    gripper_cmd = 0.0
    started = time.monotonic()

    try:
        if not args.dry_run:
            follower = NeroFollower(NeroFollowerConfig(
                id="nero_follower", interface=args.interface, channel=args.follower_channel,
                firmware=args.follower_firmware, use_gripper=True,
                speed_percent=args.speed_percent, max_relative_target=None,
                gripper_max_width_m=args.follower_gripper_width,
                gripper_force_n=args.gripper_force,
                gripper_send_threshold=0.001, disable_gripper_on_disconnect=True))
            print(f"connecting follower: channel={args.follower_channel} "
                  f"firmware={args.follower_firmware}", flush=True)
            follower.connect()
            observation = follower.get_observation()
            previous = {f"{name}.pos": float(observation[f"{name}.pos"]) for name in JOINTS}
            previous["gripper.pos"] = float(observation["gripper.pos"])

            if args.return_to_start:
                values, info = representative_start_state(args.metadata_root)
                target = {f"{name}.pos": float(values[i]) for i, name in enumerate(JOINTS)}
                print(f"start pose from {info['source']} "
                      f"({info['offset_from_centre_rad']:.3f} rad from centre)", flush=True)
                if not move_to_start(follower, target, args.return_speed, args.fps,
                                     args.return_timeout):
                    raise RuntimeError("return-to-start did not converge")
                previous = dict(target)
                previous["gripper.pos"] = float(follower.get_observation()["gripper.pos"])
                estimator.reset()
        else:
            print("DRY RUN: cameras + policy only, no follower connection.", flush=True)
            previous = {**{f"{name}.pos": 0.0 for name in JOINTS}, "gripper.pos": 0.0}

        while time.monotonic() - started < args.duration:
            tick = time.monotonic()
            snapshots, _ = cameras.snapshot_all()
            wrist_rgb, _, _ = snapshots["wrist"]
            base_rgb, _, _ = snapshots["base"]
            if wrist_rgb is None or base_rgb is None:
                time.sleep(0.02)
                continue
            ages = {name: cameras.readers[name].host_age_ms() for name in ("wrist", "base")}
            if max(ages.values()) > args.max_frame_age_ms:
                print(f"[HOLD] stale frame {ages}", flush=True)
                time.sleep(0.02)
                continue

            # The estimator wants BGR; the policy wants RGB.
            base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
            wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
            env14 = estimator.update(base_bgr, wrist_bgr, float(gripper_cmd))
            env14 = np.asarray(env14, dtype=np.float32).reshape(-1)
            if env14.size != EXPECTED_ENV_DIM:
                raise RuntimeError(f"estimator returned {env14.size} values, "
                                   f"expected {EXPECTED_ENV_DIM}")

            if not pending:
                if follower is not None:
                    observation = follower.get_observation()
                else:
                    observation = {**{f"{name}.pos": 0.0 for name in JOINTS},
                                   "gripper.pos": 0.0}
                robot_state = state_vector(observation)
                if args.state_mode == "merged22":
                    # env14 becomes the tail of the state vector; there is no separate
                    # environment_state key in this layout.
                    merged_state = np.concatenate((robot_state, env14)).astype(np.float32)
                    batch = {
                        "observation.state": torch.from_numpy(
                            merged_state).unsqueeze(0).to(device),
                        "observation.images.wrist": image_tensor(wrist_rgb).to(device),
                        "observation.images.base": image_tensor(base_rgb).to(device),
                    }
                else:
                    batch = {
                        "observation.state": torch.from_numpy(
                            robot_state).unsqueeze(0).to(device),
                        "observation.environment_state": torch.from_numpy(
                            env14).unsqueeze(0).to(device),
                        "observation.images.wrist": image_tensor(wrist_rgb).to(device),
                        "observation.images.base": image_tensor(base_rgb).to(device),
                    }
                with torch.no_grad():
                    predicted = policy.predict_action_chunk(batch)
                if not torch.isfinite(predicted).all():
                    raise RuntimeError("policy produced non-finite actions")
                actions = predicted[0, :steps_per_chunk].cpu().numpy()
                pending = [{**{f"{name}.pos": float(row[i]) for i, name in enumerate(JOINTS)},
                            "gripper.pos": float(row[-1])} for row in actions]
                cube_visible = float(env14[4])
                print(f"plan: cube_visible={cube_visible:.1f} "
                      f"cube=({env14[0]:.2f},{env14[1]:.2f}) bag=({env14[5]:.2f},{env14[6]:.2f}) "
                      f"grip_cmd={gripper_cmd:.2f} "
                      f"bag_fix={applied_correction:+.2f}rad", flush=True)

            target = pending.pop(0)
            target["gripper.pos"] = min(1.0, max(0.0, target["gripper.pos"]))
            if args.gripper_override is not None:
                target["gripper.pos"] = min(1.0, max(0.0, args.gripper_override))
            gripper_cmd = target["gripper.pos"]

            if bag_fit is not None:
                # "Holding" cannot be inferred from a single frame: at startup the cube
                # is often hidden behind the parked arm and the gripper already reads
                # closed, which made the first version fire immediately. Require the
                # sequence cube seen -> gripper opened -> gripper closed -> cube gone.
                if float(env14[4]) > 0.5:
                    seen_cube = True
                if gripper_cmd > 0.5:
                    opened_gripper = True
                holding = (seen_cube and opened_gripper
                           and gripper_cmd < 0.2 and float(env14[4]) < 0.5)
                if holding and holding_since is None:
                    holding_since = time.monotonic()
                    print("  bag correct: holding detected "
                          f"(cube seen, gripper opened then closed), ramping in over "
                          f"{args.bag_correct_ramp:.1f}s", flush=True)
                if holding_since is None:
                    applied_correction = 0.0
                else:
                    bag_cx = float(env14[5]) + float(env14[7]) / 2.0
                    bag_cy = float(env14[6]) + float(env14[8]) / 2.0
                    coef = bag_fit["coefficients"]["joint_1.pos"]
                    desired = (coef[0] * (bag_cx - bag_fit["bag_x_mean"])
                               + coef[1] * (bag_cy - bag_fit["bag_y_mean"]))
                    desired = float(np.clip(desired, -args.bag_correct_limit,
                                            args.bag_correct_limit))
                    progress = min(1.0, (time.monotonic() - holding_since)
                                   / max(args.bag_correct_ramp, 1e-3))
                    applied_correction = desired * progress
                    target["joint_1.pos"] = target["joint_1.pos"] + applied_correction

            if follower is None:
                print(f"[DRY] {sent_count}: gripper={gripper_cmd:.3f}", flush=True)
                sent_count += 1
            else:
                if args.smoothing > 0.0:
                    alpha = args.smoothing
                    target = {key: previous[key] + alpha * (target[key] - previous[key])
                              for key in target}
                raw_step = max(abs(target[f"{name}.pos"] - previous[f"{name}.pos"])
                               for name in JOINTS)
                limited, was_clamped = limit_step(target, previous, args.max_joint_speed,
                                                  max(period, 1e-3))
                clamped_count += int(was_clamped)
                follower.send_action(limited)
                previous = {key: float(value) for key, value in limited.items()}
                sent_count += 1
                step_peak = max(step_peak, raw_step)
                if sent_count % max(1, args.fps) == 0:
                    measured = follower.get_observation()
                    tracking = max(abs(limited[f"{name}.pos"] - float(measured[f"{name}.pos"]))
                                   for name in JOINTS)
                    print(f"  [jitter] max_cmd_step={np.degrees(step_peak):.2f} deg "
                          f"tracking_err={np.degrees(tracking):.2f} deg "
                          f"clamped={clamped_count}", flush=True)
                    step_peak = 0.0

            if save_log is not None and sent_count % max(1, args.save_every) == 0:
                stem = f"{sent_count:05d}"
                cv2.imwrite(str(args.save_frames / f"{stem}_wrist.jpg"), wrist_bgr)
                cv2.imwrite(str(args.save_frames / f"{stem}_base.jpg"), base_bgr)
                save_log.write(json.dumps({
                    "tick": sent_count,
                    "state": {k: float(v) for k, v in previous.items()},
                    "gripper_cmd": float(gripper_cmd),
                    "env14": [float(v) for v in env14],
                }) + "\n")
                save_log.flush()

            if args.show_preview:
                canvas = np.hstack([
                    cv2.resize(wrist_bgr, (480, 360)),
                    cv2.resize(base_bgr, (480, 360))])
                cv2.putText(canvas, "WRIST", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 2)
                cv2.putText(canvas, "BASE", (488, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 2)
                cv2.putText(canvas,
                            f"steps={sent_count} grip={gripper_cmd:.2f} "
                            f"cube_vis={env14[4]:.0f} age={max(ages.values()):.0f}ms",
                            (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)
                cv2.imshow("NERO deploy env14 (wrist | base)", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("stopped by operator", flush=True)
                    break

            elapsed = time.monotonic() - tick
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\ninterrupted by operator", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ABORT] {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if save_log is not None:
            save_log.close()
        cameras.stop()
        if follower is not None:
            try:
                follower.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"follower cleanup UNCONFIRMED: {exc}", flush=True)
            print("Disable requested for joints and gripper. This is a REQUEST: "
                  "confirm with advancing feedback timestamps before leaving the cell.",
                  flush=True)
        if args.show_preview:
            cv2.destroyAllWindows()
        print(f"steps sent={sent_count} clamped_steps={clamped_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
