r"""Capture both cameras at the training start pose, for side-by-side comparison.

The policy is trained on fixed camera views. If a camera was re-mounted (which
happened while rebuilding this machine), the policy sees a geometry it never
learned from, and no amount of retraining on the old data fixes it.

This moves the follower to the same start pose the deployment script uses (the
medoid first frame of the training episodes), then saves one frame per camera
plus the joint state. Compare those images with the training episode named in the
output: at the same pose they should look almost identical.

Motion is limited to the return-to-start move; the policy never runs.

Usage:
    python -X utf8 tools\capture_at_start_pose.py ^
        --metadata-root D:\NERO\outputs\red_cube_relation_v2\train ^
        --camera-spec wrist=serial:261622078506,size:640x480,fps:30 ^
        --camera-spec base=serial:261622071323,size:640x480,fps:30 ^
        --follower-channel 0 --follower-firmware v121 ^
        --out outputs\pose_match --movement-confirmed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

from lerobot.nero_tools.episode_capture import (
    MultiCamera,
    camera_specs_from_args,
    parse_camera_spec,
)
from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig
from nero_dual_cam_act_deploy import JOINTS, move_to_start, representative_start_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
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
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument("--return-speed", type=float, default=0.3)
    parser.add_argument("--return-timeout", type=float, default=60.0)
    parser.add_argument("--wait", type=float, default=1.0,
                        help="Seconds to settle before capturing")
    parser.add_argument("--out", type=Path, default=Path("outputs/pose_match"))
    parser.add_argument("--movement-confirmed", action="store_true")
    args = parser.parse_args()

    if not args.movement_confirmed:
        raise SystemExit("refusing to move the arm without --movement-confirmed")
    for value in args.camera_spec:
        parse_camera_spec(value)

    args.out.mkdir(parents=True, exist_ok=True)
    values, info = representative_start_state(args.metadata_root)
    target = {f"{name}.pos": float(values[i]) for i, name in enumerate(JOINTS)}
    print(f"start pose from {info['source']} "
          f"(closest of {info['episodes']} episodes, "
          f"{info['offset_from_centre_rad']:.3f} rad from centre)", flush=True)

    camera_args = SimpleNamespace(
        camera_spec=args.camera_spec, camera_type=args.camera_type, camera_index=0,
        serial=None, camera_backend=args.camera_backend,
        camera_width=args.camera_width, camera_height=args.camera_height,
        camera_fps=args.camera_fps)
    cameras = MultiCamera(camera_specs_from_args(camera_args), camera_args)
    cameras.start()

    follower = None
    try:
        follower = NeroFollower(NeroFollowerConfig(
            id="nero_follower", interface=args.interface, channel=args.follower_channel,
            firmware=args.follower_firmware, use_gripper=True,
            speed_percent=args.speed_percent, max_relative_target=None,
            gripper_max_width_m=args.follower_gripper_width, gripper_force_n=1.0,
            gripper_send_threshold=0.001, disable_gripper_on_disconnect=True))
        print(f"connecting follower: channel={args.follower_channel} "
              f"firmware={args.follower_firmware}", flush=True)
        follower.connect()
        if not move_to_start(follower, target, args.return_speed, args.camera_fps,
                             args.return_timeout):
            raise SystemExit("did not reach the start pose; not capturing a misleading frame")

        time.sleep(args.wait)
        observation = follower.get_observation()
        state = {f"{name}.pos": float(observation[f"{name}.pos"]) for name in JOINTS}
        state["gripper.pos"] = float(observation["gripper.pos"])

        snapshots, _ = cameras.snapshot_all()
        written = {}
        for name in cameras.names:
            image, _, _ = snapshots[name]
            if image is None:
                print(f"{name}: no frame", flush=True)
                continue
            path = args.out / f"{name}.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            written[name] = str(path)
            print(f"saved {path}", flush=True)

        payload = {"source_episode": info["source"], "state": state,
                   "target": target, "images": written}
        (args.out / "state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nCompare these with the training frames of {info['source']}:")
        print(f"  outputs/nero_episodes/{info['source']}/frames/wrist/frame_000000.png")
        print(f"  outputs/nero_episodes/{info['source']}/frames/base/frame_000000.png")
        print("At the same joint state the gripper and the cube must appear in the "
              "same place in both images; a shift or rotation means the camera moved.")
    finally:
        cameras.stop()
        if follower is not None:
            try:
                follower.disconnect()
                print("Disable requested for joints and gripper. This is a REQUEST: "
                      "confirm with advancing feedback timestamps before leaving the cell.",
                      flush=True)
            except Exception as exc:
                print(f"cleanup UNCONFIRMED: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
