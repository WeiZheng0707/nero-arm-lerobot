"""Disable two explicitly identified NERO arms and their AgxGripper effectors."""

import argparse
import math
import sys
import time
from pathlib import Path

from .test_enable_cycle import snapshot

PROFILE_CHOICES = ("default", "v111", "v112", "v120", "v121")


def disable_device(role, channel, profile, expected_version):
    from pyAgxArm import AgxArmFactory, create_agx_arm_config
    from read_nero_firmware import preflight_can_device

    if not preflight_can_device("agx_cando", channel):
        raise RuntimeError(f"Channel {channel}: adapter unavailable")
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot="nero", comm="can", interface="agx_cando", channel=channel,
        bitrate=1_000_000, firmeware_version=profile,
    ))
    try:
        arm.connect()
        firmware = arm.get_firmware(timeout=2.0)
        actual = firmware.get("software_version") if firmware else None
        if actual != expected_version:
            raise RuntimeError(
                f"Channel {channel}: expected {role} firmware {expected_version}, got {actual}"
            )
        gripper = None
        try:
            gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        except Exception as exc:
            print(f"Channel {channel}: effector initialization failed: {exc}", flush=True)
        time.sleep(0.3)
        before = snapshot(arm)
        grip_before = gripper.get_gripper_status() if gripper else None
        baseline = before[0] if before else [float("inf")] * 7
        grip_stamp = float(grip_before.timestamp) if grip_before else float("inf")
        print(f"Channel {channel} IDENTIFIED role={role} firmware={actual}", flush=True)
        print(f"Channel {channel} BEFORE: joints={before[1] if before else None}, "
              f"gripper_enabled={grip_before.msg.foc_status.driver_enable_status if grip_before else None}",
              flush=True)
        for attempt in range(3):
            try:
                arm.disable(255, timeout=0.0)
                print(f"Channel {channel}: joint disable request {attempt + 1} sent", flush=True)
            except Exception as exc:
                print(f"Channel {channel}: joint disable failed: {exc}", flush=True)
            if gripper:
                try:
                    gripper.disable_gripper()
                    print(f"Channel {channel}: AgxGripper disable request {attempt + 1} sent", flush=True)
                except Exception as exc:
                    print(f"Channel {channel}: AgxGripper disable failed: {exc}", flush=True)
            time.sleep(0.2)

        deadline = time.monotonic() + 5.0
        joints_ok = grip_ok = False
        while time.monotonic() < deadline:
            current = snapshot(arm)
            joints_ok = bool(current and not any(current[1]) and all(
                stamp > old for stamp, old in zip(current[0], baseline, strict=True)
            ))
            grip = gripper.get_gripper_status() if gripper else None
            grip_ok = bool(
                grip and grip.hz > 0 and math.isfinite(float(grip.timestamp))
                and grip.timestamp > grip_stamp
                and not grip.msg.foc_status.driver_enable_status
            )
            if joints_ok and grip_ok:
                break
            time.sleep(0.1)
        print(f"Channel {channel} UPDATED FEEDBACK: joints_disabled={joints_ok}, "
              f"agx_gripper_disabled={grip_ok}", flush=True)
        if joints_ok and grip_ok:
            print(f"Channel {channel}: AgxGripper confirmed; any separate pendant is not covered.", flush=True)
        return joints_ok and grip_ok
    finally:
        arm.disconnect(join_timeout=1.0)
        print(f"Channel {channel}: connection released", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--both-supported-empty-confirmed", action="store_true", required=True)
    parser.add_argument("--leader-channel", required=True)
    parser.add_argument("--follower-channel", required=True)
    parser.add_argument("--leader-firmware", choices=PROFILE_CHOICES, required=True)
    parser.add_argument("--follower-firmware", choices=PROFILE_CHOICES, required=True)
    parser.add_argument("--leader-version", required=True)
    parser.add_argument("--follower-version", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.leader_channel == args.follower_channel:
        raise ValueError("Leader and follower channels must differ")
    devices = (
        ("leader", args.leader_channel, args.leader_firmware, args.leader_version),
        ("follower", args.follower_channel, args.follower_firmware, args.follower_version),
    )
    results = []
    for role, channel, profile, expected in devices:
        try:
            results.append(disable_device(role, channel, profile, expected))
        except Exception as exc:
            print(f"Channel {channel}: UNCONFIRMED: {exc}", flush=True)
            results.append(False)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
