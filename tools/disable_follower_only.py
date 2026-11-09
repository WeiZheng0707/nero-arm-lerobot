"""Disable the follower arm's joints and its AgxGripper, then confirm from feedback.

The joint enable bits and the gripper enable bit are independent: a run that
disables the seven joints can leave the gripper powered and stiff. That has
happened on this rig (see AGENTS.md), so a disable-only tool has to touch both
and verify both.

This tool only disables. It never commands motion. `connect()` still powers the
arm, so the caller must have the arm supported before running it.

Success is reported only when the feedback timestamps advance AND every joint
reports disabled AND the gripper reports disabled. Anything less prints
UNCONFIRMED and exits non-zero.

Usage:
    python -X utf8 tools\disable_follower_only.py --channel 0 --firmware v121
"""

from __future__ import annotations

import argparse
import math
import time

from lerobot.motors.nero.read_nero_firmware import preflight_can_device
from lerobot.nero_tools.disable_pair import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0")
    parser.add_argument("--firmware", default="v121")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--verify-seconds", type=float, default=6.0)
    args = parser.parse_args()

    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    if not preflight_can_device("agx_cando", args.channel):
        raise SystemExit(f"channel {args.channel}: CAN adapter unavailable")

    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot="nero", comm="can", interface="agx_cando", channel=args.channel,
        bitrate=1_000_000, firmeware_version=args.firmware,
    ))
    try:
        arm.connect()
        firmware = arm.get_firmware(timeout=2.0)
        actual = firmware.get("software_version") if firmware else None
        print(f"FOLLOWER channel={args.channel} firmware={actual} (requested {args.firmware})")

        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        time.sleep(0.3)
        before = snapshot(arm)
        grip_before = gripper.get_gripper_status()
        baseline = before[0] if before else [float("inf")] * 7
        grip_stamp = float(grip_before.timestamp)
        print(f"BEFORE joints_enabled={before[1] if before else None} "
              f"gripper_enabled={grip_before.msg.foc_status.driver_enable_status}")

        for attempt in range(args.attempts):
            arm.disable(255, timeout=0.0)
            gripper.disable_gripper()
            print(f"disable request {attempt + 1} sent for joints and gripper", flush=True)
            time.sleep(0.25)

        deadline = time.monotonic() + args.verify_seconds
        while time.monotonic() < deadline:
            current = snapshot(arm)
            joints_ok = bool(current and not any(current[1]) and all(
                stamp > old for stamp, old in zip(current[0], baseline, strict=True)))
            grip = gripper.get_gripper_status()
            grip_ok = bool(grip and grip.hz > 0 and math.isfinite(float(grip.timestamp))
                           and grip.timestamp > grip_stamp
                           and not grip.msg.foc_status.driver_enable_status)
            if joints_ok and grip_ok:
                print("CONFIRMED: follower joints disabled and AgxGripper disabled")
                return 0
            time.sleep(0.1)

        current = snapshot(arm)
        grip = gripper.get_gripper_status()
        print("UNCONFIRMED: feedback did not advance to disabled")
        print(f"  joints_enabled={current[1] if current else None}")
        if grip:
            print(f"  gripper_enabled={grip.msg.foc_status.driver_enable_status} "
                  f"timestamp={float(grip.timestamp):.3f} baseline={grip_stamp:.3f}")
        print("  Keep the arm supported and power it off before touching the gripper.")
        return 2
    finally:
        try:
            arm.disconnect(join_timeout=1.0)
        except Exception as exc:  # noqa: BLE001
            print(f"Disconnect UNCONFIRMED: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
