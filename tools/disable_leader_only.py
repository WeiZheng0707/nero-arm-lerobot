"""Disable only the identified NERO leader arm and its AgxGripper."""

import math
import time

from lerobot.motors.nero.read_nero_firmware import preflight_can_device
from lerobot.nero_tools.disable_pair import snapshot


def main():
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    channel = "1"
    profile = "v120"
    if not preflight_can_device("agx_cando", channel):
        raise RuntimeError(f"Leader channel {channel}: CAN adapter unavailable")
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot="nero", comm="can", interface="agx_cando", channel=channel,
        bitrate=1_000_000, firmeware_version=profile,
    ))
    try:
        arm.connect()
        firmware = arm.get_firmware(timeout=2.0)
        actual = firmware.get("software_version") if firmware else None
        print(f"LEADER channel={channel} firmware={actual}")
        if actual != "1.20":
            raise RuntimeError(f"Expected leader firmware 1.20, got {actual}")
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        time.sleep(0.3)
        before = snapshot(arm)
        grip_before = gripper.get_gripper_status()
        baseline = before[0] if before else [float("inf")] * 7
        grip_stamp = float(grip_before.timestamp)
        print(f"BEFORE joints={before[1] if before else None} gripper_enabled="
              f"{grip_before.msg.foc_status.driver_enable_status}")
        for attempt in range(3):
            arm.disable(255, timeout=0.0)
            gripper.disable_gripper()
            print(f"disable request {attempt + 1} sent")
            time.sleep(0.25)
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            current = snapshot(arm)
            joints_ok = bool(current and not any(current[1]) and all(
                stamp > old for stamp, old in zip(current[0], baseline, strict=True)
            ))
            grip = gripper.get_gripper_status()
            grip_ok = bool(grip and grip.hz > 0 and math.isfinite(float(grip.timestamp))
                           and grip.timestamp > grip_stamp
                           and not grip.msg.foc_status.driver_enable_status)
            if joints_ok and grip_ok:
                print("CONFIRMED: leader joints disabled and AgxGripper disabled")
                return 0
            time.sleep(0.1)
        print("UNCONFIRMED: leader disable feedback did not advance to disabled")
        return 2
    finally:
        try:
            arm.disconnect(join_timeout=1.0)
        except Exception as exc:
            print(f"Disconnect UNCONFIRMED: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
