"""Read paired NERO effector feedback without enable, disable or movement commands."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/lerobot/motors/nero"))

PROFILE_CHOICES = ("default", "v111", "v112", "v120", "v121")


def inspect(channel, profile, expected):
    from read_nero_firmware import preflight_can_device
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    if not preflight_can_device("agx_cando", channel):
        raise RuntimeError(f"Cannot open channel {channel}")
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot="nero", comm="can", interface="agx_cando", channel=channel,
        bitrate=1_000_000, firmeware_version=profile,
    ))
    try:
        arm.connect()
        firmware = arm.get_firmware(timeout=2.0)
        if not firmware or firmware.get("software_version") != expected:
            raise RuntimeError(f"Channel identity/version changed: {firmware}")
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        print(f"CHANNEL={channel} FIRMWARE={firmware} READ ONLY", flush=True)
        time.sleep(0.3)
        previous = None
        advances = 0
        for _ in range(6):
            arm_status = arm.get_arm_status()
            if arm_status is not None:
                print(f"  arm: stamp={arm_status.timestamp} hz={arm_status.hz:.1f} "
                      f"ctrl_mode={arm_status.msg.ctrl_mode} arm_status={arm_status.msg.arm_status} "
                      f"err_code={arm_status.msg.err_code}", flush=True)
            status = gripper.get_gripper_status()
            if status is None:
                print("  effector status=None", flush=True)
            else:
                stamp = float(status.timestamp)
                if previous is not None and stamp > previous:
                    advances += 1
                previous = stamp
                msg = status.msg
                print(f"  timestamp={stamp} hz={status.hz:.1f} mode={msg.mode} "
                      f"value={msg.value} force_N={msg.force} "
                      f"enabled={msg.foc_status.driver_enable_status}", flush=True)
                faults = {name: getattr(msg.foc_status, name, None) for name in (
                    "voltage_too_low", "motor_overheating", "driver_overcurrent",
                    "driver_overheating", "sensor_status", "driver_error_status", "homing_status",
                )}
                print(f"  gripper status bits={faults}", flush=True)
            control = gripper.get_gripper_ctrl_states()
            if control is not None:
                print(f"  control: stamp={control.timestamp} hz={control.hz:.1f} "
                      f"raw_sdk_value={control.msg.value} status_code={control.msg.status_code}", flush=True)
            time.sleep(0.5)
        print(f"CHANNEL={channel} effector timestamp advances={advances}", flush=True)
    finally:
        arm.disconnect(join_timeout=1.0)
        print(f"CHANNEL={channel} connection released; no enable/disable/movement requested", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-channel", required=True)
    parser.add_argument("--follower-channel", required=True)
    parser.add_argument("--leader-firmware", choices=PROFILE_CHOICES, required=True)
    parser.add_argument("--follower-firmware", choices=PROFILE_CHOICES, required=True)
    parser.add_argument("--leader-version", required=True)
    parser.add_argument("--follower-version", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    devices = (
        (args.leader_channel, args.leader_firmware, args.leader_version),
        (args.follower_channel, args.follower_firmware, args.follower_version),
    )
    if args.leader_channel == args.follower_channel:
        raise ValueError("Leader and follower channels must differ")
    for channel, profile, expected in devices:
        inspect(channel, profile, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
