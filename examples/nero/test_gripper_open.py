"""Bounded empty-follower gripper test: open to 5 mm, then disable, never close."""

import argparse
import math
import sys
import time
from pathlib import Path

from test_enable_cycle import disable_effector, snapshot, wait_state


def read_grip(gripper):
    status = gripper.get_gripper_status()
    if status is None or status.hz <= 0 or status.timestamp <= 0:
        raise RuntimeError("Gripper feedback missing")
    msg = status.msg
    value = float(msg.value)
    if msg.mode != "width" or not math.isfinite(value) or not 0 <= value <= 0.007:
        raise RuntimeError(f"Unexpected gripper mode/position: {msg.mode}, {value}")
    for name in ("voltage_too_low", "motor_overheating", "driver_overcurrent",
                 "driver_overheating", "sensor_status", "driver_error_status"):
        if getattr(msg.foc_status, name, False):
            raise RuntimeError(f"Gripper fault: {name}")
    return float(status.timestamp), value, bool(msg.foc_status.driver_enable_status)


def check_arm_status(arm):
    status = arm.get_arm_status()
    if status is None or status.hz <= 0 or status.timestamp <= 0:
        raise RuntimeError("Arm status unavailable; refusing gripper target")
    mode = int(status.msg.ctrl_mode)
    code = int(status.msg.arm_status)
    errors = int(status.msg.err_code)
    if mode != 1 or code not in (0, 6) or errors:
        raise RuntimeError(
            f"Gripper test blocked: ctrl_mode=0x{mode:02X}, arm_status=0x{code:02X}, "
            f"err_code=0x{errors:04X}; require CAN control and no communication faults"
        )


def run_test(arm, gripper):
    joints = wait_state(arm, False, [0.0] * 7, 3.0)
    if joints is None or joints[2]:
        raise RuntimeError("Seven disabled, fault-free joint feedback frames required")
    check_arm_status(arm)
    first = read_grip(gripper)
    time.sleep(0.2)
    current = read_grip(gripper)
    if current[0] <= first[0] or current[2] or current[1] > 0.005:
        raise RuntimeError("Require updating disabled gripper, initially no wider than 5 mm")
    print(f"START width_mm={current[1] * 1000:.3f}; target_mm=5.000; force_N=1.0", flush=True)
    target = current[1]
    last_joint_stamps = joints[0]
    try:
        for step in range(1, 21):
            check_arm_status(arm)
            target = min(0.005, target + 0.0005)
            baseline = current[0]
            gripper.move_gripper_m(value=target, force=1.0)
            sent_at = time.monotonic()
            deadline = sent_at + 3.0
            while True:
                time.sleep(0.1)
                current = read_grip(gripper)
                check_arm_status(arm)
                state = snapshot(arm)
                print(f"step={step} elapsed_s={time.monotonic() - sent_at:.3f} "
                      f"target_mm={target * 1000:.3f} actual_mm={current[1] * 1000:.3f} "
                      f"timestamp={current[0]:.6f} enabled={current[2]} "
                      f"joint_flags={state[1] if state else None}", flush=True)
                if state is None or any(state[1]) or state[2]:
                    raise RuntimeError("Joint feedback missing, enabled or faulted; stopping gripper test")
                if current[0] > baseline and all(
                    new > old for new, old in zip(state[0], last_joint_stamps, strict=True)
                ) and current[2] and abs(current[1] - target) <= 0.0002:
                    last_joint_stamps = state[0]
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Gripper target/enable or fresh joint feedback not confirmed within 3s")
            print(f"target_mm={target * 1000:.3f}; actual_mm={current[1] * 1000:.3f}; "
                  f"enabled={current[2]}", flush=True)
            if target >= 0.005 - 1e-9:
                print("PASS: 5 mm target reached within 0.2 mm feedback tolerance.", flush=True)
                return
        raise RuntimeError("Step limit reached")
    finally:
        # This executes even if the first movement call raises after transmitting.
        print("Requesting effector disable; no closing target will be sent.", flush=True)
        if not disable_effector(gripper):
            raise RuntimeError("EFFECTOR DISABLE UNCONFIRMED; keep clear and retain support")
        state = snapshot(arm)
        print(f"Final joint enable feedback: {state[1] if state else 'UNAVAILABLE'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--empty-supported-clear-confirmed", action="store_true", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/lerobot/motors/nero"))
    from read_nero_firmware import preflight_can_device
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    if not preflight_can_device("agx_cando", args.channel):
        raise RuntimeError("Adapter unavailable")
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot="nero", comm="can", interface="agx_cando", channel=args.channel,
        bitrate=1_000_000, firmeware_version="v121",
    ))
    try:
        arm.connect()
        firmware = arm.get_firmware(timeout=2.0)
        print(f"Channel={args.channel}, firmware={firmware}", flush=True)
        if not firmware or firmware.get("software_version") != "1.21":
            raise RuntimeError("Expected identified 1.21 follower; refusing gripper command")
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        time.sleep(0.3)
        run_test(arm, gripper)
    finally:
        arm.disconnect(join_timeout=1.0)
        print("Connection released.", flush=True)


if __name__ == "__main__":
    main()
