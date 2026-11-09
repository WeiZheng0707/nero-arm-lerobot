"""Phased direct test for the 70 mm JXB2 NERO gripper.

Run one stage at a time: inspect -> prepare -> calibrate -> open.
No stage enables or moves the seven arm joints.
"""

import argparse
import math
import sys
import time
from pathlib import Path

from .test_enable_cycle import disable_effector, snapshot, wait_state


def grip_status(gripper):
    status = gripper.get_gripper_status()
    if status is None or status.hz <= 0 or status.timestamp <= 0:
        raise RuntimeError("Fresh gripper feedback is required")
    return status


def print_state(arm, gripper):
    arm_status = arm.get_arm_status()
    joints = snapshot(arm)
    grip = grip_status(gripper)
    params = gripper.get_gripper_teaching_pendant_param(timeout=2.0, min_interval=0.0)
    print(
        f"arm ctrl_mode={arm_status.msg.ctrl_mode if arm_status else None} "
        f"status={arm_status.msg.arm_status if arm_status else None} "
        f"err={arm_status.msg.err_code if arm_status else None}", flush=True
    )
    print(f"joint enable flags={joints[1] if joints else None}; faults={joints[2] if joints else None}",
          flush=True)
    msg = grip.msg
    print(f"gripper mode={msg.mode} width_m={msg.value} force_N={msg.force} "
          f"enabled={msg.foc_status.driver_enable_status} homed={msg.foc_status.homing_status} "
          f"status_code=0x{msg.status_code:02X}", flush=True)
    print(f"gripper params={params.msg if params else None}", flush=True)
    return arm_status, joints, grip, params


def require_joints_disabled(arm):
    state = wait_state(arm, False, [0.0] * 7, 3.0)
    if state is None or state[2]:
        raise RuntimeError("Seven fresh, disabled and fault-free joint frames are required")
    return state


def take_can_control(arm):
    before = require_joints_disabled(arm)
    baseline = arm.get_arm_status()
    if baseline is None:
        raise RuntimeError("Arm status unavailable; CAN control not requested")
    baseline_stamp = float(baseline.timestamp)
    arm.set_motion_mode(arm.OPTIONS.MOTION_MODE.P)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        status = arm.get_arm_status()
        state = snapshot(arm)
        if (status is not None and status.timestamp > baseline_stamp and int(status.msg.ctrl_mode) == 1
                and state is not None and not any(state[1]) and not state[2]
                and all(new > old for new, old in zip(state[0], before[0], strict=True))):
            print("PASS: CAN control confirmed; all seven arm joints remain disabled.", flush=True)
            return
        time.sleep(0.1)
    raise RuntimeError("CAN control or disabled-joint feedback not confirmed")


def require_params_70mm(gripper):
    params = gripper.get_gripper_teaching_pendant_param(timeout=2.0, min_interval=0.0)
    if params is None:
        raise RuntimeError("Gripper parameters unavailable")
    msg = params.msg
    if abs(float(msg.max_range_config) - 0.07) > 1e-9:
        raise RuntimeError(f"Expected max_range_config=0.07 m, got {msg.max_range_config}")
    return params


def stage_prepare(arm, gripper):
    require_joints_disabled(arm)
    take_can_control(arm)
    if not disable_effector(gripper):
        raise RuntimeError("Gripper disable unconfirmed after taking CAN control")
    ok = gripper.set_gripper_teaching_pendant_param(
        teaching_range_per=100, max_range_config=0.07, teaching_friction=1, timeout=2.0
    )
    if not ok:
        raise RuntimeError("70 mm gripper parameter write/readback failed")
    params = require_params_70mm(gripper)
    print(f"PASS: 70 mm parameter read back: {params.msg}", flush=True)
    if not disable_effector(gripper):
        raise RuntimeError("Final gripper disable unconfirmed")


def stage_calibrate(arm, gripper):
    require_joints_disabled(arm)
    take_can_control(arm)
    if not disable_effector(gripper):
        raise RuntimeError("Gripper disable unconfirmed after taking CAN control")
    require_params_70mm(gripper)
    if not gripper.calibrate_gripper(timeout=2.0):
        raise RuntimeError("Gripper zero calibration was not acknowledged")
    deadline = time.monotonic() + 3.0
    homed = False
    while time.monotonic() < deadline:
        status = grip_status(gripper)
        homed = bool(status.msg.foc_status.homing_status)
        if homed:
            break
        time.sleep(0.1)
    if not homed:
        raise RuntimeError("Calibration ACK received but homing_status stayed false")
    print("PASS: gripper zero calibration and homing feedback confirmed.", flush=True)
    if not disable_effector(gripper):
        raise RuntimeError("Final gripper disable unconfirmed")


def stage_open(arm, gripper):
    require_joints_disabled(arm)
    take_can_control(arm)
    require_params_70mm(gripper)
    start = grip_status(gripper)
    if not start.msg.foc_status.homing_status:
        raise RuntimeError("Gripper is not homed; run calibrate stage first")
    if start.msg.mode != "width" or not 0 <= float(start.msg.value) <= 0.003:
        raise RuntimeError(f"Expected closed width-mode start, got {start.msg.mode} {start.msg.value}")
    try:
        gripper.move_gripper_m(value=0.010, force=1.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            status = grip_status(gripper)
            state = snapshot(arm)
            width = float(status.msg.value)
            print(f"target_mm=10.0 actual_mm={width * 1000:.3f} "
                  f"enabled={status.msg.foc_status.driver_enable_status}", flush=True)
            if state is None or any(state[1]) or state[2]:
                raise RuntimeError("Arm joints enabled/faulted during gripper-only test")
            if status.msg.foc_status.driver_enable_status and abs(width - 0.010) <= 0.001:
                print("PASS: gripper opened to 10 mm within 1 mm tolerance.", flush=True)
                return
            time.sleep(0.2)
        raise TimeoutError("10 mm gripper target not confirmed within 5 seconds")
    finally:
        print("Requesting gripper disable; no closing target sent.", flush=True)
        if not disable_effector(gripper):
            raise RuntimeError("GRIPPER DISABLE UNCONFIRMED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "prepare", "calibrate", "open"))
    parser.add_argument("--channel", required=True)
    parser.add_argument("--firmware", choices=("v120", "v121"), required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--hardware-confirmed", action="store_true")
    parser.add_argument("--manually-closed-confirmed", action="store_true")
    args = parser.parse_args()
    if args.stage != "inspect" and not args.hardware_confirmed:
        parser.error("write stages require --hardware-confirmed")
    if args.stage == "calibrate" and not args.manually_closed_confirmed:
        parser.error("calibrate requires --manually-closed-confirmed")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/lerobot/motors/nero"))
    from read_nero_firmware import preflight_can_device
    from pyAgxArm import AgxArmFactory, create_agx_arm_config
    if not preflight_can_device("agx_cando", args.channel):
        raise RuntimeError("Adapter unavailable")
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot="nero", comm="can", interface="agx_cando", channel=args.channel,
        bitrate=1_000_000, firmeware_version=args.firmware,
    ))
    gripper = None
    try:
        arm.connect()
        firmware = arm.get_firmware(timeout=2.0)
        print(f"channel={args.channel} firmware={firmware} stage={args.stage}", flush=True)
        if not firmware or firmware.get("software_version") != args.expected_version:
            raise RuntimeError("Channel identity/version mismatch")
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        time.sleep(0.3)
        if args.stage == "inspect":
            print_state(arm, gripper)
        elif args.stage == "prepare":
            stage_prepare(arm, gripper)
        elif args.stage == "calibrate":
            stage_calibrate(arm, gripper)
        else:
            stage_open(arm, gripper)
    finally:
        if gripper is not None and args.stage != "inspect":
            disable_effector(gripper)
        arm.disconnect(join_timeout=1.0)
        print("Connection released.", flush=True)


if __name__ == "__main__":
    main()
