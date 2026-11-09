"""Supported-arm enable/disable check. Never sends position or gripper commands."""

import argparse
import math
import sys
import time
from pathlib import Path

FAULTS = (
    "voltage_too_low", "motor_overheating", "driver_overcurrent",
    "driver_overheating", "collision_status", "driver_error_status", "stall_status",
)


def snapshot(arm):
    stamps, flags, faults = [], [], []
    for joint in range(1, 8):
        state = arm.get_driver_states(joint)
        if state is None or state.timestamp <= 0 or state.hz <= 0:
            return None
        stamps.append(float(state.timestamp))
        foc = state.msg.foc_status
        flags.append(bool(foc.driver_enable_status))
        faults.extend(f"J{joint}:{name}" for name in FAULTS if getattr(foc, name, False))
    return stamps, flags, faults


def wait_state(arm, enabled, baseline, timeout):
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        result = snapshot(arm)
        if result is not None:
            stamps, flags, faults = result
            text = "".join(str(int(flag)) for flag in flags)
            if text != previous:
                print(f"Driver enable flags: {text}; faults={faults}", flush=True)
                previous = text
            if enabled and faults:
                raise RuntimeError(f"Driver fault during enable: {faults}")
            if all(stamp > old for stamp, old in zip(stamps, baseline, strict=True)):
                if all(flag == enabled for flag in flags):
                    return result
        time.sleep(0.1)
    return None


def disable_effector(gripper, timeout=5.0):
    try:
        before = gripper.get_gripper_status()
        baseline = float(before.timestamp) if before is not None else 0.0
        if not math.isfinite(baseline) or baseline < 0:
            baseline = 0.0
    except Exception:
        baseline = 0.0
    for attempt in range(3):
        try:
            gripper.disable_gripper()
        except Exception as exc:
            print(f"Effector disable attempt {attempt + 1} failed: {exc}", flush=True)
        time.sleep(0.2)
    deadline = time.monotonic() + timeout
    last_stamp = baseline
    first_disabled_at = None
    advancing_disabled = 0
    while time.monotonic() < deadline:
        try:
            status = gripper.get_gripper_status()
            stamp = float(status.timestamp) if status is not None else 0.0
            hz = float(status.hz) if status is not None else 0.0
            valid = math.isfinite(stamp) and stamp > 0 and math.isfinite(hz) and hz > 0
            if valid and stamp > last_stamp:
                last_stamp = stamp
                if status.msg.foc_status.driver_enable_status:
                    first_disabled_at = None
                    advancing_disabled = 0
                else:
                    now = time.monotonic()
                    if first_disabled_at is None:
                        first_disabled_at = now
                    advancing_disabled += 1
                    if advancing_disabled >= 3 and now - first_disabled_at >= 0.5:
                        print("PASS: effector disabled in updated feedback (stable for >=0.5s).", flush=True)
                        return True
            else:
                first_disabled_at = None
                advancing_disabled = 0
        except Exception as exc:
            first_disabled_at = None
            advancing_disabled = 0
            print(f"Effector feedback read failed during disable verification: {exc}", flush=True)
        time.sleep(0.1)
    print("EFFECTOR DISABLE UNCONFIRMED. Keep support in place.", flush=True)
    return False


def cycle(arm, gripper):
    initial = wait_state(arm, False, [0.0] * 7, 5.0)
    if initial is None:
        raise RuntimeError("Precheck failed: seven disabled driver feedback frames required; no enable sent")
    stable = wait_state(arm, False, initial[0], 2.0)
    if stable is None or stable[2]:
        raise RuntimeError("Precheck failed: driver feedback stopped or faults present; no enable sent")
    status = arm.get_arm_status()
    if status is None or status.hz <= 0:
        raise RuntimeError("Arm status unavailable; no enable sent")
    code, errors = int(status.msg.arm_status), int(status.msg.err_code)
    print(f"Precheck arm_status=0x{code:02X}, err_code=0x{errors:04X}", flush=True)
    if code not in (0, 6) or errors:
        raise RuntimeError("Arm status not suitable for enable test; no enable sent")

    try:
        print("Sending follower enable request. No target positions sent.", flush=True)
        arm.enable(255, timeout=0.0)
        enabled = wait_state(arm, True, stable[0], 5.0)
        if enabled is None:
            raise RuntimeError("Enable not confirmed within 5s")
        print("PASS: all seven enable bits confirmed in updated driver frames.", flush=True)
    finally:
        print("Requesting disable now.", flush=True)
        try:
            before = snapshot(arm)
        except Exception:
            before = None
        # No valid baseline means disable may be requested, but cannot be confirmed here.
        baseline = before[0] if before else [float("inf")] * 7
        for attempt in range(3):
            try:
                arm.disable(255, timeout=0.0)
            except Exception as exc:
                print(f"Disable attempt {attempt + 1} raised: {exc}", flush=True)
            time.sleep(0.2)
        # Effector cleanup must run even if joint-state polling fails.
        try:
            disabled = wait_state(arm, False, baseline, 5.0)
        finally:
            effector_disabled = disable_effector(gripper)
        if disabled is None:
            raise RuntimeError("DISABLE UNCONFIRMED. Keep support in place; verify on site.")
        if not effector_disabled:
            raise RuntimeError("Joint disable confirmed, but effector disable UNCONFIRMED")
        print("PASS: all seven disable bits confirmed in updated driver frames. Keep support in place.",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--supported-arm-confirmed", action="store_true", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/lerobot/motors/nero"))
    from read_nero_firmware import preflight_can_device
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    if not preflight_can_device("agx_cando", args.channel):
        raise RuntimeError("Adapter preflight failed; no enable sent")
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot="nero", comm="can", interface="agx_cando", channel=args.channel,
        bitrate=1_000_000, firmeware_version="v121",
    ))
    try:
        arm.connect()
        firmware = arm.get_firmware(timeout=2.0)
        print(f"Channel {args.channel}: actual firmware={firmware}", flush=True)
        if not firmware or firmware.get("software_version") != "1.21":
            raise RuntimeError("Expected the identified 1.21 follower; no enable sent")
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        time.sleep(0.3)
        cycle(arm, gripper)
    finally:
        arm.disconnect(join_timeout=1.0)
        print("Connection cleanup completed.", flush=True)


if __name__ == "__main__":
    main()
