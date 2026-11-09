"""Read-only real-time AgxGripper distance capture for one NERO arm."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from test_enable_cycle import snapshot


def read_sample(arm, gripper, role: str, firmware: str) -> dict:
    status = gripper.get_gripper_status()
    if status is None or status.hz <= 0 or status.timestamp <= 0:
        raise RuntimeError("No fresh gripper feedback")

    msg = status.msg
    value = float(msg.value)
    if not math.isfinite(value):
        raise RuntimeError("Gripper distance is not finite")

    joints = snapshot(arm)
    arm_status = arm.get_arm_status()
    params = gripper.get_gripper_teaching_pendant_param(timeout=0.0, min_interval=1.0)
    param_msg = params.msg if params is not None else None
    mode = str(msg.mode)
    distance_m = value if mode == "width" else None
    angle_deg = value if mode == "angle" else None

    return {
        "host_time_s": time.time(),
        "feedback_timestamp": float(status.timestamp),
        "feedback_hz": float(status.hz),
        "mode": mode,
        "distance_m": distance_m,
        "distance_mm": distance_m * 1000.0 if distance_m is not None else None,
        "angle_deg": angle_deg,
        "force_n": float(msg.force),
        "status_code": int(msg.status_code),
        "gripper_enabled": bool(msg.foc_status.driver_enable_status),
        "homing_status": bool(msg.foc_status.homing_status),
        "faults": {
            name: bool(getattr(msg.foc_status, name, False))
            for name in (
                "voltage_too_low", "motor_overheating", "driver_overcurrent",
                "driver_overheating", "sensor_status", "driver_error_status",
            )
        },
        "arm_ctrl_mode": str(arm_status.msg.ctrl_mode) if arm_status else None,
        "arm_status": str(arm_status.msg.arm_status) if arm_status else None,
        "arm_err_code": int(arm_status.msg.err_code) if arm_status else None,
        "joint_enable_flags": joints[1] if joints else None,
        "joint_faults": joints[2] if joints else None,
        "configured_range_m": (
            float(param_msg.max_range_config) if param_msg is not None else None
        ),
        "configured_range_mm": (
            float(param_msg.max_range_config) * 1000.0 if param_msg is not None else None
        ),
        "teaching_range_percent": (
            int(param_msg.teaching_range_per) if param_msg is not None else None
        ),
        "teaching_friction": (
            int(param_msg.teaching_friction) if param_msg is not None else None
        ),
        "role": role,
        "firmware": firmware,
    }


def save_capture(path: Path, metadata: dict, samples: list[dict]) -> None:
    distances = [s["distance_mm"] for s in samples if s["distance_mm"] is not None]
    result = {
        "schema": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "sample_count": len(samples),
        "distance_min_mm": min(distances) if distances else None,
        "distance_max_mm": max(distances) if distances else None,
        "distance_span_mm": (max(distances) - min(distances)) if distances else None,
        "samples": samples,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("leader", "follower"), required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--firmware", choices=("v120", "v121"), required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--seconds", type=float, default=None,
                        help="Capture duration; omit for Ctrl+C until stopped")
    parser.add_argument("--sample-period", type=float, default=0.1,
                        help="Local polling period in seconds, default 0.1")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    if not math.isfinite(args.sample_period) or not 0.02 <= args.sample_period <= 1.0:
        parser.error("--sample-period must be between 0.02 and 1.0 seconds")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/lerobot/motors/nero"))
    from pyAgxArm import AgxArmFactory, create_agx_arm_config
    from read_nero_firmware import preflight_can_device

    if not preflight_can_device("agx_cando", args.channel):
        raise RuntimeError("CAN adapter preflight failed")

    output = args.output or Path(__file__).resolve().parent / (
        f"gripper_capture_{args.role}_channel_{args.channel}.json"
    )
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot="nero", comm="can", interface="agx_cando", channel=args.channel,
        bitrate=1_000_000, firmeware_version=args.firmware,
    ))
    samples: list[dict] = []
    started = time.monotonic()
    try:
        arm.connect()
        firmware = arm.get_firmware(timeout=2.0)
        print(f"role={args.role} channel={args.channel} firmware={firmware}", flush=True)
        if not firmware or firmware.get("software_version") != args.expected_version:
            raise RuntimeError("Channel/firmware identity mismatch; capture stopped")
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        print("READ ONLY: no enable, disable, mode, calibration, or movement commands.", flush=True)
        print(f"Capture until {'Ctrl+C' if args.seconds is None else f'{args.seconds:g}s'}; "
              f"output={output}", flush=True)

        while args.seconds is None or time.monotonic() - started < args.seconds:
            sample_started = time.monotonic()
            try:
                sample = read_sample(arm, gripper, args.role, args.expected_version)
            except RuntimeError as exc:
                print(f"SAMPLE_REJECTED: {exc}", flush=True)
                time.sleep(args.sample_period)
                continue
            samples.append(sample)
            distance = (f"{sample['distance_mm']:.3f} mm"
                        if sample["distance_mm"] is not None
                        else f"{sample['angle_deg']:.3f} deg")
            print(
                f"t={sample['host_time_s']:.3f} distance={distance} "
                f"force={sample['force_n']:.3f}N mode={sample['mode']} "
                f"enabled={sample['gripper_enabled']} homed={sample['homing_status']} "
                f"range={sample['configured_range_mm']}mm "
                f"joints={sample['joint_enable_flags']}",
                flush=True,
            )
            time.sleep(max(0.0, args.sample_period - (time.monotonic() - sample_started)))
    except KeyboardInterrupt:
        print("Capture interrupted by user; no hardware state was changed.", flush=True)
    finally:
        try:
            arm.disconnect(join_timeout=1.0)
        finally:
            metadata = {
                "role": args.role, "channel": args.channel,
                "firmware": args.expected_version,
                "sample_period_s": args.sample_period,
                "read_only": True,
            }
            save_capture(output, metadata, samples)
            print(f"Saved {len(samples)} samples to {output.resolve()}", flush=True)
            print("Connection released. Disconnect does not change hardware enable state.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
