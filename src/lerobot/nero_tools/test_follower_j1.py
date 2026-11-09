"""One supported, empty follower test: J1 +3 degrees, no return trajectory."""

import argparse
import math
import sys
import time
from pathlib import Path

from .gripper_direct_test import take_can_control
from .test_enable_cycle import disable_effector, snapshot, wait_state
from .test_leader_drag import StatusMonitor, finish, status_snapshot, wait_ready_status


def pose(arm):
    msg = arm.get_joint_angles()
    if msg is None or not math.isfinite(float(msg.hz)) or msg.hz <= 0:
        raise RuntimeError("Joint feedback missing")
    stamp, values = float(msg.timestamp), list(msg.msg)
    if not math.isfinite(stamp) or stamp <= 0 or len(values) != 7 or not all(map(math.isfinite, values)):
        raise RuntimeError("Invalid joint feedback")
    return stamp, values


def target_at(start, elapsed):
    target = list(start)
    alpha = min(1.0, max(0.0, elapsed / 6.0))
    target[0] += math.radians(3) * (0.5 - 0.5 * math.cos(math.pi * alpha))
    return target


def check_limits(values, limits):
    for index, value in enumerate(values, 1):
        low, high = limits[f"joint{index}"]
        if not low <= value <= high:
            raise RuntimeError(f"J{index} outside SDK limits; refusing target rather than clipping")


def check_deviation(start, actual, target):
    delta = [math.degrees(a - b) for a, b in zip(actual, start, strict=True)]
    if any(abs(x) > 1.0 for x in delta[1:]) or not -0.5 <= delta[0] <= 3.5:
        raise RuntimeError(f"Unexpected joint movement: delta_deg={delta}")
    if abs(actual[0] - target[0]) > math.radians(1.5):
        raise RuntimeError("J1 tracking error exceeds 1.5 degrees")
    return delta


def run(arm, gripper):
    initial = wait_state(arm, False, [0.0]*7, 3)
    if initial is None or initial[2]:
        raise RuntimeError("Disabled fault-free joints required")
    health = status_snapshot(arm)
    if health is None or health[2] not in (0, 6) or health[3]:
        raise RuntimeError("Arm status precheck failed")
    limits = arm.get_config()["joint_limits"]
    stamp, start = pose(arm)
    for _ in range(5):
        time.sleep(0.2)
        updated, values = pose(arm)
        if updated <= stamp or max(abs(a-b) for a,b in zip(values,start,strict=True)) > math.radians(0.2):
            raise RuntimeError("Pose not stable/fresh; no enable sent")
        stamp = updated
    start = list(values)
    check_limits(start, limits)
    check_limits(target_at(start, 6), limits)
    print(f"START_DEG={[round(math.degrees(v),3) for v in start]}", flush=True)
    print(f"TARGET_DEG={[round(math.degrees(v),3) for v in target_at(start,6)]}", flush=True)
    try:
        take_can_control(arm)
        if not disable_effector(gripper):
            raise RuntimeError("Effector disable unconfirmed; no joint enable")
        stamp, current = pose(arm)
        if max(abs(a-b) for a,b in zip(current,start,strict=True)) > math.radians(0.2):
            raise RuntimeError("Pose changed during preparation")
        health = status_snapshot(arm)
        if health is None or health[2] not in (0,6) or health[3]:
            raise RuntimeError("Status changed during preparation")
        arm.set_joint_limits_enabled(True)
        arm.set_speed_percent(5)
        arm.set_motion_mode(arm.OPTIONS.MOTION_MODE.J)
        before = snapshot(arm)
        if before is None or any(before[1]) or before[2]:
            raise RuntimeError("Pre-enable driver status invalid")
        arm.enable(255, timeout=0.0)
        if wait_state(arm, True, before[0], 5) is None:
            raise RuntimeError("Enable unconfirmed")
        monitor = wait_ready_status(arm, "follower_after_enable", health[0])
        if not disable_effector(gripper):
            raise RuntimeError("Effector disable unconfirmed after arm enable")
        started = last_advance = time.monotonic()
        next_log = started
        sent = list(start)
        reached = 0
        while time.monotonic() - started < 9:
            now = time.monotonic()
            elapsed = now - started
            monitor.poll(arm)
            fresh, actual = pose(arm)
            if fresh < stamp:
                raise RuntimeError("Joint timestamp moved backwards")
            if fresh == stamp:
                if now - last_advance > 0.5:
                    raise TimeoutError("Joint feedback stopped")
                time.sleep(0.05)
                continue
            stamp, last_advance = fresh, now
            state = snapshot(arm)
            if state is None or not all(state[1]) or state[2]:
                raise RuntimeError("Joint enable lost or driver fault")
            check_limits(actual, limits)
            delta = check_deviation(start, actual, sent)
            sent = target_at(start, elapsed)
            check_limits(sent, limits)
            arm.move_j(sent)
            if now >= next_log:
                print(f"t={elapsed:.2f}s target_delta={math.degrees(sent[0]-start[0]):.3f} "
                      f"actual_delta_deg={[round(v,3) for v in delta]}", flush=True)
                next_log = now + 0.5
            if elapsed >= 6 and abs(delta[0]-3) <= 0.3 and max(map(abs,delta[1:])) <= 0.5:
                reached += 1
                if reached >= 3:
                    print("PASS: J1 +3deg within 0.3deg; other joints within 0.5deg.", flush=True)
                    return
            else:
                reached = 0
            time.sleep(0.05)
        raise TimeoutError("Target not confirmed within 9 seconds")
    finally:
        print("Ending: request joint and effector disable. No return motion.", flush=True)
        finish(arm, gripper, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--supported-clear-j1-positive-confirmed", action="store_true", required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src/lerobot/motors/nero"))
    from read_nero_firmware import preflight_can_device
    from pyAgxArm import AgxArmFactory, create_agx_arm_config
    if not preflight_can_device("agx_cando",args.channel):
        raise RuntimeError("Adapter unavailable")
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot="nero",comm="can",interface="agx_cando",channel=args.channel,
        bitrate=1000000,firmeware_version="v121"))
    try:
        arm.connect()
        fw = arm.get_firmware(timeout=2)
        print(f"Channel={args.channel} firmware={fw}",flush=True)
        if not fw or fw.get("software_version") != "1.21":
            raise RuntimeError("Expected identified follower 1.21")
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        time.sleep(0.3)
        run(arm,gripper)
    finally:
        arm.disconnect(join_timeout=1)
        print("Connection released.",flush=True)


if __name__ == "__main__":
    main()
