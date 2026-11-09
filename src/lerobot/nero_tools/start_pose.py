"""Persistent joint-only startup poses and bounded position-mode return."""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from .test_enable_cycle import snapshot
from .test_follower_j1 import check_limits, pose

DEFAULT_PATH = Path(__file__).resolve().parent / "saved_start_pose.json"
VERSIONS = {"leader": "1.20", "follower": "1.21"}
MAX_RETURN_DEG = 60.0


def check_return_span(current, goal, role):
    deltas = [math.degrees(b-a) for a,b in zip(current, goal, strict=True)]
    if max(map(abs, deltas)) > MAX_RETURN_DEG + 1e-9:
        raise RuntimeError(f"{role}: return exceeds {MAX_RETURN_DEG:g}deg per axis; "
                           f"delta_deg={[round(v,3) for v in deltas]}")


def validate(data):
    if data.get("schema") != 1 or data.get("unit") != "rad":
        raise ValueError("Unsupported startup pose schema/unit")
    for role, version in VERSIONS.items():
        entry = data[role]
        values = entry["joints"]
        if entry["firmware"] != version or len(values) != 7:
            raise ValueError(f"Invalid {role} version/dimensions")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
            raise ValueError(f"Invalid {role} joint angles")
    return data


def save(path, leader, follower, overwrite=False):
    data = validate({"schema": 1, "unit": "rad", "saved_at": datetime.now(timezone.utc).isoformat(),
                     "leader": {"firmware": VERSIONS["leader"], "joints": list(leader)},
                     "follower": {"firmware": VERSIONS["follower"], "joints": list(follower)}})
    # Exclusive creation prevents accidental replacement of a known starting pose.
    with Path(path).open("w" if overwrite else "x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return validate(json.load(handle))


def validate_return_tuning(speed_deg_s, controller_percent):
    if not math.isfinite(speed_deg_s) or not 0 < speed_deg_s <= 5:
        raise ValueError("--return-speed-deg-s must be finite and in (0, 5]")
    if type(controller_percent) is not int or not 1 <= controller_percent <= 20:
        raise ValueError("--return-speed-percent must be an integer in [1, 20]")


def next_target(previous, goal, dt, speed_deg_s=1.0):
    validate_return_tuning(speed_deg_s, 5)
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("Return timestep must be finite and nonnegative")
    cap = math.radians(speed_deg_s) * min(dt, 0.05)
    return [a + max(-cap, min(cap, b-a)) for a, b in zip(previous, goal, strict=True)]


def return_to_pose(arm, goal, monitor, role, observer=None,
                   speed_deg_s=1.0, controller_percent=5):
    """Bound reference speed independently of teleoperation; no collision planning."""
    validate_return_tuning(speed_deg_s, controller_percent)
    limits = arm.get_config()["joint_limits"]
    stamp, start = pose(arm)
    check_limits(start, limits)
    check_limits(goal, limits)
    span = max(abs(a-b) for a,b in zip(start, goal, strict=True))
    check_return_span(start, goal, role)
    arm.set_joint_limits_enabled(True)
    arm.set_speed_percent(controller_percent)
    arm.set_motion_mode(arm.OPTIONS.MOTION_MODE.J)
    sent = list(start)
    began = last_tick = last_fresh = time.monotonic()
    deadline = began + math.degrees(span) / speed_deg_s * 2 + 8
    print(f"RETURN {role}: reference_speed={speed_deg_s:g}deg/s "
          f"controller_speed={controller_percent}%", flush=True)
    next_log = began
    reached = 0
    while time.monotonic() < deadline:
        now = time.monotonic()
        monitor.poll(arm)
        if observer:
            observer()
        state = snapshot(arm)
        if state is None or not all(state[1]) or state[2]:
            raise RuntimeError(f"{role}: driver state invalid during return")
        new_stamp, actual = pose(arm)
        if new_stamp < stamp:
            raise RuntimeError(f"{role}: feedback timestamp regressed")
        if new_stamp == stamp:
            if now-last_fresh > 0.5:
                raise TimeoutError(f"{role}: return feedback stale")
            last_tick = now
            time.sleep(.05)
            continue
        stamp, last_fresh = new_stamp, now
        check_limits(actual, limits)
        if max(abs(a-b) for a,b in zip(actual,sent,strict=True)) > math.radians(1.5):
            raise RuntimeError(f"{role}: return tracking error >1.5deg")
        sent = next_target(sent, goal, now-last_tick, speed_deg_s)
        check_limits(sent, limits)
        arm.move_j(sent)
        error = max(abs(a-b) for a,b in zip(actual,goal,strict=True))
        command_done = max(abs(a-b) for a,b in zip(sent,goal,strict=True)) < 1e-9
        reached = reached+1 if command_done and error <= math.radians(.2) else 0
        if now >= next_log:
            print(f"RETURN {role} actual_deg={[round(math.degrees(v),3) for v in actual]} "
                  f"max_error_deg={math.degrees(error):.3f}", flush=True)
            next_log = now+.5
        if reached >= 5:
            print(f"RETURN {role}: target confirmed in five updated feedback frames.", flush=True)
            return
        last_tick = now
        time.sleep(.05)
    raise TimeoutError(f"{role}: saved start pose not reached")
