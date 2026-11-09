"""Interactive rate-limited relative teleoperation; optional saved-pose return."""

import argparse
import math
import sys
import time
from pathlib import Path

from gripper_direct_test import take_can_control
from test_enable_cycle import disable_effector, snapshot, wait_state
from test_follower_j1 import check_limits, pose
from test_leader_drag import StatusMonitor, finish, status_snapshot, wait_ready_status
import start_pose as presets


def validate_tuning(range_deg, speed_deg_s):
    for name, value, maximum in (("range-deg", range_deg, 15.0),
                                  ("speed-deg-s", speed_deg_s, 30.0)):
        if not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"--{name} must be finite and in (0, {maximum}]")


def validate_control(fps, follower_speed_percent, accel_deg_s2):
    if type(fps) is not int or not 20 <= fps <= 60:
        raise ValueError("--fps must be an integer in [20, 60]")
    if type(follower_speed_percent) is not int or not 1 <= follower_speed_percent <= 30:
        raise ValueError("--follower-speed-percent must be an integer in [1, 30]")
    if accel_deg_s2 is not None and (not math.isfinite(accel_deg_s2) or not 0 < accel_deg_s2 <= 60):
        raise ValueError("--accel-deg-s2 must be finite and in (0, 60]")


class ReferenceStep:
    """Bound discrete reference velocity and acceleration, with hard position bounds.

    An abrupt target reversal can leave a reference moving away while it brakes.
    At a hard position bound the bound takes priority and resets velocity.
    """
    def __init__(self, speed, period, acceleration=None):
        if not math.isfinite(period) or not 0 < period <= .05:
            raise ValueError("Reference period must be in (0, 0.05]")
        if acceleration is not None and (not math.isfinite(acceleration) or acceleration <= 0):
            raise ValueError("Reference acceleration must be positive")
        self.speed = speed
        self.period = period
        self.acceleration = acceleration
        self.velocity = 0.0

    def step(self, previous, desired, dt, bounds):
        dt = min(dt, self.period)
        if dt == 0:
            return previous, False, False
        distance = desired - previous
        wanted = max(-self.speed, min(self.speed, distance / dt))
        speed_limited = abs(distance) > self.speed * dt + 1e-12
        accel_limited = False
        if self.acceleration is not None:
            change = self.acceleration * dt
            # Discrete braking distance includes this step and all subsequent
            # positive-velocity steps; continuous v^2/(2a) can undershoot it.
            low, high = 0.0, abs(wanted)
            for _ in range(32):
                candidate = (low + high) / 2
                steps = math.ceil(candidate / change)
                stopping_distance = dt * (steps*candidate - change*steps*(steps-1)/2)
                if stopping_distance <= abs(distance):
                    low = candidate
                else:
                    high = candidate
            stopping_speed = low
            wanted = math.copysign(min(abs(wanted), stopping_speed), distance)
            velocity = max(self.velocity-change, min(self.velocity+change, wanted))
            accel_limited = abs(velocity-wanted) > 1e-12
        else:
            velocity = wanted
        result = previous + velocity * dt
        bounded = max(bounds[0], min(bounds[1], result))
        self.velocity = velocity if bounded == result else 0.0
        return bounded, speed_limited, accel_limited


class MotionRates:
    """Reported rates use local time between consumed samples, not the capped command dt."""
    def __init__(self, leader, target, follower, now):
        self.previous = (list(leader), list(target), list(follower))
        self.now = now
        self.rates = ([0.0]*7, [0.0]*7, [0.0]*7)

    def update(self, leader, target, follower, now):
        dt = now-self.now
        current = (list(leader), list(target), list(follower))
        if dt > 0:
            self.rates = tuple([math.degrees(a-b)/dt for a,b in zip(values, old, strict=True)]
                               for values,old in zip(current,self.previous,strict=True))
            self.previous, self.now = current, now
        return self.rates


class J1Mapper:
    def __init__(self, leader_start, follower_start, range_deg=3.0, speed_deg_s=0.5,
                 period=.05, accel_deg_s2=None):
        validate_tuning(range_deg, speed_deg_s)
        self.range_rad = math.radians(range_deg)
        self.speed_rad_s = math.radians(speed_deg_s)
        self.range_limited = []
        self.speed_limited = []
        self.leader_start = leader_start
        self.start = list(follower_start)
        self.previous = self.start[0]
        self.stepper = ReferenceStep(self.speed_rad_s, period,
                                    math.radians(accel_deg_s2) if accel_deg_s2 is not None else None)
        self.accel_limited = []

    def target(self, leader_j1, dt):
        if not math.isfinite(leader_j1) or not math.isfinite(dt) or dt < 0:
            raise ValueError("Invalid leader angle/time")
        raw = leader_j1 - self.leader_start
        change = max(-self.range_rad, min(self.range_rad, raw))
        desired = self.start[0] + change
        self.range_limited = [1] if abs(raw) > self.range_rad + 1e-12 else []
        self.previous, limited, accelerated = self.stepper.step(
            self.previous, desired, dt, (self.start[0]-self.range_rad, self.start[0]+self.range_rad))
        self.speed_limited = [1] if limited else []
        self.accel_limited = [1] if accelerated else []
        return [self.previous, *self.start[1:]]


class SevenJointMapper:
    def __init__(self, leader_start, follower_start, range_deg=3.0, speed_deg_s=0.5,
                 full_range=False, limits=None, period=.05, accel_deg_s2=None):
        validate_tuning(range_deg, speed_deg_s)
        self.range_rad = math.radians(range_deg)
        self.speed_rad_s = math.radians(speed_deg_s)
        self.range_limited = []
        self.speed_limited = []
        self.joint_limited = []
        self.full_range = full_range
        self.limits = limits
        if full_range:
            if limits is None:
                raise ValueError("Full range requires SDK joint limits")
            for index in range(1,8):
                low, high = limits[f"joint{index}"]
                if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                    raise ValueError(f"Invalid J{index} limits")
            check_limits(follower_start, limits)
        self.leader_start = list(leader_start)
        self.start = list(follower_start)
        self.previous = list(follower_start)
        self.steppers = [ReferenceStep(self.speed_rad_s, period,
                                      math.radians(accel_deg_s2) if accel_deg_s2 is not None else None)
                         for _ in range(7)]
        self.accel_limited = []

    def target(self, angles, dt):
        if len(angles) != 7 or not all(map(math.isfinite, angles)) or not math.isfinite(dt) or dt < 0:
            raise ValueError("Invalid seven-joint feedback/time")
        result = []
        self.range_limited = []
        self.speed_limited = []
        self.joint_limited = []
        self.accel_limited = []
        for index, (value, anchor, start, previous) in enumerate(zip(
                angles, self.leader_start, self.start, self.previous, strict=True), 1):
            raw = value-anchor
            delta = raw if self.full_range else max(-self.range_rad, min(self.range_rad, raw))
            if not self.full_range and abs(raw) > self.range_rad + 1e-12:
                self.range_limited.append(index)
            desired = start+delta
            if self.full_range:
                low, high = self.limits[f"joint{index}"]
                bounded = max(low, min(high, desired))
                if bounded != desired:
                    self.joint_limited.append(index)
                desired = bounded
            bounds = self.limits[f"joint{index}"] if self.full_range else (start-self.range_rad,start+self.range_rad)
            position, limited, accelerated = self.steppers[index-1].step(previous, desired, dt, bounds)
            if limited:
                self.speed_limited.append(index)
            if accelerated:
                self.accel_limited.append(index)
            result.append(position)
        self.previous = result
        return list(result)


def stable_poses(arms, timeout=10.0):
    """Require a shared stable window; never save stale or drifting poses."""
    deadline = time.monotonic() + timeout
    window_at = None
    stamps, lows, highs, latest = {}, {}, {}, {}
    count = 0
    next_log = 0.0
    reason = "waiting for feedback"
    print(f"Waiting up to {timeout:g}s for stable poses (1s, <=0.2deg span). Keep arms still.", flush=True)
    while time.monotonic() < deadline:
        now = time.monotonic()
        batch = {}
        for name, arm in arms.items():
            try:
                stamp, values = pose(arm)
            except (RuntimeError, ValueError, TypeError) as exc:
                reason = f"{name}: invalid/missing feedback: {exc}"
                break
            if name in stamps and stamp <= stamps[name]:
                reason = f"{name}: timestamp not advancing ({stamp:.6f} <= {stamps[name]:.6f})"
                break
            stamps[name] = stamp
            batch[name] = values
        else:
            if window_at is None:
                window_at = now
                lows = {name:list(values) for name,values in batch.items()}
                highs = {name:list(values) for name,values in batch.items()}
                count = 0
            for name,values in batch.items():
                lows[name] = [min(a,b) for a,b in zip(lows[name],values,strict=True)]
                highs[name] = [max(a,b) for a,b in zip(highs[name],values,strict=True)]
            spans = {name:[math.degrees(b-a) for a,b in zip(lows[name],highs[name],strict=True)]
                     for name in arms}
            moving = [(name, max(range(7),key=lambda i:span[i]), max(span))
                      for name,span in spans.items() if max(span)>0.2]
            if not moving:
                count += 1
                latest = batch
                if count >= 6 and now-window_at >= 1.0:
                    for name,values in latest.items():
                        print(f"STABLE {name}: deg={[round(math.degrees(v),3) for v in values]} "
                              f"span_deg={[round(v,4) for v in spans[name]]}",flush=True)
                    return latest
                time.sleep(.2)
                continue
            reason = "; ".join(f"{name}: J{index+1} span={span:.4f}deg exceeds 0.2deg"
                               for name,index,span in moving)
        window_at = None
        count = 0
        if now >= next_log:
            print(f"WAIT_STABLE {reason}; restarting stable window",flush=True)
            next_log = now+1
        time.sleep(.2)
    raise RuntimeError(f"No stable pose captured within {timeout:g}s: {reason}. No pose file saved.")


def stable_pose(arm):
    return stable_poses({"arm":arm})["arm"]


def enable_checked(arm, grip):
    take_can_control(arm)
    if not disable_effector(grip):
        raise RuntimeError("Effector disable unconfirmed")
    initial = snapshot(arm)
    health = status_snapshot(arm)
    if (initial is None or any(initial[1]) or initial[2] or health is None
            or health[2] not in (0,6) or health[3]):
        raise RuntimeError("Pre-enable status invalid")
    arm.enable(255, timeout=0.0)
    if wait_state(arm, True, initial[0], 5) is None:
        raise RuntimeError("Enable unconfirmed")
    monitor = wait_ready_status(arm, "pair_enable", health[0])
    if not disable_effector(grip):
        raise RuntimeError("Effector disable unconfirmed after enable")
    return monitor


def leader_pose(arm):
    msg = arm.get_leader_joint_angles()
    if msg is None or not math.isfinite(float(msg.hz)) or msg.hz <= 0:
        return None
    stamp, values = float(msg.timestamp), list(msg.msg)
    if not math.isfinite(stamp) or stamp <= 0 or len(values)!=7 or not all(map(math.isfinite,values)):
        raise RuntimeError("Invalid leader feedback")
    return stamp, values


def cleanup(devices, entered):
    errors = []
    for name in ("follower", "leader"):
        if name not in devices:
            continue
        arm, grip = devices[name]
        try:
            print(f"CLEANUP {name}", flush=True)
            finish(arm, grip, entered if name == "leader" else False)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("DISABLE UNCONFIRMED: " + "; ".join(errors))


def session(leader, lg, follower, fg, seconds, all_joints=False, saved_pose=None,
            range_deg=3.0, speed_deg_s=0.5, full_range=False, fps=20,
            follower_speed_percent=5, accel_deg_s2=None,
            return_speed_deg_s=1.0, return_speed_percent=5):
    validate_tuning(range_deg, speed_deg_s)
    validate_control(fps, follower_speed_percent, accel_deg_s2)
    presets.validate_return_tuning(return_speed_deg_s, return_speed_percent)
    period = 1.0/fps
    if full_range and not all_joints:
        raise ValueError("Full range requires --all-joints")
    for name, arm in (("leader",leader),("follower",follower)):
        state = wait_state(arm,False,[0.0]*7,3)
        if state is None or state[2]:
            raise RuntimeError(f"{name}: disabled fault-free feedback required")
    print("Both arms must remain supported for automatic disable on timeout/failure.\n"
          f"{'ALL SEVEN JOINTS FOLLOW' if all_joints else 'J1 ONLY FOLLOWS'}; "
          f"range={'SDK joint limits' if full_range else f'+/-{range_deg:g}deg from startup'}, "
          f"reference speed <={speed_deg_s:g}deg/s.\n"
          f"Teleop fps={fps}, controller speed={follower_speed_percent}%, "
          f"reference acceleration={accel_deg_s2}deg/s^2 (None = legacy slew).\n"
          f"Saved-pose return <={return_speed_deg_s:g}deg/s, controller={return_speed_percent}%; "
          "tracking-error limit remains 1.5deg.\n"
          "All seven motors are enabled. Simultaneous joint motion changes the whole arm posture.\n"
          "No gripper opening targets. End effectors will be explicitly disabled.\n"
          "Keep BOTH rotation directions clear. Physical emergency stop must be available.", flush=True)
    if saved_pose is not None:
        presets.validate(saved_pose)
        for role, arm in (("leader", leader), ("follower", follower)):
            _, current = pose(arm)
            goal = saved_pose[role]["joints"]
            check_limits(current, arm.get_config()["joint_limits"])
            check_limits(goal, arm.get_config()["joint_limits"])
            presets.check_return_span(current, goal, role)
            print(f"{role} current_deg={[round(math.degrees(v),3) for v in current]} "
                  f"saved_deg={[round(math.degrees(v),3) for v in goal]}", flush=True)
        print("START authorizes BOTH arms to return to saved joint poses, then teleoperate. "
              "Confirm entire paths are clear. This is NOT collision-aware planning. "
              "Do not touch either arm during RETURN.", flush=True)
    if input("Type START to begin: ").strip() != "START":
        print("Cancelled; no state changes requested.")
        return
    start = stable_pose(follower)
    stable_pose(leader)
    if saved_pose is not None:
        start = list(saved_pose["follower"]["joints"])
    limits = follower.get_config()["joint_limits"]
    check_limits(start,limits)
    for sign in (() if full_range else (-1,1)):
        candidate = ([v+sign*math.radians(range_deg) for v in start] if all_joints
                     else [start[0]+sign*math.radians(range_deg),*start[1:]])
        check_limits(candidate,limits)
    entered = False
    touched = {}
    try:
        touched["follower"] = (follower,fg)
        monitor = enable_checked(follower,fg)
        follower.set_joint_limits_enabled(True)
        follower.set_speed_percent(5)
        follower.set_motion_mode(follower.OPTIONS.MOTION_MODE.J)
        if saved_pose is not None:
            presets.return_to_pose(follower, start, monitor, "follower",
                                   speed_deg_s=return_speed_deg_s, controller_percent=return_speed_percent)
            start = stable_pose(follower)
            for sign in (() if full_range else (-1,1)):
                check_limits([v+sign*math.radians(range_deg) for v in start] if all_joints else
                             [start[0]+sign*math.radians(range_deg),*start[1:]], limits)
        touched["leader"] = (leader,lg)
        leader_monitor = enable_checked(leader,lg)
        if saved_pose is not None:
            def watch_follower():
                monitor.poll(follower)
                _, current = pose(follower)
                if max(abs(a-b) for a,b in zip(current,start,strict=True)) > math.radians(.3):
                    raise RuntimeError("Follower drifted while leader returned to saved pose")
            presets.return_to_pose(leader, saved_pose["leader"]["joints"], leader_monitor,
                                   "leader", observer=watch_follower,
                                   speed_deg_s=return_speed_deg_s, controller_percent=return_speed_percent)
            stable_pose(leader)
        health = status_snapshot(leader)
        if health is None or health[2]!=0 or health[3]:
            raise RuntimeError("Leader lost NORMAL status")
        previous = leader_pose(leader)
        stamp = previous[0] if previous else 0.0
        entered = True
        leader.set_leader_mode()
        optional = StatusMonitor("leader_optional",health[0],require_stream=False)
        deadline = time.monotonic()+3
        advances = 0
        while True:
            monitor.poll(follower)
            optional.poll(leader)
            reading = leader_pose(leader)
            if reading and reading[0]>stamp:
                stamp = reading[0]
                advances += 1
                if advances>=2:
                    break
            if time.monotonic()>=deadline:
                raise TimeoutError("Leader stream unavailable")
            time.sleep(.05)
        fstamp, actual = pose(follower)
        if max(abs(a-b) for a,b in zip(actual,start,strict=True))>math.radians(.3):
            raise RuntimeError(f"Follower moved during startup: delta_deg="
                               f"{[round(math.degrees(a-b),3) for a,b in zip(actual,start,strict=True)]}")
        leader_anchor = list(reading[1])
        mapper = (SevenJointMapper(leader_anchor, start, range_deg, speed_deg_s, full_range, limits,
                                   period, accel_deg_s2) if all_joints else
                  J1Mapper(reading[1][0], start, range_deg, speed_deg_s, period, accel_deg_s2))
        # Teleoperation controller speed is independent of both return movements.
        follower.set_speed_percent(follower_speed_percent)
        sent = list(start)
        began = previous_tick = last_leader = last_follower = time.monotonic()
        rates = MotionRates(leader_anchor, start, actual, began)
        next_log = began
        range_hits = [0]*7
        speed_hits = [0]*7
        joint_hits = [0]*7
        frames = 0
        print(f"READY: gently move {'leader joints' if all_joints else 'leader J1 only'}, at most a few degrees. "
              "Normal leader status can be unavailable; full fault visibility is limited.", flush=True)
        while seconds is None or time.monotonic()-began < seconds:
            now = time.monotonic()
            monitor.poll(follower)
            optional.poll(leader)
            reading = leader_pose(leader)
            fresh, actual = pose(follower)
            if fresh < fstamp or (reading and reading[0]<stamp):
                raise RuntimeError("Feedback timestamp regressed")
            lfresh = reading is not None and reading[0]>stamp
            ffresh = fresh>fstamp
            if lfresh:
                stamp, last_leader = reading[0], now
            if ffresh:
                fstamp, last_follower = fresh, now
            if now-last_leader>.5 or now-last_follower>.5:
                raise TimeoutError("Pair feedback stopped for >0.5s")
            if not lfresh or not ffresh:
                previous_tick=now
                time.sleep(max(0.0, period-(time.monotonic()-now)))
                continue
            state=snapshot(follower)
            if state is None or not all(state[1]) or state[2]:
                raise RuntimeError("Follower enable lost or faulted")
            delta=[math.degrees(a-b) for a,b in zip(actual,start,strict=True)]
            outside_test_range = (max(map(abs,delta))>range_deg+0.5 if all_joints else
                                  abs(delta[0])>range_deg+0.5 or max(map(abs,delta[1:]))>1)
            if not full_range and outside_test_range:
                raise RuntimeError(f"Unexpected follower deviation: {delta}")
            if max(abs(actual[i]-sent[i]) for i in (range(7) if all_joints else (0,)))>math.radians(1.5):
                raise RuntimeError("Follower tracking error >1.5deg")
            check_limits(actual,limits)
            tracking_error = [math.degrees(a-b) for a,b in zip(actual,sent,strict=True)]
            sent=mapper.target(reading[1] if all_joints else reading[1][0],now-previous_tick)
            check_limits(sent,limits)
            follower.move_j(sent)
            leader_rate, target_rate, follower_rate = rates.update(reading[1],sent,actual,now)
            frames += 1
            for axis in mapper.range_limited:
                range_hits[axis-1] += 1
            for axis in mapper.speed_limited:
                speed_hits[axis-1] += 1
            for axis in getattr(mapper, "joint_limited", []):
                joint_hits[axis-1] += 1
            previous_tick=now
            if now>=next_log:
                timing = f"elapsed={now-began:.1f}s" if seconds is None else f"remaining={seconds-(now-began):.1f}s"
                print(f"{timing} "
                      f"leader_delta_deg={[round(math.degrees(a-b),3) for a,b in zip(reading[1],leader_anchor,strict=True)]} "
                      f"target_delta_deg={[round(math.degrees(a-b),3) for a,b in zip(sent,start,strict=True)]} "
                      f"follower_delta_deg={[round(v,3) for v in delta]} "
                      f"range_limited_axes={mapper.range_limited} "
                      f"speed_limited_axes={mapper.speed_limited} "
                      f"accel_limited_axes={mapper.accel_limited} "
                      f"joint_limited_axes={getattr(mapper, 'joint_limited', [])} "
                      f"tracking_error_deg={[round(v,3) for v in tracking_error]} "
                      f"leader_speed_deg_s={[round(v,2) for v in leader_rate]} "
                      f"target_speed_deg_s={[round(v,2) for v in target_rate]} "
                      f"follower_speed_deg_s={[round(v,2) for v in follower_rate]} "
                      f"send_hz={frames/max(now-began,.001):.1f}",flush=True)
                next_log=now+.5
            time.sleep(max(0.0, period-(time.monotonic()-now)))
        print(f"Timed session completed: sent_frames={frames}, range_limited_frames={range_hits}, "
              f"speed_limited_frames={speed_hits}, joint_limited_frames={joint_hits}. "
              "Inspect tracking log before declaring success.",flush=True)
    finally:
        cleanup(touched,entered)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-channel",required=True)
    parser.add_argument("--follower-channel",required=True)
    parser.add_argument("--seconds",type=float,default=20)
    parser.add_argument("--all-joints",action="store_true",help="Bounded seven-joint relative teleoperation")
    parser.add_argument("--full-range",action="store_true",
                        help="Use SDK joint limits instead of a relative startup range; requires --all-joints")
    parser.add_argument("--continuous",action="store_true",help="Run until Ctrl+C, then disable both arms/effectors")
    parser.add_argument("--range-deg",type=float,default=3.0,
                        help="Per-active-joint relative target range in degrees (0,15], default 3")
    parser.add_argument("--speed-deg-s",type=float,default=0.5,
                        help="Teleoperation reference speed in deg/s (0,30], default 0.5; not return speed")
    parser.add_argument("--fps",type=int,default=20,help="Teleoperation target frequency [20,60]")
    parser.add_argument("--follower-speed-percent",type=int,default=5,
                        help="Controller move_j speed during teleop [1,30]; independent of return")
    parser.add_argument("--accel-deg-s2",type=float,default=None,
                        help="Optional reference acceleration (0,60] deg/s^2; hard limits take priority")
    parser.add_argument("--save-start",action="store_true",help="Read and save current joint poses; no motion")
    parser.add_argument("--return-to-start",action="store_true",help="Return both arms to saved poses after START")
    parser.add_argument("--return-speed-deg-s",type=float,default=1.0,
                        help="Reference speed for both saved-pose returns (0,5], default 1")
    parser.add_argument("--return-speed-percent",type=int,default=5,
                        help="Controller speed for both saved-pose returns [1,20], default 5")
    parser.add_argument("--pose-file",type=Path,default=presets.DEFAULT_PATH)
    parser.add_argument("--overwrite-start",action="store_true")
    args=parser.parse_args()
    if args.full_range and not args.all_joints:
        parser.error("--full-range requires --all-joints")
    if args.full_range and args.range_deg != 3.0:
        parser.error("--range-deg does not apply to --full-range; omit it")
    try:
        validate_tuning(args.range_deg,args.speed_deg_s)
        validate_control(args.fps,args.follower_speed_percent,args.accel_deg_s2)
        presets.validate_return_tuning(args.return_speed_deg_s,args.return_speed_percent)
    except ValueError as exc:
        parser.error(str(exc))
    if args.save_start and args.return_to_start:
        parser.error("Save and return must be separate operations")
    if args.overwrite_start and not args.save_start:
        parser.error("--overwrite-start requires --save-start")
    saved = None
    if args.return_to_start:
        try:
            saved = presets.load(args.pose_file)
        except FileNotFoundError:
            parser.error(f"Saved start pose not found: {args.pose_file.resolve()}. "
                         "Run --save-start first and wait for 'Saved joint-only start poses'. No hardware opened.")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            parser.error(f"Invalid saved pose file: {exc}. No hardware opened.")
    if args.save_start and args.pose_file.exists() and not args.overwrite_start:
        parser.error("Saved pose exists; use --overwrite-start explicitly to replace it")
    if args.leader_channel==args.follower_channel or not math.isfinite(args.seconds) or not 5<=args.seconds<=30:
        parser.error("Use different channels and a duration from 5 to 30 seconds")
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src/lerobot/motors/nero"))
    from read_nero_firmware import preflight_can_device
    from pyAgxArm import AgxArmFactory,create_agx_arm_config
    opened=[]
    devices=[]
    try:
        for name,ch,profile,version in (("leader",args.leader_channel,"v120","1.20"),
                                        ("follower",args.follower_channel,"v121","1.21")):
            if not preflight_can_device("agx_cando",ch):
                raise RuntimeError(f"Cannot open {name}")
            arm=AgxArmFactory.create_arm(create_agx_arm_config(
                robot="nero",comm="can",interface="agx_cando",channel=ch,
                bitrate=1000000,firmeware_version=profile))
            opened.append(arm)
            arm.connect()
            fw=arm.get_firmware(timeout=2)
            print(f"{name}: channel={ch} firmware={fw}",flush=True)
            if not fw or fw.get("software_version")!=version:
                raise RuntimeError("Device/version mismatch; no enable sent")
            grip=arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            time.sleep(.3)
            devices.extend((arm,grip))
        if args.save_start:
            captured = stable_poses({"leader":devices[0], "follower":devices[2]})
            leader_values = captured["leader"]
            follower_values = captured["follower"]
            check_limits(leader_values,devices[0].get_config()["joint_limits"])
            check_limits(follower_values,devices[2].get_config()["joint_limits"])
            presets.save(args.pose_file,leader_values,follower_values,args.overwrite_start)
            print(f"Saved joint-only start poses to {args.pose_file.resolve()}; no state changes sent.",flush=True)
        else:
            session(*devices,None if args.continuous else args.seconds,all_joints=args.all_joints,saved_pose=saved,
                    range_deg=args.range_deg,speed_deg_s=args.speed_deg_s,full_range=args.full_range,
                    fps=args.fps,follower_speed_percent=args.follower_speed_percent,accel_deg_s2=args.accel_deg_s2,
                    return_speed_deg_s=args.return_speed_deg_s,return_speed_percent=args.return_speed_percent)
    finally:
        for arm in reversed(opened):
            try:
                arm.disconnect(join_timeout=1)
            except Exception as exc:
                print(f"Connection cleanup failed: {exc}",flush=True)
        print("Connections released; only updated disable feedback proves disable.",flush=True)


if __name__=="__main__":
    main()
