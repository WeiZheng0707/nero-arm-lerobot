"""Interactive leader-only drag check for NERO 1.20. No position targets."""

import argparse
import math
import sys
import time
from pathlib import Path

from test_enable_cycle import disable_effector, snapshot, wait_state


def status_snapshot(arm):
    """Copy scalar fields: SDK message objects are updated in place by its reader."""
    frame = arm.get_arm_status()
    if frame is None:
        return None
    stamp, hz = float(frame.timestamp), float(frame.hz)
    if not math.isfinite(stamp) or stamp <= 0 or not math.isfinite(hz) or hz <= 0:
        return None
    msg = frame.msg
    return (stamp, hz, int(msg.arm_status), int(msg.err_code),
            int(msg.ctrl_mode), str(getattr(msg, "mode_feedback", "unavailable")))


class StatusMonitor:
    def __init__(self, phase, baseline, require_stream=True):
        self.phase = phase
        self.require_stream = require_stream
        self.stamp = baseline
        self.last_advance = time.monotonic()
        self.next_print = self.last_advance
        self.last_fields = None
        self.normal_frames = 0

    def poll(self, arm, allow_transition=False):
        now = time.monotonic()
        sample = status_snapshot(arm)
        if sample is None:
            self.normal_frames = 0
            if now >= self.next_print:
                print(f"STATUS phase={self.phase} unavailable_or_zero_hz", flush=True)
                self.next_print = now + 0.5
            if self.require_stream and not allow_transition and now - self.last_advance > 1.0:
                raise TimeoutError(f"{self.phase}: arm status missing/stale for over 1s")
            return False
        stamp, hz, code, err, ctrl, mode = sample
        fresh = stamp > self.stamp
        fields = (code, err, ctrl, mode, fresh)
        if now >= self.next_print or fields != self.last_fields:
            print(f"STATUS phase={self.phase} timestamp={stamp:.6f} hz={hz:.1f} "
                  f"advanced={fresh} status=0x{code:02X} err=0x{err:04X} "
                  f"ctrl=0x{ctrl:02X} motion_mode={mode}", flush=True)
            self.next_print = now + 0.5
            self.last_fields = fields
        if stamp < self.stamp:
            raise RuntimeError(f"{self.phase}: arm status timestamp moved backwards")
        if fresh:
            self.stamp = stamp
            self.last_advance = now
            if err or code not in (0, 6):
                raise RuntimeError(f"{self.phase}: fresh arm fault status=0x{code:02X}, err=0x{err:04X}")
            if code == 6:
                self.normal_frames = 0
                if not allow_transition:
                    raise RuntimeError(f"{self.phase}: fresh JOINT_BRAKE_NOT_RELEASED(0x06)")
            else:
                self.normal_frames += 1
        if self.require_stream and not allow_transition and now - self.last_advance > 1.0:
            raise TimeoutError(f"{self.phase}: arm status timestamp stopped for over 1s")
        return fresh and self.normal_frames >= 2


def wait_ready_status(arm, phase, baseline, timeout=3.0):
    """Bound transition grace; never continue to dragging with unresolved 0x06."""
    monitor = StatusMonitor(phase, baseline)
    deadline = time.monotonic() + timeout
    print(f"Waiting up to {timeout:g}s for two updating NORMAL status frames ({phase}); "
          "do not drag yet.", flush=True)
    while time.monotonic() < deadline:
        if monitor.poll(arm, allow_transition=True):
            print(f"PASS: {phase}: two updating NORMAL status frames.", flush=True)
            return monitor
        time.sleep(0.05)
    raise TimeoutError(f"{phase}: NORMAL state not confirmed within {timeout:g}s; "
                       "0x06, absent or stale feedback is not accepted")


def finish(arm, gripper, entered):
    if entered:
        for _ in range(3):
            try:
                arm.set_follower_mode()
            except Exception as exc:
                print(f"Exit leader mode failed: {exc}", flush=True)
            time.sleep(0.2)
    try:
        before = snapshot(arm)
    except Exception:
        before = None
    baseline = before[0] if before else [float("inf")] * 7
    for _ in range(3):
        try:
            arm.disable(255, timeout=0.0)
        except Exception as exc:
            print(f"Joint disable request failed: {exc}", flush=True)
        time.sleep(0.2)
    joints_ok = False
    try:
        joints_ok = wait_state(arm, False, baseline, 5.0) is not None
    finally:
        effector_ok = disable_effector(gripper)
        print(f"EXIT: updated joint disable confirmed={joints_ok}; "
              f"updated effector disable confirmed={effector_ok}", flush=True)
    if not joints_ok or not effector_ok:
        raise RuntimeError("DISABLE UNCONFIRMED. Keep support and verify on site.")


def run(arm, gripper, seconds):
    initial = wait_state(arm, False, [0.0] * 7, 3.0)
    if initial is None or initial[2]:
        raise RuntimeError("Need seven disabled, fault-free joint feedback frames")
    state = arm.get_arm_status()
    if (state is None or state.hz <= 0 or int(state.msg.err_code)
            or int(state.msg.arm_status) not in (0, 6)):
        raise RuntimeError("Arm status precheck failed")
    grip = gripper.get_gripper_status()
    if grip is None or grip.hz <= 0 or grip.timestamp <= 0:
        raise RuntimeError("Effector feedback required for exit verification")
    if grip.msg.foc_status.driver_enable_status:
        print("Effector is currently enabled. After START, request effector disable first; "
              "the arm will not be enabled unless effector disable is confirmed.", flush=True)
    print(f"LEADER ONLY. Up to {seconds:g}s after entry; exit DISABLES joints and effector.\n"
          "Keep support effective throughout, including timeout/fault exit.\n"
          "No follower connection, no joint targets, no gripper opening commands.\n"
          "START also authorizes effector disable on this supported, empty leader before enabling joints.\n"
          "Do not force a locked joint. Stop on unexpected motion. Physical emergency stop must be ready.",
          flush=True)
    if input("Type START only when the supported leader and work area are ready: ").strip() != "START":
        print("Cancelled; no enable or mode requests sent.")
        return
    # The confirmation may take time: require newly updated feedback before enabling.
    checked = wait_state(arm, False, initial[0], 3.0)
    if checked is None or checked[2]:
        raise RuntimeError("Joint precheck expired or faulted; no enable sent")
    before_enable = status_snapshot(arm)
    if before_enable is None or before_enable[3] or before_enable[2] not in (0, 6):
        raise RuntimeError("Arm status precheck expired or faulted; no enable sent")
    grip = gripper.get_gripper_status()
    if grip is None or grip.hz <= 0:
        raise RuntimeError("Effector feedback unavailable; no enable sent")
    entered = False
    # No automatic mode switch or joint command when preparing the end effector.
    if grip.msg.foc_status.driver_enable_status:
        if before_enable[4] != 1:
            raise RuntimeError("Effector enabled outside CAN control; no automatic mode switch or enable")
        if not disable_effector(gripper):
            raise RuntimeError("Startup effector disable UNCONFIRMED; no joint enable sent")
        # Disabling an effector can take seconds; refresh joint feedback again.
        checked = wait_state(arm, False, checked[0], 3.0)
        before_enable = status_snapshot(arm)
        if (checked is None or checked[2] or before_enable is None
                or before_enable[3] or before_enable[2] not in (0, 6)):
            raise RuntimeError("Precheck failed after effector disable; no joint enable sent")
    try:
        arm.enable(255, timeout=0.0)
        if wait_state(arm, True, checked[0], 5.0) is None:
            raise RuntimeError("Enable not confirmed")
        wait_ready_status(arm, "after_enable", before_enable[0])
        # Save the timestamp value, not the SDK's mutable message instance.
        before_mode = status_snapshot(arm)
        if before_mode is None or before_mode[2] != 0 or before_mode[3]:
            raise RuntimeError("Normal status lost before leader-mode request")
        previous_leader = arm.get_leader_joint_angles()
        leader_baseline = float(previous_leader.timestamp) if previous_leader is not None else 0.0
        if not math.isfinite(leader_baseline) or leader_baseline < 0:
            raise RuntimeError("Invalid pre-mode leader timestamp")
        entered = True
        arm.set_leader_mode()
        # v120 inherits set_leader_mode(), which explicitly disables normal CAN push.
        # New status faults remain actionable, but cached/absent status is not a heartbeat.
        monitor = StatusMonitor("leader_optional_status", before_mode[0], require_stream=False)
        print("Normal status stream may stop in leader mode; continuous fault visibility is limited. "
              "Leader-specific feedback must advance; no regular-feedback fallback.", flush=True)
        print("Leader-mode request sent. Waiting for advancing leader feedback; do not force the arm.",
              flush=True)
        started = time.monotonic()
        last_advance = started
        last_stamp = leader_baseline
        start_angles = None
        max_delta = [0.0] * 7
        count = 0
        next_print = started
        while time.monotonic() - started < seconds:
            now = time.monotonic()
            # Check health before inviting the operator to move the arm.
            monitor.poll(arm)
            message = arm.get_leader_joint_angles()
            if message is not None and math.isfinite(float(message.hz)) and message.hz > 0:
                stamp = float(message.timestamp)
                values = list(message.msg)
                if (not math.isfinite(stamp) or stamp <= 0 or len(values) != 7
                        or not all(math.isfinite(v) for v in values)):
                    raise RuntimeError("Invalid leader feedback")
                if last_stamp is not None and stamp < last_stamp:
                    raise RuntimeError("Leader timestamp moved backwards")
                if last_stamp is not None and stamp > last_stamp:
                    last_advance = now
                    count += 1
                    if start_angles is None and count >= 2:
                        start_angles = list(values)
                        print("Leader feedback advancing (SDK radians for v120). "
                              "If the arm feels normally compliant, make only a small slow movement.", flush=True)
                    if start_angles is not None:
                        max_delta = [max(old, abs(v - start)) for old, v, start in
                                     zip(max_delta, values, start_angles, strict=True)]
                    if now >= next_print and start_angles is not None:
                        print(f"remaining_s={max(0, seconds - (now - started)):.1f} "
                              f"leader_timestamp={stamp:.6f} leader_hz={message.hz:.1f} "
                              f"joints_deg={[round(math.degrees(v), 2) for v in values]}", flush=True)
                        next_print = now + 0.5
                last_stamp = stamp
            if now - last_advance > (3.0 if start_angles is None else 1.0):
                raise TimeoutError("Leader feedback not advancing; ending test")
            time.sleep(0.05)
        print(f"Observed leader timestamp advances={count}; "
              f"max_change_deg={[round(math.degrees(v), 2) for v in max_delta]}. "
              "These readings alone do not verify drag mechanics or safety.", flush=True)
    finally:
        print("Ending test: leave leader mode, disable joints and effector. Keep support in place.", flush=True)
        finish(arm, gripper, entered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 5 <= args.seconds <= 30:
        parser.error("--seconds must be between 5 and 30")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/lerobot/motors/nero"))
    from read_nero_firmware import preflight_can_device
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    if not preflight_can_device("agx_cando", args.channel):
        raise RuntimeError("Adapter unavailable")
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot="nero", comm="can", interface="agx_cando", channel=args.channel,
        bitrate=1_000_000, firmeware_version="v120",
    ))
    try:
        arm.connect()
        fw = arm.get_firmware(timeout=2.0)
        print(f"channel={args.channel} firmware={fw}", flush=True)
        if not fw or fw.get("software_version") != "1.20":
            raise RuntimeError("Expected previously identified 1.20 leader; refusing enable")
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        time.sleep(0.3)
        run(arm, gripper, args.seconds)
    finally:
        arm.disconnect(join_timeout=1.0)
        print("Connection released. Disconnection itself does not prove disable.", flush=True)


if __name__ == "__main__":
    main()
