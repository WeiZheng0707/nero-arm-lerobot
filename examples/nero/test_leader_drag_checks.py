"""Offline status-transition and cleanup tests. Never creates an SDK arm."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_leader_drag as drag
import test_enable_cycle as enable_cycle


def frame(stamp, code=0, hz=100, err=0):
    return NS(timestamp=stamp, hz=hz, msg=NS(arm_status=code, err_code=err,
                                          ctrl_mode=1, mode_feedback=1))


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.clock = patch.object(drag.time, "monotonic", side_effect=lambda: self.now)
        self.sleep = patch.object(drag.time, "sleep", side_effect=self.advance)
        self.clock.start()
        self.sleep.start()
        self.addCleanup(self.clock.stop)
        self.addCleanup(self.sleep.stop)

    def advance(self, seconds):
        self.now += seconds

    def test_copies_mutable_sdk_message(self):
        message = frame(1)
        arm = Mock(get_arm_status=Mock(return_value=message))
        saved = drag.status_snapshot(arm)
        message.timestamp = 2
        self.assertEqual(saved[0], 1)

    def test_old_brake_then_two_new_normal_frames(self):
        arm = Mock(get_arm_status=Mock(side_effect=[frame(1, 6), frame(2, 6), frame(3), frame(4)]))
        result = drag.wait_ready_status(arm, "after_enable", 1, timeout=0.5)
        self.assertEqual(result.stamp, 4)
        self.assertEqual(result.normal_frames, 2)

    def test_persistent_brake_times_out(self):
        arm = Mock(get_arm_status=Mock(side_effect=lambda: frame(1 + self.now, 6)))
        with self.assertRaises(TimeoutError):
            drag.wait_ready_status(arm, "after_enable", 0, timeout=0.2)

    def test_repeated_normal_cache_does_not_pass(self):
        arm = Mock(get_arm_status=Mock(return_value=frame(1)))
        with self.assertRaises(TimeoutError):
            drag.wait_ready_status(arm, "after_enable", 0, timeout=0.2)

    def test_new_fault_aborts_immediately(self):
        for code, err in ((5, 0), (0, 1), (11, 0)):
            with self.subTest(code=code, err=err):
                arm = Mock(get_arm_status=Mock(return_value=frame(2, code, err=err)))
                with self.assertRaises(RuntimeError):
                    drag.wait_ready_status(arm, "after_enable", 1)

    def test_missing_zero_hz_and_nonfinite_timestamp_cannot_pass(self):
        for value in (None, frame(2, hz=0), frame(float("nan"))):
            with self.subTest(value=value):
                arm = Mock(get_arm_status=Mock(return_value=value))
                with self.assertRaises(TimeoutError):
                    drag.wait_ready_status(arm, "after_enable", 1, timeout=0.2)

    def test_active_drag_stale_status_aborts(self):
        monitor = drag.StatusMonitor("drag", 1)
        self.now += 1.1
        with self.assertRaises(TimeoutError):
            monitor.poll(Mock(get_arm_status=Mock(return_value=frame(1))))

    def test_active_drag_new_brake_aborts(self):
        monitor = drag.StatusMonitor("drag", 1)
        with self.assertRaises(RuntimeError):
            monitor.poll(Mock(get_arm_status=Mock(return_value=frame(2, 6))))

    def test_timestamp_regression_aborts(self):
        monitor = drag.StatusMonitor("drag", 2)
        with self.assertRaises(RuntimeError):
            monitor.poll(Mock(get_arm_status=Mock(return_value=frame(1))))

    def test_gate_failures_always_cleanup(self):
        off = ([1.0] * 7, [False] * 7, [])
        for failed_phase in ("after_enable", "leader_feedback"):
            arm = Mock(get_arm_status=Mock(side_effect=lambda: frame(1 + self.now)))
            arm.get_leader_joint_angles.return_value = None
            grip = Mock(get_gripper_status=Mock(return_value=NS(
                hz=100, timestamp=1, msg=NS(foc_status=NS(driver_enable_status=False)))))

            def gate(_arm, phase, baseline):
                if phase == failed_phase:
                    raise TimeoutError(phase)
                return drag.StatusMonitor(phase, baseline)

            with patch("builtins.input", return_value="START"), \
                    patch.object(drag, "wait_state", return_value=off), \
                    patch.object(drag, "wait_ready_status", side_effect=gate), \
                    patch.object(drag, "finish") as cleanup:
                with self.assertRaises(TimeoutError):
                    drag.run(arm, grip, 5)
                cleanup.assert_called_once_with(arm, grip, failed_phase == "leader_feedback")
                if failed_phase == "after_enable":
                    arm.set_leader_mode.assert_not_called()
                arm.move_j.assert_not_called()
                grip.move_gripper_m.assert_not_called()

    def test_cancel_does_not_enable(self):
        off = ([1.0] * 7, [False] * 7, [])
        arm = Mock(get_arm_status=Mock(return_value=frame(1)))
        grip = Mock(get_gripper_status=Mock(return_value=NS(
            hz=100, timestamp=1, msg=NS(foc_status=NS(driver_enable_status=False)))))
        with patch("builtins.input", return_value="NO"), \
                patch.object(drag, "wait_state", return_value=off):
            drag.run(arm, grip, 5)
        arm.enable.assert_not_called()
        arm.set_leader_mode.assert_not_called()

    def test_enabled_effector_cancel_does_not_write(self):
        arm = Mock(get_arm_status=Mock(return_value=frame(1)))
        grip = Mock(get_gripper_status=Mock(return_value=NS(
            hz=100, timestamp=1, msg=NS(foc_status=NS(driver_enable_status=True)))))
        with patch("builtins.input", return_value="NO"), \
                patch.object(drag, "wait_state", return_value=([1]*7, [False]*7, [])), \
                patch.object(drag, "disable_effector") as disable:
            drag.run(arm, grip, 5)
            disable.assert_not_called()
        arm.enable.assert_not_called()

    def test_startup_disable_failure_prevents_joint_enable(self):
        arm = Mock(get_arm_status=Mock(return_value=frame(1)))
        grip = Mock(get_gripper_status=Mock(return_value=NS(
            hz=100, timestamp=1, msg=NS(foc_status=NS(driver_enable_status=True)))))
        with patch("builtins.input", return_value="START"), \
                patch.object(drag, "wait_state", return_value=([1]*7, [False]*7, [])), \
                patch.object(drag, "disable_effector", return_value=False) as disable:
            with self.assertRaisesRegex(RuntimeError, "Startup effector disable"):
                drag.run(arm, grip, 5)
            disable.assert_called_once_with(grip)
        arm.enable.assert_not_called()

    def test_effector_disable_requires_advancing_stable_feedback(self):
        def status(enabled=False, stamp=None):
            return NS(timestamp=1 + self.now if stamp is None else stamp, hz=100,
                      msg=NS(foc_status=NS(driver_enable_status=enabled)))
        grip = Mock(get_gripper_status=Mock(side_effect=lambda: status()))
        self.assertTrue(enable_cycle.disable_effector(grip, timeout=2))
        self.assertEqual(grip.disable_gripper.call_count, 3)
        self.assertGreaterEqual(self.now, 1.1)

    def test_cached_disabled_frame_cannot_confirm(self):
        grip = Mock(get_gripper_status=Mock(return_value=NS(
            timestamp=1, hz=100, msg=NS(foc_status=NS(driver_enable_status=False)))))
        self.assertFalse(enable_cycle.disable_effector(grip, timeout=1))

    def test_reenabled_effector_cannot_confirm(self):
        grip = Mock(get_gripper_status=Mock(side_effect=lambda: NS(
            timestamp=1 + self.now, hz=100,
            msg=NS(foc_status=NS(driver_enable_status=self.now > 0.8)))))
        self.assertFalse(enable_cycle.disable_effector(grip, timeout=1.5))

    def test_optional_status_missing_or_cached_is_not_health_proof(self):
        monitor = drag.StatusMonitor("leader", 1, require_stream=False)
        self.now += 2
        self.assertFalse(monitor.poll(Mock(get_arm_status=Mock(return_value=None))))
        self.assertFalse(monitor.poll(Mock(get_arm_status=Mock(return_value=frame(1)))))

    def test_optional_status_still_rejects_fresh_fault(self):
        for code in (5, 6):
            monitor = drag.StatusMonitor("leader", 1, require_stream=False)
            with self.assertRaises(RuntimeError):
                monitor.poll(Mock(get_arm_status=Mock(return_value=frame(2, code))))

    def test_leader_stream_runs_without_normal_status(self):
        off = ([1.0] * 7, [False] * 7, [])
        active = [False]
        arm = Mock()
        arm.get_arm_status.side_effect = lambda: None if active[0] else frame(1 + self.now)
        arm.set_leader_mode.side_effect = lambda: active.__setitem__(0, True)
        arm.get_leader_joint_angles.side_effect = lambda: NS(
            timestamp=2 + self.now, hz=30, msg=[self.now / 100] * 7) if active[0] else None
        grip = Mock(get_gripper_status=Mock(return_value=NS(
            hz=100, timestamp=1, msg=NS(foc_status=NS(driver_enable_status=False)))))
        with patch("builtins.input", return_value="START"), \
                patch.object(drag, "wait_state", return_value=off), \
                patch.object(drag, "wait_ready_status"), \
                patch.object(drag, "finish") as cleanup:
            drag.run(arm, grip, 5)
            cleanup.assert_called_once_with(arm, grip, True)
            arm.get_joint_angles.assert_not_called()
            arm.move_j.assert_not_called()

    def test_frozen_leader_stream_exits_and_cleans_up(self):
        off = ([1.0] * 7, [False] * 7, [])
        arm = Mock(get_arm_status=Mock(side_effect=lambda: frame(1 + self.now)))
        arm.get_leader_joint_angles.return_value = NS(timestamp=2, hz=30, msg=[0.1] * 7)
        grip = Mock(get_gripper_status=Mock(return_value=NS(
            hz=100, timestamp=1, msg=NS(foc_status=NS(driver_enable_status=False)))))
        with patch("builtins.input", return_value="START"), \
                patch.object(drag, "wait_state", return_value=off), \
                patch.object(drag, "wait_ready_status"), \
                patch.object(drag, "finish") as cleanup:
            with self.assertRaises(TimeoutError):
                drag.run(arm, grip, 5)
            cleanup.assert_called_once_with(arm, grip, True)

    def test_effector_cleanup_even_if_joint_poll_raises(self):
        arm, grip = Mock(), Mock()
        with patch.object(drag, "snapshot", return_value=None), \
                patch.object(drag, "wait_state", side_effect=RuntimeError("read failed")), \
                patch.object(drag, "disable_effector", return_value=True) as cleanup:
            with self.assertRaises(RuntimeError):
                drag.finish(arm, grip, True)
            cleanup.assert_called_once_with(grip)
            self.assertEqual(arm.disable.call_count, 3)


if __name__ == "__main__":
    unittest.main()
