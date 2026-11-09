"""Offline tests for read-only gripper capture formatting and summaries."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import gripper_capture as capture


def status(width=0.012, timestamp=10.0, enabled=False, homed=True):
    foc = NS(
        voltage_too_low=False, motor_overheating=False, driver_overcurrent=False,
        driver_overheating=False, sensor_status=False, driver_error_status=False,
        driver_enable_status=enabled, homing_status=homed,
    )
    return NS(timestamp=timestamp, hz=200.0, msg=NS(
        mode="width", value=width, force=1.2, status_code=0x80 if homed else 0,
        foc_status=foc,
    ))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.arm = Mock()
        self.arm.get_arm_status.return_value = NS(
            msg=NS(ctrl_mode="CAN_CTRL", arm_status="NORMAL", err_code=0)
        )
        self.gripper = Mock()
        self.gripper.get_gripper_status.return_value = status()
        self.gripper.get_gripper_teaching_pendant_param.return_value = NS(
            msg=NS(max_range_config=0.07, teaching_range_per=100, teaching_friction=1)
        )
        self.arm_snapshot = patch.object(capture, "snapshot", return_value=(
            [1.0] * 7, [False] * 7, []
        ))
        self.arm_snapshot.start()
        self.addCleanup(self.arm_snapshot.stop)

    def test_width_is_reported_in_millimeters_and_state_is_read_only(self):
        result = capture.read_sample(self.arm, self.gripper, "follower", "1.21")
        self.assertAlmostEqual(result["distance_mm"], 12.0)
        self.assertEqual(result["configured_range_mm"], 70.0)
        self.assertTrue(result["homing_status"])
        self.gripper.move_gripper_m.assert_not_called()
        self.gripper.disable_gripper.assert_not_called()
        self.arm.enable.assert_not_called()
        self.arm.move_j.assert_not_called()

    def test_angle_mode_has_no_fake_width(self):
        value = status()
        value.msg.mode = "angle"
        value.msg.value = 42.0
        self.gripper.get_gripper_status.return_value = value
        result = capture.read_sample(self.arm, self.gripper, "leader", "1.20")
        self.assertIsNone(result["distance_mm"])
        self.assertEqual(result["angle_deg"], 42.0)

    def test_faults_and_enable_are_captured(self):
        value = status(enabled=True, homed=False)
        value.msg.foc_status.driver_error_status = True
        self.gripper.get_gripper_status.return_value = value
        result = capture.read_sample(self.arm, self.gripper, "follower", "1.21")
        self.assertTrue(result["gripper_enabled"])
        self.assertFalse(result["homing_status"])
        self.assertTrue(result["faults"]["driver_error_status"])

    def test_summary_is_saved(self):
        samples = [
            {"distance_mm": 1.0, "host_time_s": 1},
            {"distance_mm": 7.5, "host_time_s": 2},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture.json"
            capture.save_capture(path, {"role": "follower"}, samples)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["sample_count"], 2)
        self.assertEqual(data["distance_min_mm"], 1.0)
        self.assertEqual(data["distance_max_mm"], 7.5)
        self.assertEqual(data["distance_span_mm"], 6.5)

    def test_nonfinite_feedback_is_rejected(self):
        value = status(width=float("nan"))
        self.gripper.get_gripper_status.return_value = value
        with self.assertRaises(RuntimeError):
            capture.read_sample(self.arm, self.gripper, "follower", "1.21")


if __name__ == "__main__":
    unittest.main()
