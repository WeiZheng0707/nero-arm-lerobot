from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from control_center import ConfigError, build_command, normalize_config


BASE = {
    "leader_channel": "0",
    "follower_channel": "1",
    "leader_firmware": "v120",
    "follower_firmware": "v121",
}


class ControlCenterChecks(unittest.TestCase):
    def test_read_only_command_carries_explicit_identity(self):
        config = normalize_config(BASE, "read_only")
        command = build_command("read_only", config, python="python-test")
        self.assertEqual(command[0], "python-test")
        self.assertIn("inspect_effectors.py", command[2])
        self.assertEqual(command[command.index("--leader-channel") + 1], "0")
        self.assertEqual(command[command.index("--follower-channel") + 1], "1")
        self.assertEqual(command[command.index("--leader-version") + 1], "1.20")
        self.assertEqual(command[command.index("--follower-version") + 1], "1.21")

    def test_same_channel_rejected(self):
        payload = {**BASE, "follower_channel": "0"}
        with self.assertRaises(ConfigError):
            normalize_config(payload, "read_only")

    def test_disable_requires_physical_safety_confirmation(self):
        with self.assertRaises(ConfigError):
            normalize_config(BASE, "disable")
        config = normalize_config({**BASE, "safety_confirmed": True}, "disable")
        command = build_command("disable", config, python="python-test")
        self.assertIn("--both-supported-empty-confirmed", command)
        self.assertIn("--leader-firmware", command)
        self.assertIn("--follower-firmware", command)

    def test_teleoperate_command_preserves_motion_limits(self):
        payload = {
            **BASE,
            "safety_confirmed": True,
            "fps": 30,
            "speed_percent": 20,
            "gripper_force": 1.0,
            "max_joint_speed": 0.3,
            "mapping": "relative",
            "return_to_start": True,
            "duration": 20,
        }
        config = normalize_config(payload, "teleoperate")
        command = build_command("teleoperate", config, python="python-test")
        self.assertIn("--return-to-start", command)
        self.assertEqual(command[command.index("--duration") + 1], "20")
        self.assertEqual(command[command.index("--max-joint-speed") + 1], "0.3")
        self.assertEqual(command[command.index("--mapping") + 1], "relative")

    def test_teleoperate_rejects_unchecked_safety_and_bad_limits(self):
        with self.assertRaises(ConfigError):
            normalize_config(BASE, "teleoperate")
        payload = {**BASE, "safety_confirmed": True, "fps": 80}
        with self.assertRaises(ConfigError):
            normalize_config(payload, "teleoperate")

    def test_teleoperate_requires_preverified_identity(self):
        from control_center import ControlCenter, config_signature

        controller = ControlCenter()
        payload = {
            **BASE,
            "safety_confirmed": True,
            "fps": 30,
            "speed_percent": 20,
            "gripper_force": 1.0,
            "max_joint_speed": 0.3,
            "mapping": "relative",
            "return_to_start": False,
            "duration": None,
        }
        config = normalize_config(payload, "teleoperate")
        with self.assertRaises(ConfigError):
            controller.start("teleoperate", config)
        self.assertIsNone(controller.verified_signature)
        self.assertNotEqual(controller.verified_signature, config_signature(config))


if __name__ == "__main__":
    unittest.main()
