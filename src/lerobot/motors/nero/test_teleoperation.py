"""Hardware-free regressions for NERO firmware, gripper and paired sessions."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lerobot.motors.nero import NeroMotorsBus, NeroMotorsBusConfig
from lerobot.nero_teleoperate import JOINT_KEYS, NeroTeleopConfig, TargetMapper, run
from lerobot.robots.nero_follower.test_nero_follower import _make_bus, _make_follower
from lerobot.teleoperators.nero_leader.test_nero_leader import _make_leader


@pytest.mark.parametrize("profile,value", [("default", 90.0), ("v111", 90.0),
                                           ("v112", math.pi / 2), ("v120", math.pi / 2),
                                           ("v121", math.pi / 2)])
def test_leader_units_follow_firmware(profile, value):
    bus, fake = _make_bus(role="leader", angle_source="leader")
    bus.config.firmeware_version = profile
    fake.get_leader_joint_angles = lambda: SimpleNamespace(msg=[value] * 7, hz=30)
    assert bus.read_joint_positions() == pytest.approx([math.pi / 2] * 7)


def test_sdk_new_firmware_decoder_already_returns_radians():
    from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v112.parser import Codec

    message = SimpleNamespace()
    Codec().decode_170_joint_ctrl_7(message, bytearray((90000).to_bytes(4, "big") + bytes(4)))
    assert message.joint_7 == pytest.approx(math.pi / 2)


def test_timestamp_stall_is_not_confused_with_stationary_arm():
    bus, _ = _make_bus()
    message = SimpleNamespace(hz=30, timestamp=100)
    with patch("lerobot.motors.nero.nero.time.monotonic", return_value=10):
        assert bus._feedback_is_fresh(message, "follower")
    with patch("lerobot.motors.nero.nero.time.monotonic", return_value=11):
        assert not bus._feedback_is_fresh(message, "follower")
        message.timestamp = 101
        assert bus._feedback_is_fresh(message, "follower")


def test_auto_source_falls_back_when_leader_stream_stops():
    bus, fake = _make_bus(role="leader", angle_source="auto")
    fake.get_leader_joint_angles = lambda: SimpleNamespace(msg=[90.0] * 7, hz=0)
    assert bus.read_joint_positions() == fake._joints
    assert bus.stats["angle_source"] == "follower"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_invalid_action_is_rejected_before_any_write(bad):
    follower, fake = _make_follower()
    action = dict.fromkeys(follower.action_features, 0.2)
    action["gripper.pos"] = bad
    with pytest.raises(ValueError):
        follower.send_action(action)
    assert not fake.written


def test_gripper_command_units_threshold_and_errors():
    bus, _ = _make_bus(gripper_mode="angle")
    bus.gripper = Mock()
    assert bus.write_gripper(0.5) == 0.5
    bus.gripper.move_gripper_deg.assert_called_once_with(value=50.0, force=1.0)
    assert bus.write_gripper(0.501) == 0.5
    assert bus.gripper.move_gripper_deg.call_count == 1
    bus.gripper.move_gripper_deg.side_effect = OSError("CAN down")
    with pytest.raises(RuntimeError, match="gripper command failed"):
        bus.write_gripper(0.8)
    assert bus._last_gripper_sent == 0.5


@pytest.mark.parametrize("code,value", [(1, 0.035), (5, 0.05)])
def test_leader_pendant_gripper_feedback_units(code, value):
    bus, _ = _make_bus(role="leader")
    bus.gripper = Mock()
    bus.gripper.get_gripper_ctrl_states.return_value = SimpleNamespace(
        msg=SimpleNamespace(value=value, status_code=code), hz=30,
    )
    assert bus.read_gripper() == pytest.approx(0.5)


def test_gripper_stale_feedback_is_rejected():
    bus, _ = _make_bus()
    bus.gripper = Mock()
    bus.gripper.get_gripper_status.return_value = SimpleNamespace(
        msg=SimpleNamespace(value=0.035, mode="width"), hz=0,
    )
    assert bus.read_gripper() is None


@pytest.mark.parametrize("role,enabled", [("leader", True), ("follower", False)])
def test_gripper_write_requires_enabled_follower(role, enabled):
    bus, _ = _make_bus(role=role)
    bus._is_enabled = enabled
    bus.gripper = Mock()
    with pytest.raises(RuntimeError):
        bus.write_gripper(0.5)
    bus.gripper.move_gripper_m.assert_not_called()


def test_observation_never_uses_a_command_as_feedback():
    bus, fake = _make_bus()
    bus._last_sent = [1.0] * 7
    bus._last_gripper_sent = 0.8
    fake._joints = None
    bus.gripper = None
    assert bus.present_positions() is None
    assert bus.read_gripper_or_last() is None


def test_gripper_cache_and_leader_timeout():
    leader, fake = _make_leader()
    leader.get_action()
    fake.gripper_value = None
    leader._gripper_at -= 1
    with pytest.raises(TimeoutError, match="gripper"):
        leader.get_action()


def test_leader_without_gripper_feedback_never_commands_closed():
    leader, fake = _make_leader()
    fake.gripper_value = None
    with pytest.raises(RuntimeError, match="gripper feedback"):
        leader.get_action()


def test_leader_joint_timeout():
    leader, fake = _make_leader()
    leader.get_action()
    fake.angles = None
    leader._angles_at -= 1
    with pytest.raises(TimeoutError, match="joint feedback"):
        leader.get_action()


def test_joint_limits_return_actual_sent_values():
    bus, fake = _make_bus()
    sent = bus.send_joints([100.0] * 7)
    assert sent == [bus.joint_limits()[f"joint{i}"][1] for i in range(1, 8)]
    assert fake.moved_to == sent


def test_connect_failure_releases_partial_connection():
    bus = NeroMotorsBus(NeroMotorsBusConfig(firmeware_version="v120"))
    fake = Mock()
    fake.get_config.return_value = {"firmeware_version": "v120"}
    fake.get_joints_enable_status_list.return_value = [False] * 7
    fake.init_effector.side_effect = OSError("effector init failed")
    with patch("lerobot.motors.nero.nero.AgxArmFactory.create_arm", return_value=fake):
        with pytest.raises(RuntimeError):
            bus.connect()
    fake.disable.assert_called()
    fake.disconnect.assert_called_once()
    assert not bus.is_connected


def test_camera_failure_releases_bus():
    follower, fake = _make_follower()
    follower._is_connected = False
    follower.cameras = {"front": Mock()}
    follower.cameras["front"].connect.side_effect = OSError("camera missing")
    with pytest.raises(OSError):
        follower.connect()
    assert "disconnect" in fake.calls


def state(joint=0.1, gripper=0.3):
    return {**dict.fromkeys(JOINT_KEYS, joint), "gripper.pos": gripper}


def test_relative_start_and_delayed_tick_are_bounded():
    config = NeroTeleopConfig()
    mapper = TargetMapper(config, state(1.0, 0.8), state(0.1, 0.3))
    # dt is clamped to max_control_dt_s before it is used as the speed budget.
    step_dt = min(1 / 30, config.max_control_dt_s)
    first = mapper.target(state(1.0, 0.8), 1 / 30)
    assert first["joint_1.pos"] == pytest.approx(0.1)
    assert first["gripper.pos"] == pytest.approx(0.3 + config.max_gripper_speed * step_dt)
    mapper.accept(first)
    # A delayed tick is clamped to max_control_dt_s, so one step is bounded by
    # max_joint_speed * max_control_dt_s, not by 1/fps.
    moved = mapper.target(state(2.0, 0.8), 10.0)
    assert moved["joint_1.pos"] == pytest.approx(0.1 + config.max_joint_speed * config.max_control_dt_s)
    assert moved["joint_1.pos"] - first["joint_1.pos"] <= config.max_joint_speed * config.max_control_dt_s + 1e-9


def test_absolute_start_rejects_large_pose_difference():
    with pytest.raises(ValueError, match="startup mismatch"):
        TargetMapper(NeroTeleopConfig(mapping="absolute"), state(1.0), state(0.0))


def test_normal_feedback_hz_reads_normal_channel():
    bus, fake = _make_bus(role="leader", angle_source="leader")
    assert bus.normal_feedback_hz() == pytest.approx(140.0)
    fake.hz = 0.0
    assert bus.normal_feedback_hz() == pytest.approx(0.0)
    assert bus.wait_for_normal_feedback(timeout_s=0.2) is False
    fake.hz = 220.0
    assert bus.wait_for_normal_feedback(timeout_s=0.5) is True


def devices():
    result = []
    for channel in ("0", "1"):
        device = Mock()
        device.config = SimpleNamespace(gripper_max_width_m=0.07,
                                        gripper_max_angle_deg=100.0, gripper_mode="width")
        device.action_features = dict.fromkeys(state(), float)
        device.bus.config = SimpleNamespace(interface="fake", channel=channel)
        device.bus.wait_stable_pose.return_value = [0.1] * 7
        device.bus.health_problems.return_value = []
        device.get_action.return_value = state()
        device.get_observation.return_value = state()
        device.send_action.side_effect = lambda action: dict(action)
        result.append(device)
    return result


def test_position_output_units():
    from lerobot.nero_teleoperate import format_positions

    values = dict.fromkeys(JOINT_KEYS, math.pi / 2)
    values["gripper.pos"] = 0.5
    text = format_positions(values, 0.07)
    assert "J1=90.00" in text and "J7=90.00" in text
    assert "gripper=35.00 mm" in text
    assert "gripper=50.00 mm" in format_positions(values, 0.10)
    del values["gripper.pos"]
    assert "gripper=disabled" in format_positions(values, 0.07)


def test_paired_session_sends_all_eight_axes_and_cleans_up():
    leader, follower = devices()
    assert run(leader, follower, NeroTeleopConfig(duration_s=0.07)) >= 1
    assert set(follower.send_action.call_args.args[0]) == set(state())
    leader.disconnect.assert_called_once()
    follower.disconnect.assert_called_once()


def test_second_connection_failure_cleans_up_both_devices():
    leader, follower = devices()
    follower.connect.side_effect = OSError("CAN unavailable")
    with pytest.raises(OSError):
        run(leader, follower, NeroTeleopConfig())
    leader.disconnect.assert_called_once()
    follower.disconnect.assert_called_once()
    follower.send_action.assert_not_called()


def test_same_can_channel_rejected_before_connection():
    leader, follower = devices()
    follower.bus.config.channel = leader.bus.config.channel
    with pytest.raises(ValueError, match="different CAN"):
        run(leader, follower, NeroTeleopConfig())
    leader.connect.assert_not_called()


def test_health_failure_stops_before_sending_target():
    leader, follower = devices()
    follower.bus.health_problems.return_value = ["joint communication error"]
    with pytest.raises(RuntimeError, match="communication error"):
        run(leader, follower, NeroTeleopConfig())
    follower.send_action.assert_not_called()
    leader.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# 预览白屏回归：2026-10-27 现场出现"相机卡白屏"，根因是 compose_preview 在没有
# 可用帧时返回 None，调用方 `if canvas is not None` 就跳过 imshow，窗口保持空白，
# 而且读帧失败被静默吞掉，全程零报错。
# ---------------------------------------------------------------------------
def _fake_camera(frame=None, size=(640, 480), age_ms=12.0, status="idx1 OK"):
    camera = SimpleNamespace()
    camera.size = size
    camera.latest = lambda: None if frame is None else frame.copy()
    camera.age_ms = lambda: age_ms
    camera.status_text = lambda: status
    camera.is_live = lambda: frame is not None and age_ms <= 1000.0
    return camera


def test_preview_never_returns_none_without_frames():
    """没有任何帧时必须返回可读的诊断图，绝不能返回 None（那会直接白屏）。"""
    import numpy as np

    from lerobot.nero_teleoperate import compose_preview

    cameras = [("wrist", _fake_camera()), ("base", _fake_camera())]
    canvas = compose_preview(cameras, count=10, measured_hz=60.0)
    assert canvas is not None, "没有帧时返回 None 会让 imshow 被跳过 -> 白屏"
    assert canvas.shape == (480, 1280, 3)
    # 不能是纯白（用户看到的白屏就是这个），也不能是纯黑（无法和"没画东西"区分）
    assert canvas.mean() < 200, "占位图不该是白的"
    assert canvas.mean() > 5, "占位图不该是纯黑，否则和未绘制无法区分"


def test_preview_concatenates_when_both_frames_present():
    import numpy as np

    from lerobot.nero_teleoperate import compose_preview

    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    cameras = [("wrist", _fake_camera(frame)), ("base", _fake_camera(frame))]
    canvas = compose_preview(cameras, count=10)
    assert canvas.shape == (480, 1280, 3)


def test_camera_records_read_failures_instead_of_failing_silently():
    """读帧失败必须被记账并可查询，否则上层只看到 None，用户只看到无声白屏。"""
    from lerobot.nero_teleoperate import LatestCamera

    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (False, None)
    with patch("lerobot.nero_teleoperate.cv2.VideoCapture", return_value=capture):
        camera = LatestCamera(1, "dshow", 640, 480, 30)
        try:
            assert camera.wait_first_frame(0.3) is False
            text = camera.status_text()
            failures = camera.total_failures
            assert camera.is_live() is False
        finally:
            camera.close()
    assert "NO FRAME YET" in text
    assert failures > 0, "连续读帧失败必须被计数，不能静默吞掉"
