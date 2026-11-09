"""Hardware-free regressions for the NERO episode capture control path."""

from unittest.mock import Mock
import json
import sys

import pytest
import numpy as np

from lerobot.nero_teleoperate import JOINT_KEYS, NeroTeleopConfig, TargetMapper
from lerobot.nero_tools import episode_capture as capture
from lerobot.nero_tools.episode_capture import send_mapped_target


def state(joint=0.0, gripper=0.0):
    return {**dict.fromkeys(JOINT_KEYS, joint), "gripper.pos": gripper}


def test_capture_advances_mapper_from_actual_sent_target():
    config = NeroTeleopConfig(fps=50, max_joint_speed=1.0, max_gripper_speed=1.0)
    mapper = TargetMapper(config, state(), state())
    follower = Mock()
    follower.send_action.side_effect = lambda target: dict(target)

    first = send_mapped_target(mapper, follower, state(1.0, 1.0), 0.02)
    second = send_mapped_target(mapper, follower, state(1.0, 1.0), 0.02)

    assert first["joint_1.pos"] == pytest.approx(0.02)
    assert second["joint_1.pos"] == pytest.approx(0.04)
    assert second["joint_1.pos"] > first["joint_1.pos"]


def test_next_index_preserves_gaps_and_partial_directories(tmp_path):
    assert capture.next_episode_index(tmp_path) == 0
    for name in ("episode_0000", "episode_0002", "episode_0010", "episode_notes"):
        (tmp_path / name).mkdir()
    assert capture.next_episode_index(tmp_path) == 11


def test_existing_episode_cannot_be_overwritten(tmp_path):
    old = tmp_path / "episode_0000"
    old.mkdir()
    marker = old / "metadata.json"
    marker.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        capture.save_episode(tmp_path, 0, [], [], [], [], {})
    assert marker.read_bytes() == b"original"
    assert not (old / "frames").exists()


@pytest.mark.parametrize("exit_key", [ord("q"), 27])
def test_restart_and_quit_save_without_overwrite(tmp_path, monkeypatch, exit_key):
    old = tmp_path / "episode_0002"
    old.mkdir()
    marker = old / "metadata.json"
    marker.write_bytes(b"original")
    monkeypatch.setattr(sys, "argv", ["capture", "--output", str(tmp_path),
                                       "--leader-channel", "1",
                                       "--follower-channel", "0"])
    monkeypatch.setattr(capture, "load_start_pose", lambda _: None)
    for name in ("imshow", "putText", "destroyAllWindows"):
        monkeypatch.setattr(capture.cv2, name, lambda *a, **k: None)
    for expected in (3, 4):
        keys = iter([ord("s"), exit_key])
        monkeypatch.setattr(capture.cv2, "waitKey", lambda _: next(keys))
        camera = Mock()
        camera.snapshot.side_effect = [(np.zeros((4, 4, 3), dtype=np.uint8), 1000, 1),
                                       (np.zeros((4, 4, 3), dtype=np.uint8), 1033, 2)]
        monkeypatch.setattr(capture, "create_camera", lambda _: camera)
        leader, follower = Mock(), Mock()
        connect_order = []
        leader.connect.side_effect = lambda **kwargs: connect_order.append("leader")
        follower.connect.side_effect = lambda: connect_order.append("follower")
        leader.get_action.return_value = state()
        follower.get_observation.return_value = state()
        follower.bus.read.return_value = {k.removesuffix(".pos"): v for k, v in state().items()}
        follower.send_action.side_effect = lambda target: dict(target)
        monkeypatch.setattr(capture, "NeroLeader", lambda _: leader)
        monkeypatch.setattr(capture, "NeroFollower", lambda _: follower)
        assert capture.main() == 0
        assert connect_order == ["follower", "leader"]
        saved = tmp_path / f"episode_{expected:04d}"
        assert json.loads((saved / "metadata.json").read_text())["result"] == "quit_end"
        assert len(list((saved / "frames").glob("*.png"))) == 2
        assert len((saved / "robot_state.jsonl").read_text().splitlines()) == 2
        leader.disconnect.assert_called_once()
        follower.disconnect.assert_called_once()
    assert marker.read_bytes() == b"original"


def test_background_failure_is_reported(tmp_path):
    (tmp_path / "episode_0000").mkdir()
    writer = capture.save_episode_background(tmp_path, 0, [], [], [], [], {})
    writer.join(timeout=5)
    assert not writer.is_alive()
    assert isinstance(writer.error, FileExistsError)


def test_parse_camera_spec_accepts_design_document_syntax():
    spec = capture.parse_camera_spec("wrist=index:0,backend:dshow,size:640x480,fps:30")
    assert (spec.name, spec.index, spec.backend) == ("wrist", 0, "dshow")
    assert (spec.width, spec.height, spec.fps) == (640, 480, 30)
    assert capture.parse_camera_spec("base = index = 3 , size:480x360").index == 3


def test_parse_camera_spec_rejects_unknown_option():
    with pytest.raises(ValueError, match="Unknown --camera-spec option"):
        capture.parse_camera_spec("wrist=index:0,exposure:-6")


def test_parse_camera_spec_rejects_duplicate_names():
    args = Mock(camera_spec=["wrist=index:0", "wrist=index:1"])
    with pytest.raises(ValueError, match="Duplicate camera names"):
        capture.camera_specs_from_args(args)


def test_legacy_invocation_yields_single_wrist_spec():
    args = Mock(camera_spec=None, camera_index=2, serial="s", camera_backend="msmf",
                camera_width=320, camera_height=240, camera_fps=15)
    specs = capture.camera_specs_from_args(args)
    assert [spec.name for spec in specs] == ["wrist"]
    assert specs[0].index == 2 and specs[0].fps == 15


def test_save_episode_writes_one_directory_per_camera(tmp_path):
    images = {"wrist": [np.zeros((4, 4, 3), np.uint8)] * 2,
              "base": [np.zeros((3, 3, 3), np.uint8)] * 2}
    frames = [{"frame_index": i, "cameras": {"wrist": {}, "base": {}}} for i in range(2)]
    states = [{"host_time_s": 1.0 + i} for i in range(2)]
    actions = [{"host_time_s": 1.0 + i} for i in range(2)]
    episode_dir = capture.save_episode(tmp_path, 0, images, frames, states, actions, {"result": "manual_end"})
    assert len(list((episode_dir / "frames" / "wrist").glob("*.png"))) == 2
    assert len(list((episode_dir / "frames" / "base").glob("*.png"))) == 2


def test_parse_camera_spec_accepts_serial_binding():
    # Serial binding survives re-plugging; DirectShow indices do not.
    spec = capture.parse_camera_spec("base=serial:261622071323,size:640x480,fps:30")
    assert (spec.name, spec.serial) == ("base", "261622071323")
    wrist = capture.parse_camera_spec("wrist=serial:261622078506,size:640x480,fps:30")
    assert (wrist.name, wrist.serial) == ("wrist", "261622078506")
    assert (spec.width, spec.height, spec.fps) == (640, 480, 30)


def test_index_camera_specs_open_opencv_even_with_legacy_realsense_default(monkeypatch):
    spec = capture.parse_camera_spec(
        "wrist=index:3,backend:dshow,size:640x480,fps:30"
    )
    args = Mock(camera_spec=["explicit"], camera_type="realsense", serial="unused")
    expected = object()
    monkeypatch.setattr(capture, "OpenCVCameraReader", lambda *a: expected)
    monkeypatch.setattr(capture, "RealSenseReader", Mock(side_effect=AssertionError("wrong backend")))

    assert capture.open_reader(spec, args) is expected


def test_frame_age_tolerates_device_clock():
    # RealSense device timestamps are not epoch-based; they must not read as stale.
    assert capture.frame_age_ms(0) == 0.0
    assert capture.frame_age_ms(1234.0) == 0.0


def test_host_frame_age_ignores_non_numeric_reader_values():
    # A stub/mock reader without a real host_age_ms must never leak a Mock into
    # arithmetic: that raised "'Mock' > 'float'" inside the control loop.
    stub = Mock()
    assert capture.host_frame_age_ms(stub) == 0.0

    class NumericReader:
        def host_age_ms(self):
            return 12.5

    assert capture.host_frame_age_ms(NumericReader()) == 12.5


def test_host_frame_age_uses_caller_timestamp_without_extra_snapshot():
    # The control loop already snapshotted; frame-age must not take a second one,
    # which would consume an extra frame (and exhaust a stubbed side_effect list).
    stub = Mock()
    assert capture.host_frame_age_ms(stub, 0) == 0.0
    stub.snapshot.assert_not_called()


def test_host_frame_age_uses_epoch_timestamp_when_host_age_missing():
    stub = Mock()
    age = capture.host_frame_age_ms(stub, 1234.0)
    # A non-epoch (device) timestamp is treated as 0 rather than as stale.
    assert age == 0.0
