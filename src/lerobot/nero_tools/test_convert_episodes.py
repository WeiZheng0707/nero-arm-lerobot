"""Hardware-free regressions for the 14-dim environment_state layout."""

import json

import numpy as np
import pytest

from lerobot.nero_tools.convert_episodes import (
    ENVIRONMENT_STATE_NAMES,
    configure_camera_features,
    detect_label_schema,
    environment_state_vector,
    merge_env_state,
    pi05_state_vector,
    PI05_STATE_NAMES,
)
from lerobot.nero_tools.env_state_schema import (
    ENVIRONMENT_STATE_RELATION_V2_NAMES,
    environment_state_relation_v2,
)
from env_state_batch import complete_label_file


def test_pi05_22_preserves_state_and_environment_order():
    robot = np.arange(8, dtype=np.float32)
    env = np.arange(14, dtype=np.float32) / 20
    merged = pi05_state_vector(robot, env)
    assert merged.dtype == np.float32
    np.testing.assert_array_equal(merged[:8], robot)
    np.testing.assert_array_equal(merged[8:], env)
    assert len(PI05_STATE_NAMES) == 22
    assert PI05_STATE_NAMES[8:] == ENVIRONMENT_STATE_NAMES


def test_pi05_22_rejects_wrong_shape_and_nonfinite():
    with pytest.raises(ValueError, match="requires"):
        pi05_state_vector(np.zeros(8), np.zeros(5))
    with pytest.raises(ValueError, match="non-finite"):
        pi05_state_vector(np.full(8, np.nan), np.zeros(14))


def test_camera_features_use_video_dtype_when_video_mode_is_requested():
    base = {
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    }
    converted = configure_camera_features(
        base,
        {
            "wrist": "observation.images.wrist",
            "base": "observation.images.base",
        },
        use_video=True,
    )
    assert converted["observation.images.wrist"]["dtype"] == "video"
    assert converted["observation.images.base"]["dtype"] == "video"
    assert base["observation.images.wrist"]["dtype"] == "image"
    image_features = configure_camera_features(
        base,
        {"wrist": "observation.images.wrist"},
        use_video=False,
    )
    assert image_features["observation.images.wrist"]["dtype"] == "image"


def base_frame(cube=None, bag=None, ball=None):
    frame = {}
    if cube is not None:
        frame["cube"] = cube
    if bag is not None:
        frame["bag"] = bag
    if ball is not None:
        frame["ball"] = ball
    return frame


def test_state_layout_is_fourteen_dims_in_documented_order():
    assert len(ENVIRONMENT_STATE_NAMES) == 14
    assert ENVIRONMENT_STATE_NAMES[:5] == ["cube_x", "cube_y", "cube_w", "cube_h", "cube_visible"]
    assert ENVIRONMENT_STATE_NAMES[5:9] == ["bag_x", "bag_y", "bag_w", "bag_h"]
    assert ENVIRONMENT_STATE_NAMES[9:13] == ["ball_x", "ball_y", "ball_w", "ball_h"]
    assert ENVIRONMENT_STATE_NAMES[13] == "ball_visible"


def test_bag_position_is_carried_through_so_it_can_move():
    frame = base_frame(cube={"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.05, "visible": 1},
                       bag={"x": 0.7, "y": 0.6, "w": 0.2, "h": 0.2, "visible": 1})
    vector = environment_state_vector(frame, "base")
    assert vector.shape == (14,)
    assert vector[5:9] == pytest.approx([0.7, 0.6, 0.2, 0.2])
    assert vector[9:13] == pytest.approx([0, 0, 0, 0])  # no ball present
    assert vector[13] == 0.0


def test_ball_position_is_carried_because_selection_depends_on_it():
    frame = base_frame(cube={"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.05, "visible": 1},
                       bag={"x": 0.7, "y": 0.6, "w": 0.2, "h": 0.2, "visible": 1},
                       ball={"x": 0.3, "y": 0.32, "w": 0.04, "h": 0.04, "visible": 1})
    vector = environment_state_vector(frame, "base")
    assert vector[9:13] == pytest.approx([0.3, 0.32, 0.04, 0.04])
    assert vector[13] == 1.0


def test_missing_object_fills_zeros_without_shifting_columns():
    full = base_frame(cube={"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.05, "visible": 1},
                      bag={"x": 0.7, "y": 0.6, "w": 0.2, "h": 0.2, "visible": 1},
                      ball={"x": 0.3, "y": 0.3, "w": 0.04, "h": 0.04, "visible": 1})
    partial = base_frame(cube={"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.05, "visible": 1},
                         bag={"x": 0.7, "y": 0.6, "w": 0.2, "h": 0.2, "visible": 1})
    a, b = environment_state_vector(full, "base"), environment_state_vector(partial, "base")
    assert a.shape == b.shape == (14,)
    # Everything except the ball block is identical: no column shifted.
    assert a[:9] == pytest.approx(b[:9])
    assert (a[13], b[13]) == (1.0, 0.0)


def test_legacy_five_dim_schema_is_still_convertible():
    frame = {"block_visible": 1, "block_x": 0.1, "block_y": 0.2, "block_w": 0.05, "block_h": 0.05}
    vector = environment_state_vector(frame, "legacy5")
    assert vector.shape == (5,)


def test_schema_detection_distinguishes_base_and_legacy():
    assert detect_label_schema([base_frame(cube={"visible": 1})]) == "base"
    assert detect_label_schema([{"block_visible": 0}]) == "legacy5"
    assert detect_label_schema([{"frame_index": 0}]) is None


def test_mixed_schemas_in_one_episode_are_rejected():
    with pytest.raises(ValueError, match="mixes base-camera and legacy"):
        detect_label_schema([base_frame(cube={"visible": 1}), {"block_visible": 1}])


def test_vector_is_float32_for_dataset_features():
    frame = base_frame(cube={"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.05, "visible": 1})
    assert environment_state_vector(frame, "base").dtype == np.float32


def test_relation_v2_encodes_object_visibility_and_relative_positions():
    scene = {
        "cube": {"visible": 1, "x": 0.20, "y": 0.30, "w": 0.10, "h": 0.10, "cx": 0.25, "cy": 0.35, "conf": 0.8},
        "bag": {"visible": 1, "x": 0.60, "y": 0.20, "w": 0.20, "h": 0.50, "conf": 0.6},
        "ball": {"visible": 1, "x": 0.40, "y": 0.40, "w": 0.04, "h": 0.04, "cx": 0.42, "cy": 0.42, "conf": 0.7},
    }
    wrist = {"visible": 1, "cx": 0.60, "cy": 0.25, "conf": 0.9}
    vector = environment_state_relation_v2(scene, wrist)
    assert vector.shape == (len(ENVIRONMENT_STATE_RELATION_V2_NAMES),) == (18,)
    assert vector.dtype == np.float32
    values = dict(zip(ENVIRONMENT_STATE_RELATION_V2_NAMES, vector, strict=True))
    assert values["cube_to_bag_dx"] == pytest.approx(0.45)
    assert values["cube_to_bag_dy"] == pytest.approx(-0.12)
    assert values["ball_to_cube_dx"] == pytest.approx(0.17)
    assert values["ball_to_cube_dy"] == pytest.approx(0.07)
    assert values["wrist_cube_image_center_dx"] == pytest.approx(0.20)
    assert values["wrist_cube_image_center_dy"] == pytest.approx(-0.50)
    assert values["wrist_cube_visible"] == 1.0


def test_relation_v2_missing_objects_have_explicit_visibility_bits():
    values = dict(zip(ENVIRONMENT_STATE_RELATION_V2_NAMES,
                      environment_state_relation_v2({}, None), strict=True))
    assert values["cube_base_visible"] == 0.0
    assert values["bag_visible"] == 0.0
    assert values["ball_visible"] == 0.0
    assert values["wrist_cube_visible"] == 0.0
    assert values["cube_to_bag_dx"] == values["ball_to_cube_dx"] == 0.0


def test_complete_label_file_checks_row_order_and_schema(tmp_path):
    path = tmp_path / "labels.jsonl"
    relation = {name: 0.0 for name in ENVIRONMENT_STATE_RELATION_V2_NAMES}
    path.write_text("\n".join(json.dumps({"frame_index": i, "relation_v2": relation})
                          for i in range(2)) + "\n", encoding="utf-8")
    assert complete_label_file(path, 2, "relation_v2")
    path.write_text("\n".join(json.dumps({"frame_index": i + 1, "relation_v2": relation})
                          for i in range(2)) + "\n", encoding="utf-8")
    assert not complete_label_file(path, 2, "relation_v2")
    path.write_text("{bad json\n", encoding="utf-8")
    assert not complete_label_file(path, 1, "relation_v2")


def write_env_state(path, labels):
    (path / "env_state.jsonl").write_text(
        "\n".join(json.dumps(label) for label in labels) + "\n", encoding="utf-8")


def test_merge_env_state_attaches_offline_labels(tmp_path):
    # Raw capture defers detection, so labels live in env_state.jsonl only.
    frames = [{"frame_index": 0}, {"frame_index": 1}]
    write_env_state(tmp_path, [
        {"frame_index": 0, "cube": {"visible": 1, "x": 0.1},
         "ball": {"visible": 0}, "bag": {"visible": 1, "x": 0.7}},
        {"frame_index": 1, "cube": {"visible": 0},
         "ball": {"visible": 1}, "bag": {"visible": 1, "x": 0.71}},
    ])
    merged, found = merge_env_state(tmp_path, frames)
    assert found is True
    assert merged[0]["bag"]["x"] == 0.7
    assert merged[1]["ball"]["visible"] == 1
    assert merged[0]["labels_source"] == "base_env_state_jsonl"
    assert detect_label_schema(merged) == "base"


def test_merge_env_state_is_a_noop_without_labels(tmp_path):
    frames = [{"frame_index": 0}]
    merged, found = merge_env_state(tmp_path, frames)
    assert found is False
    assert "cube" not in merged[0]
    assert detect_label_schema(merged) is None


def test_merge_env_state_rejects_short_label_file(tmp_path):
    # A truncated label file would shift every position label; it must not pass.
    frames = [{"frame_index": 0}, {"frame_index": 1}, {"frame_index": 2}]
    write_env_state(tmp_path, [{"cube": {"visible": 1}}])
    with pytest.raises(ValueError, match="re-run tools/env_state_detect.py"):
        merge_env_state(tmp_path, frames)


def test_merge_env_state_rejects_row_without_cube_key(tmp_path):
    frames = [{"frame_index": 0}]
    write_env_state(tmp_path, [{"frame_index": 0, "bag": {"visible": 1}}])
    with pytest.raises(ValueError, match="has no 'cube' key"):
        merge_env_state(tmp_path, frames)
