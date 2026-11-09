"""Hardware-free regressions for the base-view detector used by the moving-bag dataset."""

import numpy as np
import pytest

from lerobot.nero_tools.color_detector import (
    detect_color,
    detect_scene,
    drop_ball_inside_bag,
    normalize_box,
)


def scene_with_square(color_bgr, size=(240, 320), box=(120, 150, 40, 40)):
    """Draw one solid square of `color_bgr` on a dark background (BGR frame)."""
    height, width = size
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    x, y, w, h = box
    image[y:y + h, x:x + w] = color_bgr
    return image


def test_exclude_top_ratio_zero_keeps_objects_in_the_top_band():
    # A red cube placed high in an oblique base view must survive detection when
    # the wrist-tuned 0.28 top cut is disabled.
    image = scene_with_square((0, 0, 220), box=(120, 10, 40, 40))
    assert detect_color(image, "red", exclude_top_ratio=0.0) is not None
    assert detect_color(image, "red", exclude_top_ratio=0.28) is None


def test_exclude_top_ratio_out_of_range_is_rejected():
    image = scene_with_square((0, 0, 220))
    with pytest.raises(ValueError):
        detect_color(image, "red", exclude_top_ratio=1.0)


def test_normalize_box_maps_missing_detection_to_zeros():
    box = normalize_box(None, 640, 480)
    assert box["visible"] == 0
    assert all(box[key] == 0.0 for key in ("x", "y", "w", "h", "conf"))


def test_normalize_box_scales_by_frame_size():
    raw = {"x": 320, "y": 240, "w": 64, "h": 48, "center_x": 352.0, "center_y": 264.0,
           "area": 1000.0, "score": 18000.0}
    box = normalize_box(raw, 640, 480)
    assert box["visible"] == 1
    assert box["w"] == pytest.approx(0.1)
    assert box["h"] == pytest.approx(0.1)
    assert box["conf"] == pytest.approx(1.0)


def test_normalize_box_confidence_is_capped():
    raw = {"x": 0, "y": 0, "w": 10, "h": 10, "center_x": 5.0, "center_y": 5.0,
           "area": 100.0, "score": 999999.0}
    assert normalize_box(raw, 640, 480)["conf"] == pytest.approx(1.0)


def test_detect_scene_reports_cube_ball_and_bag_keys():
    image = np.full((240, 320, 3), 30, dtype=np.uint8)
    image[150:190, 120:160] = (0, 0, 220)   # red cube
    image[150:180, 40:70] = (220, 0, 0)     # blue ball
    scene = detect_scene(image, "red", ball_color="blue")
    assert set(scene) == {"cube", "ball", "bag"}
    for key, value in scene.items():
        assert set(value) == {"visible", "x", "y", "w", "h", "cx", "cy", "conf"}, key
    assert scene["cube"]["visible"] == 1
    assert scene["ball"]["visible"] == 1


def test_detect_scene_skips_ball_when_color_disabled():
    image = scene_with_square((0, 0, 220))
    scene = detect_scene(image, "red", ball_color=None)
    assert scene["ball"]["visible"] == 0


def test_cube_and_ball_colours_do_not_overlap():
    # Red cube and blue ball must be separable by colour alone.
    red_only = scene_with_square((0, 0, 220))
    assert detect_color(red_only, "blue") is None
    blue_only = scene_with_square((220, 0, 0))
    assert detect_color(blue_only, "red") is None


def scene_dict(ball_visible=True, ball_cx=0.46, ball_cy=0.95,
               bag_visible=True, bag=(0.38, 0.61, 0.21, 0.32)):
    scene = {
        "cube": normalize_box(None, 1, 1),
        "ball": {"visible": 1 if ball_visible else 0,
                 "x": ball_cx - 0.02, "y": ball_cy - 0.02, "w": 0.04, "h": 0.04,
                 "cx": ball_cx, "cy": ball_cy, "conf": 0.01},
        "bag": {"visible": 1 if bag_visible else 0,
                "x": bag[0], "y": bag[1], "w": bag[2], "h": bag[3],
                "cx": bag[0] + bag[2] / 2, "cy": bag[1] + bag[3] / 2, "conf": 1.0},
    }
    if not ball_visible:
        scene["ball"] = normalize_box(None, 1, 1)
    return scene


def test_ball_printing_inside_the_bag_is_rejected():
    # A blue blob inside the bag mouth is printing, not the distractor ball.
    # Bag box spans y 0.61-0.933, so 0.80 is comfortably inside the inset area.
    scene = drop_ball_inside_bag(scene_dict(ball_cx=0.46, ball_cy=0.80))
    assert scene["ball"]["visible"] == 0
    assert scene["ball_rejected"] == "inside_bag_print"


def test_ball_below_the_bag_bottom_is_not_caught_by_this_rule():
    # Documents the measured case: the real false positive sat at y~=0.93, just
    # below the bag's lower edge (0.933), so this geometric rule does not catch
    # it. The saturation floor in detect_scene is what removes that one.
    scene = drop_ball_inside_bag(scene_dict(ball_cx=0.46, ball_cy=0.95))
    assert scene["ball"]["visible"] == 1
    assert "ball_rejected" not in scene


def test_ball_next_to_the_bag_is_kept():
    # A real distractor ball sits on the table, left of the bag.
    scene = drop_ball_inside_bag(scene_dict(ball_cx=0.20, ball_cy=0.70))
    assert scene["ball"]["visible"] == 1
    assert "ball_rejected" not in scene


def test_ball_above_the_bag_is_kept():
    scene = drop_ball_inside_bag(scene_dict(ball_cx=0.46, ball_cy=0.40))
    assert scene["ball"]["visible"] == 1


def test_ball_without_a_bag_is_kept():
    scene = drop_ball_inside_bag(scene_dict(ball_cx=0.46, ball_cy=0.93, bag_visible=False))
    assert scene["ball"]["visible"] == 1


def test_bag_edge_does_not_swallow_a_nearby_ball():
    # inset=0.9 shrinks the box so a ball hugging the left edge survives.
    scene = drop_ball_inside_bag(scene_dict(ball_cx=0.381, ball_cy=0.70))
    assert scene["ball"]["visible"] == 1
