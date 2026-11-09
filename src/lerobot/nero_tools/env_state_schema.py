"""Versioned environment-state feature layouts for NERO datasets."""

from __future__ import annotations

import numpy as np

ENVIRONMENT_STATE_BASE14_NAMES = [
    'cube_cx', 'cube_cy', 'cube_w', 'cube_h', 'cube_visible',
    'bag_cx', 'bag_cy', 'bag_w', 'bag_h', 'bag_visible',
    'cube_to_bag_dx', 'cube_to_bag_dy', 'cube_area_ratio', 'bag_area_ratio',
]


def validate_base14(values):
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError('base14 requires 14 finite values')
    if any(result[i] not in (0, 1) for i in (4, 9)):
        raise ValueError('base14 visibility must be binary')
    if np.any(result[:10] < 0) or np.any(result[:10] > 1) or np.any(result[12:] < 0) or np.any(result[12:] > 1):
        raise ValueError('base14 coordinates and areas must be normalized')
    for offset, area in ((0, 12), (5, 13)):
        if not result[offset + 4] and (np.any(result[offset:offset + 4]) or result[area]):
            raise ValueError('Missing objects must have zero geometry and area')
    expected = result[5:7] - result[:2] if result[4] and result[9] else np.zeros(2)
    if not np.allclose(result[10:12], expected, atol=1e-6):
        raise ValueError('base14 relative offsets do not match visible centers')
    return result


def environment_state_base14(scene):
    """Normalized base boxes; area_ratio is contour area / full image area."""
    values, areas = [], []
    for name in ('cube', 'bag'):
        box = scene.get(name) or {}
        if box.get('visible', 0):
            center = _center(box)
            if center is None:
                raise ValueError('Visible base14 object requires geometry')
            values.extend([*center, box['w'], box['h'], 1.0])
            areas.append(box['area_ratio'])
        else:
            values.extend([0.0] * 5)
            areas.append(0.0)
    values.extend([values[5] - values[0], values[6] - values[1]]
                  if values[4] and values[9] else [0.0, 0.0])
    return validate_base14(values + areas)

ENVIRONMENT_STATE_RELATION_V2_NAMES = [
    "cube_base_cx", "cube_base_cy", "cube_base_visible", "cube_base_score_proxy",
    "bag_mouth_base_cx", "bag_mouth_base_cy", "bag_visible", "bag_area_proxy",
    "cube_to_bag_dx", "cube_to_bag_dy",
    "ball_to_cube_dx", "ball_to_cube_dy", "ball_visible", "ball_score_proxy",
    "wrist_cube_image_center_dx", "wrist_cube_image_center_dy",
    "wrist_cube_visible", "wrist_cube_score_proxy",
]


def _center(box):
    if not box or not box.get("visible", 0):
        return None
    if "cx" in box and "cy" in box:
        return float(box["cx"]), float(box["cy"])
    if all(key in box for key in ("x", "y", "w", "h")):
        return float(box["x"]) + float(box["w"]) / 2.0, float(box["y"]) + float(box["h"]) / 2.0
    return None


def environment_state_relation_v2(base_scene, wrist_cube):
    """Build the 18-value v2 vector from normalized detector outputs.

    Relative offsets are image-plane coordinates, not calibrated table-plane
    measurements. The wrist-camera offset is relative to the image centre, not
    the gripper. Score fields are uncalibrated detector-quality proxies, not
    probabilities. No picked-up state is inferred from visibility alone.
    """
    cube = base_scene.get("cube") or {}
    bag = base_scene.get("bag") or {}
    ball = base_scene.get("ball") or {}
    cube_center = _center(cube)
    ball_center = _center(ball)
    wrist_center = _center(wrist_cube or {})
    cube_visible = float(cube_center is not None)
    bag_visible = float(bool(bag.get("visible", 0)))
    ball_visible = float(ball_center is not None)
    wrist_visible = float(wrist_center is not None)

    if bag_visible:
        bag_mouth_x = float(bag.get("x", 0.0)) + float(bag.get("w", 0.0)) / 2.0
        bag_mouth_y = float(bag.get("y", 0.0)) + 0.06 * float(bag.get("h", 0.0))
    else:
        bag_mouth_x = bag_mouth_y = 0.0

    cube_x, cube_y = cube_center if cube_center is not None else (0.0, 0.0)
    ball_x, ball_y = ball_center if ball_center is not None else (0.0, 0.0)
    if cube_center is not None and bag_visible:
        cube_to_bag_dx, cube_to_bag_dy = bag_mouth_x - cube_x, bag_mouth_y - cube_y
    else:
        cube_to_bag_dx = cube_to_bag_dy = 0.0
    if cube_center is not None and ball_center is not None:
        ball_to_cube_dx, ball_to_cube_dy = ball_x - cube_x, ball_y - cube_y
    else:
        ball_to_cube_dx = ball_to_cube_dy = 0.0

    wrist_dx, wrist_dy = (
        (2.0 * wrist_center[0] - 1.0, 2.0 * wrist_center[1] - 1.0)
        if wrist_center is not None else (0.0, 0.0)
    )
    values = [
        cube_x, cube_y, cube_visible, float(cube.get("conf", 0.0)) if cube_visible else 0.0,
        bag_mouth_x, bag_mouth_y, bag_visible, float(bag.get("conf", 0.0)) if bag_visible else 0.0,
        cube_to_bag_dx, cube_to_bag_dy,
        ball_to_cube_dx, ball_to_cube_dy, ball_visible, float(ball.get("conf", 0.0)) if ball_visible else 0.0,
        wrist_dx, wrist_dy, wrist_visible, float((wrist_cube or {}).get("conf", 0.0)) if wrist_visible else 0.0,
    ]
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("relation_v2 contains non-finite detector values")
    return result
