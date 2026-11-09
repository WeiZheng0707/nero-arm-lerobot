"""Eye-in-hand detector for colored objects and the black gripper."""

import cv2
import numpy as np


def _mask_for_color(hsv, color, min_sat=None, min_val=None):
    """HSV mask for one object colour.

    `min_sat` / `min_val` override the per-colour saturation and value floors.
    The defaults are tuned for the wrist camera, where the only blue thing in
    frame is the object. On the base camera the floor beyond the table edge is
    pale blue-grey and clears S>=30 easily, so the ball role passes a much
    higher saturation floor; see detect_scene.
    """
    if color == "red":
        sat = 70 if min_sat is None else min_sat
        val = 45 if min_val is None else min_val
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array((0, sat, val)), np.array((10, 255, 255))),
            cv2.inRange(hsv, np.array((170, sat, val)), np.array((179, 255, 255))),
        )
        # OpenCV hue 170-179 wraps to magenta in a naive red-channel check.
        # Red should dominate blue for hue 0-10, while hue 170-179 is already
        # covered by the wraparound threshold and must remain accepted.
        low_hue = cv2.inRange(hsv, np.array((0, sat, val)), np.array((10, 255, 255)))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        red_dominant = (
            bgr[:, :, 2].astype(np.int16) > bgr[:, :, 0].astype(np.int16) + 25
        ).astype(np.uint8) * 255
        return cv2.bitwise_or(
            cv2.bitwise_and(low_hue, red_dominant),
            cv2.inRange(hsv, np.array((170, sat, val)), np.array((179, 255, 255))),
        )
    if color == "green":
        return cv2.inRange(hsv,
                           np.array((35, 55 if min_sat is None else min_sat,
                                     45 if min_val is None else min_val)),
                           np.array((90, 255, 255)))
    if color == "blue":
        return cv2.inRange(hsv,
                           np.array((85, 30 if min_sat is None else min_sat,
                                     65 if min_val is None else min_val)),
                           np.array((135, 255, 255)))
    if color == "yellow":
        return cv2.inRange(hsv,
                           np.array((18, 70 if min_sat is None else min_sat,
                                     60 if min_val is None else min_val)),
                           np.array((38, 255, 255)))
    raise ValueError(f"Unsupported block color: {color}")


def _shape_name(contour, w, h):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    ratio = w / max(h, 1)
    if circularity >= 0.68 and 0.72 <= ratio <= 1.4:
        return "ball"
    if 0.42 <= ratio <= 2.4 and 4 <= len(approx) <= 10:
        return "cube"
    return "unknown"


def detect_color_objects(image_bgr, color, max_objects=5, roi=None, exclude_top_ratio=0.28,
                         morphology_kernel=3, min_sat=None, min_val=None):
    """Detect objects of `color` inside an optional ROI.

    `exclude_top_ratio` blanks the top band of the (ROI-cropped) image before
    contour extraction. The 0.28 default was tuned for the wrist camera's
    top-down view, where the gripper body occupies the top of the frame. A
    static third-person (base) camera looks at the scene from an oblique angle,
    so the same band would cut off the bag mouth or the block; base callers must
    pass a small value such as 0.0-0.05.
    """
    offset_x = offset_y = 0
    if roi is not None:
        rx, ry, rw, rh = roi
        offset_x, offset_y = rx, ry
        image_bgr = image_bgr[ry:ry + rh, rx:rx + rw]
    if not 0.0 <= exclude_top_ratio < 1.0:
        raise ValueError(f"exclude_top_ratio must be in [0, 1): {exclude_top_ratio}")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = _mask_for_color(hsv, color, min_sat=min_sat, min_val=min_val)
    if exclude_top_ratio > 0.0:
        mask[: int(image_bgr.shape[0] * exclude_top_ratio)] = 0
    if morphology_kernel < 1 or morphology_kernel % 2 == 0:
        raise ValueError("morphology_kernel must be a positive odd integer")
    kernel = np.ones((morphology_kernel, morphology_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        fill = area / max(w * h, 1)
        ratio = w / max(h, 1)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / max(hull_area, 1.0)
        if area < 180 or area > image_bgr.shape[0] * image_bgr.shape[1] * 0.25:
            continue
        if not 0.45 <= ratio <= 1.8 or fill < 0.45 or solidity < 0.75:
            continue
        shape = _shape_name(contour, w, h)
        score = min(area, 18000.0) * fill * solidity
        candidates.append((score, x, y, w, h, shape, area, fill, solidity))
    candidates.sort(reverse=True)
    return [{"x": x + offset_x, "y": y + offset_y, "w": w, "h": h,
             "center_x": x + offset_x + w / 2.0,
             "center_y": y + offset_y + h / 2.0, "shape": shape, "score": score,
             "area": area, "fill": fill, "solidity": solidity, "visible": 1}
            for score, x, y, w, h, shape, area, fill, solidity in candidates[:max_objects]]


def detect_color(image_bgr, color, roi=None, exclude_top_ratio=0.28,
                 morphology_kernel=3, min_sat=None, min_val=None):
    objects = detect_color_objects(image_bgr, color, max_objects=1, roi=roi,
                                   exclude_top_ratio=exclude_top_ratio,
                                   morphology_kernel=morphology_kernel,
                                   min_sat=min_sat, min_val=min_val)
    return objects[0] if objects else None


def detect_gripper_in_base(image_bgr, exclude_rects=(), min_area=600, max_area=9000,
                           max_value=70, max_sat=110, prefer_lowest=True):
    """Locate the gripper fingers in the static base view.

    The wrist-oriented `detect_gripper` returns the largest dark blob, which in
    the base view is the robot's black mounting base (measured: bounding box
    67802 px against a gripper measured in the low thousands). Because that base
    never moves, the caller can pass `exclude_rects` covering it, and an area
    window then rejects whatever else is large or tiny.

    Thresholds are deliberately tighter than the wrist detector: at
    `max_value=105` the brown bag's shadow merged with the fingers into one
    blob spanning both (bounding box 41664 px, centre 60 px off the fingers).
    `max_value=70` keeps only near-black pixels.

    `prefer_lowest` breaks ties toward the blob with the largest y centre, on the
    observation that the fingers reach down to the table while the wrist joint
    and cables stay above it.

    Returns a pixel box like the other detectors, or None.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 0, 0)), np.array((180, max_sat, max_value)))
    for rect in exclude_rects:
        rx, ry, rw, rh = rect
        mask[ry:ry + rh, rx:rx + rw] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # Gripper fingers are compact; a long thin blob is a cable or a shadow edge.
        if not 0.35 <= w / max(h, 1) <= 2.8:
            continue
        # A blob that touches the frame edge is usually a cropped artefact.
        if x <= 1 or y <= 1:
            continue
        key = y + h if prefer_lowest else -area
        candidates.append((key, area, x, y, w, h))
    if not candidates:
        return None
    _, area, x, y, w, h = max(candidates)
    return {"x": x, "y": y, "w": w, "h": h, "center_x": x + w / 2.0,
            "center_y": y + h / 2.0, "area": float(area), "visible": 1}


def detect_gripper(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 0, 0)), np.array((180, 125, 105)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if 150 <= area <= image_bgr.shape[0] * image_bgr.shape[1] * 0.20:
            candidates.append((area, x, y, w, h))
    if not candidates:
        return None
    _, x, y, w, h = max(candidates)
    return {"x": x, "y": y, "w": w, "h": h, "center_x": x + w / 2.0,
            "center_y": y + h / 2.0, "visible": 1}


def detect_bag(image_bgr, roi=None, exclude_top_ratio=0.20):
    """Detect the brown paper bag as a broad, warm low-saturation region.

    The bag is allowed to move between episodes: nothing here assumes a fixed
    bag position. `roi` narrows the search to the known table area so a brown
    tabletop or a warm-lit wall is less likely to win the largest-blob contest.
    """
    offset_x = offset_y = 0
    if roi is not None:
        rx, ry, rw, rh = roi
        offset_x, offset_y = rx, ry
        image_bgr = image_bgr[ry:ry + rh, rx:rx + rw]
    if not 0.0 <= exclude_top_ratio < 1.0:
        raise ValueError(f"exclude_top_ratio must be in [0, 1): {exclude_top_ratio}")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((5, 25, 35)), np.array((28, 190, 220)))
    if exclude_top_ratio > 0.0:
        mask[: int(image_bgr.shape[0] * exclude_top_ratio)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area >= 1200 and w >= 45 and h >= 45:
            candidates.append((area, x, y, w, h))
    if not candidates:
        return None
    _, x, y, w, h = max(candidates)
    return {"x": x + offset_x, "y": y + offset_y, "w": w, "h": h,
            "center_x": x + offset_x + w / 2.0, "center_y": y + offset_y + h / 2.0,
            "visible": 1}


def normalize_box(box, width, height, score_scale=18000.0):
    """Convert a pixel box to normalized coordinates and a score proxy.

    The detector exposes no confidence, but `score = min(area, 18000)*fill*
    solidity` already ranks candidates. Dividing by the same 18000 cap (and by
    the fill/solidity maxima of 1) yields a 0-1 ranking proxy, not a calibrated
    probability. Non-color detectors (bag/gripper) have no contour score, so
    they fall back to an area-based ratio.
    """
    if box is None:
        return {"visible": 0, "x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0,
                "cx": 0.0, "cy": 0.0, "conf": 0.0}
    area = float(box.get("area", box["w"] * box["h"]))
    conf = box.get("score")
    conf = min(1.0, float(conf) / score_scale) if conf is not None else min(1.0, area / score_scale)
    return {
        "visible": 1,
        "x": box["x"] / width,
        "y": box["y"] / height,
        "w": box["w"] / width,
        "h": box["h"] / height,
        "cx": box["center_x"] / width,
        "cy": box["center_y"] / height,
        "conf": conf,
    }


def drop_ball_inside_bag(scene, inset=0.9):
    """Discard a ball detection whose centre falls inside the bag box.

    The paper bag carries dark-blue printing, and on a 640x480 base view those
    print blobs land in the same size range as the distractor ball (both around
    25-30 px), so neither area nor confidence separates them.

    Scope note, measured on a cube-only episode: the false "ball" that appeared
    on 281 of 598 frames sat at y~=0.93 while the bag box ended at y~=0.933, i.e.
    on the bag's lower edge and partly below it. That particular false positive
    is NOT caught by this rule (it is outside the inset box); it is caught by the
    saturation floor in detect_scene, because the pale blue-grey floor beyond the
    table is far less saturated than the real ball. This function stays as a
    guard for a genuinely blue blob inside the bag mouth.

    `inset` shrinks the box slightly so a ball genuinely beside the bag is not
    swallowed by the box edge. Returns the scene with the ball zeroed on reject.
    """
    ball = scene.get("ball") or {}
    bag = scene.get("bag") or {}
    if not (ball.get("visible") and bag.get("visible")):
        return scene
    margin_x = bag["w"] * (1.0 - inset) / 2.0
    margin_y = bag["h"] * (1.0 - inset) / 2.0
    left = bag["x"] + margin_x
    right = bag["x"] + bag["w"] - margin_x
    top = bag["y"] + margin_y
    bottom = bag["y"] + bag["h"] - margin_y
    if left <= ball["cx"] <= right and top <= ball["cy"] <= bottom:
        scene["ball"] = normalize_box(None, 1, 1)
        scene["ball_rejected"] = "inside_bag_print"
    return scene


def detect_scene(image_bgr, cube_color, ball_color="blue", roi=None,
                 exclude_top_ratio=0.03, bag_roi=None, bag_exclude_top_ratio=0.0,
                 morphology_kernel=3, suppress_ball_in_bag=True,
                 ball_min_sat=110, ball_min_val=70, ball_roi=None,
                 ball_max_size_ratio=0.12):
    """Detect cube, ball and bag in one pass for the wrist-independent base view.

    All returned values are dicts from `normalize_box`, so callers get identical
    keys whether or not each object was found.

    Defences against false balls, each added after a measured failure:

    * `ball_min_sat` / `ball_min_val` raise the blue saturation and value floors
      for the ball only. Measured on a cube-only episode, the pale blue-grey
      floor beyond the table edge cleared the wrist defaults (S>=30), producing
      a "ball" on 281 of 598 frames at the bottom edge of the image.
    * `suppress_ball_in_bag` drops a detection whose centre lies inside the bag
      box, because the paper bag carries dark-blue printing.
    * `ball_roi` searches for the ball in a tighter rectangle than the cube. The
      base view includes a blue chair leg in the top-right corner, which the
      detector locked onto on every frame of episode_0224 while ignoring the real
      ball in the middle of the table.
    * `ball_max_size_ratio` rejects anything wider or taller than this fraction of
      the frame; a real ball is roughly 14 px across.
    """
    height, width = image_bgr.shape[:2]
    cube = detect_color(image_bgr, cube_color, roi=roi, exclude_top_ratio=exclude_top_ratio,
                        morphology_kernel=morphology_kernel)
    ball = None
    if ball_color and ball_color != cube_color:
        search = ball_roi if ball_roi is not None else roi
        ball = detect_color(image_bgr, ball_color, roi=search,
                            exclude_top_ratio=exclude_top_ratio,
                            morphology_kernel=morphology_kernel,
                            min_sat=ball_min_sat, min_val=ball_min_val)
        if ball is not None:
            raw_w = ball["w"] / max(width, 1)
            raw_h = ball["h"] / max(height, 1)
            if max(raw_w, raw_h) > ball_max_size_ratio:
                ball = None
    bag = detect_bag(image_bgr, roi=bag_roi, exclude_top_ratio=bag_exclude_top_ratio)
    scene = {
        "cube": normalize_box(cube, width, height),
        "ball": normalize_box(ball, width, height),
        "bag": normalize_box(bag, width, height),
    }
    if suppress_ball_in_bag:
        scene = drop_ball_inside_bag(scene)
    return scene
