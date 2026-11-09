"""Shared causal base-only detector for offline labels and live inference."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import cv2
import numpy as np
from .color_detector import _mask_for_color
from .env_state_schema import environment_state_base14


@dataclass(frozen=True)
class Base14Config:
    base_roi: tuple[int, int, int, int]
    bag_roi: tuple[int, int, int, int] | None = None
    cube_color: str = 'red'
    morphology_kernel: int = 3
    cube_min_area: float = 100
    cube_max_area: float = 6000
    bag_min_area: float = 1200
    bag_max_area: float = 60000
    cube_max_jump: float = 0.10
    bag_max_jump: float = 0.06
    hold_frames: int = 3
    max_age_s: float = 0.34
    bag_min_size_ratio: float = 0.0

    def __post_init__(self):
        if not 0 <= self.bag_min_size_ratio <= 1:
            raise ValueError('Invalid bag size ratio')
        if self.morphology_kernel not in (3, 5, 7):
            raise ValueError('Morphology kernel must be 3, 5 or 7')
        if not 0 <= self.hold_frames <= 10 or not 0 < self.max_age_s <= 1:
            raise ValueError('Invalid bounded carry duration')
        for role in ('cube', 'bag'):
            lo, hi = getattr(self, role + '_min_area'), getattr(self, role + '_max_area')
            jump = getattr(self, role + '_max_jump')
            if not all(math.isfinite(v) for v in (lo, hi, jump)) or not 0 < lo < hi or not 0 < jump <= 1:
                raise ValueError('Invalid area/jump thresholds')
        for roi in (self.base_roi, self.bag_roi):
            if roi is not None and (len(roi) != 4 or any(type(v) is not int for v in roi) or min(roi[:2]) < 0 or min(roi[2:]) <= 0):
                raise ValueError('ROI requires nonnegative x,y and positive width,height')
        if self.base_roi is None:
            raise ValueError('base14 requires an explicit base ROI')

    def fingerprint(self):
        values = asdict(self)
        if self.bag_min_size_ratio == 0:
            values.pop('bag_min_size_ratio')  # Preserve existing disabled-guard configurations.
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class StableBox:
    def __init__(self, max_jump, hold_frames, max_age_s, min_size_ratio=0.0):
        self.max_jump, self.hold_frames, self.max_age_s = max_jump, hold_frames, max_age_s
        self.box = None
        self.age = 0
        self.accepted_time = None
        self.reference = None
        self.min_size_ratio = min_size_ratio
        self.size_rejected = False

    def update(self, candidate, timestamp, aspect):
        expired = self.accepted_time is not None and timestamp - self.accepted_time > self.max_age_s
        if expired:
            self.box = None
            self.reference = None
        self.size_rejected = bool(candidate and self.reference and self.min_size_ratio and
            any(candidate[key] < self.reference[key] * self.min_size_ratio for key in ('w', 'h')))
        rejected = bool(candidate and self.box and math.hypot(
            candidate['cx'] - self.box['cx'], (candidate['cy'] - self.box['cy']) * aspect) > self.max_jump)
        invalid = rejected or self.size_rejected
        if candidate and not invalid:
            self.box, self.age, self.accepted_time = dict(candidate), 0, timestamp
            self.reference = dict(candidate)
            return dict(self.box), 'detected', False
        self.age += 1
        if self.box and self.age <= self.hold_frames and not expired:
            return dict(self.box), 'carried', rejected
        self.box = None
        return {'visible': 0}, 'rejected' if invalid else 'missing', rejected


class Base14Detector:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        c = self.config
        self.trackers = {name: StableBox(getattr(c, name + '_max_jump'), c.hold_frames, c.max_age_s,
                                       c.bag_min_size_ratio if name == 'bag' else 0)
                         for name in ('cube', 'bag')}
        self.last_time = None

    def _candidate(self, image, role):
        c = self.config
        height, width = image.shape[:2]
        x, y, w, h = c.base_roi
        if role == 'bag' and c.bag_roi is not None:
            bx, by, bw, bh = c.bag_roi
            if bx < x or by < y or bx + bw > x + w or by + bh > y + h:
                raise ValueError('Bag ROI must lie within base ROI')
            x, y, w, h = c.bag_roi
        if x + w > width or y + h > height:
            raise ValueError('ROI exceeds image dimensions')
        hsv = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        mask = (_mask_for_color(hsv, c.cube_color) if role == 'cube' else
                cv2.inRange(hsv, np.array((5, 25, 35)), np.array((28, 190, 220))))
        kernel = np.ones((c.morphology_kernel,) * 2, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates, rejected, edge = [], 0, 0
        for contour in contours:
            area = cv2.contourArea(contour)
            px, py, pw, ph = cv2.boundingRect(contour)
            if not getattr(c, role + '_min_area') <= area <= getattr(c, role + '_max_area'):
                rejected += 1
                continue
            if px <= 0 or py <= 0 or px + pw >= w or py + ph >= h:
                rejected += 1
                edge += 1
                continue
            ratio, fill = pw / ph, area / (pw * ph)
            solidity = area / max(1.0, cv2.contourArea(cv2.convexHull(contour)))
            valid = (0.45 <= ratio <= 1.8 and fill >= .45 and solidity >= .75) if role == 'cube' else (pw >= 30 and ph >= 30 and .25 <= ratio <= 4)
            if not valid:
                rejected += 1
                continue
            box = dict(x=(x+px)/width, y=(y+py)/height, w=pw/width, h=ph/height,
                       cx=(x+px+pw/2)/width, cy=(y+py+ph/2)/height, visible=1, area_ratio=area/(width*height))
            candidates.append((area * fill * solidity if role == 'cube' else area, box))
        return (max(candidates, key=lambda item: item[0])[1] if candidates else None), rejected, edge

    def update(self, image_bgr, timestamp):
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError('base14 requires strictly increasing frame timestamps; reset per episode')
        self.last_time = timestamp
        scene, diagnostics = {}, {}
        for role in ('cube', 'bag'):
            candidate, rejected, edge = self._candidate(image_bgr, role)
            box, status, jumped = self.trackers[role].update(candidate, timestamp, image_bgr.shape[0]/image_bgr.shape[1])
            scene[role] = box
            diagnostics[role] = dict(status=status, jump_rejected=jumped, candidates_rejected=rejected, edge_rejected=edge)
            diagnostics[role]['size_rejected'] = self.trackers[role].size_rejected
        return scene, environment_state_base14(scene), diagnostics

    def overlay(self, image, scene, diagnostics):
        image = image.copy()
        h, w = image.shape[:2]
        for roi in (self.config.base_roi, self.config.bag_roi):
            if roi:
                x, y, rw, rh = roi
                cv2.rectangle(image, (x, y), (x+rw, y+rh), (0, 255, 0), 2)
        centers = []
        for index, (role, color) in enumerate((('cube', (0, 0, 255)), ('bag', (0, 255, 255)))):
            box = scene[role]
            if box.get('visible'):
                x, y, bw, bh = [int(box[key] * scale) for key, scale in zip(('x','y','w','h'), (w,h,w,h))]
                center = (int(box['cx']*w), int(box['cy']*h))
                centers.append(center)
                cv2.rectangle(image, (x,y), (x+bw,y+bh), color, 2)
                cv2.circle(image, center, 3, color, -1)
            cv2.putText(image, role + ': ' + diagnostics[role]['status'], (10, 22+index*22), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        if len(centers) == 2:
            cv2.arrowedLine(image, centers[0], centers[1], (0,255,255), 1)
        return image
