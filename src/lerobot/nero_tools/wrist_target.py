"""Wrist color target candidates. No fingertip, grasp-center or success inference."""
import math
import cv2
import numpy as np
from .color_detector import _mask_for_color


class WristTargetDetector:
    def __init__(self, color='red', min_area=100, max_jump=.18, max_gap_s=.2):
        self.color, self.min_area = color, min_area
        self.max_jump, self.max_gap_s = max_jump, max_gap_s
        self.previous = None
        self.last_seen = None
        self.last_time = None

    def update(self, image, timestamp):
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError('Expected increasing finite timestamps')
        self.last_time = timestamp
        if self.last_seen is not None and timestamp-self.last_seen > self.max_gap_s:
            self.previous = None
        height,width = image.shape[:2]
        mask = _mask_for_color(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),self.color)
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            x,y,w,h=cv2.boundingRect(contour)
            fill=area/(w*h)
            solidity=area/max(1,cv2.contourArea(cv2.convexHull(contour)))
            clipped=x<=1 or y<=1 or x+w>=width-1 or y+h>=height-1
            # A close object can occupy most of the frame or be partly outside it.
            if area<self.min_area or fill<.3 or solidity<.6:
                continue
            if not clipped and not .35<=w/h<=2.8:
                continue
            cx,cy=(x+w/2)/width,(y+h/2)/height
            distance=(math.hypot(cx-self.previous['cx'],(cy-self.previous['cy'])*height/width)
                      if self.previous is not None else 0)
            candidates.append(dict(x=x,y=y,w=w,h=h,cx=cx,cy=cy,area=float(area),
                                   clipped=clipped,distance=distance,score=area*fill*solidity))
        if self.previous is not None:
            eligible=[c for c in candidates if c['distance']<=self.max_jump]
            eligible.sort(key=lambda c:(c['distance'],-c['score']))
        else:
            eligible=sorted(candidates,key=lambda c:c['score'],reverse=True)
        if not eligible:
            return dict(status='jump_rejected' if candidates else 'missing',box=None,
                        candidate_count=len(candidates),localization_usable=False)
        box=eligible[0]
        ambiguous=len(eligible)>1 and (abs(eligible[1]['distance']-box['distance'])<.03
                                      if self.previous is not None else eligible[1]['score']>.6*box['score'])
        if ambiguous:
            return dict(status='ambiguous',box=box,candidate_count=len(candidates),localization_usable=False)
        self.previous=dict(box)
        self.last_seen=timestamp
        return dict(status='clipped' if box['clipped'] else 'candidate',box=box,
                    candidate_count=len(candidates),localization_usable=not box['clipped'])
