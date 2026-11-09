"""Offline sample of wrist red-object and legacy dark-region detector; no grasp success inference."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
from lerobot.nero_tools.color_detector import detect_color, detect_gripper

root=Path('outputs/nero_episodes')
out=Path('outputs/wrist_grasp_review')
out.mkdir(parents=True,exist_ok=True)
reports=[]
for name in ('episode_0000','episode_0004','episode_0007'):
    ep=root/name
    rows=[json.loads(x) for x in (ep/'timestamps.jsonl').read_text().splitlines() if x.strip()]
    states=[json.loads(x) for x in (ep/'robot_state.jsonl').read_text().splitlines() if x.strip()]
    tiles=[]
    for index in (0,90,150,180,210,240,300,390):
        image=cv2.imread(str(ep/rows[index]['cameras']['wrist']['image']))
        if image is None:
            raise ValueError('Unreadable frame')
        cube=detect_color(image,'red',exclude_top_ratio=0,morphology_kernel=3)
        dark=detect_gripper(image)
        overlay=image.copy()
        for box,color,label in ((cube,(0,0,255),'red candidate'),(dark,(255,0,255),'dark region NOT calibrated')):
            if box:
                x,y,w,h=[box[k] for k in ('x','y','w','h')]
                cv2.rectangle(overlay,(x,y),(x+w,y+h),color,2)
                cv2.putText(overlay,label,(x,max(16,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        cv2.putText(overlay,f'{name} f{index} grip={states[index]["gripper"]:.3f}',(8,465),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1)
        cv2.imwrite(str(out/f'{name}_{index:04d}.jpg'),overlay)
        tiles.append(cv2.resize(overlay,(320,240)))
        reports.append(dict(episode=name,frame_index=index,gripper_state=states[index]['gripper'],cube=cube,dark_region=dark))
    cv2.imwrite(str(out/f'{name}_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in (0,4)]))
(out/'detections.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
print('24 sampled frames; overlays are detector candidates, not grasp labels.')
