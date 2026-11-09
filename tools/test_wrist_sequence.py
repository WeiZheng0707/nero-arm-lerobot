"""Offline continuous-frame comparison; output is review material, not grasp truth."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
from lerobot.nero_tools.wrist_target import WristTargetDetector
from lerobot.nero_tools.color_detector import detect_color


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/nero_episodes'))
    p.add_argument('--output',type=Path,default=Path('outputs/wrist_sequence_review'))
    p.add_argument('--episode',action='append',default=None)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for name in args.episode or ['episode_0000','episode_0004','episode_0007']:
        ep=args.source/name
        rows=[json.loads(line) for line in (ep/'timestamps.jsonl').read_text().splitlines() if line.strip()]
        states=[json.loads(line) for line in (ep/'robot_state.jsonl').read_text().splitlines() if line.strip()]
        if len(rows)!=len(states):
            raise ValueError('State/image row count mismatch')
        output=args.output/name
        output.mkdir(exist_ok=True)
        detector=WristTargetDetector()
        counts=Counter()
        tiles=[]
        with (output/'detections.jsonl').open('w',encoding='utf-8') as handle:
            for i,row in enumerate(rows):
                image=cv2.imread(str(ep/row['cameras']['wrist']['image']))
                if image is None:
                    raise ValueError('Unreadable frame')
                old=detect_color(image,'red',exclude_top_ratio=0,morphology_kernel=3)
                result=detector.update(image,float(row['host_time_s']))
                counts[result['status']]+=1
                counts['old_detected']+=int(old is not None)
                counts['new_candidate']+=int(result['box'] is not None)
                counts['new_only']+=int(old is None and result['box'] is not None)
                handle.write(json.dumps(dict(frame_index=i,host_time_s=row['host_time_s'],
                    gripper_state=states[i]['gripper'],old_detected=old is not None,**result))+chr(10))
                if i%30==0 or i in (150,180,210,240,300,390):
                    overlay=image.copy()
                    box=result['box']
                    if box:
                        x,y,w,h=[box[k] for k in ('x','y','w','h')]
                        color=(0,165,255) if not result['localization_usable'] else (0,0,255)
                        cv2.rectangle(overlay,(x,y),(x+w,y+h),color,2)
                        if result['localization_usable']:
                            cv2.circle(overlay,(int(box['cx']*image.shape[1]),int(box['cy']*image.shape[0])),4,color,-1)
                    cv2.putText(overlay,f'f{i} {result["status"]} old={int(old is not None)}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,0),2)
                    cv2.putText(overlay,'NO calibrated grasp center / NO success verdict',(8,465),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,0,0),1)
                    cv2.imwrite(str(output/f'frame_{i:06d}.jpg'),overlay)
                    if i in (0,90,150,180,210,240,300,390):
                        tiles.append(cv2.resize(overlay,(320,240)))
        cv2.imwrite(str(output/'contact_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in (0,4)]))
        summaries.append(dict(episode=name,frames=len(rows),counts=dict(counts)))
        print(summaries[-1],flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
