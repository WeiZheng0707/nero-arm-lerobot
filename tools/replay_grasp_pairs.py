"""Create offline paired review MP4s and closure-event sheets. No success classifier."""
import argparse
import json
from pathlib import Path
import av
import cv2
import numpy as np


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def closure_events(openings, window=6, threshold=.05, separation=30):
    """Falling normalized command opening; events are not contact evidence."""
    events=[]
    for i in range(window,len(openings)):
        if openings[i-window]-openings[i]>=threshold and (not events or i-events[-1]>=separation):
            events.append(i)
    return events


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/nero_episodes'))
    p.add_argument('--output',type=Path,default=Path('outputs/grasp_pair_review'))
    p.add_argument('--episode',action='append')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    summary=[]
    for name in args.episode or ('episode_0000','episode_0004','episode_0007'):
        ep=args.source/name
        times,states,actions=[rows(ep/f) for f in ('timestamps.jsonl','robot_state.jsonl','actions.jsonl')]
        if not len(times)==len(states)==len(actions):
            raise ValueError('Row mismatch')
        commands=[a['gripper.pos'] for a in actions]
        events=closure_events(commands)
        chosen=sorted({min(len(times)-1,max(0,i+d)) for i in events for d in (-30,0,30,60)})
        video=args.output/f'{name}.mp4'
        if video.exists():
            raise FileExistsError(video)
        sheets={}
        with av.open(str(video),'w') as container:
            stream=container.add_stream('libx264',rate=30,options={'crf':'20','preset':'fast'})
            stream.width,stream.height,stream.pix_fmt=1280,580,'yuv420p'
            for i,row in enumerate(times):
                images=[cv2.imread(str(ep/row['cameras'][cam]['image'])) for cam in ('base','wrist')]
                if any(image is None for image in images):
                    raise ValueError('Unreadable frame')
                canvas=np.zeros((580,1280,3),np.uint8)
                canvas[:480]=np.hstack(images)
                elapsed=row['host_time_s']-times[0]['host_time_s']
                text=f'{name} frame={i} source t={elapsed:.3f}s  pair gap={row["pair_error_ms"]:.1f}ms'
                cv2.putText(canvas,text,(10,505),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
                cv2.putText(canvas,f'BASE (left) / WRIST (right)   grip observed={states[i]["gripper"]:.3f} command={commands[i]:.3f}',(10,535),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),1)
                cv2.putText(canvas,'Row-aligned at 30fps; source time shown. Closing is NOT proof of grasp.',(10,565),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                for packet in stream.encode(av.VideoFrame.from_ndarray(canvas,format='bgr24')):
                    container.mux(packet)
                if i in chosen:
                    sheets[i]=cv2.resize(canvas,(640,290))
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(video)) as container:
            count=sum(1 for _ in container.decode(video=0))
        if count!=len(times):
            raise ValueError('Video frame count mismatch')
        for event in events:
            ids=[min(len(times)-1,max(0,event+d)) for d in (-30,0,30,60)]
            cv2.imwrite(str(args.output/f'{name}_close_{event:04d}.jpg'),np.vstack([np.hstack([sheets[i] for i in ids[:2]]),np.hstack([sheets[i] for i in ids[2:]])]))
        summary.append(dict(episode=name,frames=count,closure_candidates=events,
                            source_duration_s=times[-1]['host_time_s']-times[0]['host_time_s'],
                            video_duration_s=count/30,success='UNREVIEWED'))
        print(summary[-1],flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
