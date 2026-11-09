"""Verify converted robot rows against local raw captures; no robot access."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image,ImageDraw

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('D:/1104/nero_cube_300_pi05'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.dataset/'conversion_manifest.json').read_text(encoding='utf-8'))
    names=[f'joint_{i}' for i in range(1,8)]+['gripper']
    result={'hardware_access':False,'physical_zero_direction_calibration':'UNVERIFIED: offline source consistency does not verify physical calibration',
        'source_resolution':'Use workspace outputs/nero_episodes by source episode key; verify all numeric rows, not path name alone.',
        'capture_semantics':'episode_capture.py reads follower observation before sending mapped command, then saves observation/command with the same host time. Command-feedback lag is expected and not by itself label misalignment.',
        'episodes':[]};panels=[]
    for split in ('train','val'):
        entries=manifest['splits'][split]['episodes']
        paths={int(p.stem.split('_')[-1]):p for p in (a.dataset/split/'data').glob('*/*.parquet')}
        for e in entries:
            raw=Path('outputs/nero_episodes')/e['source_episode']
            s=rows(raw/'robot_state.jsonl');c=rows(raw/'actions.jsonl');f=rows(raw/'timestamps.jsonl')
            if not len(s)==len(c)==len(f):raise RuntimeError('Raw row counts differ')
            rs=np.asarray([[r.get(n,r.get(n+'.pos')) for n in names] for r in s],np.float32)
            rc=np.asarray([[r[n+'.pos'] for n in names] for r in c],np.float32)
            ts=np.array([r['host_time_s'] for r in f]);relative=ts-ts[0]
            t=pq.read_table(paths[e['episode_index']],columns=['observation.state','action'])
            ps=np.asarray(t['observation.state'].to_pylist(),np.float32);pc=np.asarray(t['action'].to_pylist(),np.float32)
            grid=np.arange(len(t))/30;right=np.clip(np.searchsorted(relative,grid),0,len(ts)-1);left=np.maximum(right-1,0)
            ix=np.where(abs(relative[left]-grid)<=abs(relative[right]-grid),left,right)
            sd=float(abs(ps[:,:8]-rs[ix]).max());cd=float(abs(pc-rc[ix]).max())
            if max(sd,cd)>1e-6:raise RuntimeError(f'Source mismatch {split} {e["episode_index"]} {sd} {cd}')
            lags=[float(abs(pc[:-lag,7]-ps[lag:,7]).mean()) if lag else float(abs(pc[:,7]-ps[:,7]).mean()) for lag in range(16)]
            row={'split':split,'episode':e['episode_index'],'source_episode':e['source_episode'],'frames':len(t),
                'state_max_abs_diff':sd,'action_max_abs_diff':cd,
                'first_action_vs_state_per_joint_deg':(abs(pc[0,:7]-ps[0,:7])*180/np.pi).tolist(),
                'gripper_command_vs_future_feedback_MAE_by_lag_0_to_15':lags,
                'gripper_best_lag_frames':int(np.argmin(lags)),
                'raw_row_timestamp_max_error_s':float(max(max(abs(r['host_time_s']-ts[j]) for j,r in enumerate(s)),max(abs(r['host_time_s']-ts[j]) for j,r in enumerate(c)))),
                'historical_mapping':{k:e['source_metadata'].get(k) for k in ('leader_channel','follower_channel','leader_gripper_width_m','follower_gripper_width_m')}}
            result['episodes'].append(row)
            if split=='val' and e['episode_index'] in (0,10,20,30,40,50):
                panel=Image.new('RGB',(640,270),'white');draw=ImageDraw.Draw(panel)
                for j,cam in enumerate(('base','wrist')):
                    image=Image.open(raw/f[0]['cameras'][cam]['image']).convert('RGB');image.thumbnail((320,240));panel.paste(image,(320*j,25))
                draw.text((4,5),f'{split} ep{e["episode_index"]} {e["source_episode"]} first frame: base / wrist',fill='black');panels.append(panel)
        print('RAW_SPLIT_PASS',split,len(entries),flush=True)
    result['summary']={'episodes':len(result['episodes']),'frames':sum(x['frames'] for x in result['episodes']),
        'all_source_robot_values_match':True,
        'first_action_state_joint_MAE_deg':np.mean([x['first_action_vs_state_per_joint_deg'] for x in result['episodes']],axis=0).tolist(),
        'best_gripper_lag_episode_counts':dict(Counter(x['gripper_best_lag_frames'] for x in result['episodes'])),
        'caution':'Lag selection is descriptive over 0..15 frames and can be affected by long stationary periods. No target shifting applied.'}
    result['status']='PASS'
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(640,270*len(panels)),'white')
    for j,panel in enumerate(panels):sheet.paste(panel,(0,j*270))
    sheet.save(a.output/'first_frames.png');print(json.dumps(result['summary']),flush=True)

if __name__=='__main__':main()
