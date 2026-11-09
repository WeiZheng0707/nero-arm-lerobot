"""Run three bounded single-factor arms sequentially on one frozen manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args()
    root=a.experiment;manifest=root/'manifest.json'
    spec=json.loads(manifest.read_text());steps=spec['steps']
    arms=[('reference_r8_lr5e5',8,5e-5),('low_lr_r8_lr1e5',8,1e-5),('capacity_r16_lr5e5',16,5e-5)]
    state={'status':'RUNNING','hardware_access':False,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
           'steps_per_arm':steps,'training_seed':42,'arms':[],
           'limits':'One training seed, two inference seeds, 400 steps; offline subset, not task success. Rank comparison holds alpha/rank=2. No inference offset patch.'}
    out=root/'comparison.json'
    def write():out.write_text(json.dumps(state,indent=2),encoding='utf-8')
    write();baseline=None
    for name,rank,lr in arms:
        folder=root/name
        cmd=[sys.executable,'-u','tools/pi05_lora_stage_train.py','--steps',str(steps),'--manifest',str(manifest),
             '--lora-rank',str(rank),'--lr',str(lr),'--output',str(folder)]
        print('START',name,flush=True);started=time.perf_counter()
        with (root/(name+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        if child.returncode:
            state.update(status='FAILED',failed_arm=name,returncode=child.returncode);write();raise RuntimeError('Arm failed: '+name)
        r=json.loads((folder/'report.json').read_text())
        if r['status']!='PASS' or r['sampling_manifest_sha256']!=state['manifest_sha256']:
            raise RuntimeError('Arm verification mismatch')
        z=np.load(folder/'trajectories.npz')
        values={k:z[k] for k in z.files if '_step_0_seed_' in k}
        if baseline is None:baseline=values
        elif set(values)!=set(baseline) or any(not np.array_equal(v,baseline[k]) for k,v in values.items()):
            state.update(status='FAILED',failed_arm=name,error='Initial functional baseline differs');write();raise RuntimeError('Initial baseline differs')
        state['arms'].append({'name':name,'rank':rank,'lr':lr,'seconds':time.perf_counter()-started,
            'status':r['status'],'initial_baseline_bitexact':True,'trainable_parameters':r['trainable_parameters'],
            'best_trained_step':r['best_trained_step'],'best_beats_base':r['best_beats_base_selection_score'],
            'evaluations':r['evaluations'],'reload_max_difference':r['reload_action_max_abs_difference'],
            'cuda_peak_allocated_GB':r['cuda_peak_allocated_GB'],'max_rss_GB':r['max_rss_GB']})
        write();print('FINISHED',name,'best',r['best_trained_step'],'beats_base',r['best_beats_base_selection_score'],flush=True)
    state['status']='PASS';write();print('POSITION_COMPARISON_PASS',flush=True)

if __name__=='__main__':main()
