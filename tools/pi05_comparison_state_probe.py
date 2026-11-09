"""Matched-noise state sensitivity of frozen comparison checkpoints, offline only."""
import argparse
import gc
import json
from pathlib import Path
from pi05_train_preflight import build_config
from pi05_lora_short_train import make_dataset,get_batch
import numpy as np
import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['base','reference_r8_lr5e5','low_lr_r8_lr1e5','capacity_r16_lr5e5'])
    p.add_argument('--output',default='state_probe.json')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    root=a.experiment;m=json.loads((root/'manifest.json').read_text());datasetroot=Path(m['dataset'])
    choices=m['evaluation']['validation_stages'];groups=sorted({x['position_group'] for x in choices})
    selected=[x for x in choices if x['position_group'] in (groups[0],groups[-1])]
    if len(selected)!=8 or any(selected[i]['stage']!=selected[(i+4)%8]['stage'] for i in range(8)):
        raise RuntimeError('Expected two positions with matched four-stage pairs')
    ds=make_dataset(datasetroot/'val','val',30);batches=[get_batch(ds,x['dataset_index']) for x in selected]
    report={'hardware_access':False,'samples':selected,'models':{},'noise_seeds':[31415,27182],
            'note':'Swap joints or bag xy with other position episode at same stage; images fixed. Cross-modal inconsistency means sensitivity is not evidence of correct state use.'}
    if a.resume and (root/a.output).exists():
        prior=json.loads((root/a.output).read_text())
        if prior['samples']!=selected or prior['noise_seeds']!=report['noise_seeds']:
            raise RuntimeError('Cannot resume mismatched probe')
        report['models']=prior['models']
    with torch.no_grad():
        for label in a.models:
            if label in report['models']:continue
            if label=='base':
                cfg,_=build_config(Path('outputs/model'),datasetroot/'train',Path('outputs/model/tokenizer'),'bfloat16')
                policy=PI05Policy.from_pretrained('outputs/model',config=cfg,strict=True,local_files_only=True)
            elif label=='delta_base':
                from pi05_delta_experiment import ExperimentalDeltaPI05Policy
                cfg,_=build_config(Path('outputs/model'),datasetroot/'train',Path('outputs/model/tokenizer'),'bfloat16')
                cfg.quantile_stats_path=str((root/'delta_stats.json').resolve())
                policy=ExperimentalDeltaPI05Policy.from_pretrained('outputs/model',config=cfg,strict=True,local_files_only=True)
            elif label.startswith('delta_'):
                from pi05_delta_experiment import ExperimentalDeltaPI05Policy
                policy=ExperimentalDeltaPI05Policy.from_pretrained(root/label/'checkpoint_best',strict=True,local_files_only=True)
            else:policy=PI05Policy.from_pretrained(root/label/'checkpoint_best',strict=True,local_files_only=True)
            policy.eval()
            # Loading transiently constructs base copies. Release these before
            # inference to avoid Windows GPU memory spilling into host RAM.
            gc.collect();torch.cuda.empty_cache()
            print('PROBE_MODEL',label,'allocated_GB',torch.cuda.memory_allocated()/1e9,
                  'reserved_GB',torch.cuda.memory_reserved()/1e9,flush=True)
            effects={'joints':[],'bag_xy':[]};inputdelta={k:[] for k in effects}
            residual_effects={k:[] for k in effects}
            for seed in report['noise_seeds']:
                for i,b in enumerate(batches):
                    obs={k:b[k] for k in ('observation.state','observation.images.base','observation.images.wrist','task')}
                    noise=torch.randn((1,50,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed+i))
                    correct=policy.predict_action_chunk(obs,noise=noise,num_steps=10)
                    for name,sl in [('joints',slice(0,7)),('bag_xy',slice(13,15))]:
                        modified=dict(obs);modified['observation.state']=obs['observation.state'].clone()
                        modified['observation.state'][:,sl]=batches[(i+4)%8]['observation.state'][:,sl]
                        inputdelta[name].append(float((modified['observation.state'][:,sl]-obs['observation.state'][:,sl]).abs().mean()))
                        altered=policy.predict_action_chunk(modified,noise=noise,num_steps=10)
                        effects[name].append((altered-correct).abs().cpu().numpy())
                        residual_delta=(altered-correct).clone()
                        residual_delta[...,:7]-=(modified['observation.state'][:,:7]-obs['observation.state'][:,:7])[:,None,:]
                        residual_effects[name].append(residual_delta.abs().cpu().numpy())
            result={}
            for name,rows in effects.items():
                d=np.concatenate(rows)
                result[name]={'input_mean_abs':float(np.mean(inputdelta[name])),
                              'input_joint_deg':float(np.mean(inputdelta[name])*180/np.pi) if name=='joints' else None,
                              'output_joint_change_deg':float(d[...,:7].mean()*180/np.pi),
                              'output_first_joint_change_deg':float(d[:,0,:7].mean()*180/np.pi),
                              'output_per_joint_change_deg':(d[...,:7].mean((0,1))*180/np.pi).tolist(),
                              'action_minus_start_joint_change_deg':float(np.concatenate(residual_effects[name])[...,:7].mean()*180/np.pi),
                              'output_gripper_change':float(d[...,7].mean())}
            report['models'][label]=result
            (root/a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(label,json.dumps(result),flush=True)
            del policy,correct,altered;gc.collect();torch.cuda.empty_cache()
    report['status']='PASS';(root/a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPARISON_STATE_PROBE_PASS',flush=True)

if __name__=='__main__':main()
