"""Offline ACT assessment: episode-balanced samples, padding mask, separate units, coherent env interventions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from lerobot.nero_tools.env_state_schema import validate_base14, ENVIRONMENT_STATE_BASE14_NAMES


def shifted_cube(vector, shift):
    v=np.asarray(vector,dtype=np.float32).copy()
    validate_base14(v)
    if v[4]:
        v[0]=np.clip(v[0]+shift,v[2]/2,1-v[2]/2)
        v[10:12]=v[5:7]-v[:2] if v[9] else 0
    return validate_base14(v)


def metrics(prediction,target,valid):
    if prediction.shape != target.shape or prediction.shape[-1]!=8 or valid.shape!=prediction.shape[:-1]:
        raise ValueError('Shape mismatch')
    if not valid.any() or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Empty or nonfinite evaluation')
    error=np.abs(prediction-target)
    per_dim=error[valid].mean(axis=0)
    return dict(joint_mae_rad=per_dim[:7].tolist(),gripper_mae=float(per_dim[7]),valid_steps=int(valid.sum()),
                per_horizon=[error[:,i][valid[:,i]].mean(axis=0).tolist() if valid[:,i].any() else None for i in range(error.shape[1])])


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples-per-episode',type=int,default=8)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.samples_per_episode<1:
        p.error('samples-per-episode must be positive')
    policy=ACTPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    chunk=policy.config.chunk_size
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta=LeRobotDatasetMetadata(args.repo_id,root=args.root)
    if meta.features['observation.environment_state']['names']!=ENVIRONMENT_STATE_BASE14_NAMES:
        raise ValueError('Dataset is not base14')
    dataset=LeRobotDataset(args.repo_id,root=args.root,video_backend='pyav',delta_timestamps={'action':[i/meta.fps for i in range(chunk)]})
    uses_env='observation.environment_state' in policy.config.input_features
    if uses_env and getattr(policy.config,'environment_state_schema',None)!='base14':
        raise ValueError('Checkpoint schema is not base14')
    expected={'observation.state','observation.images.wrist'}|({'observation.environment_state'} if uses_env else set())
    if set(policy.config.input_features)!=expected:
        raise ValueError('Expected wrist/state A or wrist/state/base14 B')
    indices=[]
    for lo,hi in zip(dataset.episode_data_index['from'],dataset.episode_data_index['to']):
        indices.extend(np.unique(np.linspace(int(lo),int(hi)-1,min(args.samples_per_episode,int(hi-lo)),dtype=int)).tolist())
    envs=[dataset[i]['observation.environment_state'].numpy() for i in indices]
    donors=[i for i,e in enumerate(envs) if e[4] and e[9]]
    if uses_env and len(donors)<2:
        raise ValueError('Insufficient visible env samples for interventions')
    predictions={name:[] for name in (('correct','hold','zero','swap','left','right') if uses_env else ('correct','hold'))}
    targets, masks, rows=[],[],[]
    for n,index in enumerate(indices):
        sample=dataset[index]
        batch={key:sample[key].unsqueeze(0).to(args.device) for key in policy.config.input_features}
        def predict(b):
            with torch.no_grad():
                return policy.predict_action_chunk(b)[0].cpu().numpy()
        predictions['correct'].append(predict(batch))
        predictions['hold'].append(np.repeat(sample['observation.state'].numpy()[None,:],chunk,axis=0))
        targets.append(sample['action'].numpy())
        masks.append(~sample['action_is_pad'].numpy().astype(bool))
        donor=None
        if uses_env:
            candidates=[i for i in donors if i!=n]
            donor=max(candidates,key=lambda i:float(np.linalg.norm(envs[i][:2]-envs[n][:2])))
            variants=dict(zero=np.zeros(14,np.float32),swap=envs[donor],left=shifted_cube(envs[n],-.08),right=shifted_cube(envs[n],.08))
            for name,env in variants.items():
                changed=dict(batch)
                changed['observation.environment_state']=torch.from_numpy(env).unsqueeze(0).to(args.device)
                predictions[name].append(predict(changed))
        rows.append(dict(index=index,episode_index=int(sample['episode_index']),
                         cube_x=float(envs[n][0]),cube_visible=bool(envs[n][4]),
                         donor_index=indices[donor] if donor is not None else None))
    target,valid=np.stack(targets),np.stack(masks)
    report=dict(checkpoint=str(args.checkpoint),samples=rows,metrics={name:metrics(np.stack(values),target,valid) for name,values in predictions.items()},
                interpretation='Intervention errors use original targets: sensitivity/association only, not correct actions for counterfactual scenes. No automatic deployment approval.')
    report['prediction_changes']={name:metrics(np.stack(values),np.stack(predictions['correct']),valid) for name,values in predictions.items() if name not in ('correct','hold')}
    groups={f'episode_{episode}':[i for i,row in enumerate(rows) if row['episode_index']==episode]
            for episode in sorted({row['episode_index'] for row in rows})}
    groups.update({f'cube_x_bin_{bin_index}':[i for i,row in enumerate(rows)
                   if row['cube_visible'] and min(2,int(row['cube_x']*3))==bin_index] for bin_index in range(3)})
    report['groups']={group:{name:metrics(np.stack(values)[selection],target[selection],valid[selection])
                     for name,values in predictions.items()} for group,selection in groups.items() if selection}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(args.output)


if __name__=='__main__':
    main()
