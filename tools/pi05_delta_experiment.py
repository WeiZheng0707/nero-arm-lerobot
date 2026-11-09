"""Isolated joint-delta ablation. Never use its checkpoints with the normal loader.

Targets are future absolute joint commands minus measured joints at the window
start, not frame-to-frame differences. Gripper commands remain absolute.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from pi05_train_preflight import build_config  # offline setup before policy imports
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

CONTRACT = {'schema': 1, 'representation': 'future_joint_command_minus_window_start_measured_joint',
            'joint_indices': list(range(7)), 'gripper_index': 7, 'gripper': 'absolute',
            'restore': 'add_window_start_measured_joint_once_after_unnormalizing',
            'external_relative_processor': False, 'deployment': 'OFFLINE_EXPERIMENT_ONLY'}
CHECKPOINT_TYPE = 'pi05_delta_experimental_requires_explicit_loader'

def relative_targets(actions, state):
    if actions.ndim != 3 or actions.shape[-1] != 8 or state.ndim != 2 or state.shape[-1] != 22:
        raise ValueError('Expected actions BxHx8 and raw state Bx22')
    if actions.shape[0] != state.shape[0]:
        raise ValueError('Batch dimensions differ')
    result = actions.clone()
    result[..., :7] -= state[:, None, :7]
    return result

def restore_absolute(relative, state):
    result = relative.clone()
    result[..., :7] += state[:, None, :7]
    return result

def compute_delta_stats(train_root, horizon=50):
    train_root = Path(train_root)
    if train_root.name != 'train':
        raise ValueError('Statistics must be fitted on the train split only')
    original = json.loads((train_root/'meta/quantile_stats.json').read_text(encoding='utf-8'))
    paths = sorted((train_root/'data').glob('*/*.parquet'))
    if not paths or horizon < 1:
        raise ValueError('No train parquet files or invalid horizon')
    episodes = []
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode()); digest.update(hashlib.sha256(path.read_bytes()).digest())
        t = pq.read_table(path, columns=['observation.state','action'])
        s = np.asarray(t['observation.state'].to_pylist(), np.float32)
        a = np.asarray(t['action'].to_pylist(), np.float32)
        if s.shape != (len(t),22) or a.shape != (len(t),8) or not np.isfinite(s).all() or not np.isfinite(a).all():
            raise ValueError('Invalid training rows')
        if len(t) >= horizon:
            episodes.append((s,a))
    if not episodes:
        raise ValueError('No complete action windows')
    lows=[]; highs=[]; minimum=[]; maximum=[]
    count=sum((len(a)-horizon+1)*horizon for s,a in episodes)
    for j in range(7):
        values=np.concatenate([(np.lib.stride_tricks.sliding_window_view(a[:,j],horizon)
                               -s[:len(a)-horizon+1,j,None]).reshape(-1) for s,a in episodes])
        lo,hi=np.quantile(values,[.01,.99]);lows.append(float(lo));highs.append(float(max(hi,lo+1e-6)))
        minimum.append(float(values.min()));maximum.append(float(values.max()))
    # Preserve absolute gripper scaling from the existing train-only statistics.
    grip=original['action']; lows.append(grip['q01'][7]);highs.append(grip['q99'][7])
    result={'observation.state':original['observation.state'],
            'action':{'q01':lows,'q99':highs,'joint_min':minimum,'joint_max':maximum},
            'experimental_delta_action':CONTRACT,
            'fit':{'split':'train','root':str(train_root.resolve()),'horizon':horizon,
                   'source_frames':sum(len(a) for s,a in episodes),'episodes':len(episodes),
                   'complete_windows':count//horizon,'joint_target_pairs':count,
                   'parquet_sha256':digest.hexdigest(),
                   'weighting':'all complete window starts x all horizon steps, uniform',
                   'gripper_scaling':'unchanged original train absolute q01/q99'}}
    return result

class ExperimentalDeltaPI05Policy(PI05Policy):
    def __init__(self, config, **kwargs):
        stats=json.loads(Path(config.quantile_stats_path).read_text(encoding='utf-8'))
        if stats.get('experimental_delta_action') != CONTRACT or stats.get('fit',{}).get('split') != 'train':
            raise ValueError('Missing verified experimental delta-action contract')
        if config.use_relative_actions:
            raise ValueError('External relative processor would risk double conversion')
        super().__init__(config, **kwargs)

    def prepare_action(self, batch):
        converted=dict(batch)
        converted['action']=relative_targets(batch['action'].float(),batch['observation.state'].float())
        return super().prepare_action(converted)

    @torch.no_grad()
    def predict_relative_chunk(self, batch, **kwargs):
        return super().predict_action_chunk(batch, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        relative=self.predict_relative_chunk(batch, **kwargs)
        return restore_absolute(relative,batch['observation.state'].to(relative.device,dtype=relative.dtype))

    def _save_pretrained(self, save_directory):
        super()._save_pretrained(save_directory)
        p=Path(save_directory)/'config.json'; payload=json.loads(p.read_text())
        # Fail closed in the ordinary PI05 loader rather than silently returning
        # residuals as absolute robot targets. The explicit loader decodes below.
        payload['type']=CHECKPOINT_TYPE
        p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        (Path(save_directory)/'delta.json').write_text(json.dumps(CONTRACT,indent=2),encoding='utf-8')

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
        if config is None:
            import draccus
            from lerobot.policies.pi05.configuration_pi05 import PI05Config
            folder=Path(pretrained_name_or_path)
            payload=json.loads((folder/'config.json').read_text())
            if payload.pop('type',None) != CHECKPOINT_TYPE:
                raise ValueError('Use an explicit delta configuration to initialize from the base')
            if json.loads((folder/'delta.json').read_text()) != CONTRACT:
                raise ValueError('Checkpoint delta contract mismatch')
            config=draccus.decode(PI05Config,payload)
            config.quantile_stats_path=str((folder/'quantile_stats.json').resolve())
        return super().from_pretrained(pretrained_name_or_path,config=config,**kwargs)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-root',type=Path,default=Path('D:/1104/nero_cube_300_pi05/train'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    stats=compute_delta_stats(a.train_root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print('DELTA_TRAIN_STATS_PASS',json.dumps(stats['fit']),flush=True)

if __name__=='__main__': main()
