import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from pi05_delta_experiment import (CONTRACT, CHECKPOINT_TYPE, ExperimentalDeltaPI05Policy,
                                   compute_delta_stats, relative_targets, restore_absolute)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.prompt import normalize_quantiles, unnormalize_quantiles

def test_delta_roundtrip_anchor_at_start_and_gripper_absolute():
    state=torch.randn(2,22); actions=torch.randn(2,50,8)
    before=actions.clone(); delta=relative_targets(actions,state)
    assert torch.equal(actions,before)
    assert torch.equal(delta[...,7],actions[...,7])
    assert torch.allclose(delta[...,0],actions[...,0]-state[:,0,None])
    assert torch.allclose(restore_absolute(delta,state),actions,atol=5e-7)
    q01=torch.full((8,),-2.);q99=torch.full((8,),2.)
    normalized=normalize_quantiles(delta,q01,q99)
    restored=restore_absolute(unnormalize_quantiles(normalized,q01,q99),state)
    assert torch.allclose(restored,actions,atol=8e-7)

def test_stats_use_all_complete_train_windows_and_keep_state_gripper(tmp_path):
    train=tmp_path/'train'; (train/'meta').mkdir(parents=True); (train/'data/chunk-000').mkdir(parents=True)
    original={'observation.state':{'q01':[0]*22,'q99':[1]*22},'action':{'q01':[-3]*8,'q99':[4]*8}}
    (train/'meta/quantile_stats.json').write_text(json.dumps(original))
    state=np.zeros((5,22),np.float32);state[:,0]=np.arange(5)*.5
    actions=np.zeros((5,8),np.float32);actions[:,0]=np.arange(5)*2
    pq.write_table(pa.table({'observation.state':state.tolist(),'action':actions.tolist()}),train/'data/chunk-000/episode_000000.parquet')
    stats=compute_delta_stats(train,horizon=3)
    expected=np.array([[0,2,4],[1.5,3.5,5.5],[3,5,7]])
    assert np.isclose(stats['action']['q01'][0],np.quantile(expected,.01))
    assert np.isclose(stats['action']['q99'][0],np.quantile(expected,.99))
    assert stats['action']['q01'][7] == -3 and stats['action']['q99'][7] == 4
    assert stats['observation.state'] == original['observation.state']
    assert stats['fit']['complete_windows'] == 3 and stats['fit']['joint_target_pairs'] == 9
    with pytest.raises(ValueError,match='train split'):compute_delta_stats(tmp_path/'val')

def test_predict_adds_start_once_without_changing_gripper():
    p=object.__new__(ExperimentalDeltaPI05Policy);torch.nn.Module.__init__(p)
    residual=torch.ones(1,50,8)*.2
    p.predict_relative_chunk=lambda batch,**kwargs:residual
    state=torch.zeros(1,22);state[:,:7]=.4
    result=p.predict_action_chunk({'observation.state':state})
    assert torch.allclose(result[...,:7],torch.full((1,50,7),.6))
    assert torch.equal(result[...,7],residual[...,7])

def test_checkpoint_explicit_loader_and_normal_loader_fail_closed(tmp_path):
    config=PreTrainedConfig.from_pretrained(Path('outputs/model'),local_files_only=True)
    config._save_pretrained(tmp_path)
    payload=json.loads((tmp_path/'config.json').read_text());payload['type']=CHECKPOINT_TYPE
    (tmp_path/'config.json').write_text(json.dumps(payload))
    (tmp_path/'delta.json').write_text(json.dumps(CONTRACT))
    with pytest.raises(Exception):PreTrainedConfig.from_pretrained(tmp_path,local_files_only=True)
    with patch.object(PI05Policy,'from_pretrained',return_value='loaded') as loader:
        assert ExperimentalDeltaPI05Policy.from_pretrained(tmp_path,local_files_only=True) == 'loaded'
        cfg=loader.call_args.kwargs['config']
        assert cfg.quantile_stats_path == str((tmp_path/'quantile_stats.json').resolve())
        assert cfg.use_relative_actions is False
    (tmp_path/'delta.json').write_text('{}')
    with pytest.raises(ValueError,match='contract mismatch'):ExperimentalDeltaPI05Policy.from_pretrained(tmp_path)
