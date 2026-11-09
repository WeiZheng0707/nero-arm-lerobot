from collections import Counter
from copy import deepcopy
from pathlib import Path
import pytest
from pi05_position_manifest import balanced_schedule,position_examples
from pi05_lora_stage_train import validate_manifest
from pi05_stage_sampling import STAGES

def fixture():
    audit={'eligible':[{'episode':e,'offset':e*600,'frames':600,'windows':{s:[80,90,100] for s in STAGES}} for e in range(9)]}
    groups={e:{'position_group':str(e//3),'bag_xy':[.1,.2]} for e in range(9)}
    return audit,groups

def test_schedule_deterministic_and_balances_each_stage_position():
    audit,groups=fixture();rows=balanced_schedule(audit,groups,400)
    assert rows==balanced_schedule(audit,groups,400)
    assert all({x['stage'] for x in rows[i:i+4]}==set(STAGES) for i in range(0,400,4))
    for stage in STAGES:
        counts=Counter(x['position_group'] for x in rows if x['stage']==stage)
        assert max(counts.values())-min(counts.values())<=1
    assert all(x['dataset_index']==x['episode']*600+x['frame'] and x['frame']+50<=600 for x in rows)

def test_manifest_rejects_tampered_offsets_and_nonstage_windows(tmp_path):
    a,g=fixture();samples=position_examples(a,g)
    m={'dataset':str(tmp_path),'training':balanced_schedule(a,g,16),
       'evaluation':{'train_probe':samples,'validation_stages':samples,'validation_unfiltered':[]}}
    validate_manifest(m,{'train':a,'val':a},tmp_path,16)
    bad=deepcopy(m);bad['training'][0]['dataset_index']+=1
    with pytest.raises(RuntimeError,match='candidate/offset'):validate_manifest(bad,{'train':a,'val':a},tmp_path,16)
    bad=deepcopy(m);bad['training'][0]['frame']=580
    with pytest.raises(RuntimeError,match='candidate/offset'):validate_manifest(bad,{'train':a,'val':a},tmp_path,16)
