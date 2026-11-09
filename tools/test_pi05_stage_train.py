import json

import numpy as np
import pytest

from pi05_lora_stage_train import check_disjoint_sources, summarize_predictions, training_schedule
from pi05_stage_sampling import STAGES


def test_balanced_schedule_spans_episodes_deterministically():
    audit = {'eligible': [{'episode': e, 'offset': e*600,
                          'windows': {s: [100, 110] for s in STAGES}} for e in range(202)]}
    schedule = training_schedule(audit, 800)
    assert schedule == training_schedule(audit, 800)
    assert len(schedule) == 800
    assert len({x['episode'] for x in schedule}) == 202
    for i in range(0, 800, 4):
        assert {x['stage'] for x in schedule[i:i+4]} == set(STAGES)
    for s in STAGES:
        rows = [x for x in schedule if x['stage'] == s]
        assert len(rows) == 200
        assert len({x['episode'] for x in rows}) == 200
    assert all(x['dataset_index'] == x['episode']*600+x['frame'] for x in schedule)


def test_metric_reports_raw_violation_and_transition_direction():
    target = np.zeros((2, 50, 8))
    target[0, :, 7] = np.linspace(1, 0, 50)
    target[1, :, 7] = np.linspace(0, 1, 50)
    predictions = np.stack([target.copy(), target.copy()])
    predictions[:, :, :, 7] += .1
    summary = summarize_predictions(predictions, target, np.ones((2, 50), bool), np.ones(8))
    assert summary['mean_errors']['50']['joint_MAE_deg'] == 0
    assert np.isclose(summary['mean_errors']['50']['gripper_MAE_0_1'], .1)
    assert summary['gripper_transition_direction_accuracy'] == 1
    assert summary['gripper_transition_samples'] == 2
    assert np.isclose(summary['gripper_net_change_MAE'], 0)
    assert summary['gripper_mean_bound_violation'] > 0


def test_original_split_identities_disjoint(tmp_path):
    def manifest(train_id, val_id):
        return {'splits': {s: {'episodes': [{'source_dataset': 'same_source', 'source_episode': e}]}
                           for s, e in [('train', train_id), ('val', val_id)]}}
    p = tmp_path / 'conversion_manifest.json'
    p.write_text(json.dumps(manifest('ep_01', 'ep_02')), encoding='utf-8')
    assert check_disjoint_sources(tmp_path) == {'train': 1, 'val': 1}
    p.write_text(json.dumps(manifest('ep_01', 'ep_01')), encoding='utf-8')
    with pytest.raises(RuntimeError, match='share original episode'):
        check_disjoint_sources(tmp_path)
