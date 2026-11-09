import numpy as np
import pytest

from pi05_stage_sampling import CHUNK, STAGES, first_stable, fixed_examples, stage_windows


def example():
    return np.r_[np.zeros(80), np.linspace(0, .85, 20), np.full(120, .85),
                 np.linspace(.85, .1, 30), np.full(170, .1),
                 np.linspace(.1, .9, 30), np.full(50, .9), np.linspace(.9, 0, 30), np.zeros(70)]


def test_windows_cover_transition_and_exclude_reset():
    g = example()
    events, windows = stage_windows(g)
    assert 90 <= events['opened'] <= 100
    assert 230 <= events['closed'] <= 250
    assert 420 <= events['released'] <= 450
    for s in STAGES:
        assert windows[s]
        assert all(0 <= i and i + CHUNK <= len(g) for i in windows[s])
    assert all(g[i]-g[i+49] >= .25 for i in windows['grasp'])
    assert all(g[i+49]-g[i] >= .25 for i in windows['release'])
    assert max(windows['release']) < 450
    assert max(windows['approach'])+49 < events['close_start']
    assert min(windows['transport']) >= events['closed']+5
    assert max(windows['transport'])+49 < events['release_start']


def test_short_spike_not_a_stable_open():
    g = example()
    g[15:18] = .9
    assert stage_windows(g)[0]['opened'] > 80
    assert first_stable(np.array([False, True, True, False, True, True, True]), dwell=3) == 4


@pytest.mark.parametrize('g', [np.zeros(600), np.ones(600), np.zeros(100),
                              np.full(600, np.nan), example()[:400]])
def test_missing_events_rejected(g):
    with pytest.raises(ValueError):
        stage_windows(g)


def test_too_short_transport_rejected():
    g = np.r_[np.zeros(50), np.ones(100), np.zeros(15), np.ones(100)]
    with pytest.raises(ValueError):
        stage_windows(g)


def test_fixed_samples_use_own_split_offsets():
    def audit(offset):
        return {'eligible': [{'episode': i, 'offset': offset+i*600,
                              'windows': {s: [100, 101, 102] for s in STAGES}} for i in range(8)]}
    a, b = fixed_examples(audit(0), 4), fixed_examples(audit(10000), 4)
    assert len(a) == 16
    assert len({(x['episode'], x['stage']) for x in a}) == 16
    assert [x['dataset_index']+10000 for x in a] == [x['dataset_index'] for x in b]
