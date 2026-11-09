"""Gripper-event stage proxies for offline sampling; not ground-truth task labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

STAGES = ('approach', 'grasp', 'transport', 'release')
CHUNK = 50


def first_stable(mask, start=0, dwell=5):
    for i in range(start, len(mask) - dwell + 1):
        if mask[i:i+dwell].all():
            return i
    raise ValueError('missing stable gripper event')


def stage_windows(gripper):
    g = np.asarray(gripper, dtype=float)
    if g.ndim != 1 or len(g) < 200 or not np.isfinite(g).all():
        raise ValueError('invalid/short gripper trajectory')
    opened = first_stable(g >= .6)
    closed = first_stable(g <= .3, opened + 5)
    released = first_stable(g >= .6, closed + 5)
    # Find the last open position before closure, and closed position before release.
    close_start = int(np.flatnonzero(g[opened:closed] >= .6)[-1] + opened)
    release_start = int(np.flatnonzero(g[closed:released] <= .3)[-1] + closed)
    candidates = {}
    candidates['approach'] = list(range(opened, close_start - CHUNK + 1))
    candidates['transport'] = list(range(closed + 5, release_start - CHUNK + 1))
    for name, lo, hi in [('grasp', close_start, closed), ('release', release_start, released)]:
        midpoint = (lo + hi) // 2
        starts = range(max(0, midpoint-35), min(len(g)-CHUNK, midpoint-15)+1)
        candidates[name] = [i for i in starts if
                            (g[i] - g[i+CHUNK-1] if name == 'grasp' else g[i+CHUNK-1]-g[i]) >= .25]
    if any(not candidates[s] for s in STAGES):
        raise ValueError('missing full stage window or gripper transition >= 0.25')
    events = {'opened': opened, 'close_start': close_start, 'closed': closed,
              'release_start': release_start, 'released': released}
    return events, {s: candidates[s] for s in STAGES}


def audit_split(root):
    episodes, rejected, actions_all = [], [], []
    offset = 0
    paths = sorted((root / 'data').glob('*/*.parquet'))
    for path in paths:
        table = pq.read_table(path, columns=['action', 'episode_index'])
        actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        episode = int(table['episode_index'][0].as_py())
        actions_all.append(actions)
        try:
            events, windows = stage_windows(actions[:, 7])
            episodes.append({'episode': episode, 'offset': offset, 'frames': len(actions),
                             'events': events, 'windows': windows})
        except ValueError as exc:
            rejected.append({'episode': episode, 'reason': str(exc)})
        offset += len(actions)
    if not actions_all:
        raise ValueError('No episode parquet files')
    actions = np.concatenate(actions_all)
    return {'episode_count': len(paths), 'frames': offset, 'eligible': episodes, 'rejected': rejected,
            'candidate_counts': {s: sum(len(e['windows'][s]) for e in episodes) for s in STAGES},
            'action_std': actions.std(axis=0).clip(1e-6).tolist()}


def fixed_examples(audit, count):
    eligible = audit['eligible']
    if len(eligible) < count:
        raise ValueError('Too few eligible episodes')
    selected = []
    for j in range(count):
        e = eligible[j * len(eligible) // count]
        for stage in STAGES:
            window = e['windows'][stage]
            frame = window[len(window)//2]
            selected.append({'episode': e['episode'], 'stage': stage, 'frame': frame,
                             'dataset_index': e['offset'] + frame})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('D:/1104/nero_cube_300_pi05'))
    parser.add_argument('--output', type=Path, default=Path('outputs/pi05_stage_audit_20261105.json'))
    args = parser.parse_args()
    audit = {split: audit_split(args.dataset / split) for split in ('train', 'val')}
    audit['note'] = 'Ordered open-close-open heuristic; phase names are proxies, not task success labels.'
    args.output.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    for split in ('train', 'val'):
        a = audit[split]
        print(split, json.dumps({'total': a['episode_count'], 'eligible': len(a['eligible']),
                                'rejected': a['rejected'], 'candidate_counts': a['candidate_counts']}))
    print('AUDIT='+str(args.output))


if __name__ == '__main__':
    main()
