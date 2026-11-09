"""Freeze offline stage x image-space bag-position sampling, without robot access."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import pyarrow.parquet as pq
from pi05_stage_sampling import STAGES, audit_split

def bag_groups(root):
    result = {}
    for path in sorted((root/'data').glob('*/*.parquet')):
        t = pq.read_table(path, columns=['episode_index', 'observation.state'])
        s = np.asarray(t['observation.state'].to_pylist(), np.float32)
        xy = np.median(s[:, 13:15], axis=0)
        # Fixed bins, no fitting to validation labels. Proxy only, not physical positions.
        key = ','.join(str(int(v)) for v in np.floor(xy/.1+.5))
        result[int(t['episode_index'][0].as_py())] = {'position_group': key, 'bag_xy': xy.tolist()}
    return result

def balanced_schedule(audit, groups, steps, seed=42):
    rng = random.Random(seed)
    pools = defaultdict(list)
    for e in audit['eligible']:
        pools[groups[e['episode']]['position_group']].append(e)
    keys = sorted(pools)
    if not keys:
        raise ValueError('No eligible position groups')
    cycles = {}
    for stage in STAGES:
        for key in keys:
            episodes = list(pools[key]); rng.shuffle(episodes)
            cycles[stage, key] = episodes
    seen = Counter(); group_orders = {}; result = []
    for i in range(steps):
        if i % 4 == 0:
            order = list(STAGES); rng.shuffle(order)
        stage = order[i % 4]; n = i//4
        if n % len(keys) == 0 or stage not in group_orders:
            group_orders[stage] = list(keys); rng.shuffle(group_orders[stage])
        key = group_orders[stage][n % len(keys)]
        episodes = cycles[stage, key]; j = seen[stage, key]
        e = episodes[j % len(episodes)]; seen[stage, key] += 1
        frame = rng.choice(e['windows'][stage])
        result.append({'episode': e['episode'], 'stage': stage, 'frame': frame,
                       'dataset_index': e['offset']+frame, **groups[e['episode']]})
    return result

def position_examples(audit, groups, episodes_per_group=1):
    pools = defaultdict(list)
    for e in audit['eligible']:
        pools[groups[e['episode']]['position_group']].append(e)
    result = []
    for key in sorted(pools):
        episodes = pools[key]
        for j in range(min(episodes_per_group, len(episodes))):
            e = episodes[j*len(episodes)//min(episodes_per_group,len(episodes))]
            for stage in STAGES:
                w = e['windows'][stage]; frame = w[len(w)//2]
                result.append({'episode': e['episode'], 'stage': stage, 'frame': frame,
                               'dataset_index': e['offset']+frame, **groups[e['episode']]})
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('D:/1104/nero_cube_300_pi05'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=400)
    a = p.parse_args()
    audits = {s: audit_split(a.dataset/s) for s in ('train','val')}
    groups = {s: bag_groups(a.dataset/s) for s in audits}
    selected = {'train_probe': position_examples(audits['train'], groups['train']),
                'validation_stages': position_examples(audits['val'], groups['val'])}
    # Independent unfiltered panel, including stage-rejected episodes.
    unfiltered = []; offset = 0
    paths = sorted((a.dataset/'val/data').glob('*/*.parquet'))
    for j,path in enumerate(paths):
        t = pq.read_table(path, columns=['episode_index']); ep = int(t['episode_index'][0].as_py())
        if j in {k*len(paths)//16 for k in range(16)}:
            frame = int((len(t)-50)*(.1,.35,.6,.85)[len(unfiltered)%4])
            unfiltered.append({'episode': ep, 'stage':'unfiltered','frame':frame,
                               'dataset_index': offset+frame, **groups['val'][ep]})
        offset += len(t)
    selected['validation_unfiltered'] = unfiltered
    schedule = balanced_schedule(audits['train'],groups['train'],a.steps)
    result = {'schema':1,'seed':42,'steps':a.steps,'dataset':str(a.dataset.resolve()),
              'position_note':'Fixed 0.1 image-coordinate grid of episode-median bag xy. Proxy, not physical placement or success labels. Rare cells oversampled; report support.',
              'training':schedule,'evaluation':selected,
              'support':{s:dict(Counter(groups[s][e['episode']]['position_group'] for e in audits[s]['eligible'])) for s in audits},
              'stage_position_counts':dict(Counter(x['stage']+'|'+x['position_group'] for x in schedule)),
              'groups':groups}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'manifest':str(a.output),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
                      'support':result['support'],'evaluation_counts':{k:len(v) for k,v in selected.items()},
                      'stage_position_counts':result['stage_position_counts']},indent=2))

if __name__=='__main__':main()
