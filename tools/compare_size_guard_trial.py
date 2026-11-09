import json
from pathlib import Path
for name in ('episode_0030','episode_0099'):
    paths=[Path('D:/NERO/outputs/base14_labels_full_v1')/name/'env_state_base14.jsonl',
           Path('outputs/base14_size_guard_trial')/name/'env_state_base14.jsonl']
    data=[[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
    summary=dict(episode=name,size_rejected=sum(r['diagnostics']['bag'].get('size_rejected',False) for r in data[1]),
                 changed_frames=sum(a['base14']!=b['base14'] for a,b in zip(*data)))
    print(summary)
    for i in (270,285,300,315,330):
        print(i,[(rows[i]['diagnostics']['bag']['status'],round(rows[i]['base14']['bag_w'],3),round(rows[i]['base14']['bag_h'],3)) for rows in data])
