"""Read-only full label validation, with reports written to workspace."""
import json
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lerobot.nero_tools.base14 import Base14Config
from lerobot.nero_tools.env_state_schema import ENVIRONMENT_STATE_BASE14_NAMES,validate_base14

def main():
    source=Path('outputs/nero_episodes')
    labels=Path('D:/NERO/outputs/base14_labels_full_v1')
    config=Base14Config(**json.loads(Path('configs/base14_spread_candidate.json').read_text()))
    expected=config.fingerprint()
    errors=[]
    records=[]
    total=Counter()
    names={ep.name for ep in source.glob('episode_*') if ep.is_dir()}
    present={ep.name for ep in labels.glob('episode_*') if ep.is_dir()}
    if names!=present:
        errors.append(dict(episode='ALL',error=f'folder mismatch missing={names-present} extra={present-names}'))
    for name in sorted(names):
        try:
            stamps=[json.loads(x) for x in (source/name/'timestamps.jsonl').read_text().splitlines() if x.strip()]
            rows=[json.loads(x) for x in (labels/name/'env_state_base14.jsonl').read_text().splitlines() if x.strip()]
            report=json.loads((labels/name/'base14_report.json').read_text())
            assert len(rows)==len(stamps)==report['frames']
            assert report['detector_config_hash']==expected
            assert Base14Config(**report['config']).fingerprint()==expected
            counts={role:Counter() for role in ('cube','bag')}
            run=dict(cube=0,bag=0)
            for i,(row,stamp) in enumerate(zip(rows,stamps,strict=True)):
                assert row['schema']=='base14' and row['frame_index']==stamp['frame_index']==i
                assert row['host_time_s']==stamp['host_time_s']
                assert row['detector_config_hash']==expected
                assert set(row['base14'])==set(ENVIRONMENT_STATE_BASE14_NAMES)
                validate_base14([row['base14'][key] for key in ENVIRONMENT_STATE_BASE14_NAMES])
                for role in counts:
                    visible=int(row['base14'][role+'_visible'])
                    status=row['diagnostics'][role]['status']
                    assert status in ('detected','carried','rejected','missing')
                    assert bool(visible)==(status in ('detected','carried'))
                    counts[role]['visible']+=visible
                    counts[role][status]+=1
                    if i<150:
                        counts[role]['first150_visible']+=visible
                    run[role]=0 if visible else run[role]+1
                    counts[role]['longest_missing_run']=max(counts[role]['longest_missing_run'],run[role])
                    for key in ('jump_rejected','edge_rejected','candidates_rejected'):
                        counts[role][key]+=int(row['diagnostics'][role][key])
            for role in counts:
                for key,val in counts[role].items():
                    if key!='first150_visible':
                        assert report['quality'][role].get(key,0)==val,(role,key)
                    total[role+'_'+key]+=val
            total['frames']+=len(rows)
            records.append(dict(episode=name,frames=len(rows),quality=counts))
        except Exception as exc:
            errors.append(dict(episode=name,error=f'{type(exc).__name__}: {exc}'))
    result=dict(episodes=len(records),config_hash=expected,totals=total,errors=errors,records=records)
    Path('outputs/base14_full_label_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(episodes=len(records),totals=total,errors=errors),indent=2))
    for role in ('cube','bag'):
        print(role,'worst early:',[(r['episode'],r['quality'][role]['first150_visible'],r['quality'][role]['visible']) for r in sorted(records,key=lambda r:r['quality'][role]['first150_visible'])[:12]])
        print(role,'worst overall:',[(r['episode'],r['quality'][role]['visible'],r['quality'][role]['longest_missing_run']) for r in sorted(records,key=lambda r:r['quality'][role]['visible'])[:8]])
    return bool(errors)

if __name__=='__main__':
    raise SystemExit(main())
