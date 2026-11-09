"""Inventory and package code/reports only. Never moves or copies raw images."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

def main():
    root=Path.cwd()
    source=root/'outputs/nero_episodes'
    out=root/'outputs/base14_handoff'
    out.mkdir(exist_ok=True)
    inventory=[]
    for ep in sorted(source.glob('episode_*')):
        rows=[json.loads(x) for x in (ep/'timestamps.jsonl').read_text().splitlines() if x.strip()]
        for name in ('timestamps.jsonl','robot_state.jsonl','actions.jsonl','metadata.json'):
            path=ep/name
            inventory.append(dict(path=str(path.relative_to(root)),size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for row in rows:
            for cam in ('base','wrist'):
                path=(ep/row['cameras'][cam]['image']).resolve()
                if not path.is_relative_to(ep.resolve()) or not path.is_file():
                    raise ValueError(f'Missing/invalid image {path}')
                inventory.append(dict(path=str(path.relative_to(root)),size=path.stat().st_size,sha256=None))
    (out/'source_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    review=out/'episode_review.csv'
    if not review.exists():
        with review.open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.writer(handle)
            writer.writerow(['episode','grasp','lift','placement','label_quality','notes'])
            for ep in sorted(source.glob('episode_*')):
                writer.writerow([ep.name,'UNREVIEWED','UNREVIEWED','UNREVIEWED','UNREVIEWED',''])
    patterns=['src/lerobot/nero_tools/base14.py','src/lerobot/nero_tools/env_state_schema.py',
        'src/lerobot/nero_tools/convert_episodes.py','src/lerobot/nero_tools/wrist_target.py',
        'src/lerobot/policies/act/configuration_act.py','src/lerobot/policies/factory.py',
        'tools/*base14*.py','tools/*wrist*.py','tools/replay_grasp_pairs.py','tools/nero_dual_cam_act_deploy.py',
        'configs/base14*.json','docs/base14*.md','docs/*grasp*findings.md','tests/test_base14*.py','tests/test_wrist_target.py',
        'outputs/base14_audit_summary.json','outputs/base14_data_audit.json']
    files=sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
    with zipfile.ZipFile(out/'code_and_reports.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,path.relative_to(root))
    result=dict(source=str(source),episodes=len(list(source.glob('episode_*'))),files=len(inventory),
                bytes=sum(x['size'] for x in inventory),image_hashes='not computed; paths and sizes verified',
                dataset_transferred=False,training_ready=False)
    (out/'handoff_status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':
    main()
