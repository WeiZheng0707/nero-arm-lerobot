"""Verify a user-copied project root against the pre-transfer inventory."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--project-root',type=Path,required=True)
    args=p.parse_args()
    root=args.project_root.resolve()
    records=json.loads(args.inventory.read_text(encoding='utf-8'))
    errors=[]
    for row in records:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Inventory path escapes destination')
        if not path.is_file():
            errors.append(f'MISSING {row["path"]}')
        elif path.stat().st_size!=row['size']:
            errors.append(f'SIZE {row["path"]}')
        elif row['sha256'] and hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            errors.append(f'HASH {row["path"]}')
    print(json.dumps(dict(files=len(records),errors=len(errors),first_errors=errors[:20],
                         note='Image paths/sizes verified, not image byte hashes.'),indent=2))
    return 1 if errors else 0

if __name__=='__main__':
    raise SystemExit(main())
