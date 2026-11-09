"""Generate separate base14 labels and reviews; never convert or move data."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cv2
from lerobot.nero_tools.base14 import Base14Config, Base14Detector
from lerobot.nero_tools.env_state_schema import ENVIRONMENT_STATE_BASE14_NAMES


def label_episode(episode, destination, config, review_every=30):
    rows = [json.loads(line) for line in (episode/'timestamps.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty episode')
    destination.mkdir(parents=True, exist_ok=True)
    output = destination/'env_state_base14.jsonl'
    if output.exists():
        raise FileExistsError(output)
    review = destination/'review_base14'
    if review_every > 0:
        review.mkdir(exist_ok=True)
    detector = Base14Detector(config)
    counts = {role: Counter() for role in ('cube', 'bag')}
    missing_run = dict(cube=0, bag=0)
    temporary = output.with_suffix('.jsonl.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            for index, row in enumerate(rows):
                if row.get('frame_index') != index:
                    raise ValueError('Non-contiguous source frame indices')
                path = (episode / row['cameras']['base']['image']).resolve()
                if not path.is_relative_to(episode.resolve()):
                    raise ValueError('Image path escapes episode')
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f'Unreadable base frame: {path}')
                scene, vector, diagnostic = detector.update(image, float(row['host_time_s']))
                label = dict(frame_index=index, host_time_s=row['host_time_s'], schema='base14',
                             detector_config_hash=config.fingerprint(), base14=dict(zip(ENVIRONMENT_STATE_BASE14_NAMES, vector.tolist())),
                             diagnostics=diagnostic, **scene)
                handle.write(json.dumps(label, allow_nan=False)+'\n')
                for role in counts:
                    d = diagnostic[role]
                    counts[role][d['status']] += 1
                    counts[role]['visible'] += int(bool(scene[role].get('visible')))
                    for key in ('jump_rejected', 'candidates_rejected', 'edge_rejected'):
                        counts[role][key] += int(d[key])
                    missing_run[role] = 0 if scene[role].get('visible') else missing_run[role]+1
                    counts[role]['longest_missing_run'] = max(counts[role]['longest_missing_run'], missing_run[role])
                if review_every > 0 and index % review_every == 0:
                    if not cv2.imwrite(str(review/f'frame_{index:06d}.jpg'), detector.overlay(image, scene, diagnostic)):
                        raise OSError('Could not write review image')
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    report = dict(schema='base14', frames=len(rows), source=str(episode.resolve()),
                  config=asdict(config), detector_config_hash=config.fingerprint(), quality=counts,
                  note='Visibility includes short carries. False positives require human review.')
    (destination/'base14_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Separate label root; source images remain untouched')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--max-episodes', type=int, default=10)
    parser.add_argument('--episode', action='append', help='Exact episode name; repeatable')
    parser.add_argument('--review-every', type=int, default=30)
    args = parser.parse_args()
    if args.max_episodes < 1 or args.review_every < 0:
        parser.error('Invalid count')
    config = Base14Config(**json.loads(args.config.read_text(encoding='utf-8')))
    episodes = sorted(args.source.glob('episode_*'))
    if args.episode:
        requested=set(args.episode)
        if requested-{ep.name for ep in episodes}:
            parser.error('Requested episode missing')
        episodes=[ep for ep in episodes if ep.name in requested]
    else:
        episodes=episodes[:args.max_episodes]
    if not episodes:
        parser.error('No episodes found')
    for episode in episodes:
        report = label_episode(episode, args.output/episode.name, config, args.review_every)
        print(episode.name, report['quality'], flush=True)


if __name__ == '__main__':
    main()
