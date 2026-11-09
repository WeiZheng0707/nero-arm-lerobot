"""Validate every raw episode in a capture directory before conversion.

Checks the things that silently poison a dataset: unpaired rows, missing or
mis-sized images, non-monotonic timestamps, non-finite state/action values and
gripper values outside [0, 1]. Read-only; nothing is written.

Usage:
    python -X utf8 tools/validate_capture.py --source outputs/nero_episodes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


NAMES = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_episode(episode, expect_size):
    problems = []
    try:
        frames = read_rows(episode / "timestamps.jsonl")
        states = read_rows(episode / "robot_state.jsonl")
        actions = read_rows(episode / "actions.jsonl")
    except FileNotFoundError as exc:
        return [f"missing file: {exc.filename}"], 0
    if not frames:
        return ["timestamps.jsonl is empty"], 0
    if not len(frames) == len(states) == len(actions):
        problems.append(f"row counts differ: frames={len(frames)} states={len(states)} "
                        f"actions={len(actions)}")

    metadata_path = episode / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        problems.append("metadata.json missing")

    result = metadata.get("result")
    if result not in ("auto_end", "manual_end"):
        problems.append(f"result={result!r} needs review before conversion")

    # Timestamps must advance; equal stamps break the 30 Hz resampling grid.
    stamps = np.asarray([row["host_time_s"] for row in frames], dtype=np.float64)
    if not np.isfinite(stamps).all():
        problems.append("non-finite host_time_s")
    elif len(stamps) > 1 and np.any(np.diff(stamps) <= 0):
        problems.append("host_time_s not strictly increasing")

    try:
        state = np.asarray([[row.get(n, row.get(n + ".pos")) for n in NAMES]
                            for row in states], dtype=np.float64)
        action = np.asarray([[row[n + ".pos"] for n in NAMES] for row in actions],
                            dtype=np.float64)
    except KeyError as exc:
        return problems + [f"missing state/action key: {exc}"], len(frames)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        problems.append("non-finite state/action values")
    else:
        if state[:, -1].min() < 0 or state[:, -1].max() > 1:
            problems.append(f"state gripper out of [0,1]: "
                            f"{state[:, -1].min():.3f}..{state[:, -1].max():.3f}")
        if action[:, -1].min() < 0 or action[:, -1].max() > 1:
            problems.append(f"action gripper out of [0,1]: "
                            f"{action[:, -1].min():.3f}..{action[:, -1].max():.3f}")

    # Image presence and size, per camera.
    cameras = (frames[0].get("cameras") or {})
    if not cameras:
        problems.append("no 'cameras' key: this is the legacy single-image layout")
        return problems, len(frames)
    for name in cameras:
        expected = expect_size.get(name)
        missing = 0
        wrong = 0
        for row in frames:
            entry = (row.get("cameras") or {}).get(name)
            if not entry:
                missing += 1
                continue
            path = episode / entry["image"]
            if not path.exists():
                missing += 1
                continue
            try:
                with Image.open(path) as image:
                    if expected and image.size != expected:
                        wrong += 1
            except Exception:
                problems.append(f"{name}: unreadable image {path.name}")
                break
        if missing:
            problems.append(f"{name}: {missing} image(s) missing")
        if wrong:
            problems.append(f"{name}: {wrong} image(s) not {expected[0]}x{expected[1]}")
    return problems, len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expect-size", default="wrist=640x480,base=640x480",
                        help="Per-camera expected WxH, e.g. wrist=640x480,base=640x480")
    parser.add_argument("--quiet", action="store_true", help="Only print problems")
    args = parser.parse_args()

    expect_size = {}
    for chunk in args.expect_size.split(","):
        if not chunk.strip():
            continue
        name, _, size = chunk.partition("=")
        width, _, height = size.partition("x")
        expect_size[name.strip()] = (int(width), int(height))

    episodes = sorted(path for path in args.source.glob("episode_*") if path.is_dir())
    if not episodes:
        raise SystemExit(f"No episode_* directories under {args.source}")

    bad = {}
    frame_counts = {}
    results = {}
    total_frames = 0
    for episode in episodes:
        problems, count = check_episode(episode, expect_size)
        total_frames += count
        frame_counts[episode.name] = count
        metadata = episode / "metadata.json"
        if metadata.exists():
            label = json.loads(metadata.read_text(encoding="utf-8")).get("result")
            results[label] = results.get(label, 0) + 1
        if problems:
            bad[episode.name] = problems
        elif not args.quiet:
            print(f"OK   {episode.name}: {count} frames", flush=True)

    print(f"\nSUMMARY: episodes={len(episodes)} total_frames={total_frames}")
    print(f"  result labels: {results}")
    counts = sorted(frame_counts.values())
    if counts:
        print(f"  frames/episode: min={counts[0]} median={counts[len(counts) // 2]} "
              f"max={counts[-1]}")
    if not bad:
        print("  structural problems: none")
        return 0
    print(f"  episodes with problems: {len(bad)}")
    for name, problems in list(bad.items())[:20]:
        print(f"  {name}:")
        for problem in problems:
            print(f"    - {problem}")
    if len(bad) > 20:
        print(f"  ... and {len(bad) - 20} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
