r"""Fit bag position -> transport joint angles, and report how well it explains them.

The ACT policy reliably approaches, grasps and lifts the cube, but it ignores the
bag fields in `observation.environment_state` (measured: moving the bag across the
frame changes the prediction by 0.07%), so it always drops at the bag's average
spot even though the operator's joint_1 correlates 0.85 with the bag's x.

This fits a small linear model per joint:

    joint_k(transport) ~ a_k * bag_x + b_k * bag_y + c_k

A high R^2 means the transport pose is almost a function of the bag position, so a
scripted place phase can replace the network for that segment without retraining.
A low R^2 means the operator's transport motion is not explained by the bag alone
and a simple correction will not be enough.

Read-only. Writes the fitted coefficients to JSON for the deployment script.

Usage:
    python -X utf8 tools\fit_bag_to_joint.py --split-root D:\1104\nero_cube_300\train --out D:\1104\outputs\bag_regression.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

JOINT_NAMES = [f"joint_{i}.pos" for i in range(1, 8)] + ["gripper.pos"]


def episode_arrays(path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["observation.environment_state", "observation.state",
                                         "action"])
    env = np.asarray([np.asarray(v, dtype=np.float32)
                      for v in table.column("observation.environment_state").to_pylist()])
    state = np.asarray([np.asarray(v, dtype=np.float32)
                        for v in table.column("observation.state").to_pylist()])
    action = np.asarray([np.asarray(v, dtype=np.float32)
                         for v in table.column("action").to_pylist()])
    return env, state, action


def release_index(action: np.ndarray) -> int | None:
    """Frame where the commanded gripper opens to drop the cube.

    A fixed frame window compares different phases across episodes: at frame 300
    one demo is still lifting while another is already at the bag. Aligning on the
    release moment makes the joint averages comparable.

    The gripper closes on the cube (low values) and opens to release (high), so the
    release is the last sustained rise above 0.5 that follows a closed period.
    """
    grip = action[:, -1]
    above = grip > 0.5
    # Walk backwards to the last transition from closed to open.
    for index in range(len(grip) - 1, 10, -1):
        if not above[index] and above[index - 1]:
            return index
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=Path(r"D:\1104\nero_cube_300\train"))
    parser.add_argument("--transport-window", default="300,460",
                        help="Frame range used as the transport phase when --window=fixed")
    parser.add_argument("--window", choices=("auto", "fixed"), default="auto",
                        help="auto aligns on the release moment (gripper opening) so all "
                             "episodes are compared at the same phase; fixed uses "
                             "--transport-window for every episode")
    parser.add_argument("--release-lookback", type=int, default=45,
                        help="With --window=auto, average this many frames before release")
    parser.add_argument("--out", type=Path, default=Path(r"D:\1104\outputs\bag_regression.json"))
    args = parser.parse_args()

    start, end = (int(v) for v in args.transport_window.split(","))
    files = sorted((args.split_root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.split_root}")

    rows, targets = [], []
    used_window = args.transport_window
    for path in files:
        env, state, action = episode_arrays(path)
        # Bag presence is encoded by a non-zero box; there is no bag_visible column.
        if float((env[:, 7] > 0.005).mean()) < 0.5:
            continue
        if args.window == "auto":
            release = release_index(action)
            if release is None or release <= args.release_lookback:
                continue
            window = state[release - args.release_lookback:release]
            used_window = f"{args.release_lookback} frames before release"
        else:
            window = state[start:end]
        if len(window) < 20:
            continue
        bag_cx = float(np.median(env[:, 5] + env[:, 7] / 2.0))
        bag_cy = float(np.median(env[:, 6] + env[:, 8] / 2.0))
        rows.append([bag_cx, bag_cy])
        targets.append(window.mean(axis=0))

    if not rows:
        raise SystemExit("no usable episodes")

    X = np.asarray(rows, dtype=np.float64)          # (N, 2)
    Y = np.asarray(targets, dtype=np.float64)       # (N, 8)
    design = np.hstack([X, np.ones((len(X), 1))])   # (N, 3)
    coef, *_ = np.linalg.lstsq(design, Y, rcond=None)   # (3, 8)

    predicted = design @ coef
    ss_res = ((Y - predicted) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    # R^2 alone is misleading here: a joint that barely moves gets a low R^2 while
    # still being predicted accurately. What matters for a scripted target is the
    # absolute residual in degrees.
    rmse = np.sqrt(((Y - predicted) ** 2).mean(axis=0))
    spread = Y.std(axis=0)

    print(f"episodes used: {len(X)}")
    print(f"bag x range: {X[:,0].min():.3f}..{X[:,0].max():.3f}   "
          f"bag y range: {X[:,1].min():.3f}..{X[:,1].max():.3f}")
    print(f"transport window: {used_window}\n")
    print(f"{'joint':16s} {'a':>9s} {'b':>9s} {'c':>9s} {'R^2':>7s} "
          f"{'rmse deg':>9s} {'spread deg':>11s}")
    for i, name in enumerate(JOINT_NAMES):
        a, b, c = coef[0, i], coef[1, i], coef[2, i]
        print(f"{name:16s} {a:9.3f} {b:9.3f} {c:9.3f} {r2[i]:7.3f} "
              f"{np.degrees(rmse[i]):9.2f} {np.degrees(spread[i]):11.2f}")

    payload = {
        "window": [start, end],
        "episodes": int(len(X)),
        "bag_x_mean": float(X[:, 0].mean()),
        "bag_y_mean": float(X[:, 1].mean()),
        "coefficients": {name: [float(coef[0, i]), float(coef[1, i]), float(coef[2, i])]
                         for i, name in enumerate(JOINT_NAMES)},
        "r2": {name: float(r2[i]) for i, name in enumerate(JOINT_NAMES)},
        "rmse_deg": {name: float(np.degrees(rmse[i])) for i, name in enumerate(JOINT_NAMES)},
        "spread_deg": {name: float(np.degrees(spread[i])) for i, name in enumerate(JOINT_NAMES)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\ncoefficients written to {args.out}")

    worst = float(np.degrees(rmse[:7]).max())
    if worst < 5.0:
        print(f"\n>> Worst joint residual is {worst:.1f} deg. The regression predicts the "
              f"transport pose closely enough to script the place phase.")
    elif worst < 10.0:
        print(f"\n>> Worst joint residual is {worst:.1f} deg. Usable as a target with the "
              f"speed limiter on, but expect a few centimetres of placement error.")
    else:
        print(f"\n>> Worst joint residual is {worst:.1f} deg - too large to script the "
              f"whole pose. Correct joint_1 only, or retrain with the bag inside "
              f"observation.state.")
        strong = [name for i, name in enumerate(JOINT_NAMES[:7]) if r2[i] > 0.6]
        if strong:
            print(f">> {', '.join(strong)} remains well explained and can be corrected "
                  f"directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
