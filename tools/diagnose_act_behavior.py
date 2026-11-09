"""Diagnose whether an ACT checkpoint actually uses its inputs.

Low training loss on this dataset is not evidence of task learning: consecutive
frames 33 ms apart have almost identical actions, so "predict the current state"
already scores well. This tool answers the questions that matter instead.

  1. How does the policy compare with two trivial baselines?
       - global-mean action (ignores the observation entirely)
       - persistence (action := current state)
     If the policy is no better than persistence, it is copying rather than seeing.

  2. What happens to the prediction when one input is destroyed?
       zero the wrist image / zero the base image / zero the env-state vector.
     A large change means the input matters. A change near zero means the policy
     ignores it, which is exactly the failure where the arm moves somewhere
     plausible but never toward the object.

Read-only: no robot, no cameras.

Usage:
    python -X utf8 tools/diagnose_act_behavior.py ^
        --checkpoint outputs/train/nero_red_cube_act_v1/checkpoints/050000/pretrained_model ^
        --root outputs/red_cube_300_video/val --repo-id local/nero_red_cube_bag_val ^
        --stride 120 --samples 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


def to_batch(sample, device):
    """Observations only; the target action chunk is returned separately.

    `action` must not be forwarded: ACT.forward runs its VAE branch whenever the
    key is present, and a single frame is not an action chunk.
    """
    batch = {}
    target = None
    for key, value in sample.items():
        if not torch.is_tensor(value):
            continue
        if key == "action":
            target = value.unsqueeze(0).to(device)
        elif key.startswith("observation."):
            batch[key] = value.unsqueeze(0).to(device)
    return batch, target


def predict(policy, batch):
    with torch.no_grad():
        return policy.predict_action_chunk(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/nero_red_cube_bag_val")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    chunk = policy.config.chunk_size

    # Future actions are required: step 0 of a chunk is almost the current joint
    # state, so judging the policy (or an ablation) on step 0 alone says nothing
    # about whether it models the task.
    dataset = LeRobotDataset(
        args.repo_id, root=args.root, video_backend=args.video_backend,
        delta_timestamps={"action": [i / args.fps for i in range(chunk)]},
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"dataset={dataset.num_episodes} episodes, chunk={chunk}, "
          f"using {args.samples} frames\n", flush=True)

    index_list = list(range(0, len(dataset), args.stride))[:args.samples]
    policy_chunks = []
    truth_chunks = []
    state_rows = []
    deltas = {"wrist": [], "base": [], "env_state": []}

    for count, index in enumerate(index_list, start=1):
        sample = dataset[index]
        batch, target = to_batch(sample, device)
        if target is None or target.shape[1] != chunk:
            continue
        base_pred = predict(policy, batch)
        policy_chunks.append(base_pred.squeeze(0).cpu().numpy())
        truth_chunks.append(target.squeeze(0).cpu().numpy())
        state_rows.append(batch["observation.state"].squeeze(0).cpu().numpy())

        for name, key in (("wrist", "observation.images.wrist"),
                          ("base", "observation.images.base"),
                          ("env_state", "observation.environment_state")):
            ablated = dict(batch)
            ablated[key] = torch.zeros_like(batch[key])
            change = (predict(policy, ablated) - base_pred).abs().mean().item()
            deltas[name].append(float(change))
        if count % 10 == 0:
            print(f"  processed {count}/{len(index_list)}", flush=True)

    preds = np.stack(policy_chunks)          # (N, chunk, 8)
    truth = np.stack(truth_chunks)           # (N, chunk, 8)
    states = np.stack(state_rows)            # (N, 8)

    policy_mae = float(np.abs(preds - truth).mean())
    # Trivial baseline 1: hold the current pose for the whole chunk.
    persistence = np.repeat(states[:, None, :], chunk, axis=1)
    persistence_mae = float(np.abs(persistence - truth).mean())
    # Trivial baseline 2: emit the dataset's mean action chunk.
    mean_chunk = truth.mean(axis=0, keepdims=True)
    mean_mae = float(np.abs(mean_chunk - truth).mean())
    scale = float(np.abs(truth).mean())

    print("\n=== error over the WHOLE chunk (lower is better)")
    print(f"  policy                        {policy_mae:.4f}")
    print(f"  constant: hold current pose   {persistence_mae:.4f}")
    print(f"  constant: mean action chunk   {mean_mae:.4f}")
    print(f"  (mean |action| = {scale:.4f} rad for scale)")

    print("\n=== per-step error (step index into the 1s chunk)")
    print(f"  {'step':>5s} {'policy':>9s} {'hold':>9s}")
    for step in (0, 5, 10, 20, chunk - 1):
        policy_step = float(np.abs(preds[:, step] - truth[:, step]).mean())
        hold_step = float(np.abs(persistence[:, step] - truth[:, step]).mean())
        print(f"  {step:>5d} {policy_step:>9.4f} {hold_step:>9.4f}")

    if policy_mae >= persistence_mae * 0.98:
        print("\n  >> NOT better than holding the pose. The model learned "
              "'action := current state', so the arm makes small moves and never "
              "heads for the object. This is a data/model problem, not a deployment "
              "problem.")
    elif policy_mae >= persistence_mae * 0.8:
        print("\n  >> Only slightly better than holding the pose. Late-chunk steps "
              "decide whether this is real: if 'hold' wins at step 20-29, the model "
              "has not learned the task dynamics.")
    else:
        print("\n  >> Clearly better than holding the pose; it models the trajectory.")

    print("\n=== input ablation over the whole chunk")
    print("    mean |change| of the prediction; near zero means the input is ignored")
    for name, values in deltas.items():
        value = float(np.mean(values))
        share = 100.0 * value / max(scale, 1e-9)
        verdict = "IGNORED" if share < 5.0 else "used"
        print(f"  zero {name:10s} {value:.4f}  ({share:5.1f}% of action scale)  {verdict}")

    print("\nHow to read this:")
    print("  * 'hold' winning at late steps -> the policy never learned to react.")
    print("  * a camera ablation near zero -> that view carried no usable signal in")
    print("    training; re-encoding video or adding cameras will not fix it.")
    print("  * env_state near zero but the cameras used -> the detector labels were")
    print("    too noisy to learn from, so the policy fell back to images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
