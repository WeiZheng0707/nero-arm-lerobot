"""Check that environment_state actually reaches the ACT encoder.

Two training runs, two chunk sizes, two datasets, and the env-state ablation
came out at 0.0001 both times. That is too exact to be "the model learned to
ignore it"; something upstream is flat. This prints the wiring and the tensors
so the cause is visible instead of guessed.

Read-only: builds the policy, runs two forwards on one dataset sample.

Usage:
    python -X utf8 tools/check_env_wiring.py ^
        --checkpoint outputs/train/nero_red_cube_act_v2/checkpoints/030000/pretrained_model ^
        --root outputs/red_cube_300_v2/val --repo-id local/nero_red_cube_bag_v2_val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/nero_red_cube_bag_v2_val")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    config = policy.config

    print("=== config wiring")
    print(f"  env_state_feature        : {config.env_state_feature}")
    print(f"  robot_state_feature      : {config.robot_state_feature}")
    print(f"  image_features           : {list(config.image_features)}")
    print(f"  input_features keys      : {sorted(config.input_features)}")
    print(f"  normalization_mapping    : { {str(k): str(v) for k, v in config.normalization_mapping.items()} }")
    has_proj = hasattr(policy.model, "encoder_env_state_input_proj")
    print(f"  has env state projection : {has_proj}")
    if hasattr(policy.model, "encoder_1d_feature_pos_embed"):
        print(f"  1d pos embed rows        : "
              f"{policy.model.encoder_1d_feature_pos_embed.weight.shape[0]} "
              f"(1 latent + 1 robot + 1 env = 3 expected)")

    print("\n=== dataset sample")
    dataset = LeRobotDataset(args.repo_id, root=args.root, video_backend=args.video_backend)
    sample = dataset[args.index]
    env = sample["observation.environment_state"]
    print(f"  env sample shape : {tuple(env.shape)}")
    print(f"  env sample values: {[round(float(v), 4) for v in env]}")
    print(f"  nonzero dims     : {int((env != 0).sum())} / {env.numel()}")

    # Normalisation is the prime suspect: if the stored stats flatten the vector,
    # the encoder receives a constant no matter what the detector reports.
    if hasattr(policy, "normalize_inputs"):
        module = policy.normalize_inputs
        buffers = dict(module.named_buffers())
        for suffix in ("observation.environment_state.mean",
                       "observation.environment_state.std"):
            key = next((k for k in buffers if k.endswith(suffix)), None)
            if key is not None:
                print(f"  {suffix:38s}: "
                      f"{[round(float(v), 4) for v in buffers[key].flatten()]}")
        batch_norm = {"observation.environment_state": env.unsqueeze(0).to(device)}
        with torch.no_grad():
            normalized = module(batch_norm)["observation.environment_state"]
        print(f"  after normalisation: {[round(float(v), 4) for v in normalized[0]]}")
        print(f"  normalised std across dims: {float(normalized.std()):.5f} "
              f"(near zero means the vector is effectively constant)")

    print("\n=== decoder response to the env vector")
    batch = {"observation.environment_state": env.unsqueeze(0).to(device)}
    for key in ("observation.state",):
        batch[key] = sample[key].unsqueeze(0).to(device)
    for key in config.image_features:
        batch[key] = sample[key].unsqueeze(0).to(device)

    with torch.no_grad():
        base = policy.predict_action_chunk(batch)
        zeroed = dict(batch)
        zeroed["observation.environment_state"] = torch.zeros_like(batch["observation.environment_state"])
        blank = policy.predict_action_chunk(zeroed)
        if has_proj:
            flat = batch["observation.environment_state"]
            projected = policy.model.encoder_env_state_input_proj(flat)
            print(f"  projected env token norm : {float(projected.norm()):.4f}")
            print(f"  projection weight norm   : "
                  f"{float(policy.model.encoder_env_state_input_proj.weight.norm()):.4f}")

    delta = float((base - blank).abs().mean())
    scale = float(base.abs().mean())
    print(f"  |Δ action| when env zeroed: {delta:.6f}  "
          f"({100 * delta / max(scale, 1e-9):.3f}% of action scale)")

    print("\nHow to read this:")
    print("  * env_state_feature None or pos embed rows == 2 -> the token was never")
    print("    added; the checkpoint simply has no env pathway.")
    print("  * normalised std ~ 0 -> the stats flatten the vector, so every frame")
    print("    looks identical to the encoder and it is correctly ignored.")
    print("  * projection weight norm ~ 0 -> the layer was trained into silence.")
    print("  * everything healthy but |Δ| ~ 0 -> the encoder attends elsewhere;")
    print("    the fix is feature design (relative goal vector), not more training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
