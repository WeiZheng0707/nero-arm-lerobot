r"""Gate checks for the pi0.5 + LoRA pipeline.

The previous handoff trained for 800 steps on four samples and read the resulting
2.65 deg as progress. It never established whether the pipeline can fit a single
sample at all, so the 2.65 deg is ambiguous: it could mean "not enough capacity"
or "the plumbing is wrong". These gates separate the two before any long run.

    G0  single-sample memorisation: can the model drive ONE action chunk below
        --g0-target-deg (default 0.5 deg)? If not, stop; training longer will not
        fix a wiring bug.
    G1  action scale sanity: fraction of generated actions outside the training
        q01/q99 range and outside [0, 1] for the gripper. A healthy pipeline keeps
        these small; the earlier run had 74% of gripper outputs below zero.
    G2  determinism: identical noise seed must give bit-identical actions.

Unlike the earlier diagnostic this script derives everything it needs from the
dataset and the base model: no dependency on a previous run's report, no hard
coded validation indices, no borrowed action std.

Run from the repository root with the pi05 venv:

    "D:\NERO\envs\pi05_lora\Scripts\python.exe" -u tools\pi05_gate_checks.py ^
        --dataset D:\1104\nero_cube_300_pi05 --gates 0,1,2

Nothing here touches robot hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from pi05_lora_short_train import get_batch, make_dataset
from pi05_train_preflight import build_config

from lerobot.policies.pi05.modeling_pi05 import PI05Policy

FRAME_HORIZON = 50


# --------------------------------------------------------------------------- #
# Norms derived from the dataset itself, never from an older run
# --------------------------------------------------------------------------- #
def compute_train_action_std(root: Path, episodes: int = 12) -> np.ndarray:
    actions = []
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))[:episodes]
    for path in files:
        table = pq.read_table(path, columns=["action"])
        actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
    stacked = np.concatenate(actions, axis=0)
    return stacked.std(axis=0)


def load_quantiles(root: Path):
    stats = json.loads((root / "meta" / "quantile_stats.json").read_text(encoding="utf-8"))
    action = stats["action"]
    return np.asarray(action["q01"], dtype=np.float32), np.asarray(action["q99"],
                                                                   dtype=np.float32)


def pick_moving_samples(dataset, root: Path, count: int) -> list[int]:
    """Indices whose next 50 actions differ most from the current state.

    A high-motion window exercises the whole chunk instead of a static tail, so
    G0 is not satisfied by predicting "hold".
    """
    picked = []
    total = len(dataset.episode_data_index["from"])
    for k in range(count):
        episode = (total * k) // count
        path = root / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
        table = pq.read_table(path, columns=["action", "observation.state"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if len(actions) <= FRAME_HORIZON:
            continue
        scores = [np.abs(actions[i:i + FRAME_HORIZON, :7] - states[i, :7]).mean()
                  for i in range(len(actions) - FRAME_HORIZON + 1)]
        local = int(np.argmax(scores))
        base = int(dataset.episode_data_index["from"][episode])
        picked.append(base + local)
    return picked


# --------------------------------------------------------------------------- #
def generate(policy, batch, seed: int, num_steps: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn((1, FRAME_HORIZON, 32), generator=generator, device="cuda")
    observation = {key: batch[key] for key in
                   ("observation.state", "observation.images.base",
                    "observation.images.wrist", "task")}
    with torch.no_grad():
        return policy.predict_action_chunk(observation, noise=noise, num_steps=num_steps)


def action_error_deg(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray,
                     horizon: int) -> float:
    delta = (prediction[:, :horizon] - target[:, :horizon])[valid[:, :horizon]]
    return float(np.abs(delta[:, :7]).mean() * 180 / math.pi)


def targets(batch):
    target = batch["action"][0].cpu().numpy()
    valid = ~batch["action_is_pad"][0].cpu().numpy().astype(bool)
    return target, valid


# --------------------------------------------------------------------------- #
def gate_g0(policy, dataset, indices, args, report):
    samples = [get_batch(dataset, i) for i in indices]
    std = compute_train_action_std(args.dataset / "train")
    print(f"G0: {len(samples)} sample(s), target {args.g0_target_deg} deg on frame 1",
          flush=True)

    trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    if not trainable:
        raise RuntimeError("G0 needs trainable LoRA parameters; none found")
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=args.g0_lr, betas=(0.9, 0.95),
                                  weight_decay=0.0)
    history = []
    policy.train()
    started = time.perf_counter()
    best = float("inf")
    for step in range(1, args.g0_steps + 1):
        batch = samples[(step - 1) % len(samples)]
        lr = args.g0_lr * min(1.0, step / 10)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = policy(batch)[0]
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss in G0")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
        optimizer.step()

        if step % args.g0_eval_every == 0 or step == args.g0_steps:
            policy.eval()
            errors = []
            with torch.no_grad():
                for index, batch in zip(indices, samples):
                    prediction = generate(policy, batch, 10000 + index, args.inference_steps)
                    target, valid = targets(batch)
                    errors.append(action_error_deg(prediction[0].cpu().numpy(), target,
                                                   valid, horizon=1))
            policy.train()
            value = float(np.mean(errors))
            best = min(best, value)
            history.append({"step": step, "loss": loss.item(), "frame1_joint_MAE_deg": value,
                            "elapsed_s": time.perf_counter() - started})
            print(f"  step {step:4d}  loss {loss.item():.4f}  "
                  f"frame1 MAE {value:6.2f} deg", flush=True)
            if value <= args.g0_target_deg:
                print(f"  reached target at step {step}", flush=True)
                break

    passed = best <= args.g0_target_deg
    report["g0"] = {"passed": passed, "best_frame1_joint_MAE_deg": best,
                    "target_deg": args.g0_target_deg, "history": history,
                    "samples": indices}
    print(f"G0 {'PASS' if passed else 'FAIL'}: best {best:.2f} deg", flush=True)
    if not passed:
        print("  A single chunk cannot be memorised, so the pipeline is not learning "
              "the action space. Fix the wiring (padding mask, timestep, "
              "de-normalisation) before any longer run.", flush=True)
    return passed


def gate_g1(policy, dataset, indices, args, report):
    std = compute_train_action_std(args.dataset / "train")
    q01, q99 = load_quantiles(args.dataset / "train")
    policy.eval()
    outside_low = outside_high = total = 0
    gripper_outside = gripper_total = 0
    per_dim_outside = np.zeros(8, dtype=np.int64)
    per_dim_total = np.zeros(8, dtype=np.int64)
    with torch.no_grad():
        for index in indices:
            batch = get_batch(dataset, index)
            prediction = generate(policy, batch, 20000 + index, args.inference_steps)
            array = prediction[0].cpu().numpy()
            target, valid = targets(batch)
            mask = valid
            total += int(mask.sum())
            per_dim_total += mask.sum(axis=0)
            low = array < q01[None, :]
            high = array > q99[None, :]
            outside_low += int((low & mask).sum())
            outside_high += int((high & mask).sum())
            per_dim_outside += ((low | high) & mask).sum(axis=0)
            grip = array[..., 7]
            gripper_total += int(mask.sum())
            gripper_outside += int((((grip < 0) | (grip > 1)) & mask).sum())

    fraction = (outside_low + outside_high) / max(total, 1)
    gripper_fraction = gripper_outside / max(gripper_total, 1)
    passed = fraction <= args.g1_max_outside and gripper_fraction <= args.g1_max_outside
    report["g1"] = {
        "passed": passed,
        "action_outside_q01_q99_fraction": fraction,
        "gripper_outside_0_1_fraction": gripper_fraction,
        "limit": args.g1_max_outside,
        "per_dim_outside_fraction": (per_dim_outside / np.maximum(per_dim_total, 1)).tolist(),
        "action_std_train": std.tolist(),
    }
    print(f"G1 {'PASS' if passed else 'FAIL'}: actions outside q01/q99 "
          f"{100 * fraction:.1f}%, gripper outside [0,1] {100 * gripper_fraction:.1f}% "
          f"(limit {100 * args.g1_max_outside:.1f}%)", flush=True)
    if fraction > args.g1_max_outside:
        print("  The generated actions sit outside the training range the model was "
              "normalised for. That points at the de-normalisation path, not at the "
              "policy's skill.", flush=True)
    return passed


def gate_g2(policy, dataset, indices, args, report):
    policy.eval()
    batch = get_batch(dataset, indices[0])
    with torch.no_grad():
        first = generate(policy, batch, 31415, args.inference_steps)
        second = generate(policy, batch, 31415, args.inference_steps)
    difference = float((first - second).abs().max())
    passed = difference == 0.0
    report["g2"] = {"passed": passed, "max_abs_difference_same_seed": difference}
    print(f"G2 {'PASS' if passed else 'FAIL'}: same-seed max abs difference "
          f"{difference:.3e}", flush=True)
    return passed


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/model"))
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\1104\nero_cube_300_pi05"))
    parser.add_argument("--tokenizer", type=Path, default=Path("outputs/model/tokenizer"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank. The comparison arm uses 16 with all attention "
                             "projections; G0 only needs any trainable set, so the "
                             "previously verified default stays at 8.")
    parser.add_argument("--lora-targets", default="q_proj,v_proj",
                        help="Comma separated module names for LoRA")
    parser.add_argument("--gates", default="0,1,2")
    parser.add_argument("--g0-samples", type=int, default=1)
    parser.add_argument("--g0-steps", type=int, default=400)
    parser.add_argument("--g0-lr", type=float, default=1e-4)
    parser.add_argument("--g0-target-deg", type=float, default=0.5)
    parser.add_argument("--g0-eval-every", type=int, default=25)
    parser.add_argument("--g1-samples", type=int, default=8)
    parser.add_argument("--g1-max-outside", type=float, default=0.05)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    wanted = {int(v) for v in args.gates.split(",") if v.strip()}
    output = args.output or Path("outputs/train") / time.strftime("pi05_gates_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "hardware_access": False, "gates": sorted(wanted),
              "dataset": str(args.dataset), "model": str(args.model)}

    def write():
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    write()
    print(f"OUTPUT={output}", flush=True)
    try:
        config, info = build_config(args.model, args.dataset / "train",
                                    args.tokenizer, args.dtype,
                                    lora_rank=args.lora_rank,
                                    lora_targets=tuple(t.strip() for t in
                                                       args.lora_targets.split(",") if t.strip()))
        train = make_dataset(args.dataset / "train", "train", info["fps"])
        print(f"dataset: {len(train)} frames", flush=True)
        policy = PI05Policy.from_pretrained(args.model, config=config, strict=True,
                                            local_files_only=True)
        indices = pick_moving_samples(train, args.dataset / "train",
                                      max(args.g0_samples, args.g1_samples))
        results = {}
        if 0 in wanted:
            results["g0"] = gate_g0(policy, train, indices[:args.g0_samples], args, report)
            write()
        if 1 in wanted:
            results["g1"] = gate_g1(policy, train, indices[:args.g1_samples], args, report)
            write()
        if 2 in wanted:
            results["g2"] = gate_g2(policy, train, indices[:1], args, report)
            write()
        report["status"] = "PASS" if all(results.values()) else "FAIL"
        report["results"] = results
        write()
        print(f"\nGATES {report['status']}  report={output / 'report.json'}", flush=True)
        print("A FAIL is informative, not a crash: fix the named stage before scaling up.",
              flush=True)
    except BaseException as exc:  # noqa: BLE001 - report any failure into the JSON
        report["status"] = "ERROR"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write()
        raise


if __name__ == "__main__":
    main()
