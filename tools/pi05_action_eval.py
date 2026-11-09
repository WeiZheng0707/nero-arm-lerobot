"""Offline generated-action errors and controlled condition sensitivity probes."""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

from pi05_lora_short_train import make_dataset, get_batch
from pi05_train_preflight import build_config

import numpy as np
import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def errors(predictions, targets, valid, std):
    delta = (predictions - targets)[valid]
    return {'joint_MAE_rad': float(np.abs(delta[:, :7]).mean()),
            'joint_MAE_deg': float(np.abs(delta[:, :7]).mean() * 180 / math.pi),
            'joint_RMSE_rad': float(np.sqrt(np.square(delta[:, :7]).mean())),
            'gripper_MAE_0_1': float(np.abs(delta[:, 7]).mean()),
            'per_dim_MAE': np.abs(delta).mean(axis=0).tolist(),
            'mean_abs_error_over_train_std': float((np.abs(delta) / std).mean()),
            'valid_actions': int(valid.sum()),
            'gripper_outside_0_1_fraction': float(((predictions[..., 7][valid] < 0) | (predictions[..., 7][valid] > 1)).mean())}


def sensitivity(reference, altered, std, valid):
    difference = (altered - reference) / std
    return {'first_action_mean_abs_delta_over_train_std': float(np.abs(difference[:, 0]).mean()),
            'first_action_per_dim_abs_delta_over_train_std': np.abs(difference[:, 0]).mean(axis=0).tolist(),
            'valid_chunk_mean_abs_delta_over_train_std': float(np.abs(difference)[valid].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-report', type=Path, default=Path('outputs/train/pi05_lora_smoke_20261105_152737/report.json'))
    parser.add_argument('--dataset', type=Path, default=Path('D:/1104/nero_cube_300_pi05'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    training = json.loads(args.training_report.read_text())
    output = args.output or args.training_report.parent / ('action_eval_' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'hardware_access': False, 'models': {},
              'sample_indices': training['validation_indices'], 'noise_seeds': [31415, 27182],
              'inference_steps': 10, 'horizons': [1, 25, 50],
              'note': 'Fixed offline validation subset; sensitivity does not measure task success or correct response direction.'}
    path = output / 'report.json'
    def write():
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    write()
    try:
        checkpoint = Path(training['checkpoint'])
        config_json = json.loads((checkpoint / 'config.json').read_text())
        model_path = Path(config_json['base_model_path'])
        tokenizer = Path(config_json['tokenizer_name_or_path'])
        config, info = build_config(model_path, args.dataset / 'train', tokenizer, 'bfloat16')
        dataset = make_dataset(args.dataset / 'val', 'val', info['fps'])
        # Compute scale from train actions only; do not fit anything to validation.
        import pyarrow.parquet as pq
        train_actions = np.concatenate([np.asarray(pq.read_table(p, columns=['action'])['action'].to_pylist(), dtype=np.float32)
                                        for p in sorted((args.dataset / 'train/data').glob('*/*.parquet'))])
        std = train_actions.std(axis=0).clip(1e-6)
        report['train_action_std'] = std.tolist()
        names = info['features']['observation.state']['names']
        bag_index = names.index('bag_x')
        cube_index = names.index('cube_x')
        q = json.loads((checkpoint / 'quantile_stats.json').read_text())['observation.state']
        report['perturbation'] = {'x_shift_raw_normalized_image_units': 0.1,
                                  'clamped_to_train_state_q01_q99': True,
                                  'out_of_train_quantile_range': 'leave original coordinate unchanged',
                                  'probe_samples': 8, 'probe_seed': 31415,
                                  'camera_black': 'raw image zeroed; camera present mask retained',
                                  'state_probe': 'only named x coordinate changes; images and other state stay fixed'}
        batches = [get_batch(dataset, i) for i in report['sample_indices']]
        targets = np.stack([b['action'][0].cpu().numpy() for b in batches])
        valid = np.stack([~b['action_is_pad'][0].cpu().numpy().astype(bool) for b in batches])
        hold = np.stack([np.repeat(b['observation.state'][0, :8].cpu().numpy()[None], 50, axis=0) for b in batches])
        report['hold_current_state_baseline'] = {str(h): errors(hold[:, :h], targets[:, :h], valid[:, :h], std) for h in (1, 25, 50)}
        arrays = {'targets': targets, 'valid': valid, 'hold_current_state': hold}
        for label, location in [('pretrained', model_path), ('lora_100', checkpoint)]:
            torch.manual_seed(42)
            policy = PI05Policy.from_pretrained(location, config=config if label == 'pretrained' else None,
                                                 strict=True, local_files_only=True)
            policy.eval()
            torch.cuda.empty_cache()
            results = []
            def predict(batch, sample, seed):
                # Explicit noise, no target action or padding mask passed to inference.
                generator = torch.Generator(device='cuda').manual_seed(seed + sample)
                noise = torch.randn((1, 50, 32), device='cuda', generator=generator)
                observation = {k: v for k, v in batch.items() if k in
                               ('observation.state', 'observation.images.base', 'observation.images.wrist', 'task')}
                output_actions = policy.predict_action_chunk(observation, noise=noise, num_steps=10)
                if not torch.isfinite(output_actions).all():
                    raise RuntimeError('Nonfinite generated actions')
                return output_actions[0].cpu().numpy()
            with torch.no_grad():
                for seed in report['noise_seeds']:
                    generated = []
                    for i, batch in enumerate(batches):
                        generated.append(predict(batch, i, seed))
                        if (i + 1) % 4 == 0:
                            print(f'{label} seed={seed}: {i+1}/{len(batches)} action chunks', flush=True)
                    results.append(np.stack(generated))
                reference = results[0][:8]
                repeat = predict(batches[0], 0, report['noise_seeds'][0])
                repeat_diff = float(np.abs(repeat-reference[0]).max())
                if repeat_diff > 1e-6:
                    raise RuntimeError('Repeated inference with identical conditions differs')
                probes = {}
                for condition in ('base_black', 'wrist_black', 'bag_x_minus', 'bag_x_plus', 'cube_x_minus', 'cube_x_plus'):
                    modified = []
                    raw_deltas = []
                    for i, batch in enumerate(batches[:8]):
                        changed = dict(batch)
                        if condition.endswith('_black'):
                            key = 'observation.images.' + condition.split('_')[0]
                            changed[key] = torch.zeros_like(batch[key])
                        else:
                            index = bag_index if condition.startswith('bag') else cube_index
                            changed['observation.state'] = batch['observation.state'].clone()
                            old = batch['observation.state'][0, index].item()
                            new = (min(q['q99'][index], max(q['q01'][index],
                                   old + (-0.1 if condition.endswith('minus') else 0.1)))
                                   if q['q01'][index] <= old <= q['q99'][index] else old)
                            changed['observation.state'][0, index] = new
                            raw_deltas.append(new-old)
                        modified.append(predict(changed, i, report['noise_seeds'][0]))
                    altered = np.stack(modified)
                    probes[condition] = sensitivity(reference, altered, std, valid[:8])
                    if raw_deltas:
                        probes[condition]['actual_raw_x_deltas'] = raw_deltas
                    arrays[label + '_' + condition] = altered
                    print(label + ' PROBE ' + condition + ' ' + json.dumps(probes[condition]), flush=True)
            per_seed = []
            for seed, generated in zip(report['noise_seeds'], results, strict=True):
                per_seed.append({'seed': seed, 'errors': {str(h): errors(generated[:, :h], targets[:, :h], valid[:, :h], std) for h in (1, 25, 50)}})
                arrays[label + '_seed_' + str(seed)] = generated
            report['models'][label] = {'per_seed': per_seed, 'sensitivity': probes, 'repeat_max_abs_delta': repeat_diff}
            write()
            np.savez_compressed(output / 'trajectories.npz', **arrays)
            del policy
            gc.collect()
            torch.cuda.empty_cache()
        report['status'] = 'PASS'
        write()
        print(f'PI05_OFFLINE_ACTION_EVAL_PASS report={path}', flush=True)
    except BaseException as exc:
        report.update({'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'})
        write()
        raise


if __name__ == '__main__':
    main()
