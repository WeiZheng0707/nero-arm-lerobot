"""Offline bounded LoRA training, fixed validation and fresh-model reload check.

Run from the repository root. Does not access robot hardware or publish files.
This smoke run checks the training pipeline, not task success or generalization.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import time
from pathlib import Path

# Import first: configures offline mode, short cache paths and local src imports.
from pi05_train_preflight import build_config

import psutil
import torch
from torch.utils.data import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.lora import LoRALinear
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def make_dataset(root, split, fps):
    return LeRobotDataset(
        f'local/nero_cube_pi05_{split}', root=root, video_backend='pyav',
        delta_timestamps={'action': [i / fps for i in range(50)]},
    )


def get_batch(dataset, index):
    batch = default_collate([dataset[index]])
    return {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def evaluate(policy, dataset, indices):
    """Use the same samples and flow noise/time without disturbing training RNG."""
    was_training = policy.training
    policy.eval()
    losses = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
        for offset, index in enumerate(indices):
            torch.manual_seed(10000 + offset)
            value = policy(get_batch(dataset, index))[0].item()
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite validation loss')
            losses.append(value)
    policy.train(was_training)
    return {'mean_loss': sum(losses) / len(losses), 'sample_losses': losses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('outputs/model'))
    parser.add_argument('--dataset', type=Path, default=Path('D:/1104/nero_cube_300_pi05'))
    parser.add_argument('--tokenizer', type=Path, default=Path('outputs/model/tokenizer'))
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--val-samples', type=int, default=16)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100 or not 1 <= args.val_samples <= 60:
        parser.error('This bounded smoke check supports 1..100 steps and 1..60 validation samples')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with bf16 support required')
    output = args.output or Path('outputs/train') / time.strftime('pi05_lora_smoke_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'report.json'
    report = {'status': 'RUNNING', 'seed': 42, 'steps_requested': args.steps,
              'purpose': 'training stability and adapter reload smoke check',
              'hardware_access': False, 'validation': [], 'training': []}

    def write_report():
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    write_report()
    try:
        torch.manual_seed(42)
        rng = random.Random(42)
        config, info = build_config(args.model, args.dataset / 'train', args.tokenizer, 'bfloat16')
        config.optimizer_lr = 1e-4
        config.optimizer_weight_decay = 0.01
        config.scheduler_warmup_steps = min(10, args.steps)
        config.scheduler_decay_steps = args.steps
        config.scheduler_decay_lr = 1e-5
        train = make_dataset(args.dataset / 'train', 'train', info['fps'])
        val_info = json.loads((args.dataset / 'val/meta/info.json').read_text(encoding='utf-8'))
        if any(val_info['features'][key]['shape'] != info['features'][key]['shape']
               for key in ('observation.state', 'action', 'observation.images.base', 'observation.images.wrist')):
            raise RuntimeError('Train/validation feature shapes differ')
        val = make_dataset(args.dataset / 'val', 'val', val_info['fps'])
        val_indices = []
        val_rng = random.Random(2026)
        episode_count = len(val.episode_data_index['from'])
        for i in range(args.val_samples):
            ep = i * episode_count // args.val_samples
            begin, end = (int(val.episode_data_index[key][ep]) for key in ('from', 'to'))
            val_indices.append(val_rng.randrange(begin, end))
        report.update({'gpu': torch.cuda.get_device_name(), 'batch_size': 1,
                       'train_frames': len(train), 'val_frames': len(val),
                       'validation_indices': val_indices, 'train_quantiles_only': True,
                       'sampling': 'uniform random train frames with replacement',
                       'dataset_balancing': False, 'lr_peak': 1e-4, 'lr_end': 1e-5,
                       'warmup_steps': config.scheduler_warmup_steps})
        print(f'OUTPUT={output}', flush=True)
        policy = PI05Policy.from_pretrained(args.model, config=config, strict=True, local_files_only=True)
        trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
        allowed = ('action_in_proj.', 'action_out_proj.', 'time_mlp_in.', 'time_mlp_out.')
        if any('lora_' not in n and not any(n.startswith('model.' + head) for head in allowed) for n in trainable):
            raise RuntimeError('Unexpected trainable parameters')
        if any(p.requires_grad for p in policy.model.paligemma_with_expert.paligemma.parameters()):
            raise RuntimeError('PaliGemma is not frozen')
        report['lora_layers'] = sum(isinstance(m, LoRALinear) for m in policy.modules())
        report['trainable_parameters'] = sum(p.numel() for p in trainable.values())
        if report['lora_layers'] != 36 or report['trainable_parameters'] != 2792480:
            raise RuntimeError('Unexpected LoRA architecture')
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = evaluate(policy, val, val_indices)
        report['validation'].append({'step': 0, **baseline})
        print('VAL ' + json.dumps(report['validation'][-1]), flush=True)
        write_report()
        policy.train()
        started = time.perf_counter()
        with (output / 'metrics.jsonl').open('w', encoding='utf-8') as log:
            for step in range(1, args.steps + 1):
                warmup = config.scheduler_warmup_steps
                lr = 1e-4 * step / warmup if step <= warmup else (
                    1e-5 + (1e-4 - 1e-5) * (1 + math.cos(math.pi * (step-warmup) / max(1, args.steps-warmup))) / 2)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                index = rng.randrange(len(train))
                batch = get_batch(train, index)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = policy(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss at step {step}')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable.values()):
                    raise RuntimeError(f'Missing/nonfinite trainable gradient at step {step}')
                if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
                    raise RuntimeError('Frozen parameter received gradients')
                norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in trainable.values()):
                    raise RuntimeError('Nonfinite updated parameter')
                metrics = {'step': step, 'index': index, 'loss': loss.item(), 'lr': lr,
                           'grad_norm': norm.item(), 'elapsed_seconds': time.perf_counter()-started,
                           'cuda_peak_allocated_GB': torch.cuda.max_memory_allocated()/1e9,
                           'cuda_peak_reserved_GB': torch.cuda.max_memory_reserved()/1e9,
                           'rss_GB': psutil.Process().memory_info().rss/1e9}
                report['training'].append(metrics)
                log.write(json.dumps(metrics) + '\n')
                log.flush()
                if step % 10 == 0 or step == 1:
                    print('TRAIN ' + json.dumps(metrics), flush=True)
                if step % 25 == 0 or step == args.steps:
                    report['validation'].append({'step': step, **evaluate(policy, val, val_indices)})
                    print('VAL ' + json.dumps(report['validation'][-1]), flush=True)
                    write_report()
        optimizer.zero_grad(set_to_none=True)
        checkpoint = output / 'checkpoint_last'
        checkpoint.mkdir()
        shutil.copyfile(args.dataset / 'train/meta/quantile_stats.json', checkpoint / 'quantile_stats.json')
        policy.config.quantile_stats_path = str((checkpoint / 'quantile_stats.json').resolve())
        policy.save_pretrained(checkpoint)
        expected = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        final_losses = report['validation'][-1]['sample_losses']
        del optimizer, trainable, policy, loss, batch
        gc.collect()
        torch.cuda.empty_cache()
        print('RELOAD: fresh base model + saved adapter', flush=True)
        restored = PI05Policy.from_pretrained(checkpoint, local_files_only=True, strict=True)
        torch.cuda.empty_cache()
        actual = {n: p for n, p in restored.named_parameters() if p.requires_grad}
        if set(actual) != set(expected):
            raise RuntimeError('Reloaded trainable tensor names differ')
        if any(not torch.equal(actual[n].detach().cpu(), expected[n]) for n in expected):
            raise RuntimeError('Reloaded trainable weights differ')
        reloaded_eval = evaluate(restored, val, val_indices)
        max_difference = max(abs(a-b) for a, b in zip(final_losses, reloaded_eval['sample_losses'], strict=True))
        if max_difference > 1e-6:
            raise RuntimeError(f'Reload validation differs: {max_difference}')
        report.update({'status': 'PASS', 'checkpoint': str(checkpoint.resolve()),
                       'checkpoint_bytes': sum(p.stat().st_size for p in checkpoint.iterdir() if p.is_file()),
                       'reload_trainable_tensors_equal': len(expected),
                       'reload_validation_max_abs_difference': max_difference,
                       'reload_validation': reloaded_eval, 'optimizer_state_saved': False,
                       'note': 'Adapter checkpoint supports inference/further fine-tuning, not exact optimizer resume.'})
        write_report()
        print(f'PI05_100_STEP_SMOKE_PASS report={report_path}', flush=True)
    except BaseException as exc:
        report.update({'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'})
        write_report()
        raise


if __name__ == '__main__':
    main()
