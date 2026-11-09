"""Bounded four-example memorization diagnostic, with generated action evaluation."""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from pathlib import Path

from pi05_train_preflight import build_config
from pi05_lora_short_train import make_dataset, get_batch
from pi05_action_eval import errors

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def select_moving_examples(dataset, root):
    """One complete, high-motion 50-frame target from each of four episodes."""
    selected = []
    count = len(dataset.episode_data_index['from'])
    for episode in (0, count // 4, count // 2, 3 * count // 4):
        path = root / 'data' / f'chunk-{episode // 1000:03d}' / f'episode_{episode:06d}.parquet'
        table = pq.read_table(path, columns=['action', 'observation.state'])
        actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        if len(actions) < 50:
            raise RuntimeError('Selected episode is shorter than the diagnostic chunk')
        scores = np.asarray([np.abs(actions[i:i+50, :7] - states[i, :7]).mean()
                             for i in range(len(actions)-49)])
        local = int(scores.argmax())
        selected.append({'episode': episode, 'frame_in_episode': local,
                         'dataset_index': int(dataset.episode_data_index['from'][episode])+local,
                         'hold_joint_MAE_rad': float(scores[local])})
    return selected


def targets_and_hold(batches):
    target = np.stack([b['action'][0].cpu().numpy() for b in batches])
    valid = np.stack([~b['action_is_pad'][0].cpu().numpy().astype(bool) for b in batches])
    hold = np.stack([np.repeat(b['observation.state'][0, :8].cpu().numpy()[None], 50, axis=0) for b in batches])
    return target, valid, hold


def evaluate(policy, batches, std, arrays, key):
    was_training = policy.training
    policy.eval()
    target, valid, _ = targets_and_hold(batches)
    result = {'flow_sample_losses': [], 'per_seed': []}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
        for i, batch in enumerate(batches):
            torch.manual_seed(10000+i)
            result['flow_sample_losses'].append(float(policy(batch)[0].item()))
        for seed in (31415, 27182):
            predictions = []
            for i, batch in enumerate(batches):
                generator = torch.Generator(device='cuda').manual_seed(seed+i)
                noise = torch.randn((1, 50, 32), generator=generator, device='cuda')
                observation = {k: batch[k] for k in ('observation.state', 'observation.images.base',
                                                   'observation.images.wrist', 'task')}
                pred = policy.predict_action_chunk(observation, noise=noise, num_steps=10)
                if not torch.isfinite(pred).all():
                    raise RuntimeError('Nonfinite generated action')
                predictions.append(pred[0].cpu().numpy())
            predictions = np.stack(predictions)
            arrays[f'{key}_seed_{seed}'] = predictions
            result['per_seed'].append({'seed': seed, 'errors': {str(h): errors(
                predictions[:, :h], target[:, :h], valid[:, :h], std) for h in (1, 25, 50)}})
    result['flow_mean_loss'] = float(np.mean(result['flow_sample_losses']))
    result['mean_errors'] = {str(h): {k: float(np.mean([s['errors'][str(h)][k] for s in result['per_seed']]))
                            for k in ('joint_MAE_deg', 'gripper_MAE_0_1', 'mean_abs_error_over_train_std',
                                      'gripper_outside_0_1_fraction')} for h in (1, 25, 50)}
    policy.train(was_training)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('D:/1104/nero_cube_300_pi05'))
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 800:
        parser.error('Bounded diagnostic supports 1..800 steps')
    output = args.output or Path('outputs/train') / time.strftime('pi05_lora_overfit_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'hardware_access': False, 'purpose': 'four-example memorization, not generalization',
              'steps_requested': args.steps, 'initialization': 'fresh pretrained base, zero-initialized LoRA B',
              'batch_size': 1, 'lr': 1e-4, 'warmup_steps': 10, 'weight_decay': 0.0,
              'lora_dropout': 0.05, 'train_evaluations': [], 'validation_evaluations': [], 'training': []}
    arrays = {}
    def write():
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if arrays:
            np.savez_compressed(output / 'trajectories.npz', **arrays)
    write()
    try:
        torch.manual_seed(42)
        config, info = build_config(Path('outputs/model'), args.dataset / 'train', Path('outputs/model/tokenizer'), 'bfloat16')
        train = make_dataset(args.dataset / 'train', 'train', info['fps'])
        val = make_dataset(args.dataset / 'val', 'val', info['fps'])
        selected = select_moving_examples(train, args.dataset / 'train')
        batches = [get_batch(train, x['dataset_index']) for x in selected]
        previous = json.loads(Path('outputs/train/pi05_lora_smoke_20261105_152737/report.json').read_text())
        val_indices = previous['validation_indices'][::4]
        validation = [get_batch(val, i) for i in val_indices]
        std = np.asarray(json.loads(Path('outputs/train/pi05_lora_smoke_20261105_152737/action_eval_20261105_154436/report.json').read_text())['train_action_std'])
        report.update({'selected_train_examples': selected, 'validation_indices': val_indices,
                       'selection': 'largest 50-frame joint change versus current state, episodes 0,60,120,180',
                       'train_action_std': std.tolist(), 'inference_steps': 10, 'noise_seeds': [31415, 27182]})
        for label, data in [('train', batches), ('validation', validation)]:
            target, valid, hold = targets_and_hold(data)
            arrays.update({label+'_targets': target, label+'_valid': valid, label+'_hold': hold})
            report[label+'_hold_baseline'] = {str(h): errors(hold[:, :h], target[:, :h], valid[:, :h], std) for h in (1, 25, 50)}
        if not arrays['train_valid'].all():
            raise RuntimeError('Training examples must have full 50-frame targets')
        print('OUTPUT='+str(output), flush=True)
        print('SAMPLES '+json.dumps(selected), flush=True)
        policy = PI05Policy.from_pretrained('outputs/model', config=config, strict=True, local_files_only=True)
        trainable = {n:p for n,p in policy.named_parameters() if p.requires_grad}
        if sum(p.numel() for p in trainable.values()) != 2792480:
            raise RuntimeError('Unexpected trainable architecture')
        if any(p.requires_grad for p in policy.model.paligemma_with_expert.paligemma.parameters()):
            raise RuntimeError('PaliGemma must stay frozen')
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, betas=(0.9,0.95), weight_decay=0.0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for label, data in [('train', batches), ('validation', validation)]:
            value = {'step': 0, **evaluate(policy, data, std, arrays, label+'_step_0')}
            report[label+'_evaluations'].append(value)
            print(label.upper()+'_EVAL '+json.dumps({'step':0, 'flow':value['flow_mean_loss'], 'errors':value['mean_errors']}), flush=True)
        write()
        policy.train()
        started = time.perf_counter()
        for step in range(1, args.steps+1):
            index = (step-1) % len(batches)
            lr = 1e-4 * min(1.0, step/10)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = policy(batches[index])[0]
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable.values()):
                raise RuntimeError('Missing/nonfinite trainable gradient')
            if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen parameter received gradients')
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in trainable.values()):
                raise RuntimeError('Nonfinite trainable parameter')
            metrics = {'step':step, 'sample':index, 'loss':loss.item(), 'lr':lr, 'grad_norm':norm.item(),
                       'elapsed_seconds':time.perf_counter()-started}
            report['training'].append(metrics)
            if step % 25 == 0:
                print('TRAIN '+json.dumps(metrics), flush=True)
            if step in (100,200,400,800) or step == args.steps:
                value = {'step':step, **evaluate(policy, batches, std, arrays, f'train_step_{step}')}
                report['train_evaluations'].append(value)
                print('TRAIN_EVAL '+json.dumps({'step':step, 'flow':value['flow_mean_loss'], 'errors':value['mean_errors']}), flush=True)
                write()
        optimizer.zero_grad(set_to_none=True)
        value = {'step':args.steps, **evaluate(policy, validation, std, arrays, f'validation_step_{args.steps}')}
        report['validation_evaluations'].append(value)
        print('VALIDATION_EVAL '+json.dumps({'step':args.steps,'flow':value['flow_mean_loss'],'errors':value['mean_errors']}), flush=True)
        checkpoint = output / 'checkpoint_last'
        checkpoint.mkdir()
        shutil.copyfile(args.dataset / 'train/meta/quantile_stats.json', checkpoint / 'quantile_stats.json')
        policy.config.quantile_stats_path = str((checkpoint / 'quantile_stats.json').resolve())
        policy.save_pretrained(checkpoint)
        expected = {n:p.detach().cpu().clone() for n,p in trainable.items()}
        report.update({'checkpoint':str(checkpoint.resolve()), 'cuda_peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,
                       'cuda_peak_reserved_GB':torch.cuda.max_memory_reserved()/1e9, 'optimizer_state_saved':False})
        del policy, optimizer, trainable, loss
        gc.collect()
        torch.cuda.empty_cache()
        restored = PI05Policy.from_pretrained(checkpoint, strict=True, local_files_only=True)
        torch.cuda.empty_cache()
        actual = {n:p for n,p in restored.named_parameters() if p.requires_grad}
        if set(actual) != set(expected) or any(not torch.equal(actual[n].detach().cpu(), expected[n]) for n in expected):
            raise RuntimeError('Saved adapter weights differ after fresh reload')
        reloaded = evaluate(restored, batches, std, arrays, 'train_reload')
        max_diff = max(float(np.abs(arrays[f'train_step_{args.steps}_seed_{s}']-arrays[f'train_reload_seed_{s}']).max()) for s in (31415,27182))
        if max_diff > 1e-6:
            raise RuntimeError('Generated actions differ after fresh reload')
        report.update({'status':'PASS', 'reload_tensor_count':len(expected), 'reload_action_max_abs_difference':max_diff,
                       'note':'PASS means diagnostic execution succeeded; consult action errors for fit quality. No task success measured.'})
        write()
        print('PI05_OVERFIT_DIAGNOSTIC_PASS report='+str(output/'report.json'), flush=True)
    except BaseException as exc:
        report.update({'status':'FAILED', 'error':f'{type(exc).__name__}: {exc}'})
        write()
        raise


if __name__ == '__main__':
    main()
