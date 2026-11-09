"""Offline bounded stage-balanced experiment with independent generated-action validation."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

from pi05_train_preflight import build_config
from pi05_lora_short_train import get_batch, make_dataset
from pi05_lora_overfit import targets_and_hold
from pi05_action_eval import errors
from pi05_stage_sampling import STAGES, audit_split, fixed_examples

import numpy as np
import psutil
import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

SEEDS = (31415, 27182)
METRICS = ('joint_MAE_deg', 'gripper_MAE_0_1', 'mean_abs_error_over_train_std',
           'gripper_outside_0_1_fraction')


def summarize_predictions(predictions, target, valid, std):
    per_seed = [{str(h): errors(p[:, :h], target[:, :h], valid[:, :h], std)
                 for h in (1, 25, 50)} for p in predictions]
    result = {'mean_errors': {str(h): {k: float(np.mean([v[str(h)][k] for v in per_seed]))
                                      for k in METRICS} for h in (1, 25, 50)},
              'per_seed': [{'seed': s, 'errors': v} for s, v in zip(SEEDS, per_seed)]}
    grip = predictions[..., 7][:, valid]
    violation = np.maximum(-grip, 0) + np.maximum(grip-1, 0)
    result['gripper_raw_range'] = [float(grip.min()), float(grip.max())]
    result['first_per_joint_MAE_deg'] = (np.abs(predictions[:,:,0,:7]-target[None,:,0,:7]).mean((0,1))*180/np.pi).tolist()
    result['h50_per_joint_MAE_deg'] = (np.abs(predictions[...,:7]-target[None,...,:7]).mean((0,1,2))*180/np.pi).tolist()
    result['seed_joint_difference_deg'] = float(np.abs(predictions[0,...,:7]-predictions[1,...,:7]).mean()*180/np.pi)
    result['gripper_mean_bound_violation'] = float(violation.mean())
    # Absolute error in net open/close change, plus direction accuracy on clear transitions.
    t_delta = target[:, -1, 7] - target[:, 0, 7]
    p_delta = predictions[:, :, -1, 7] - predictions[:, :, 0, 7]
    transitions = np.abs(t_delta) >= .25
    result['gripper_transition_samples'] = int(transitions.sum())
    if transitions.any():
        result['gripper_net_change_MAE'] = float(np.abs(p_delta[:, transitions]-t_delta[transitions]).mean())
        result['gripper_transition_direction_accuracy'] = float(
            (np.sign(p_delta[:, transitions]) == np.sign(t_delta[transitions])).mean())
    return result


def evaluate(policy, batches, selected, std, arrays, key):
    was_training = policy.training
    policy.eval()
    target, valid, _ = targets_and_hold(batches)
    if not valid.all():
        raise RuntimeError('Every evaluated window must have 50 real target actions')
    losses, generated = [], []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
        for i, batch in enumerate(batches):
            torch.manual_seed(10000+i)
            loss = float(policy(batch)[0].item())
            if not math.isfinite(loss):
                raise RuntimeError('Nonfinite evaluation flow loss')
            losses.append(loss)
        for seed in SEEDS:
            predictions = []
            for i, batch in enumerate(batches):
                generator = torch.Generator(device='cuda').manual_seed(seed+i)
                noise = torch.randn((1, 50, 32), generator=generator, device='cuda')
                observation = {k: batch[k] for k in ('observation.state', 'observation.images.base',
                                                   'observation.images.wrist', 'task')}
                pred = policy.predict_action_chunk(observation, noise=noise, num_steps=10)
                if not torch.isfinite(pred).all():
                    raise RuntimeError('Nonfinite generated actions')
                predictions.append(pred[0].cpu().numpy())
            generated.append(np.stack(predictions))
            arrays[f'{key}_seed_{seed}'] = generated[-1]
    generated = np.stack(generated)
    result = summarize_predictions(generated, target, valid, std)
    result.update({'flow_mean_loss': float(np.mean(losses)), 'flow_sample_losses': losses, 'by_stage': {}})
    for stage in STAGES:
        mask = np.array([x['stage'] == stage for x in selected])
        if mask.any():
            result['by_stage'][stage] = summarize_predictions(generated[:, mask], target[mask], valid[mask], std)
    result['by_position'] = {}
    for group in sorted({x['position_group'] for x in selected if 'position_group' in x}):
        mask = np.array([x.get('position_group') == group for x in selected])
        result['by_position'][group] = summarize_predictions(generated[:, mask], target[mask], valid[mask], std)
    policy.train(was_training)
    return result


def check_disjoint_sources(root):
    manifest = json.loads((root / 'conversion_manifest.json').read_text(encoding='utf-8'))
    identities = {s: {(e['source_dataset'], e['source_episode']) for e in manifest['splits'][s]['episodes']}
                  for s in ('train', 'val')}
    if identities['train'] & identities['val']:
        raise RuntimeError('Train/validation share original episode identity')
    return {s: len(v) for s, v in identities.items()}


def training_schedule(audit, steps):
    rng = random.Random(42)
    by_stage = {}
    for stage in STAGES:
        episodes = list(audit['eligible'])
        rng.shuffle(episodes)
        by_stage[stage] = episodes
    result = []
    order = []
    for step in range(steps):
        if step % 4 == 0:
            order = list(STAGES)
            rng.shuffle(order)
        stage = order[step % 4]
        e = by_stage[stage][(step // 4) % len(by_stage[stage])]
        frame = rng.choice(e['windows'][stage])
        result.append({'episode': e['episode'], 'stage': stage, 'frame': frame,
                       'dataset_index': e['offset']+frame})
    return result


def validate_manifest(manifest, audit, root, steps):
    if Path(manifest['dataset']).resolve() != root.resolve() or len(manifest['training']) != steps:
        raise RuntimeError('Manifest dataset/step mismatch')
    for panel, choices in [('training', manifest['training']), *manifest['evaluation'].items()]:
        split = 'train' if panel in ('training','train_probe') else 'val'
        eligible = {e['episode']:e for e in audit[split]['eligible']}
        for x in choices:
            if x['stage'] == 'unfiltered':
                continue  # Dataset target-padding check covers this independent panel.
            e = eligible.get(x['episode'])
            if e is None or x['frame'] not in e['windows'][x['stage']] or x['dataset_index'] != e['offset']+x['frame']:
                raise RuntimeError('Manifest candidate/offset mismatch')
    if set(manifest['evaluation']) != {'train_probe','validation_stages','validation_unfiltered'}:
        raise RuntimeError('Unexpected evaluation panels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('D:/1104/nero_cube_300_pi05'))
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--delta-stats', type=Path, help='Isolated joint-delta ablation; explicit experimental loader required')
    args = parser.parse_args()
    policy_class = PI05Policy
    stats_source = args.dataset / 'train/meta/quantile_stats.json'
    if args.delta_stats:
        from pi05_delta_experiment import ExperimentalDeltaPI05Policy
        policy_class = ExperimentalDeltaPI05Policy
        stats_source = args.delta_stats
    if not 1 <= args.steps <= 800:
        parser.error('Bounded experiment supports 1..800 steps')
    if not 0 < args.lr <= 1e-4 or args.lora_rank not in (8,16):
        parser.error('Bounded comparison supports lr <= 1e-4 and rank 8/16')
    output = args.output or Path('outputs/train') / time.strftime('pi05_lora_stages_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'hardware_access': False, 'seed': 42,
              'purpose': 'bounded stage-balanced training; offline errors do not prove task success',
              'steps_requested': args.steps, 'batch_size': 1, 'max_lr': args.lr, 'end_lr': args.lr/10,
              'lora_rank': args.lora_rank, 'lora_alpha': 2*args.lora_rank, 'lora_targets': ['q_proj','v_proj'],
              'warmup_steps': min(50, args.steps), 'weight_decay': .01,
              'initialization': 'fresh pretrained base; no previous overfit adapter',
              'noise_seeds': list(SEEDS), 'inference_steps': 10, 'training': [], 'evaluations': [],
              'selection_metric': '50-action normalized MAE averaged equally over stage and unfiltered validation sets',
              'stage_note': 'Ordered gripper events are heuristic stage proxies. No automatic task-success labels.'}
    report['action_representation'] = 'joint_delta_gripper_absolute' if args.delta_stats else 'absolute'
    report['statistics_source'] = str(stats_source.resolve())
    arrays = {}
    def write():
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if arrays:
            np.savez_compressed(output / 'trajectories.npz', **arrays)
    def checkpoint(policy, step):
        folder = output / f'checkpoint_{step:06d}'
        folder.mkdir()
        shutil.copyfile(stats_source, folder / 'quantile_stats.json')
        policy.config.quantile_stats_path = str((folder / 'quantile_stats.json').resolve())
        policy.save_pretrained(folder)
        return folder
    write()
    try:
        print('OUTPUT='+str(output), flush=True)
        report['source_episode_disjoint_check'] = check_disjoint_sources(args.dataset)
        audit = {s: audit_split(args.dataset / s) for s in ('train', 'val')}
        (output / 'stage_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        report['stage_eligible_episodes'] = {s: len(audit[s]['eligible']) for s in audit}
        schedule = training_schedule(audit['train'], args.steps)
        selected = {'train_probe': fixed_examples(audit['train'], 4),
                    'validation_stages': fixed_examples(audit['val'], 8)}
        frozen_manifest = None
        if args.manifest:
            frozen_manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
            validate_manifest(frozen_manifest, audit, args.dataset, args.steps)
            schedule = frozen_manifest['training']
            selected = frozen_manifest['evaluation']
            report['sampling_manifest_sha256'] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
            report['position_support'] = frozen_manifest['support']
        torch.manual_seed(42)
        config, info = build_config(Path('outputs/model'), args.dataset / 'train', Path('outputs/model/tokenizer'), 'bfloat16', lora_rank=args.lora_rank)
        config.optimizer_lr = report['max_lr']
        config.optimizer_weight_decay = report['weight_decay']
        config.scheduler_warmup_steps = report['warmup_steps']
        config.scheduler_decay_steps = args.steps
        config.scheduler_decay_lr = report['end_lr']
        if args.delta_stats:
            config.quantile_stats_path = str(stats_source.resolve())
        train = make_dataset(args.dataset / 'train', 'train', info['fps'])
        val = make_dataset(args.dataset / 'val', 'val', info['fps'])
        val_info = json.loads((args.dataset / 'val/meta/info.json').read_text(encoding='utf-8'))
        if val_info['fps'] != info['fps'] or any(val_info['features'][k] != info['features'][k]
             for k in ('observation.state', 'action', 'observation.images.base', 'observation.images.wrist')):
            raise RuntimeError('Train/validation schema differs')
        for split, dataset in [('train', train), ('val', val)]:
            for e in audit[split]['eligible']:
                if int(dataset.episode_data_index['from'][e['episode']]) != e['offset']:
                    raise RuntimeError('Parquet offsets and dataset indices differ')
        # A separate panel uses all episode IDs without stage eligibility filtering.
        # Phase fractions span the valid episode, rather than centering all samples.
        unfiltered = []
        for j in range(16):
            ep = j * val_info['total_episodes'] // 16
            start, end = (int(val.episode_data_index[k][ep]) for k in ('from', 'to'))
            frac = (.1, .35, .6, .85)[j % 4]
            frame = int((end-start-50)*frac)
            unfiltered.append({'episode': ep, 'frame': frame, 'stage': 'unfiltered', 'dataset_index': start+frame})
        if frozen_manifest is None:
            selected['validation_unfiltered'] = unfiltered
        (output / 'sampling_manifest.json').write_text(json.dumps({'training': schedule, 'evaluation': selected}, indent=2), encoding='utf-8')
        std = np.asarray(audit['train']['action_std'])
        report.update({'train_action_std': std.tolist(), 'selected': selected,
                       'training_stage_counts': dict(Counter(x['stage'] for x in schedule)),
                       'training_unique_episodes': len({x['episode'] for x in schedule}),
                       'training_unique_windows': len({x['dataset_index'] for x in schedule}),
                       'validation_is_fixed': True, 'optimizer_state_saved': False})
        batches = {}
        for label, choices in selected.items():
            dataset = train if label == 'train_probe' else val
            batches[label] = [get_batch(dataset, x['dataset_index']) for x in choices]
            target, valid, hold = targets_and_hold(batches[label])
            if not valid.all():
                raise RuntimeError('Evaluation padding present')
            arrays.update({label+'_target': target, label+'_valid': valid, label+'_hold': hold})
            report[label+'_hold_baseline'] = {str(h): errors(hold[:, :h], target[:, :h], valid[:, :h], std)
                                             for h in (1, 25, 50)}
        print('LOADING_FRESH_BASE', flush=True)
        policy = policy_class.from_pretrained('outputs/model', config=config, strict=True, local_files_only=True)
        trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
        allowed = ('model.action_in_proj.', 'model.action_out_proj.', 'model.time_mlp_in.', 'model.time_mlp_out.')
        if len(trainable) != 80 or any('lora_' not in n and not n.startswith(allowed) for n in trainable):
            raise RuntimeError('Unexpected trainable architecture')
        if any(p.requires_grad for p in policy.model.paligemma_with_expert.paligemma.parameters()):
            raise RuntimeError('PaliGemma must remain frozen')
        report['trainable_parameters'] = sum(p.numel() for p in trainable.values())
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=args.lr, betas=(.9, .95), weight_decay=.01)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        best_score, best_step, best_expected = float('inf'), None, None
        def evaluation(step, include_train=False):
            entry = {'step': step}
            for label in selected:
                if label == 'train_probe' and not include_train:
                    continue
                entry[label] = evaluate(policy, batches[label], selected[label], std, arrays, f'{label}_step_{step}')
                print('EVAL '+json.dumps({'step': step, 'panel': label,
                     'flow': entry[label]['flow_mean_loss'], 'errors': entry[label]['mean_errors']}), flush=True)
            entry['selection_score'] = float(np.mean([entry[label]['mean_errors']['50']['mean_abs_error_over_train_std']
                           for label in ('validation_stages', 'validation_unfiltered')]))
            report['evaluations'].append(entry)
            write()
            return entry['selection_score']
        evaluation(0, True)
        # Architecture initialization consumes different RNG counts at different ranks.
        # Restart training RNG after loading/evaluation for matched time/noise/dropout.
        torch.manual_seed(42)
        started = time.perf_counter()
        policy.train()
        for step, sample in enumerate(schedule, 1):
            warmup = report['warmup_steps']
            if step <= warmup:
                lr = report['max_lr'] * step/warmup
            else:
                progress = (step-warmup)/max(1, args.steps-warmup)
                lr = report['end_lr'] + .5*(report['max_lr']-report['end_lr'])*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:
                group['lr'] = lr
            batch = get_batch(train, sample['dataset_index'])
            if batch['action_is_pad'].any():
                raise RuntimeError('Training target crosses episode boundary')
            optimizer.zero_grad(set_to_none=True)
            loss = policy(batch)[0]
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable.values()):
                raise RuntimeError('Missing/nonfinite gradient')
            if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen parameter received gradients')
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1., error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in trainable.values()):
                raise RuntimeError('Nonfinite updated weight')
            metrics = {'step': step, **sample, 'loss': loss.item(), 'lr': lr, 'grad_norm': norm.item(),
                       'elapsed_seconds': time.perf_counter()-started, 'rss_GB': psutil.Process().memory_info().rss/1e9}
            report['training'].append(metrics)
            if step % 25 == 0:
                print('TRAIN '+json.dumps(metrics), flush=True)
                write()
            if step in (200, 400, 800) or step == args.steps:
                optimizer.zero_grad(set_to_none=True)
                score = evaluation(step, step == args.steps)
                folder = checkpoint(policy, step)
                if score < best_score:
                    best_score, best_step = score, step
                    best_expected = {n: p.detach().cpu().clone() for n, p in trainable.items()}
                report.update({'best_trained_step': best_step, 'best_selection_score': best_score,
                               'last_checkpoint': str(folder.resolve())})
                write()
        best_folder = output / f'checkpoint_{best_step:06d}'
        best_copy = output / 'checkpoint_best'
        shutil.copytree(best_folder, best_copy)
        last_copy = output / 'checkpoint_last'
        shutil.copytree(output / f'checkpoint_{args.steps:06d}', last_copy)
        report.update({'checkpoint_best': str(best_copy.resolve()), 'checkpoint_last': str(last_copy.resolve()),
                       'cuda_peak_allocated_GB': torch.cuda.max_memory_allocated()/1e9,
                       'cuda_peak_reserved_GB': torch.cuda.max_memory_reserved()/1e9,
                       'max_rss_GB': max(m['rss_GB'] for m in report['training'])})
        del policy, optimizer, trainable, loss, batch
        gc.collect()
        torch.cuda.empty_cache()
        restored = policy_class.from_pretrained(best_copy, strict=True, local_files_only=True)
        actual = {n: p for n, p in restored.named_parameters() if p.requires_grad}
        if set(actual) != set(best_expected) or any(not torch.equal(actual[n].detach().cpu(), best_expected[n]) for n in actual):
            raise RuntimeError('Best adapter weights differ after fresh reload')
        evaluate(restored, batches['validation_stages'][:4], selected['validation_stages'][:4], std, arrays, 'reload_best_first4')
        difference = max(float(np.abs(arrays[f'reload_best_first4_seed_{s}']-
                           arrays[f'validation_stages_step_{best_step}_seed_{s}'][:4]).max()) for s in SEEDS)
        if difference > 1e-6:
            raise RuntimeError('Reload generated actions differ')
        report.update({'status': 'PASS', 'reload_tensor_count': len(actual), 'reload_action_max_abs_difference': difference,
                       'best_beats_base_selection_score': best_score < report['evaluations'][0]['selection_score'],
                       'note': 'PASS means execution and reload checks passed. Assess quality using held-out action errors; no task success measured.'})
        write()
        print('PI05_STAGE_TRAIN_PASS report='+str(output / 'report.json'), flush=True)
    except BaseException as exc:
        report.update({'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'})
        write()
        raise


if __name__ == '__main__':
    main()
