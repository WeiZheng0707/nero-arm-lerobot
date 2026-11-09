"""Offline pi05_22 dual-camera LoRA check using real data.

Runs a bounded number of optimizer steps in memory; never saves model weights
or accesses robot hardware. Only the JSON verification report is persisted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# Keep Windows lock paths short and prevent remote fallback during this check.
os.environ.setdefault('HF_DATASETS_CACHE', str(Path(tempfile.gettempdir()) / 'pi05_verify_cache'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import psutil
import torch
from torch.utils.data import default_collate

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.lora import LoRALinear
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def build_config(model, root, tokenizer, dtype, lora_rank=8, lora_alpha=None,
                 lora_targets=('q_proj', 'v_proj'), lora_dropout=0.05):
    """Build the training config.

    The LoRA knobs are parameters because the comparison experiment needs more than
    one arm: rank 8 on q/v only (2.79M parameters) may be too small to adapt a new
    embodiment, and a false negative there would be misread as "pi0.5 is worse than
    ACT". Defaults keep the previously verified configuration.
    """
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    features = info['features']
    if features['observation.state']['shape'] != [22] or features['action']['shape'] != [8]:
        raise ValueError('Expected the converted pi05_22 dataset, state=22/action=8')
    cameras = ('observation.images.base', 'observation.images.wrist')
    config = PreTrainedConfig.from_pretrained(model, local_files_only=True)
    config.device = 'cuda'
    config.dtype = dtype
    config.push_to_hub = False
    config.lora_enabled = True
    config.train_expert_only = False
    config.freeze_vision_encoder = True
    config.lora_rank = int(lora_rank)
    config.lora_alpha = float(lora_alpha) if lora_alpha is not None else 2.0 * int(lora_rank)
    config.lora_dropout = float(lora_dropout)
    config.lora_target_modules = tuple(lora_targets)
    config.gradient_checkpointing = True
    config.compile_model = False
    config.empty_cameras = 1
    config.chunk_size = 50
    config.n_action_steps = 25
    config.tokenizer_name_or_path = str(tokenizer.resolve())
    config.quantile_stats_path = str((root / 'meta/quantile_stats.json').resolve())
    config.input_features = {}
    for key in cameras:
        h, w, c = features[key]['shape']
        config.input_features[key] = PolicyFeature(FeatureType.VISUAL, (c, h, w))
    config.input_features['observation.state'] = PolicyFeature(FeatureType.STATE, (22,))
    config.output_features = {'action': PolicyFeature(FeatureType.ACTION, (8,))}
    return config, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('outputs/model'))
    parser.add_argument('--root', type=Path, default=Path('D:/1104/nero_cube_300_pi05/train'))
    parser.add_argument('--tokenizer', type=Path, default=Path('outputs/model/tokenizer'))
    parser.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--report', type=Path, default=Path('outputs/pi05_train_preflight/report.json'))
    args = parser.parse_args()
    if not 1 <= args.steps <= 10:
        parser.error('--steps must be between 1 and 10 for this bounded check')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support bfloat16')
    torch.manual_seed(42)
    config, info = build_config(args.model, args.root, args.tokenizer, args.dtype)
    print('DATA: state=22 action=8 cameras=base,wrist + 1 masked slot; chunk=50', flush=True)
    dataset = LeRobotDataset('local/nero_cube_pi05_train', root=args.root, video_backend='pyav',
                            delta_timestamps={'action': [i / info['fps'] for i in range(50)]})
    print('LOADING MODEL (strict=True)...', flush=True)
    start = time.perf_counter()
    policy = PI05Policy.from_pretrained(args.model, config=config, strict=True, local_files_only=True)
    print(f'MODEL_LOADED seconds={time.perf_counter()-start:.2f}', flush=True)
    adapters = [(n, m) for n, m in policy.named_modules() if isinstance(m, LoRALinear)]
    trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    allowed = ('action_in_proj.', 'action_out_proj.', 'time_mlp_in.', 'time_mlp_out.')
    unexpected = [n for n in trainable if 'lora_' not in n and not any(n.startswith('model.'+v) for v in allowed)]
    if not adapters or unexpected:
        raise RuntimeError(f'Unexpected trainable parameters: {unexpected}')
    if not all(not p.requires_grad for p in policy.model.paligemma_with_expert.paligemma.parameters()):
        raise RuntimeError('PaliGemma must be fully frozen')
    policy.eval()
    with torch.no_grad():
        max_delta = 0.0
        for _, layer in adapters:
            x = torch.randn(2, layer.in_features, device='cuda', dtype=layer.weight.dtype)
            max_delta = max(max_delta, (layer(x)-layer.base(x)).abs().max().item())
        if max_delta != 0:
            raise RuntimeError('Initial LoRA is not a no-op')
    dtype_counts = Counter()
    for p in policy.parameters():
        dtype_counts[str(p.dtype)] += p.numel()
    report = {
        'model': str(args.model.resolve()), 'dataset': str(args.root.resolve()),
        'gpu': torch.cuda.get_device_name(), 'seed': 42, 'state_dim': 22, 'action_dim': 8,
        'chunk_size': 50, 'batch_size': 1, 'gradient_checkpointing': True,
        'lora_layers': len(adapters), 'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': 0.05,
        'total_parameters': sum(p.numel() for p in policy.parameters()),
        'trainable_parameters': sum(p.numel() for p in trainable.values()),
        'trainable_tensors': len(trainable), 'dtype_parameter_counts': dict(dtype_counts),
        'initial_adapter_max_delta': max_delta, 'paligemma_frozen': True, 'steps': [],
    }
    print(json.dumps(report, ensure_ascii=True), flush=True)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, betas=(0.9,0.95), weight_decay=0.01)
    policy.train()
    torch.cuda.synchronize()
    # Release unused initialization blocks before measuring steady training.
    report['cuda_reserved_after_load_GB'] = torch.cuda.memory_reserved() / 1e9
    torch.cuda.empty_cache()
    report['cuda_reserved_after_cache_cleanup_GB'] = torch.cuda.memory_reserved() / 1e9
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        index = int(dataset.episode_data_index['from'][step])
        batch = default_collate([dataset[index]])
        batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        before = {n:p.detach().clone() for n,p in trainable.items() if '.lora_B.' in n}
        torch.cuda.synchronize()
        start = time.perf_counter()
        print(f'STEP {step+1}: forward sample={index}', flush=True)
        loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        print(f'STEP {step+1}: backward loss={loss.item():.6f}', flush=True)
        loss.backward()
        bad = [n for n,p in trainable.items() if p.grad is None or not torch.isfinite(p.grad).all()]
        if bad:
            raise RuntimeError(f'Missing/nonfinite gradients: {bad}')
        if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen parameter received gradients')
        grad_norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0,error_if_nonfinite=True)
        b_grad = sum(p.grad.abs().sum().item() for n,p in trainable.items() if '.lora_B.' in n)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in trainable.values()):
            raise RuntimeError('Nonfinite updated parameters')
        changed = sum(not torch.equal(before[n],trainable[n].detach()) for n in before)
        if not b_grad > 0 or changed != len(before):
            raise RuntimeError('LoRA B parameters did not all receive an effective update')
        torch.cuda.synchronize()
        metrics = {'step':step+1, 'sample_index':index, 'loss':loss.item(),
                   'grad_norm_before_clip':grad_norm.item(), 'updated_lora_B_tensors':changed,
                   'seconds_forward_backward_optimizer':time.perf_counter()-start,
                   'cuda_peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,
                   'cuda_peak_reserved_GB':torch.cuda.max_memory_reserved()/1e9,
                   'process_rss_GB':psutil.Process().memory_info().rss/1e9}
        report['steps'].append(metrics)
        print(json.dumps(metrics), flush=True)
    report['status'] = 'PASS'
    report['model_weights_saved'] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'PI05_REAL_BATCH_PREFLIGHT_PASS report={args.report}',flush=True)


if __name__ == '__main__':
    main()
