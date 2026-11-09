"""Summarize a completed bounded stage experiment; never train or access hardware."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--no-plot', action='store_true', help='Write summary and verify arrays without matplotlib')
    args = parser.parse_args()
    r = json.loads((args.run / 'report.json').read_text(encoding='utf-8'))
    if r['status'] != 'PASS':
        raise RuntimeError('Summarize only after execution and reload verification complete')
    audit = json.loads((args.run / 'stage_audit.json').read_text(encoding='utf-8'))
    rows = r['evaluations']
    baseline, last = rows[0], rows[-1]
    best = next(e for e in rows if e['step'] == r['best_trained_step'])
    stage_names = {'approach': '接近', 'grasp': '夹取闭合', 'transport': '搬运', 'release': '释放张开'}
    panels = {'validation_stages': '阶段验证集（8 段轨迹 × 4 阶段）',
              'validation_unfiltered': '未按阶段筛选的验证集（16 段轨迹各 1 窗口）'}
    lines = ['# π0.5 + LoRA 四阶段有界训练结果（2026-11-05）', '',
             f"已完成 {r['steps_requested']} 步，运行/重载检查 {r['status']}。全程离线，无机器人操作。", '',
             '## 实验设置', '',
             f"- 从本地原始基础权重开始，不继承之前四样本过拟合 adapter。可训练参数 {r['trainable_parameters']:,}。",
             f"- batch=1；最高学习率 {r['max_lr']}，预热 {r['warmup_steps']} 步，余弦衰减至 {r['end_lr']}；AdamW，weight_decay={r['weight_decay']}。",
             f"- 四阶段各 200 个窗口，共 {r['training_unique_windows']} 个不同窗口，覆盖 {r['training_unique_episodes']} 段训练轨迹。",
             '- 每窗口有完整 50 帧目标，保持原始 30fps；训练只使用 train，归一化分位数与动作标准差只由 train 得到。',
             '- 原始 episode 身份检查：train=240、val=60，无重叠。固定采样清单见 sampling_manifest.json。',
             '- 验证：固定噪声 31415 / 27182、10 步生成；报告 1 / 25 / 50 帧的原始关节与夹爪误差。无目标动作输入推理。',
             '- 两个验证面板来自同一 val 分区，并非互相独立的两个测试集；面板之间可能共享轨迹。它们都与训练原始轨迹隔离。',
             '- 不裁剪生成夹爪值，原始越界比例与幅度均保留。每阶段方向正确率只评价明显开合窗口，方向正确不代表完整动作或任务成功。', '',
             '## 阶段覆盖与局限', '']
    for split in ('train', 'val'):
        a = audit[split]
        lines.append(f"- {split}：{len(a['eligible'])}/{a['episode_count']} 段满足稳定张开≥0.6、闭合≤0.3、再次张开≥0.6的顺序条件；{len(a['rejected'])} 段没有进入四阶段采样，详细原因见 stage_audit.json。")
    lines.extend(['- 稳定事件连续至少 5 帧；夹取/释放窗口内夹爪净变化至少 0.25。接近/搬运窗口限定在对应事件之间。',
                  '- 阶段是夹爪事件的启发式候选标签，不是人工真值。单个画面与夹爪变化都不能证明抓取成功。',
                  '- 抽查验证轨迹 0/13/26/43 的四阶段画面，见 ../../pi05_stage_validation_preview.jpg；没有完成所有轨迹的人工成功标注。', '',
                  '## 固定验证结果', '',
                  '| 步数 | 面板 | Flow loss | 50帧关节 MAE（°） | 50帧夹爪 MAE（0–1） | 归一化动作 MAE | 夹爪越界比例 |',
                  '|---:|---|---:|---:|---:|---:|---:|'])
    for e in rows:
        for label, name in [('validation_stages', '阶段'), ('validation_unfiltered', '未筛选')]:
            v, m = e[label], e[label]['mean_errors']['50']
            lines.append(f"| {e['step']} | {name} | {v['flow_mean_loss']:.6f} | {m['joint_MAE_deg']:.3f} | {m['gripper_MAE_0_1']:.4f} | {m['mean_abs_error_over_train_std']:.4f} | {m['gripper_outside_0_1_fraction']:.2%} |")
    lines.extend(['', '归一化误差为各动作维度除以训练集标准差后的绝对误差平均；仅用于固定口径比较。检查点选择分数是两个验证面板的 50 帧归一化误差等权平均。', '',
                  f"最优已训练检查点：第 {r['best_trained_step']} 步，分数 {best['selection_score']:.6f}；预训练基线分数 {baseline['selection_score']:.6f}。",
                  'checkpoint_best 仅指本轮已训练检查点中分数最低者，不意味着优于预训练模型或可以部署。', ''])
    if best['selection_score'] < baseline['selection_score']:
        lines.append('本轮最优已训练检查点在所选验证综合指标上优于预训练基线；仍需逐阶段、逐维度检查，且这不是独立测试集或机器人成功率。')
    else:
        lines.append('本轮已训练检查点未超过预训练基线的验证综合指标。不要因为 Flow loss 降低而把本轮权重当作可部署模型；暂不启动正式长训练。')
    lines.extend(['', '## 首帧与前25帧（最终步）', '',
                  '| 面板 | 预测长度 | 预训练关节 MAE（°） | 最终关节 MAE（°） | 保持当前状态关节 MAE（°） | 预训练夹爪 MAE | 最终夹爪 MAE |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for label, name in panels.items():
        for h in ('1', '25'):
            pre, post = baseline[label]['mean_errors'][h], last[label]['mean_errors'][h]
            hold = r[label+'_hold_baseline'][h]
            lines.append(f"| {name} | {h} | {pre['joint_MAE_deg']:.3f} | {post['joint_MAE_deg']:.3f} | {hold['joint_MAE_deg']:.3f} | {pre['gripper_MAE_0_1']:.4f} | {post['gripper_MAE_0_1']:.4f} |")
    lines.extend(['', '保持当前状态只是一条离线数值基线，不能完成任务；它误差低时提示模型尚未充分利用当前姿态或生成动作仍有系统偏移。', '',
                  '## 分阶段表现（预训练 → 最终步，50帧）', '',
                  '| 阶段 | 关节 MAE（°） | 夹爪 MAE（0–1） | 开合方向正确率 |',
                  '|---|---:|---:|---:|'])
    for s, name in stage_names.items():
        pre, post = baseline['validation_stages']['by_stage'][s], last['validation_stages']['by_stage'][s]
        a, b = pre['mean_errors']['50'], post['mean_errors']['50']
        direction = (f"{pre['gripper_transition_direction_accuracy']:.1%} → {post['gripper_transition_direction_accuracy']:.1%}"
                     if 'gripper_transition_direction_accuracy' in pre else '不适用（无明显净开合）')
        lines.append(f"| {name} | {a['joint_MAE_deg']:.3f} → {b['joint_MAE_deg']:.3f} | {a['gripper_MAE_0_1']:.4f} → {b['gripper_MAE_0_1']:.4f} | {direction} |")
    lines.extend(['', '## 保存与验证', '',
                  f"- checkpoint_000200 / checkpoint_000400 / checkpoint_000800 为阶段检查点；checkpoint_best 对应第 {r['best_trained_step']} 步，checkpoint_last 对应第 {r['steps_requested']} 步。",
                  '- 检查点是 adapter（包含动作/时间投影等可训练层），依赖原始 outputs/model 基础权重与本地 tokenizer；不是独立全量模型。',
                  '- 未保存 optimizer/RNG 状态，不能据此声称精确断点续训；后续正式训练需要补充保存与恢复。',
                  f"- 新实例重载 best：{r['reload_tensor_count']} 个可训练张量完全相同；固定前4验证窗口/两组噪声的生成动作最大差 {r['reload_action_max_abs_difference']}。",
                  f"- PyTorch峰值 allocated={r['cuda_peak_allocated_GB']:.2f}GB，reserved={r['cuda_peak_reserved_GB']:.2f}GB；训练阶段进程最大 RSS={r['max_rss_GB']:.2f}GB（不含初始加载瞬时峰值）。",
                  '- report.json 保存逐步 loss、学习率、梯度范数与评估；trajectories.npz 保存目标、掩码、保持状态基线和生成动作。', '',
                  '## 下一步', '',
                  '先检查最优检查点在相同当前姿态下的首帧/分关节偏差、夹爪开合时序以及归一化反变换；若动作误差未改善，再设计有限的学习率、采样与可训练容量对照。每次只改变一个因素，不直接延长本轮训练。',
                  '扩大人工抽查，确认阈值未漏掉夹爪闭合程度不同的真实抓取轨迹；筛除规则不能当作任务失败标签。',
                  '后续扩大验证覆盖后应另留不参与检查点选择的测试集。当前没有实机推理延迟、闭环稳定性或任务成功率结论。', '',
                  '复现入口（在项目根目录的 Anaconda Prompt，重新运行会生成新实验）：', '',
                  '~~~bat', '"D:/NERO/envs/pi05_lora/Scripts/python.exe" -u tools/pi05_lora_stage_train.py --steps 800',
                  '~~~', ''])
    (args.run / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    with np.load(args.run / 'trajectories.npz') as z:
        if any(not np.isfinite(z[k]).all() for k in z.files):
            raise RuntimeError('Nonfinite saved trajectories')
        print('FINITE_ARRAYS', len(z.files))
    print('SUMMARY='+str(args.run / 'summary.md'))
    if args.no_plot:
        return
    # Keep optional plot/font caches next to the experiment, not in the user profile.
    os.environ.setdefault('MPLCONFIGDIR', str((args.run / 'plot_cache').resolve()))
    try:
        import matplotlib
    except ModuleNotFoundError:
        print('PLOT_SKIPPED: matplotlib not installed; summary and array verification completed')
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    plots = [('flow_mean_loss', 'Flow loss'), ('joint_MAE_deg', '50-action joint MAE (degrees)'),
             ('gripper_MAE_0_1', '50-action gripper MAE (0-1)'),
             ('mean_abs_error_over_train_std', '50-action normalized MAE')]
    for ax, (key, title) in zip(axes.flat, plots):
        for label, name in [('validation_stages', 'Stage validation (32)'), ('validation_unfiltered', 'Unfiltered validation (16)')]:
            values = [e[label][key] if key == 'flow_mean_loss' else e[label]['mean_errors']['50'][key] for e in rows]
            ax.plot([e['step'] for e in rows], values, marker='o', label=name)
        ax.set_title(title)
        ax.set_xlabel('Training step')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Bounded stage-balanced LoRA: fixed offline validation; no task success measured')
    fig.savefig(args.run / 'validation_errors.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
