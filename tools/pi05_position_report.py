"""Summarize frozen-manifest offline experiments from saved evidence only."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

ARMS = ['reference_r8_lr5e5', 'low_lr_r8_lr1e5', 'capacity_r16_lr5e5', 'delta_r8_lr5e5']
PANELS = ['validation_stages', 'validation_unfiltered']

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def table(headers, rows):
    return ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
            *['| '+' | '.join(map(str,row))+' |' for row in rows], '']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    args=p.parse_args();root=args.experiment
    comparison=read(root/'comparison.json')
    if comparison['status']!='PASS': raise RuntimeError('Absolute comparison incomplete')
    manifest=read(root/'manifest.json')
    reports={a:read(root/a/'report.json') for a in ARMS if (root/a/'report.json').exists()}
    if any(r['status']!='PASS' for r in reports.values()): raise RuntimeError('An arm is incomplete')
    details={'hardware_access':False,'arms':{},'finite_arrays':{},'manifest_sha256':comparison['manifest_sha256']}
    lines=['# π0.5 + LoRA：阶段×位置受控实验', '', '日期：2026-11-05。仅离线诊断与有界训练；未访问机器人。', '',
      '## 核对结论', '',
      '- 300 段、179,911 帧，按采集时间重建的原始状态/动作与转换数据逐值一致，最大差为 0。首动作与首状态逐关节平均差均小于 0.023°。这只验证记录与转换一致，不能验证实体零位、关节方向或相机外参。',
      '- 采集代码先读状态、后发送命令，再记录同一主机时间。夹爪命令相对未来反馈的描述性最优滞后为 2 帧（261 段）或 3 帧（39 段）；静止区间会影响该统计，未平移动作标签。',
      '- 真实模型两窗口探针：BF16 完整/缓存 Euler 关节差 0.0159–0.1123°；同权重 FP32 差小于 0.000001°。缓存手动路径与生产预测 API 最大差 0。FP32 仅隔离算术精度，不恢复 BF16 权重舍入；有限样本不能排除所有实现问题。',
      '- 归一化往返误差及提示词截断已在前轮复核；未对绝对动作模型事后补起始偏置。独立增量实验在训练和推理中采用成对变换。', '',
      '证据：raw_alignment/report.json、raw_alignment/first_frames.png、numerical_probe_v2/report.json。numerical_probe 为失败的旧探针，不作为结论来源。', '',
      '## 采样与比较边界', '',
      '- 400 个训练样本，四阶段各 100，固定同一顺序、训练噪声种子和评估噪声。位置由每段袋子图像 xy 中位数按 0.1 网格分组，是代理标签。训练 13 组，每阶段×组约 7–8 样本；少数只有一段轨迹的组被重复抽样。',
      '- 阶段候选覆盖 train 202/240 段、val 43/60 段。验证阶段面板 44 窗口，未按阶段筛选面板 16 窗口；两面板可能共享验证轨迹，均与训练原始轨迹隔离。验证位置 2,4 无对应的阶段合格训练支持。',
      '- 一次训练种子、两次推理噪声、10 步 Euler、batch 1、每样本 50 帧。LoRA q/v + 动作/时间投影可训练；rank8/16 都固定 alpha/rank=2。400 步不是收敛实验。',
      '- best 由这两个固定验证面板的标准化误差平均选取；验证已参与选模，不能当成独立测试集或闭环成功率。PASS 只指运行与重载检查通过。',
      '- 绝对动作三组初始生成数组逐位相同，统计同一 manifest SHA256：'+comparison['manifest_sha256']+'.', '',
      '## 主要结果', '',
      '关节误差单位为度；夹爪 MAE 在原始 0–1 单位上计算，未裁剪预测。两个数字依次对应阶段面板／未筛选面板。', '']
    ref=reports[ARMS[0]];base=ref['evaluations'][0]
    rows=[]
    def pair(entry,h,key):
        return ' / '.join(f'{entry[p]["mean_errors"][str(h)][key]:.4f}' for p in PANELS)
    rows.append(['绝对动作初始模型',0,pair(base,1,'joint_MAE_deg'),pair(base,50,'joint_MAE_deg'),pair(base,50,'gripper_MAE_0_1'),f'{base["selection_score"]:.6f}'])
    hold=' / '.join(f'{ref[p+"_hold_baseline"]["50"]["joint_MAE_deg"]:.4f}' for p in PANELS)
    firsthold=' / '.join(f'{ref[p+"_hold_baseline"]["1"]["joint_MAE_deg"]:.4f}' for p in PANELS)
    griphold=' / '.join(f'{ref[p+"_hold_baseline"]["50"]["gripper_MAE_0_1"]:.4f}' for p in PANELS)
    rows.append(['保持当前状态', '—',firsthold,hold,griphold,'—'])
    best_entries={}
    for arm,r in reports.items():
        best=next(e for e in r['evaluations'] if e['step']==r['best_trained_step']);best_entries[arm]=best
        if arm.startswith('delta_'):
            e=r['evaluations'][0]
            rows.append([arm+' 初始',0,pair(e,1,'joint_MAE_deg'),pair(e,50,'joint_MAE_deg'),pair(e,50,'gripper_MAE_0_1'),f'{e["selection_score"]:.6f}'])
        rows.append([arm,r['best_trained_step'],pair(best,1,'joint_MAE_deg'),pair(best,50,'joint_MAE_deg'),pair(best,50,'gripper_MAE_0_1'),f'{best["selection_score"]:.6f}'])
        z=np.load(root/arm/'trajectories.npz')
        if any(not np.isfinite(z[k]).all() for k in z.files): raise RuntimeError('Nonfinite saved array')
        details['finite_arrays'][arm]=len(z.files)
        motion={}
        for panel in PANELS:
            pred=np.stack([z[f'{panel}_step_{r["best_trained_step"]}_seed_{seed}'] for seed in r['noise_seeds']])
            initial_pred=np.stack([z[f'{panel}_step_0_seed_{seed}'] for seed in r['noise_seeds']])
            target=z[panel+'_target'];anchor=z[panel+'_hold']
            pr=(pred[...,:7]-anchor[None,...,:7])*180/np.pi
            initial_pr=(initial_pred[...,:7]-anchor[None,...,:7])*180/np.pi
            tr=(target[...,:7]-anchor[...,:7])*180/np.pi
            moving={}
            for threshold in (1.,5.):
                mask=np.abs(tr).mean((1,2))>=threshold
                moving[str(threshold)]={'windows':int(mask.sum()),
                    'hold_MAE_deg':float(np.abs(tr[mask]).mean()) if mask.any() else None,
                    'initial_model_MAE_deg':float(np.abs(initial_pr[:,mask]-tr[None,mask]).mean()) if mask.any() else None,
                    'model_MAE_deg':float(np.abs(pr[:,mask]-tr[None,mask]).mean()) if mask.any() else None}
            motion[panel]={'predicted_motion_mean_abs_deg':float(np.abs(pr).mean()),
                           'initial_motion_mean_abs_deg':float(np.abs(initial_pr).mean()),
                           'initial_pooled_motion_correlation':float(np.corrcoef(initial_pr.flatten(),np.broadcast_to(tr,initial_pr.shape).flatten())[0,1]),
                           'target_motion_mean_abs_deg':float(np.abs(tr).mean()),
                           'pooled_motion_correlation':float(np.corrcoef(pr.flatten(),np.broadcast_to(tr,pr.shape).flatten())[0,1]),
                           'motion_subsets':moving}
        details['arms'][arm]={'best':best,'hold':{p:r[p+'_hold_baseline'] for p in PANELS},'motion':motion}
    lines+=table(['模型','所选步数','首动作关节 MAE','50 帧关节 MAE','50 帧夹爪 MAE','综合分数↓'],rows)
    absolute_arms=[a for a in reports if not a.startswith('delta_')]
    improved=[a for a in absolute_arms if best_entries[a]['selection_score']<base['selection_score']]
    lines+=['结果解读：'+('绝对动作组中综合分数优于初始模型的组为 '+', '.join(improved)+'。' if improved else
             '绝对动作三组所选已训练检查点的综合分数均未优于同一初始模型；降低学习率和增加本次 q/v LoRA 的 rank 未修复验证动作误差。'), '']
    for arm in reports:
        if not arm.startswith('delta_'):continue
        initial=reports[arm]['evaluations'][0];best=best_entries[arm]
        change=(best['selection_score']/initial['selection_score']-1)*100
        beats_hold=all(best[p]['mean_errors']['50']['joint_MAE_deg']<ref[p+'_hold_baseline']['50']['joint_MAE_deg'] for p in PANELS)
        lines += [f'{arm} 的综合分数相对自己的零步变化 {change:+.2f}%；'
                  +('两个面板的 50 帧关节误差均低于保持状态基线。' if beats_hold else '未同时优于两个面板的保持状态基线，逐面板结果见上表。')
                  +'必须同时检查运动段、夹爪及扣除起点后的状态敏感性，不能把机械加回起点的收益归为学会运动。', '']
    lines+=['保持当前状态在大量平稳样本上很强，因此增量动作输出接近起点本身不是学习成功。需要结合残差变化、相对零步改善、运动段及夹爪指标判断。增量表示改变了目标分布和归一化，Flow loss 不能与绝对表示直接横向比较。', '', '## 各步结果', '']
    rows=[]
    for arm,r in reports.items():
        for e in r['evaluations']:
            rows.append([arm,e['step'],pair(e,50,'joint_MAE_deg'),pair(e,50,'gripper_MAE_0_1'),f'{e["selection_score"]:.6f}'])
    lines+=table(['组','步数','50 帧关节 MAE','50 帧夹爪 MAE','综合分数↓'],rows)
    lines+=['## 分阶段、分关节与位置', '', '以下为各组所选检查点；未筛选面板的逐关节与按位置明细保存在 details.json 和各组 report.json。', '']
    rows=[]
    for arm,e in best_entries.items():
        for stage,v in e['validation_stages']['by_stage'].items():
            rows.append([arm,stage,f'{v["mean_errors"]["1"]["joint_MAE_deg"]:.4f}',f'{v["mean_errors"]["50"]["joint_MAE_deg"]:.4f}',f'{v["mean_errors"]["50"]["gripper_MAE_0_1"]:.4f}'])
    lines+=table(['组','阶段','首动作°','50 帧°','夹爪 MAE'],rows)
    rows=[]
    for arm,e in best_entries.items():
        rows.append([arm,*[f'{v:.3f}' for v in e['validation_stages']['h50_per_joint_MAE_deg']]])
    lines+=table(['组',*[f'J{i+1}°' for i in range(7)]],rows)
    rows=[]
    for arm,e in best_entries.items():
        for position,v in e['validation_stages']['by_position'].items():
            rows.append([arm,position,manifest['support']['train'].get(position,0),f'{v["mean_errors"]["50"]["joint_MAE_deg"]:.4f}'])
    lines+=table(['组','验证位置代理组','阶段合格训练轨迹支持数','50 帧关节 MAE°'],rows)
    lines+=['## 状态依赖探针', '', '固定图像与噪声，交换两个位置同阶段窗口的关节或袋子 xy。8 窗口×两噪声；干预会造成图像/状态不一致，只测敏感性，不证明正确使用状态。增量组必须扣除确定性的起点加回，观察残差是否随状态变化。', '']
    rows=[]
    for filename in ['state_probe.json','delta_state_probe.json']:
        if not (root/filename).exists(): continue
        probe=read(root/filename)
        if probe.get('status')!='PASS': raise RuntimeError('State probe incomplete')
        for arm,effects in probe['models'].items():
            for kind,v in effects.items():
                rows.append([arm,kind,f'{v["input_mean_abs"]:.5f}',f'{v["output_joint_change_deg"]:.4f}',f'{v["action_minus_start_joint_change_deg"]:.4f}',f'{v["output_gripper_change"]:.5f}'])
    lines+=table(['模型','交换输入','输入差(rad/图像单位)','绝对输出关节变化°','扣除起点后变化°','夹爪变化'],rows)
    if (root/'delta_state_probe.json').exists():
        dp=read(root/'delta_state_probe.json')['models']
        if 'delta_base' in dp and 'delta_r8_lr5e5' in dp:
            before=dp['delta_base']['joints']['action_minus_start_joint_change_deg']
            after=dp['delta_r8_lr5e5']['joints']['action_minus_start_joint_change_deg']
            lines += [f'增量模型扣除起点后的关节残差敏感性：零步 {before:.4f}° → 400 步 {after:.4f}°。'
                      '绝对输出随起点近等量移动主要来自确定性还原；这项不一致输入探针未显示学习到的残差对关节输入更敏感，也不能据此单独认定状态通道是唯一根因。', '']
    lines+=['## 动作幅度与夹爪转换', '']
    rows=[]
    for arm,d in details['arms'].items():
        for panel,v in d['motion'].items():
            e=d['best'][panel]
            rows.append([arm,panel,f'{v["initial_motion_mean_abs_deg"]:.4f}',f'{v["predicted_motion_mean_abs_deg"]:.4f}',f'{v["target_motion_mean_abs_deg"]:.4f}',f'{v["initial_pooled_motion_correlation"]:.4f}',f'{v["pooled_motion_correlation"]:.4f}',e['gripper_transition_samples'],e.get('gripper_transition_direction_accuracy','N/A')])
    lines+=table(['组','面板','初始预测幅度°','所选预测幅度°','目标幅度°','初始残差相关','所选残差相关','明确开合窗口数','净变化方向准确率'],rows)
    lines+=['夹爪净变化方向仅在目标首尾差 ≥0.25 的窗口上统计，不是转换时刻误差，不代表抓取/释放成功。未通过此项修正控制命令时序。', '',
      '按真实目标相对当前关节的 50 帧平均幅度筛选运动窗口，下面阈值为离线描述性指标，没有据此选检查点。', '']
    rows=[]
    for arm,d in details['arms'].items():
        for panel,v in d['motion'].items():
            for threshold,s in v['motion_subsets'].items():
                rows.append([arm,panel,threshold,s['windows'],s['hold_MAE_deg'],s['initial_model_MAE_deg'],s['model_MAE_deg']])
    lines+=table(['组','面板','目标幅度阈值°','窗口数','保持状态 MAE°','初始模型 MAE°','所选模型 MAE°'],rows)
    lines+=['## 工程验证与后续判断', '']
    for arm,r in reports.items():
        lines.append(f'- {arm}: {len(r["training"])} 步；可训练 {r["trainable_parameters"]:,} 参数；全新重载动作最大差 {r["reload_action_max_abs_difference"]}；CUDA allocated 峰值 {r["cuda_peak_allocated_GB"]:.2f} GB、训练 RSS 峰值 {r["max_rss_GB"]:.2f} GB。')
    lines+=['', '- 所有已保存数组均有限；详见 details.json。检查点未保存 optimizer/RNG 状态，不能精确断点续训。',
      '- 增量实验独立类、训练集统计、训练/推理成对变换，夹爪保持绝对值；检查点设专用 type 和 delta.json，普通加载器会拒绝。仅供离线实验。',
      '- 这批对照用于判断设计假设；不能凭降低 loss、单种子或 400 步负结果断言 π0.5/LoRA 整体无效。也不能因为增量表示机械加回起点就认定状态依赖已解决。',
      '- 若所选增量检查点仍不优于保持状态或没有运动/夹爪改善，应先做运动段独立评估与训练集多轨迹可拟合性检查，再决定引入结构化状态条件或扩大可训练模块。不要直接长训或上线。', '',
      '## 设计判断与下一轮建议', '',
      '- 当前证据不支持把问题仅归于学习率或本次 q/v LoRA rank；降低学习率与 rank16 都没有在固定验证综合分数上超过初始模型。尚未检验其他注入位置、全量 expert 或更长收敛训练，不能把本轮负结果扩大为 π0.5/LoRA 整体无效。',
      '- 增量表示降低了初始的绝对位置负担，400 步进一步改善自身零步指标，但合并运动残差相关仍接近零，运动子集也没有超过保持状态。当前更接近“动作误差减小、条件运动学习尚未证实”，而非已学会抓取。',
      '- 下一轮先从训练集中固定 8–16 条跨位置轨迹，覆盖接近、夹取、搬运、释放的明确运动窗口，验证同一模型能否拟合这些轨迹的非零残差及夹爪变化。保留整条验证轨迹，另留未参与选模的测试集。先约定有界步数、相同窗口/噪声与检查点规则；不得以训练 loss 代替动作指标。',
      '- 在可拟合性检查中并列记录首动作/分关节误差、50 帧误差、相对保持状态的增益、预测与目标残差相关、夹爪转换命中与时差；仅有 2/4 个大运动窗口的现有子集不够支持泛化结论。',
      '- 若仍无法拟合多轨迹条件运动，再设计连续状态投影/状态 token 与当前文本化状态的受控对照；改变结构时应单独验证初始化和加载兼容性。不要直接认定结构改造一定有效。',
      '- 实体零位、方向、外参与真实控制时序仍需后续单独确认；本轮只有离线数据和模型证据，无机器人访问、使能、运动或闭环成功率。', '',
      '代码：tools/pi05_position_manifest.py、tools/pi05_raw_alignment_audit.py、tools/pi05_numerical_state_probe.py、tools/pi05_position_comparison.py、tools/pi05_comparison_state_probe.py、tools/pi05_delta_experiment.py。', '']
    (root/'details.json').write_text(json.dumps(details,indent=2,ensure_ascii=False),encoding='utf-8')
    (root/'结果报告.md').write_text(chr(10).join(lines),encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str((root/'plot_cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Hold','Absolute base']+list(reports)
    delta_arms=[a for a in reports if a.startswith('delta_')]
    labels += [a+' initial' for a in delta_arms]
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,panel,title in zip(axes,PANELS,['Stage panel (44 windows)','Unfiltered panel (16 windows)']):
        values=[ref[panel+'_hold_baseline']['50']['joint_MAE_deg'],base[panel]['mean_errors']['50']['joint_MAE_deg']]
        values += [best_entries[a][panel]['mean_errors']['50']['joint_MAE_deg'] for a in reports]
        values += [reports[a]['evaluations'][0][panel]['mean_errors']['50']['joint_MAE_deg'] for a in delta_arms]
        colors=['#747d87','#a0aab5']+['#2f8579' if a.startswith('delta_') else '#426c98' for a in reports]+['#a1c4bd']*len(delta_arms)
        bars=ax.barh(labels,values,color=colors)
        ax.set_xlim(0,max(values)*1.17);ax.invert_yaxis();ax.set_title(title)
        ax.set_xlabel('Mean joint absolute error over 50 frames (degrees)')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for bar,value in zip(bars,values):ax.text(value+.12,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center',fontsize=9)
        for side in ['top','right']:ax.spines[side].set_visible(False)
    fig.suptitle('Bounded offline comparison: selected trained checkpoints; lower is better')
    fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)
    print('POSITION_REPORT_PASS',json.dumps(details['finite_arrays']))

if __name__=='__main__':main()
