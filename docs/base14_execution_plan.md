# ACT base14：离线结果与执行方案

日期：2026-11-01。目标是利用 base 的目标位置改善抓取，不把重新编码视频或 loss 降低当成成功。

## 已完成与证据边界

已实现独立 base14 schema、共享检测和时序过滤、独立标签输出、审核图、视频转换接入、ACT 输入选择及部署 schema/配置检查。
新增 audit_base14_data.py、summarize_base14_audit.py、evaluate_base14_policy.py，补齐数据统计与模型评估工具。
30 项离线测试通过，包括缺失/跳变、标签错帧、旧版兼容、H.264 编解码、评估补齐帧排除及单位分离。
未转移原始数据、未执行正式数据集转换、未训练 base14 checkpoint、未运行真机。模型评估工具的数值逻辑已测试，尚无新 checkpoint 可做端到端效果评估。

全量原始数值/时间记录审计：300 条、179857 帧，state/action 有限，帧号连续、主机时间递增，夹爪范围无越界。不是全量图片内容或任务成功审计。

| 实测项目 | 结果 | 含义 |
|---|---|---|
| state/action/记录行时间差 | 最大 0 ms | 使用共同记录时间，不能证明传感器同步 |
| 相机配对时间差 | 最大约 78 ms；各 episode P95 的中位数约 32 ms | 运动阶段需专门核对，不宜盲目增加固定偏移 |
| base 帧龄 | 各 episode P95 的中位数约 32 ms | env 使用的画面可能早于当前动作 |
| 各关节 action-state 平均绝对差 | 约 0.0013～0.0055 rad | 保持姿态基线可能很强，必须评估未来动作 |
| 480 帧后仍有单步关节变化 >0.02 rad | 217/300 条 | 尾段不是统一静止；不能直接判断是有效放置还是回位 |
| 前 10 条初始方块中心范围 | x 0.349～0.527，y 0.506～0.724 | 有位置变化，但不能代表全量覆盖 |
| 前 10 条相近初始姿态、不同目标位置的配对 | 10 对（姿态 RMS<0.05 rad，图像距离>0.05） | 具有初步对照样本，不证明动作标签足够或任务成功 |

10 条标签试跑共 5996 帧：cube 可见 45.68%，bag 可见 99.68%，含短暂 carried。
按时间四等分，cube 可见率约 99.33%、27.27%、51.13%、4.89%。时间四等分不是人工确认的抓取阶段。首段检测更完整，末段大量不可见与遮挡/入袋相符，仍需审核误检。
审核总览：outputs/base14_review_v2/contact_sheet.jpg。详细证据：outputs/base14_data_audit.json、outputs/base14_audit_summary.json。

## 推荐路径

1. 标签与时序审核。保留当前原始数据和旧模型。冻结 base14 字段；对更多位置、不同袋子位置、干扰物以及配对误差大的样本做审核。记录接近、夹持、入袋、回位时刻；实际看见时才统计检测召回。袋子框中心只作为全局条件，不当作精确袋口或三维落点。
2. 冻结检测参数。保存 JSON 及其哈希。只有审核后才能全量标签生成。沿用不超过 3 帧且不超过 0.34 秒，超限清零。不能为了可见率提高缓存时间。前 10 条参数不宣称适用于全量。
3. 用户转移数据。由用户决定目标磁盘/训练机并执行转移。保留原始数据，传递已审核标签、配置、代码和审核记录。不自动移动或覆盖旧数据集。
4. 转换新版本。H.264 MP4 保留 wrist/base 原图，Parquet 保存 state/action/base14；训练只选 wrist。默认保留完整轨迹，不照搬 max-frames=480；有人工阶段标记后再裁剪，并对 A/B 使用同一裁剪。标签与图像使用同一重采样源行，回读并全量解码核对帧数。建议 pyav 解码。
5. 首轮 A/B 小训练。A=wrist+state，B=wrist+state+base14。同一 train/val episode 划分、seed、批量、优化器、chunk_size 和训练步数。建议先在 2k、5k steps 评估，按趋势决定是否继续；这些是实验检查点，不是保证收敛的步数。先不叠加 image/env dropout，不改动作平滑。ENV 归一化策略显式记录且只用训练集统计。
6. 离线决策。运行 evaluate_base14_policy.py，两组使用完全相同的验证集与抽样。输出各关节 rad MAE、夹爪 MAE、各预测步误差、按 episode/目标横向区间统计，以及保持姿态基线。补齐帧不计入。对 B 做零 env、真实标签置换、左右移动 cube 且同步更新相对位移的敏感性分析。模型输出变化不代表方向正确；置换图像不一致的 env 只能作诊断证据。
7. 是否继续。B 应在目标位置变化的未见 episode 上稳定优于 A，正确 env 优于置换 env，未来动作优于保持姿态基线，夹爪闭合时序不恶化。检查多个 episode 的一致性，不能以固定“5%动作变化”自动批准。若证据弱，先排查数据/标签/输入，再考虑残差动作、辅助目标损失或补采数据；这些不并入第一轮以免无法归因。
8. 真机阶段。仅在离线证据充分后另行开始，先只读识别设备，再空载支撑确认、低速测试并逐关节限速。记录指令截断比例、目标误差、夹爪时序。退出关节与末端分别失能并用推进反馈验证。当前不会启动这一阶段。

## 操作入口（准备好后执行）

标签：tools/label_base14.py --source RAW --output LABELS --config CONFIG --max-episodes 10 --review-every 30。
全量处理需将 max-episodes 显式改成实际条数；先审核，输出目录必须是新目录。

视频转换：python -m lerobot.nero_tools.convert_episodes，使用 --source RAW --output NEW_OUTPUT、两路 --image-key 映射、--labels-root LABELS --env-state-schema base14 --video --video-codec h264 --video-crf 20 --video-backend pyav。人工成功审核参数必须如实填写。

A 配置 policy.input_feature_keys=[observation.images.wrist, observation.state]。
B 再加入 observation.environment_state，policy.environment_state_schema=base14；检测哈希由转换清单写入 checkpoint 配置。

评估：tools/evaluate_base14_policy.py --checkpoint CHECKPOINT --root VALIDATION_ROOT --repo-id REPO_ID --output REPORT.json --samples-per-episode 8。

当前交付为代码、离线审计与实验方案。最终抓取成功率仍需新数据集转换、训练对照和真机验收，不作“已经解决乱抓”的承诺。
