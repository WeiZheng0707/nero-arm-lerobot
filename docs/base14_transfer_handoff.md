# 数据转移交接：停在这里，由用户操作

本次没有搬移原始数据、没有运行正式 LeRobot 转换、没有训练或操作机械臂。
“转移文件”与“转换为视频数据集”是两步。用户可先搬移原始数据到处理位置，再生成全量标签及MP4；也可在原电脑处理后再搬运视频数据集。当前交付不包含已转换的训练视频数据集。

## 已准备

- 原始数据：outputs/nero_episodes，300条，179857个记录行，两路图片。
- 全量数值/时间审计：outputs/base14_data_audit.json、base14_audit_summary.json。
- 前10条标签审核：outputs/base14_review_v2。
- 分散10条扩展审核：outputs/base14_spread_v2。覆盖早中晚采集和蓝球干扰样本。
- 腕部连续测试：outputs/wrist_sequence_review。
- 3条双机位回放：outputs/grasp_pair_review。回放视频是审核材料，不是LeRobot训练视频。
- 代码及报告包：outputs/base14_handoff/code_and_reports.zip。它是同版本项目的代码补丁包，不是独立安装包。
- 文件清单：outputs/base14_handoff/source_inventory.json。图片检查路径/大小，4类数值元数据检查SHA256。
- 审核空表：outputs/base14_handoff/episode_review.csv。不是转换器的success CSV，所有条目默认UNREVIEWED，不可直接当作成功训练清单。

## 新发现与配置状态

旧 configs/base14_review_initial.json 的袋子ROI只适用于前部样本，扩展抽查发现4/10条袋子整段不可见，不能作为全量配置。
新增 configs/base14_spread_candidate.json 扩大base ROI并取消袋子专用小ROI。扩展10条中9条袋子全程有检测，第30条537/600帧有检测。审核图中的袋子候选位置合理，但不等于逐帧准确率通过。
新配置仍标为candidate；不要混用不同配置哈希的标签。转移后需用同一冻结配置重新生成所选全部episode的标签，再审核异常。
当前只有抽查标签，没有准备完整300条新标签，因此不能直接正式转换、训练。

## 用户转移内容

如果迁移到另一台机器，建议复制整个同版本项目（包含src、tools、configs、docs和依赖说明）以及 outputs/nero_episodes，保留原目录结构；还需复制上述审核结果和 base14_handoff。
如果只是换数据磁盘，可只复制整个 nero_episodes 目录和标签/报告，代码继续使用当前项目。不要合并到旧 relation_v2 输出目录，不删除源文件。
目标机器路径和可用空间尚未知，因此未生成带真实目标盘的搬运命令，避免写错位置。

## 转移后核对

若保留项目结构，运行：

    python tools/verify_base14_transfer.py --inventory INVENTORY --project-root DEST_PROJECT

该检查核对全部图片路径/大小及数值文件SHA256，不声称图片字节级完整校验。文件复制工具还应提供自身校验。
随后按以下顺序继续：

1. 对新候选配置审核，确定一致参数并生成全量独立标签。
2. 审核抓取/抬起/放置及失败样本，不能把闭合或auto_end当成功。
3. 使用两路视频映射、--env-state-schema base14、--labels-root、--video --video-codec h264 --video-crf 20 --video-backend pyav，输出到全新目录。
4. 回读、全量解码检查及A/B小训练。保留完整轨迹，未经时刻审核不沿用480帧截断。

本轮33项离线测试通过；没有新模型成功率结果。当前到达文件转移交接点，标签全量验收与训练仍是后续工作。
