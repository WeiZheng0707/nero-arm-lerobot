# base14 第一阶段实现与交接

## 旧方案失败记录与本次验收约束（2026-11-01 补充）

用户反馈旧 relation_v2 模型反复乱抓、抓不到；不能仅换成 base14 后重复长训练。
用户粘贴的旧消融结果为 0.0%、0.084%、0.02%，目前是历史记录，尚未独立复测旧 checkpoint。
视频编码只解决存储问题，不能保证模型利用 env-state。保留 H.264 视频形式。

已核对的风险：

- 旧标签命令只有 ball-roi，没有 base-roi；旧部署命令设置 base-roi。按代码默认行为，两端目标搜索范围不一致。
- 当前 relation_v2 离线 wrist 检测 exclude_top_ratio=0，部署为 0.28；标签与实时输入可能不同。
- 旧诊断脚本把关节弧度和归一化夹爪混合求均值，再除以绝对 action 大小，并用固定 5% 判断使用与否；这不是可靠的抓取验收标准。
- 旧诊断只抽取前部稀疏帧、未排除 action_is_pad，并用评估样本自身均值作常量基线；不能据此确认泛化。
- 整个轨迹中关节状态可能编码任务阶段，但这仍是待检验假设，不是已证实的唯一根因。
- 旧文档“只有 wrist 两个量随运动变化”不准确；base 坐标、尺寸和可见性也可能随物体移动、遮挡变化。

本次推进条件：

1. 转换前：对未遮挡接近阶段、夹持阶段、入袋阶段分别审核标签；不以全段可见率代替准确率。
2. 数据检查：核实相近初始关节姿态下是否存在不同目标位置和对应不同正确动作；检查 state/action 差值、夹爪闭合时刻、相机实际采集时间与源行对齐。
3. 不照搬 --max-frames 480：先审核各 episode 实际完成时刻，防止裁掉放置或保留无用等待。
4. 先短程 A/B：相同划分、seed、步数，A=wrist+state，B=wrist+state+base14；标签确认前不进行正式训练。
5. 验证集按 episode 和目标位置均衡抽样，排除 action_is_pad；分别报告各关节弧度误差、夹爪误差、整段未来动作与保持当前姿态基线。
6. env 测试同时做零值、有效标签置换和物理一致的左右目标扰动；移动 cube 时同步更新相对位移。保持 wrist/state 不变只是敏感性诊断，可能制造不一致观察，不能独立证明正确控制。
7. 判断“有用”：正确 env 的预测误差应优于置换 env；B 在目标位置变化样本上应优于 A。单纯输出有变化、梯度非零、loss 降低都不足以通过。
8. 不通过则暂停真机部署和长训练，先定位标签、数据覆盖或输入路径；不能靠增加 dropout、扩大图像或堆训练步数当作已解决。

以上为下一阶段必须补齐的评估，不宣称当前 27 项工程测试已经验证模型理解目标，也不承诺无需增加数据就一定成功。

用户要求：视频格式；真实数据集转移由用户执行。当前不移动原始数据、不批量转换、不训练、不运行硬件。

## 可行性与限制

现有 ACT 支持 ENV token。新增 base14 无需修改网络主体，但旧 legacy14 同为 14 维，必须核验字段名和 schema。
base14 使用袋子外框中心，不能声称它是精确袋口。面积定义为 HSV 轮廓面积/整张 base 图像面积。
方块进入袋子或被夹爪遮挡后，base 图像无法提供其真实位置。不能为了达到 90% 可见率而无限缓存。
短暂沿用保持 visible=1，并在 diagnostics 标为 carried；超过 3 帧或 0.34 秒清零。
离线与部署共用 Base14Detector，但不同采样频率仍会影响按帧计数的过滤；训练与部署应使用相近相机采样率。
首次试验配置只审核了前 10 条，不代表覆盖全部 300 条的摆放位置。矩形 ROI 也不能完全剔除斜桌边之外的背景。

## 文件与流程

- src/lerobot/nero_tools/base14.py：硬 ROI、颜色分割、形态学、独立袋子筛选、时序过滤。
- src/lerobot/nero_tools/env_state_schema.py：固定字段顺序、缺失和相对坐标校验。
- tools/label_base14.py：读取原始 timestamps 和 base PNG，输出独立标签目录、审核图、质量报告；不改原始图片。
- configs/base14_review_initial.json：可调整初始参数，不是已验收标定。
- tools/nero_dual_cam_act_deploy.py：增加 base14 选择，校验 checkpoint schema、字段名、检测参数哈希；未做真机测试。

标签输出：LABEL_ROOT/episode_xxxx/env_state_base14.jsonl、base14_report.json、review_base14/。
每个 episode 重置过滤器。报告包含 detected、carried、missing、跳变拒绝、边缘拒绝和最长连续缺失；误检率需人工审核。

## 视频转换（用户转移数据后再执行）

现有转换器新增 --env-state-schema base14 和 --labels-root。base14 必须显式 --video，默认标签名 env_state_base14.jsonl。
应使用两路映射：

    --image-key wrist=observation.images.wrist
    --image-key base=observation.images.base
    --env-state-schema base14 --labels-root LABEL_ROOT
    --video --video-codec h264 --video-crf 20

结合原转换器的 --source、--output 和人工演示审核参数执行。不要伪造 --visual-reviewed 或任务成功确认。
输出 wrist/base MP4，state/action/base14 写入 Parquet。原始数据保持不变。
转换核对标签帧号、原始时间戳、14 维值和配置哈希；重采样时数值与图像使用同一 source_row。
已有转换器会重新加载首末帧；新增 base14 转换完成后全量解码视频并核对每个 episode 的两路帧数。此检查代码尚待真实转换执行验证。
尚未运行真实 LeRobot 数据集端到端转换；已用小型合成图片实测 H.264 编码和逐帧解码。

## 后续 A/B 训练配置

A 的 policy.input_feature_keys：observation.images.wrist、observation.state。
B 的 policy.input_feature_keys：observation.images.wrist、observation.state、observation.environment_state。
B 设置 policy.environment_state_schema=base14。工厂从 conversion_manifest.json 读取检测配置哈希并写入 checkpoint 配置。
ENV 可选 MEAN_STD；A/B 需显式保持相同数据划分、seed、训练步数及其余训练超参数，统计量仅来自训练集。
数据集保留 base 视频不等于模型使用 base：input_feature_keys 控制实际模型输入。
部署 --env-schema base14 --base14-config 必须匹配训练时配置；旧 14 维 checkpoint 会被拒绝。

## 当前边界

先审核标签质量，再由用户转移数据，接着正式视频转换与全量对齐检查，然后训练 A/B。
未训练新模型，未证明 env-state 能改善任务成功率，未做新模型消融或真机验收。

2026-11-01：27 项离线测试通过，涵盖旧标签回归、schema、时序过滤、标签对齐和真实 H.264 编解码。
第二轮前 10 条共 5996 帧，cube visible 2739（45.68%）、bag visible 5977（99.68%），包含短期 carried。
这些是检测输出比例，不是准确率；方块入袋后不可见，其他漏检仍需逐图检查。
审核目录 outputs/base14_review_v2，第一轮保留 outputs/base14_review_trial 便于比较。
