# NERO Eye-in-Hand 自动数据清洗设计

## 目标

从红色和蓝色眼在手原始数据中筛出真正完成“抓取并放入棕色纸袋”的完整 Episode，供 ACT 重新训练。

原始数据必须只读。任何清洗结果都写入新的目录，不覆盖：

```text
outputs/eye_in_hand_red_cube_20261016
outputs/eye_in_hand_blue_cube_20261016
outputs/eye_in_hand_red_cube_v2
outputs/eye_in_hand_blue_cube_v1
```

当前原始 Episode 的 `metadata.result=auto_end` 只表示录制时间结束，不表示任务成功，不能作为成功标签。

## 推荐流程

```text
原始 Episode
    -> 结构完整性检查
    -> 视觉模型查看完整时序
    -> 关节/夹爪轨迹规则检查
    -> AI 生成候选标签和证据
    -> 低置信度及冲突样本人工复核
    -> 仅导出 success Episode
    -> 按 Episode 划分 train/val
    -> 重新计算归一化统计
    -> 离线抽样验收
```

不允许只看首帧、末帧或固定几张图片判定成功。必须查看整个 Episode 的时序，至少覆盖：恢复点、接近物块、下降、夹爪闭合、抬升、移动到袋子、释放、结束。

## 输入契约

每个 Episode 目录应包含：

```text
episode_NNNN/
  metadata.json
  timestamps.jsonl
  robot_state.jsonl
  actions.jsonl
  frames/frame_XXXXXX.png
```

每帧需要按相同行号配对：

- `timestamps.jsonl`：图像路径、`host_time_s`、图像中的物体标注
- `robot_state.jsonl`：7 个关节和 `gripper`
- `actions.jsonl`：7 个关节目标和 `gripper.pos`
- 图像：640x480 RGB，腕部相机视角

目标颜色必须与数据集一致：红色数据只接受红色方块，蓝色数据只接受蓝色方块。棕色袋子不是目标物块。

## 标签定义

AI 必须为每个 Episode 输出以下四选一标签：

```text
success
failure
uncertain
invalid
```

### success

只有以下条件全部满足才可标记 `success`：

1. 目标颜色物块在过程中被清楚识别。
2. 机械臂接近的是目标物块，而不是袋子、夹具或背景物。
3. 下降时夹爪从物块两侧接近，没有明显插入桌面或撞击桌面。
4. 夹爪在物块附近闭合，并且物块被夹爪带离桌面。
5. 抬升后物块仍被夹住，没有中途掉落。
6. 机械臂移动到棕色纸袋上方或袋口附近。
7. 夹爪打开后物块进入袋内或明确落入袋口。
8. 结束时物块不再留在原桌面位置，且没有明显碰撞或人工介入。

### failure

以下任一项成立即可标记 `failure`：

- 没有抓到目标物块。
- 夹爪夹空、夹偏或夹到目标以外的物体。
- 物块被夹住但抬升前掉落。
- 机械臂插入桌面、明显碰撞、卡住或越过安全区域。
- 目标物块仍在桌面上，任务未完成。
- 抓到物块但没有放入袋子。
- 释放在袋子外，或释放后物块明显掉在桌面上。
- 出现人工干预、重新摆放物块、手进入工作区或人为修改动作。
- 相机严重花屏、旧帧、遮挡或无法判断关键动作。

### uncertain

当关键动作被遮挡、图像质量不足、袋口被遮挡、无法确认是否抬升或无法确认是否入袋时，必须标记 `uncertain`，不能为了提高成功数量而猜测成功。

### invalid

结构或数据本身不可用：

- 三个 JSONL 行数不一致。
- 图像缺失、损坏、分辨率错误或无法解码。
- 时间戳不递增或状态/动作存在 NaN、Inf。
- 关节维度不是 8，夹爪值超出 `[0, 1]`。
- 颜色数据集和 Episode 内目标颜色不一致。

`invalid` 不进入训练集，也不要与任务失败混为一谈。

## 视觉审核要求

AI 应把 Episode 读取为完整视频或按时间均匀采样至少 24 帧；当检测到动作变化、夹爪变化、物块可见性变化或袋子可见性变化时，必须额外采样邻近帧。

每个 Episode 至少输出以下阶段判断：

```text
object_visible
approach_ok
pregrasp_ok
descent_ok
grasp_closed
object_lifted
object_retained
bag_visible
release_ok
final_success
```

每个字段取值为：

```text
true / false / unknown
```

判定规则：

- `grasp_closed` 不能只根据夹爪数值判断，必须结合图像确认夹爪实际闭合。
- `object_lifted` 必须比较闭合前后物块与桌面的相对位置。
- `object_retained` 必须检查抬升和水平移动阶段，不允许只看闭合瞬间。
- `release_ok` 必须看到打开后物块在袋口内或袋内，单纯靠近袋子不算成功。
- 如果关键阶段不可见，填 `unknown`，最终不得为 `success`。

## 轨迹信号检查

视觉结果必须和运动数据交叉验证。AI 或辅助脚本至少计算：

```text
state_action_max_abs_error
max_joint_step
max_joint_velocity_estimate
gripper_min
gripper_max
gripper_transition_count
gripper_close_time
gripper_open_time
timestamp_monotonic
frame_count
duration_s
```

这些数值只用于发现异常，不能单独证明任务成功。

异常提示：

- 夹爪在接近阶段多次开合：优先 `failure` 或 `uncertain`，等待视觉确认。
- 夹爪在物块尚未对准时闭合：不能标记成功。
- 关节目标持续大幅变化但末端没有有效接近：标记失败风险。
- 末端明显下降到桌面高度以下或发生接触：标记 `failure`。
- 录制到 20 秒但没有完整抓放：仍然是 `failure`，不能使用 `auto_end` 推断成功。

## 自动判定策略

建议使用两级输出，而不是强制 AI 对所有样本做二元判断：

```text
confidence >= 0.90 且所有硬条件满足 -> success_candidate
confidence >= 0.90 且任一失败硬条件满足 -> failure_candidate
其余 -> uncertain
```

只有 `success_candidate` 才能进入自动清洗后的训练集。`failure_candidate`、`uncertain`、`invalid` 全部排除；`uncertain` 单独保存供人工复核。

### 严格成功门槛

以下字段必须全部为 `true`：

```text
object_visible
approach_ok
pregrasp_ok
descent_ok
grasp_closed
object_lifted
object_retained
bag_visible
release_ok
final_success
```

任何 `false` 或 `unknown` 都不能进入 `success` 集合。

## 输出文件

### 1. 审核结果 CSV

文件：

```text
outputs/eye_in_hand_red_review_ai.csv
outputs/eye_in_hand_blue_review_ai.csv
```

字段：

```text
dataset,episode,label,confidence,reason,review_version
```

`label` 必须是 `success/failure/uncertain/invalid` 之一。

### 2. 详细结果 JSONL

文件：

```text
outputs/eye_in_hand_red_review_ai.jsonl
outputs/eye_in_hand_blue_review_ai.jsonl
```

每行示例：

```json
{
  "dataset": "eye_in_hand_red_cube_20261016",
  "episode": "episode_0042",
  "label": "success",
  "confidence": 0.94,
  "stage": {
    "object_visible": true,
    "approach_ok": true,
    "pregrasp_ok": true,
    "descent_ok": true,
    "grasp_closed": true,
    "object_lifted": true,
    "object_retained": true,
    "bag_visible": true,
    "release_ok": true,
    "final_success": true
  },
  "evidence": {
    "approach_frame": 142,
    "grasp_frame": 238,
    "lift_frame": 276,
    "release_frame": 486,
    "final_frame": 598
  },
  "motion_metrics": {
    "frame_count": 601,
    "duration_s": 20.0,
    "gripper_transition_count": 2
  },
  "reason": "目标被夹起并在最后阶段落入棕色纸袋，未见碰撞或人工干预。",
  "review_version": "nero-eye-in-hand-v1"
}
```

### 3. 成功 Episode 清单

另外输出纯文本清单，便于脚本使用：

```text
outputs/eye_in_hand_red_success_episodes.txt
outputs/eye_in_hand_blue_success_episodes.txt
```

每行一个原始目录名，例如：

```text
episode_0003
episode_0017
episode_0042
```

## 给其他 AI 的执行提示词

将以下提示词连同本设计文档、原始 Episode 目录和对应帧图像一起提供给多模态 AI：

```text
你是机器人操作数据质量审核器。请逐条审核一个完整的 NERO 眼在手 Episode。

任务是判断“抓取指定颜色方块并放入棕色纸袋”是否完整成功。必须查看完整时序，不能只看第一帧、最后一帧或录制结果 auto_end。请依次确认：目标颜色、接近、预抓取、下降、夹爪闭合、物块离桌、抬升后保持、移动到袋口、释放入袋、最终状态。

硬规则：
1. 任何关键阶段看不清都标记 uncertain，不要猜 success。
2. 物块没有被明确抬离桌面，不能 success。
3. 只看到夹爪闭合不能证明抓住物块。
4. 只到达袋子附近不能证明放入袋子。
5. 插桌、碰撞、夹空、掉落、人工干预都标记 failure。
6. auto_end 不是 success。
7. 输出必须是严格 JSON，不要输出额外说明。

返回格式：
{
  "label": "success|failure|uncertain|invalid",
  "confidence": 0.0,
  "stage": {
    "object_visible": true,
    "approach_ok": true,
    "pregrasp_ok": true,
    "descent_ok": true,
    "grasp_closed": true,
    "object_lifted": true,
    "object_retained": true,
    "bag_visible": true,
    "release_ok": true,
    "final_success": true
  },
  "evidence": {
    "approach_frame": null,
    "grasp_frame": null,
    "lift_frame": null,
    "release_frame": null,
    "final_frame": null
  },
  "failure_type": "none|missed_object|bad_alignment|collision|dropped|bad_release|camera_bad|manual_intervention|data_invalid|unknown",
  "reason": "一句话说明判定依据"
}
```

## 数据集重建规则

1. 只使用 `label=success` 的 Episode。
2. `uncertain` 不得自动加入训练集。
3. `failure` 和 `invalid` 只保留审核结果，不复制到训练输出。
4. train/val 必须按 Episode 划分，不能按帧随机划分。
5. 红色和蓝色仍然分别建数据集，不混合颜色模型。
6. 重新转换后必须重新计算 action/state 统计，不能复用包含失败 Episode 的旧统计。
7. 输出目录必须使用新名字，例如：

```text
outputs/eye_in_hand_red_cube_success_v1
outputs/eye_in_hand_blue_cube_success_v1
```

8. 转换 manifest 必须记录：源目录、成功 Episode 清单、审核文件路径、审核版本、train/val Episode 列表和统计信息。

## 验收标准

清洗完成后，必须检查：

- 成功 Episode 数量至少为 10；少于 10 时先不要训练，继续采集或人工复核。
- 红蓝各自 train/val 都有 Episode，且不重叠。
- 所有输出 Episode 的图像、状态、动作行数一致。
- 无 NaN、Inf、损坏图像和越界夹爪值。
- 随机抽查至少 10 个 success Episode，视觉上都确实完成抓放。
- 随机抽查所有 uncertain，不得混入 success 清单。
- 训练前打印成功 Episode 列表和帧数，不接受只报告总帧数。

清洗的目的不是让数据量最大，而是让每条进入训练集的轨迹都表达同一个正确任务：安全接近、正确抓取、抬升保持、移动到袋口并可靠释放。
