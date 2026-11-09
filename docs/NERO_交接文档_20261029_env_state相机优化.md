# NERO 相机与 env-state 优化交接文档

**日期**：2026-10-29  
**项目目录**：`C:\Users\Administrator\Desktop\lerobot_nero_standalone_backup_20261019_before_changes\lerobot_nero_standalone_backup_20261019_before_changes`  
**Python**：`D:\Anaconda3\envs\lerobot\python.exe`

## 当前结论

- 已采集原始数据位于 `outputs/nero_episodes`，共 300 个 episode。原始图片、状态和动作没有删除或重写。
- 原始采集使用 OpenCV DirectShow：`wrist=index:3` 是腕部近景，`base=index:1` 是固定全景。这个映射已经从 `metadata.json` 和实际帧交叉确认。
- 按相机职责确认的 RealSense 序列号映射是：
  - `261622078506`：wrist，近距离看夹爪和方块；
  - `261622071323`：base，固定拍摄桌面、袋子和机械臂。
- 原来的 `outputs/camera_roles/wrist.png` 与 `base.png` 文件名曾经标反；它们是旧快照，不代表当前正式采集数据。新的示例命令已按上面的映射更新。
- 采集帧整体比旧快照更中性。旧 RealSense 快照有明显绿色偏色和高亮裁剪；代码现在打开 RealSense 彩色流时会尝试启用并回读自动曝光、自动白平衡，并在快照中报告中性区域 RGB。当前 Windows 会话运行 RealSense 时遇到 `SHGetKnownFolderPath ... 0x80070002`，因此还没有重新取得硬件实测白平衡结果。

## 已完成的代码调整

### 相机

- `src/lerobot/nero_tools/episode_capture.py`
  - RealSense 打开后启用并确认自动曝光、自动白平衡。
  - 保留每路相机独立读取线程和帧龄检查。
  - 默认单 RealSense 序列号改为 wrist 的 `261622078506`。
- `tools/snapshot_cameras.py`
  - 示例改为 `wrist=261622078506`、`base=261622071323`。
  - 快照会输出中心亮区域的 RGB 中位数以及绿色通道相对红蓝均值的偏差；这只是诊断，不会偷偷改变训练图片。
- `src/lerobot/nero_tools/realsense_preview.py`、`realsense_record.py`
  - 默认序列号同步为 wrist 的 `261622078506`。
- `tools/nero_dual_cam_act_deploy.py`、`tools/realsense_bandwidth_check.py` 和采集交接文档中的示例映射已同步。

### env-state relation_v2

- 新增 `src/lerobot/nero_tools/env_state_schema.py`，定义 18 维 `relation_v2`：
  - base 方块中心、可见位和检测分数代理；
  - 袋口粗略中心、可见位和面积代理；
  - 方块到袋口的图像平面相对位移；
  - 蓝球到方块的相对位移、可见位和分数代理；
  - wrist 画面中方块相对图像中心的偏差、可见位和分数代理。
- 这些坐标都是归一化图像坐标，不是毫米，也不是夹爪坐标。wrist 偏差表示“方块在腕部画面中的位置”，不能解释为“方块到夹爪的距离”。
- 缺失目标使用显式 `*_visible=0`，坐标才填 0；没有根据目标消失直接推断“已经抓起”。
- 分数字段已经命名为 `*_score_proxy`/`bag_area_proxy`，它们是检测质量代理，不是概率。
- `tools/env_state_batch.py` 新增 `--schema relation_v2`、逐帧完整性检查、原子写入、`--episode` 精确试跑和安全的 `--only-missing`。不传 `--overwrite` 时，已有但不完整的文件不会被覆盖。
- `convert_episodes.py` 支持 `--env-state-schema relation_v2`；请求 v2 但标签不存在时会直接报错，不会悄悄生成没有 env-state 的数据集。

## 当前数据状态

- `outputs/nero_episodes/env_state.jsonl`：原有 legacy 标签，300 集完整，不能删除。
- `outputs/nero_episodes/env_state_groups.csv`：已根据 300 集 legacy 标签恢复为全量摘要。之前的两集 relation_v2 试跑不会再覆盖这个全量文件。
- `episode_0000` 和 `episode_0001` 下有临时试跑文件 `env_state_relation_v2_test.jsonl`；它们只是检查用，不是正式转换输入。
- 当前全部 episode 的 metadata 都记录 `ball_color=none`，但 legacy 标签中后段有 61 集出现蓝色物体检测。这说明旧采集元数据不能作为是否有蓝球的依据；正式 v2 运行时要用 `--ball-color blue`，并抽查 overlay。
- 两集 relation_v2 试跑的 base 方块检测约为 40% 左右；方块被夹爪遮挡或已经离开 base 视野时，`visible=0` 是预期现象。正式全量生成前必须抽查有方块、接近夹取、已夹起、接近袋子和蓝球干扰的帧。

## 重新打开项目

在 CMD 中执行：

```bat
conda activate lerobot
cd /d "C:\Users\Administrator\Desktop\lerobot_nero_standalone_backup_20261019_before_changes\lerobot_nero_standalone_backup_20261019_before_changes"
set "PYTHONPATH=%CD%\src;%CD%\tools"
python -c "import sys; print(sys.executable)"
```

最后一行必须显示：`D:\Anaconda3\envs\lerobot\python.exe`。

## 只读相机检查

相机接好后，使用新输出目录，不要覆盖旧快照：

```bat
python -X utf8 tools\snapshot_cameras.py ^
  --camera-spec wrist=serial:261622078506,size:640x480,fps:30 ^
  --camera-spec base=serial:261622071323,size:640x480,fps:30 ^
  --out outputs\camera_roles_20261029_verified ^
  --warmup 4
```

然后查看：

- `wrist.png`：应是夹爪和方块近景；
- `base.png`：应是固定桌面全景。

如果出现 `SHGetKnownFolderPath ... 0x80070002`，这是当前 Windows 会话的 RealSense 初始化问题，不是机器人故障；先不要据此修改序列号。

USB 带宽只读检查：

```bat
python -X utf8 tools\realsense_bandwidth_check.py ^
  --serial-base 261622071323 ^
  --serial-wrist 261622078506 ^
  --concurrent --seconds 20
```

## v2 标签的安全试跑

先只跑有蓝球的代表集和一个无蓝球集，输出到独立文件：

```bat
python -X utf8 tools\env_state_batch.py ^
  --source outputs\nero_episodes ^
  --schema relation_v2 ^
  --output-name env_state_relation_v2_test.jsonl ^
  --review-dir relation_v2_test_review ^
  --ball-color blue ^
  --episode episode_0000 ^
  --episode episode_0224 ^
  --review-every 120 ^
  --overwrite
```

人工检查两个 episode 的 `relation_v2_test_review`：方块框应在红方块上，袋框应在袋子上，蓝球框只应在真实蓝球上。确认后再用正式文件名生成全量 v2；不要直接把 v2 写入旧 `env_state.jsonl`。

## 正式转换前的原则

1. 先完成 v2 overlay 抽查，再生成全量 `env_state_relation_v2.jsonl`。
2. 转换使用单独输出目录，例如 `outputs/red_cube_300_relation_v2_video`，不覆盖旧 video 数据集。
3. 固定随机种子和 train/val 划分，保证三组对照只改变输入：
   - 图像 + 关节状态；
   - env-state + 关节状态；
   - 图像 + env-state + 关节状态。
4. 第一轮 env-state dropout 设为 0；确认模型确实读取后，再测试 0.1 或 0.2。
5. 验收要做 env-state 清零、跨场景替换和按方块/袋子位置分组的整段动作误差，不能只看首帧误差或训练 loss。

## 未完成事项

- 尚未重新取得两台 RealSense 的实时快照，自动曝光/白平衡只能确认代码路径，不能宣称硬件实测通过。
- relation_v2 仍是归一化图像关系；还没有桌面标定点，因此不能输出真实工作平面毫米坐标。
- 还没有可靠的“方块已抓起”标签。方块暂时不可见可能是遮挡、出视野或进入袋子，不能仅靠 `visible=0` 判定抓取成功。
- 尚未用 relation_v2 生成正式 LeRobot 视频数据集，也尚未开始三组对照训练。
- 旧的两个 `episode_capture` 自动化测试仍有 2 个测试假设与当前 Q 键“丢弃当前未完成 episode”的行为不一致；这不影响已采集数据，但后续应更新测试期望。

## 关键文件

- [episode_capture.py](../src/lerobot/nero_tools/episode_capture.py)
- [env_state_schema.py](../src/lerobot/nero_tools/env_state_schema.py)
- [env_state_batch.py](../tools/env_state_batch.py)
- [convert_episodes.py](../src/lerobot/nero_tools/convert_episodes.py)
- [snapshot_cameras.py](../tools/snapshot_cameras.py)
- [realsense_bandwidth_check.py](../tools/realsense_bandwidth_check.py)
