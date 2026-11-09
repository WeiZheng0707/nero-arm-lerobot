# NERO LeRobot 项目重置交接文档

最后更新：2026-10-07。电脑重置后请先读本文件，再读项目内 `HANDOFF.md`、`AGENTS.md` 和 `outputs/green_cube_lerobot/README.md`。

## 项目与目标

项目目录：`C:\Users\XIOT\Desktop\test\lerobot_nero_standalone`

目标：用 LeRobot 控制松灵 NERO 七轴双臂主从遥操作，并用固定外部相机采集绿色方块等抓取任务数据。

原项目：`C:\Users\XIOT\Desktop\test\测试版lerobot_piper2-main`

历史完整备份：`C:\Users\XIOT\Desktop\test\lerobot_full_backup_20260922_145610`

## 已确认硬件

最后确认的映射（USB 插拔后必须重新确认，不能永久假定）：

| 角色 | CAN 通道 | 固件 | SDK |
|---|---:|---|---|
| 主臂 leader | 0 | 1.20 | v120 |
| 从臂 follower | 1 | 1.21 | v121 |

两台臂使用两个独立 USB-CAN 适配器，CAN bitrate 为 `1000000`。夹具软件量程：主臂 100 mm，从臂 70 mm，遥操作映射为主臂 0~100 mm 到从臂 0~70 mm。夹具当前常用力度是 1.0 N；采集器支持 `--gripper-force`，建议一次只增加 0.5 N。之前夹具 prepare/calibrate 有反馈未确认，未完成硬件零位标定，不要盲目重复写入标定。

所有运动、使能、归位和采集前，两臂必须可靠支撑、夹具空载、工作区清空、实体急停可用。退出时七轴和夹具必须分别确认失能，不能只看 `NERO disconnected`。

## 当前软件环境

当前环境名：`lerobot`。当前解释器：`D:\Users\XIOT\miniconda3\envs\lerobot\python.exe`。

已验证：Python 3.12.14、pyAgxArm 1.0.0（Git `841a625f5f4920e776f20b934eb13048b747e6d0`）、python-can-agx-cando 0.0.1（Git `b222c4027ad4f6599f7634c72c67184619177972`）、python-can 4.6.1、numpy 2.2.6、draccus 0.10.0、pytest 9.1.1、torch 2.8.0+cu128、torchvision 0.23.0+cu128、torchaudio 2.8.0+cu128、torchcodec 0.11.1。

重置后：

```cmd
conda create -n lerobot python=3.12.14 -y
conda activate lerobot
cd /d "C:\Users\XIOT\Desktop\test\lerobot_nero_standalone"
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pip install python-can==4.6.1
python -m pip install --no-deps "git+https://github.com/agilexrobotics/python-can-agx-cando.git@b222c4027ad4f6599f7634c72c67184619177972"
python -m pip install --no-deps "git+https://github.com/agilexrobotics/pyAgxArm.git@841a625f5f4920e776f20b934eb13048b747e6d0"
set PYTHONPATH=C:\Users\XIOT\Desktop\test\lerobot_nero_standalone\src
python -c "import pyAgxArm, can, draccus, torch; print('dependencies ok')"
```

如果需要当前 CUDA PyTorch：

```cmd
python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install torchcodec==0.11.1
```

检查 CAN 枚举：

```cmd
python -c "import can; print(can.detect_available_configs(interfaces=['agx_cando']))"
```

## 代码结构

核心硬件：`src/lerobot/motors/nero/nero.py`。主从层：`src/lerobot/robots/nero_follower/`、`src/lerobot/teleoperators/nero_leader/`。正式遥操作：`src/lerobot/nero_teleoperate.py`、`examples/nero/teleoperate.py`。采集器：`src/lerobot/nero_tools/episode_capture.py`。数据转换：`src/lerobot/nero_tools/convert_episodes.py`。测试：`src/lerobot/motors/nero/test_teleoperation.py`、`src/lerobot/robots/nero_follower/test_nero_follower.py`、`src/lerobot/teleoperators/nero_leader/test_nero_leader.py`、`src/lerobot/nero_tools/test_episode_capture.py`。

最近核心离线测试通过：`162 passed`。离线测试不证明真机安全或 CAN 已连接。

## 起始姿态与真机只读检查

起始姿态：`examples/nero/saved_start_pose.json`，另有 `src/lerobot/nero_tools/saved_start_pose.json`。最近保存时使用：

```cmd
python -X utf8 -m lerobot.nero_tools.test_pair_j1 --leader-channel 0 --follower-channel 1 --save-start --overwrite-start
```

重置后先只读检查：

```cmd
python -X utf8 -m lerobot.motors.nero.read_nero_firmware --interface agx_cando --channel 0 --firmware v120 --no-enable --no-disable --no-mode-change
python -X utf8 -m lerobot.motors.nero.read_nero_firmware --interface agx_cando --channel 1 --firmware v121 --no-enable --no-disable --no-mode-change
```

确认固件、时间戳和关节反馈持续更新，再做运动测试。USB-CAN 被 Web 上位机或其他工具占用会导致打开失败。

## 正式遥操作

```cmd
python -X utf8 examples/nero/teleoperate.py --interface agx_cando --leader-channel 0 --follower-channel 1 --leader-firmware v120 --follower-firmware v121 --leader-gripper-width 0.10 --follower-gripper-width 0.07 --gripper-mode width --leader-gripper-source auto --gripper-force 1.0 --max-joint-speed 0.6 --speed-percent 15 --fps 50 --return-to-start --return-duration 5
```

正式入口会先归位，再让主臂进入零力拖拽，主臂动作相对映射到从臂。采集器的关键修复是发送后调用 `mapper.accept(sent)`，避免从臂每帧重复第一小步而抖动。

## 相机和采集器

已验证 Intel RealSense D435i（序列号 `261622071323`，曾为 USB 2.x）和普通 UVC/OpenCV 相机。当前采集使用 OpenCV DirectShow，640x480@30，控制循环约 50 Hz。相机应固定拍到红色袋子、目标物体、夹具和夹具附近的机械臂。

采集命令示例（绿色方块/绿色小球只需改输出目录）：

```cmd
python -X utf8 -m lerobot.nero_tools.episode_capture --camera-type opencv --camera-index 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 30 --episode-seconds 15 --output outputs\green_cube --gripper-force 1.5
```

按键：`S` 开始，`E` 保存，`D` 丢弃，`R` 从臂自动归位并提示人工拖主臂回起点，`Q/Esc` 默认请求关节和夹具失能。可以中途 Q/Esc，下一次使用相同 `--output` 会从最大已有 episode 编号继续，不覆盖旧目录。录制中 Q/Esc 保存的片段标记为 `quit_end`，可能不完整，训练前人工排除。只有特殊调试才使用 `--keep-enabled-on-exit`。

## 当前原始数据

原始绿色方块数据：`outputs/green_cube`。实际有 31 条（0000~0030），约 13,899 张图片，约 4.71 GiB。每条包含 `frames/`、`robot_state.jsonl`、`actions.jsonl`、`timestamps.jsonl`、`metadata.json`。状态是从臂实际七轴弧度和夹具值；actions 是发送目标；timestamps 是图像与时间；metadata 是相机、通道、夹具等配置。

## 当前标准 LeRobot 数据

转换输出：`outputs/green_cube_lerobot`。说明和划分清单：`outputs/green_cube_lerobot/README.md`、`conversion_manifest.json`。

```text
训练集：outputs/green_cube_lerobot/train
repo_id: local/nero_green_cube_train
25 episodes，11197 frames

验证集：outputs/green_cube_lerobot/val
repo_id: local/nero_green_cube_val
6 episodes，2702 frames
```

任务：`Pick up the green cube and put it into the red bag.`。feature：`observation.images.front` 为 RGB 480x640x3；`observation.state` 和 `action` 顺序均为 `joint_1.pos` 到 `joint_7.pos`、`gripper.pos`；关节为弧度、夹具为 0~1。按完整 episode 以随机种子 42 划分，验证源 episode 为 0006、0010、0014、0019、0024、0027，无 episode 重叠。标准数据已使用 LeRobot 加载器重新读取并检查边界样本；没有上传到 Hugging Face。

转换只做了结构、图像、数值和时间对齐检查，不代表每条抓取都成功；模型尚未训练，尚未部署到机械臂。

如果需要重做转换（目标不存在时）：

```cmd
set PYTHONPATH=C:\Users\XIOT\Desktop\test\lerobot_nero_standalone\src
python -X utf8 -m lerobot.nero_tools.convert_episodes --source outputs\green_cube --output outputs\green_cube_lerobot
```

## 下一步建议

1. 人工抽查原始 31 条，排除夹空、掉落、遮挡、跟随异常和 `quit_end`。
2. 确认标准训练/验证数据仍与审核结果一致。
3. 训练单任务 ACT 或项目支持的模仿学习模型，先只做“绿色方块放入红色袋子”。
4. 先离线推理检查八维输出、弧度单位、夹具 0~1、无 NaN 和无跳变。
5. 真机只连从臂、空载、低速先测试接近，再逐步测试夹取、提起、放入。
6. 连续成功后再处理蓝色小球、绿色小球、红色方块，多物体任务最后做。

当前没有已经验证的训练命令或自动部署入口，不要把“标准数据已生成”当作“模型已训练成功”。

## 环境补充与已知限制（恢复时必读）

2026-10-07 实际查询：opencv-python 5.0.0.93、pyrealsense2 2.58.4.10922、datasets 3.6.0、pyarrow 25.0.1、Pillow 12.3.0。采集器在模块顶层导入 pyrealsense2，即使使用 OpenCV 相机也需要安装它。

项目依赖声明包含 opencv-python-headless，但实时预览需要 GUI 版，安装项目后处理如下：

```cmd
python -m pip uninstall -y opencv-python-headless
python -m pip install opencv-python==5.0.0.93 pyrealsense2==2.58.4.10922 datasets==3.6.0 pyarrow==25.0.1 Pillow==12.3.0
python -c "import cv2, pyrealsense2, datasets, pyarrow; print(cv2.__version__)"
python -c "import lerobot; print(lerobot.__file__)"
```

安装顺序和包来源需结合恢复时的可用版本检查；这些是本机版本记录，不是已在干净系统完整重装验证的锁文件。Windows 图像型数据集不需要视频解码器；不要为当前训练强行安装不兼容的 torchcodec，按实际导入错误判断。CUDA wheel 需要兼容 NVIDIA 驱动，不代表必须安装独立 CUDA Toolkit。备份到外部盘或另一台机器；同一台电脑的桌面文件也可能随重置丢失。

重要修正：原始图像数量是 **13,911**，30 Hz 重采样后标准数据才是 **13,899**。原始状态键为 joint_1 等裸名，action 键为 joint_1.pos；转换已统一为固定八维顺序。v2.1 图像以字节嵌入 Parquet，train/val 各自是独立数据集，内部 HF split 名均是 train。验证时必须复用训练集归一化统计。尚未完成相机与机器人硬件时钟同步，只保留了原始配对行和时间映射。

以下限制尚未修复，不要用历史对话中的肯定结论代替实机证据：

- R 的主臂自动回位没有成功；当前是从臂自动回位、主臂人工拖回。set_normal_mode 在新 SDK 档位是空操作；试过的私有接口方案已移除，不能宣称升级 1.21 一定能或一定不能解决。
- 主从回位容差现为 10 度、超限警告；主臂等 30 秒后可能继续。wait_for_leader_near 返回的是历史最佳误差姿态，不一定是最新姿态，重新锚定前仍需修正；从未有有效反馈时还可能返回 None。
- 初始归位的成功文案只经过稳定/有效反馈检查，不等于逐轴目标误差全部已核实。
- 采集时从臂读失败会沿用缓存，未额外限制这层缓存时长；可能保存陈旧状态，训练前应审查。
- 普通失能清理存在反馈未确认警告，尤其夹具。Q/Esc 默认请求失能；这不等于已确认失能。主臂保持零力模式时“不失能关闭连接”也不保证位置锁定。
- Ctrl+C/异常可进入清理，但当前未录完片段只有 Q/Esc 路径自动保存；不要假设所有异常都保存当前片段。
- 相机读取在独立线程，控制与窗口刷新仍在同一循环；没有完成独立固定频率控制线程的重构。
- 保存起点默认写 src/lerobot/nero_tools/saved_start_pose.json，采集默认读 examples/nero/saved_start_pose.json。未来建议保存时显式加 --pose-file examples/nero/saved_start_pose.json，避免再次不一致。
- 旧备份是 9 月版本，不包含本次采集续号和转换修复。只保留旧备份无法恢复最新项目。

用于本任务的离线检查（不会连接硬件）：

```cmd
python -X utf8 -m pytest -q -p no:cacheprovider src/lerobot/motors/nero/test_teleoperation.py src/lerobot/robots/nero_follower/test_nero_follower.py src/lerobot/teleoperators/nero_leader/test_nero_leader.py src/lerobot/nero_tools/test_episode_capture.py
```

162 是历史测试数量；续号修复之后单独运行 test_episode_capture.py 为 6 passed，不要把历史数量作为唯一验收条件。

## 重置前备份

最简单是压缩并备份整个 `C:\Users\XIOT\Desktop\test\lerobot_nero_standalone`。至少不能漏：两个 `saved_start_pose.json`、`outputs/green_cube`、`outputs/green_cube_lerobot`、夹具采集 JSON、日志、固件升级工具、USB-CAN 驱动、`cando.dll` 和厂商安装包。标准数据中的图像已嵌入 Parquet，不能只备份 metadata。

## 给重置后 Codex 的最短说明

```text
请先读取 PROJECT_HANDOFF.md。项目在 C:\Users\XIOT\Desktop\test\lerobot_nero_standalone。先恢复 Conda lerobot 环境并检查 outputs\green_cube_lerobot 的训练/验证数据；当前模型尚未训练，暂时不要连接机械臂。
```
