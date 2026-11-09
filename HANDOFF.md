# NERO 项目使用说明书与交接记录

本文件是电脑重置前的主交接文档。恢复或继续开发时，先读取本文件，再读取 README.md 和 docs/nero_teleoperation.md。

最后整理日期：2026-09-24。

## 2026-10-07 采集续存修复

`episode_capture` 启动时从输出目录最大 `episode_数字` 编号加一继续保存，不再从零覆盖。
保存目录以独占方式创建，遇到已有目录拒绝写入；残缺目录也保留并跳过编号。
Q/Esc 在录制中退出时保存非空片段，metadata.result 标为 `quit_end`，不表示任务成功。
默认先请求两臂关节和夹具失能，再等待所有后台保存完成；保存成功打印 `SAVED`，
写盘失败明确报错。失能请求不等于反馈确认，仍须留意硬件层 UNCONFIRMED 警告。
重新启动使用相同 `--output` 即可续采。旧版本已经覆盖的数据无法通过此次修复恢复。
离线测试覆盖编号续存、旧目录拒绝覆盖、Q/Esc 保存退出及后台写盘失败，无硬件操作。

2026-10-05 采集器排障：三个摄像头都出现从臂只在小范围抖动。根因不是相机帧率，
而是 `src/lerobot/nero_tools/episode_capture.py` 在发送 `mapper.target(...)` 后漏掉
`mapper.accept(sent)`，导致每一帧都从从臂起始姿态重新计算并重复发送第一小步。已增加
`send_mapped_target()` 封装和离线回归测试 `test_episode_capture.py`。正式遥操作
`src/lerobot/nero_teleoperate.py` 原本已有 `mapper.accept()`，所以不受此具体 bug 影响。

## 1. 项目状态

- 独立项目：C:\Users\XIOT\Desktop\test\lerobot_nero_standalone
- 完整备份：C:\Users\XIOT\Desktop\test\lerobot_full_backup_20260922_145610
- 原项目：C:\Users\XIOT\Desktop\test\测试版lerobot_piper2-main
- Python 环境：conda lerobot，D:\Users\XIOT\miniconda3\envs\lerobot\python.exe
- 独立包名：lerobot-nero；源码结构：src/lerobot
- 备份含 557 个文件，逐文件哈希校验 0 个不匹配

运行独立项目前建议设置：

    conda activate lerobot
    $env:PYTHONPATH = "C:\Users\XIOT\Desktop\test\lerobot_nero_standalone\src"

## 2. 项目目标

基于 LeRobot 实现松灵 NERO 七轴机械臂主从遥操作：主臂读取七关节和夹具，从臂下发七关节和夹具，支持起始姿态归位、夹具采集、直接测试、显式失能和硬件无关测试。

## 3. 硬件身份

以下为 2026-09-22 已验证的映射。USB 插拔后必须重新确认，不要永久假定：

| 角色 | 通道 | 主控固件 | SDK 档位 |
|---|---:|---|---|
| 主臂 leader | 0 | 1.20 | v120 |
| 从臂 follower | 1 | 1.21 | v121 |

两臂各使用独立 USB-CAN 适配器，CAN bitrate 为 1000000。

    python -X utf8 -c "import can; print(can.detect_available_configs(interfaces=['agx_cando']))"
    python -X utf8 src/lerobot/motors/nero/read_nero_firmware.py --interface agx_cando --channel 1 --firmware v121 --expected-version 1.21 --no-enable --no-disable
    python -X utf8 src/lerobot/motors/nero/read_nero_firmware.py --interface agx_cando --channel 0 --firmware v120 --expected-version 1.20 --no-enable --no-disable

读不到固件不等于连接失败。必须确认时间戳持续推进、关节角持续更新。

## 4. 代码结构

核心硬件层：src/lerobot/motors/nero/nero.py，负责 CAN、SDK、固件、七关节、反馈、使能失能、限位、速度、夹具归一化和退出清理。

机器人与遥操作层：

- src/lerobot/robots/nero_follower/：从臂配置、观测和动作下发。
- src/lerobot/teleoperators/nero_leader/：主臂配置和动作读取。
- src/lerobot/nero_teleoperate.py：连接、稳定检查、映射、限速、归位、循环和清理。

NERO 工具包 src/lerobot/nero_tools/ 包含 start_pose.py、gripper_capture.py、gripper_direct_test.py、test_pair_j1.py、test_leader_drag.py、test_follower_j1.py、test_enable_cycle.py、disable_pair.py、disable_follower.py 和 saved_start_pose.json。

examples/nero/ 保留为兼容入口和历史日志、JSON 数据入口。

## 5. 正式遥操作流程

1. 读取 saved_start_pose.json。
2. 主臂以普通位置模式连接，不立即进入零力模式。
3. 从臂归位到保存姿态。
4. 主臂归位到保存姿态。
5. 等待两侧稳定反馈。
6. 主臂进入零力拖拽和 leader 反馈。
7. 创建相对姿态映射器。
8. 主臂七关节和夹具映射到从臂。
9. 异常、超时或 Ctrl+C 后分别断开并请求失能。

关键默认参数：

- --return-to-start：启动归位。
- --max-relative-target 默认关闭。
- --max-joint-speed 默认 0.3 rad/s。
- --max-gripper-speed 默认 1.0 归一化值/秒。
- --gripper-force 默认 1.0 N。
- gripper_send_threshold=0.005，约 0.5% 变化才发送。
- 主臂 leader 模式普通 CAN hz=0 正常。
- 七关节和夹具失能必须分别用更新反馈确认。



## 6. 推荐启动命令

先设置环境：

    conda activate lerobot
    cd "C:\Users\XIOT\Desktop\test\lerobot_nero_standalone"
    $env:PYTHONPATH = "$PWD\src"

带自动归位的正式主从遥操作：

    python -X utf8 examples/nero/teleoperate.py --interface agx_cando --leader-channel 0 --follower-channel 1 --leader-firmware v120 --follower-firmware v121 --leader-gripper-width 0.10 --follower-gripper-width 0.07 --gripper-mode width --leader-gripper-source auto --gripper-force 1.0 --max-joint-speed 0.6 --speed-percent 15 --fps 50 --return-to-start --return-duration 5

预期顺序：

    START POSE: moving follower and leader to saved joint poses
    START POSE: both saved joint poses confirmed; starting teleoperation
    LEADER mapped | ...
    FOLLOWER feedback | ...

去掉 --return-to-start 和 --return-duration 5 可跳过归位，只用于当前姿态已经确认时的快速调试。

保存新的起始姿态（七关节，不包含夹具）：

    python -X utf8 examples/nero/test_pair_j1.py --leader-channel 0 --follower-channel 1 --save-start

保存前两臂必须支撑、空载、静止；覆盖已有姿态必须使用显式覆盖参数。

## 7. 夹具状态和量程

只读采集结果：

| 侧 | 通道/固件 | 配置/观测 | 样本 | 使能 | homing | 故障 |
|---|---|---|---:|---|---|---|
| 主臂 | 0 / 1.20 | 配置 100 mm，观测约 100.533 mm | 291 | 否 | 否 | 未发现 |
| 从臂 | 1 / 1.21 | 配置 70 mm，观测 0~70 mm | 293 | 否 | 否 | 未发现 |

当前正式遥操作采用软件映射：

    主臂 0~100 mm -> 归一化 0~1 -> 从臂 0~70 mm

主臂参数 --leader-gripper-width 0.10，从臂参数 --follower-gripper-width 0.07。当前没有执行硬件零位标定，homed=False 不等于已完成机械零位；只说明软件比例映射可以工作。

只读采集不会发送使能、失能、标定、模式切换或开合指令：

    python -X utf8 src/lerobot/nero_tools/gripper_capture.py --role follower --channel 1 --firmware v121 --expected-version 1.21 --seconds 30 --output examples/nero/gripper_capture_follower.json
    python -X utf8 src/lerobot/nero_tools/gripper_capture.py --role leader --channel 0 --firmware v120 --expected-version 1.20 --seconds 30 --output examples/nero/gripper_capture_leader.json

夹具四阶段测试顺序为 inspect、prepare、calibrate、open。此前 prepare 曾出现 70 mm 参数写读回失败，calibrate 曾出现 zero calibration 未确认；没有现场确认时不要反复执行写入阶段。标定成功必须看到 homing_status=True。

## 8. 已完成验证

已确认：

- 主臂单独拖拽，反馈持续更新，退出时关节和末端失能确认。
- 从臂 J1 约 +3 度单轴测试，目标约 +2.997 度，其余关节基本不动。
- 双臂 J1 和七轴受限跟随，证明从臂能执行主臂映射目标。
- 保存并自动返回主从起始姿态。
- 正式入口完成归位后再进入跟随。
- 两臂夹具只读采集，反馈持续更新。
- 夹具加入遥操作后的基本开合功能已报告可用。
- 核心离线测试最近为 160 passed；另有 159 passed 记录。

离线测试不证明真实 CAN、实际负载或机械安全已经验证。

## 9. 下一步待验证

1. 重置电脑后重新确认两个 USB-CAN 适配器通道，不能直接假定主臂 0、从臂 1。
2. 检查 saved_start_pose.json 是否仍对应当前工作空间；不匹配则重新保存。
3. 运行带 --return-to-start 的低速遥操作，确认归位方向。
4. 观察主臂 0~100 mm 到从臂 0~70 mm 的映射方向和实际开口。
5. 分别确认退出时七关节和夹具的推进反馈失能。
6. 每次只提高一个速度参数，不同时提高速度和夹持力。
7. 若需要真正硬件零位，先 inspect 和现场闭合夹具，再审查 calibrate。

## 10. 安全规则

- 每次运动、使能、归位前，两臂必须有可靠支撑，空载，工作区无障碍。
- 必须准备实体急停；网页控制台和 Python 异常退出不是急停系统。
- 两个适配器必须是两个独立 CAN 设备，主从通道不能相同。
- 夹具和七关节使能相互独立；测试中启用夹具，退出必须单独请求夹具失能。
- 失能只以推进中的反馈时间戳和状态为证据，不能只看 Connection released 或 NERO disconnected。
- 主臂 leader 模式普通 CAN hz=0 正常，但必须确认专用 leader 反馈仍在更新。
- 不要执行 move_js，代码禁止无轨迹平滑的高风险运动模式。
- 不要未经确认修改夹具固件、零位、最大行程或夹持力。
- 程序不是硬实时安全 PLC，不能替代硬件安全回路。

## 11. 重置后的最短恢复路径

1. 安装 Miniconda，创建名为 lerobot 的环境。
2. 恢复 lerobot_nero_standalone，或从完整备份恢复原项目。
3. 按第 14 节安装完整 LeRobot 测试依赖，并固定 NERO SDK 与 CAN 后端 Git 提交。
4. 设置 PYTHONPATH 到独立项目 src。
5. 运行：
       python -X utf8 -m pytest -q src/lerobot/motors/nero/test_teleoperation.py src/lerobot/teleoperators/nero_leader/test_nero_leader.py src/lerobot/robots/nero_follower/test_nero_follower.py
6. 只读检查两个 CAN 通道和两臂固件、关节反馈。
7. 确认夹具、机械臂支撑和实体急停。
8. 运行第 6 节的低速正式遥操作。
9. 确认归位、主从映射、夹具映射和退出失能。

## 12. 文档索引

- 本文件：恢复上下文和交接记录，优先读取。
- README.md：项目总览和推荐入口。
- docs/nero_teleoperation.md：差异分析、完整说明和历史实机记录。
- examples/nero/：兼容入口、日志、起始姿态和夹具 JSON。
- src/lerobot/nero_tools/：已整合工具源码。
- AGENTS.md：硬件测试安全约束。

## 13. 重置前必须保存的内容

重置前请至少复制或压缩以下目录：

- lerobot_nero_standalone\
- lerobot_full_backup_20260922_145610\
- 测试版lerobot_piper2-main\（如仍需原项目）

必须额外保存的非代码数据：

- examples/nero/saved_start_pose.json
- src/lerobot/nero_tools/saved_start_pose.json
- examples/nero/gripper_capture_leader.json
- examples/nero/gripper_capture_follower.json
- examples/nero/*.log
- 两台臂的固件文件和升级工具
- USB-CAN 适配器驱动、cando.dll 或厂商安装包
- Python 环境中的 pyAgxArm 安装包或版本记录

建议压缩保存整个 lerobot_nero_standalone，而不是只保存 src。

## 14. LeRobot 运行环境

### 14.1 当前实测环境

- 操作系统：Windows 10 x64 22H2，build 19045。
- Conda：Miniconda，conda 26.7.1。
- Conda 环境名：`lerobot`。
- Python：3.12.14，64 位，MSC v.1942。
- Python 路径：`D:\Users\XIOT\miniconda3\envs\lerobot\python.exe`。
- LeRobot：当前独立项目使用 `src` 路径或 editable 安装；不要让 Python 解析到旧项目副本。
- USB-CAN：两个 `agx_cando` 设备，CAN bitrate `1000000`。
- 固件升级工具：松灵网页上位机及两台机械臂、夹具固件包。

项目声明支持 Python 3.10 及以上。为重置后尽量保持一致，建议固定使用 Python 3.12。

### 14.2 已实测的关键 Python 包

| 包 | 当前版本 |
|---|---|
| `pyAgxArm` | 1.0.0，Git 提交 `841a625f5f4920e776f20b934eb13048b747e6d0` |
| `python-can-agx-cando` | 0.0.1，Git 提交 `b222c4027ad4f6599f7634c72c67184619177972` |
| `python-can` | 4.6.1 |
| `pyserial` | 3.5 |
| `numpy` | 2.2.6 |
| `draccus` | 0.10.0 |
| `pydantic` | 2.13.5 |
| `pydantic_core` | 2.46.5 |
| `pytest` | 9.1.1 |
| `torch` | 2.8.0+cu128 |
| `torchvision` | 0.23.0+cu128 |
| `torchaudio` | 2.8.0+cu128 |
| `torchcodec` | 0.11.1 |

NERO 遥操作必须保持一致的核心是 Python、`pyAgxArm`、`python-can-agx-cando` 和 `python-can`。PyTorch 用于 LeRobot 导入、策略、数据处理和训练；只做遥操作时仍可能被 LeRobot 顶层模块导入，因此环境里需要有 PyTorch。

### 14.3 重置后创建 LeRobot 环境

安装 Miniconda 后执行：

    conda create -n lerobot python=3.12.14 -y
    conda activate lerobot
    cd "C:\Users\XIOT\Desktop\test\lerobot_nero_standalone"
    python -m pip install --upgrade pip
    python -m pip install -e ".[test]"

如需与当前机器一致的 CUDA 12.8 PyTorch：

    python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install torchcodec==0.11.1

不训练策略时也可使用 CPU 或其他兼容 PyTorch，但必须重新运行 NERO 离线测试确认兼容。

### 14.4 固定 NERO SDK 与 CAN 后端

当前实测代码使用 Git 固定提交，不是只写 `pyAgxArm==1.0.0` 就必然等价。建议按提交安装：

    python -m pip install python-can==4.6.1
    python -m pip install --no-deps "git+https://github.com/agilexrobotics/python-can-agx-cando.git@b222c4027ad4f6599f7634c72c67184619177972"
    python -m pip install --no-deps "git+https://github.com/agilexrobotics/pyAgxArm.git@841a625f5f4920e776f20b934eb13048b747e6d0"

若已经安装 `pyproject.toml` 中的 `pyAgxArm==1.0.0`，再执行 Git 安装时请使用 `--force-reinstall`，确保实际覆盖到指定提交。

### 14.5 Windows 与 CAN 运行条件

- Windows 10/11 x64。
- 使用 x64 Python 和 `agx_cando\bin\x64\cando.dll`。
- x32 DLL 位于 `agx_cando\bin\x32`，但当前 Python 是 x64，应使用 x64 版本。
- USB-CAN 设备需要 Windows 正常枚举并允许 `python-can-agx-cando` 打开。
- 能枚举不代表 `cando_open` 成功；后者失败通常是 Windows USB 驱动或 DLL 状态问题。
- 一个适配器只能由一个程序打开，Web 上位机或其他 CAN 工具不能同时占用同一通道。
- 如使用 NVIDIA 训练，还需要与 PyTorch 匹配的显卡驱动/CUDA 运行库；纯遥操作不要求独立 CUDA Toolkit。

### 14.6 环境验证

依次执行：

    python --version
    python -c "import sys; print(sys.executable)"
    python -c "import pyAgxArm, can, draccus, torch; print('dependencies ok')"
    python -c "import can; print(can.detect_available_configs(interfaces=['agx_cando']))"
    python -X utf8 -m pytest -q src/lerobot/motors/nero/test_teleoperation.py src/lerobot/teleoperators/nero_leader/test_nero_leader.py src/lerobot/robots/nero_follower/test_nero_follower.py

恢复成功的最低标准：

1. Python 显示 3.12.x，路径位于 Conda 环境 `lerobot`。
2. `pyAgxArm`、`python-can-agx-cando`、`python-can` 可导入。
3. `detect_available_configs` 能列出实际连接的 `agx_cando` 设备。
4. NERO 离线测试为 `160 passed`。

离线测试通过但设备打不开时，优先检查 Windows USB/cando 驱动；Python 包无法导入时，先恢复环境，不要连接机械臂。
## 15. 文件校验值

HANDOFF.md 生成时 SHA256：

    由恢复者在最终修改后重新计算

该值会因后续追加记录而变化，主要用于确认当前这份交接文件未被意外截断。完整备份此前已逐文件哈希校验，结果为 0 个不匹配。
