# NERO 毕设最终版交接文档

更新时间：2026-11-06（北京时间）

请后续 AI 先读本文件，再执行任何训练、部署或硬件命令。

## 目录边界

最终版目录：C:\Users\Administrator\Desktop\最终版

最终版应保留核心源码、NERO 主臂/从臂驱动、遥操作、相机和 CAN 诊断、训练入口、离线评估、部署脚本，以及保留的训练权重。原始数据集、MP4/Parquet、数据转换代码、测试目录、Python 缓存和临时实验输出不属于干净代码副本。

当前采用安全的非破坏整理方式：最终版已有文件和权重不删除；干净源码副本放在最终版的 code_clean 子目录。

## 最终权重

权重目录：C:\Users\Administrator\Desktop\最终版\weights_nero_act22_50k

该模型是 ACT，不是 PI0/PI0.5。使用 wrist 和 base 两路图像，22 维 observation.state，输出 8 维绝对动作（7 个关节加夹爪）。

22 维 state 顺序必须保持：

0 到 7：joint_1.pos 到 joint_7.pos、gripper.pos
8 到 12：cube_x、cube_y、cube_w、cube_h、cube_visible
13 到 16：bag_x、bag_y、bag_w、bag_h
17 到 21：ball_x、ball_y、ball_w、ball_h、ball_visible

坐标是 base 相机归一化坐标 0 到 1，不是毫米。不要使用需要 8 维 state 或独立 environment_state 的部署脚本加载该权重。

## 环境

默认 Python：D:\Anaconda3\envs\lerobot\python.exe

在 Anaconda Prompt 中：

conda activate lerobot
cd /d "C:\Users\Administrator\Desktop\最终版\code_clean"
set "PYTHONPATH=%CD%\src;%CD%\tools"
set "PYTHONIOENCODING=utf-8"

先验证：python -c "import sys; print(sys.executable)"

## 相机与 CAN

wrist 相机序列号：261622071323，应看到夹爪近景。
base 相机序列号：261622078506，应看到桌面全景。

必须通过画面确认相机角色。CAN 通道不是永久身份；USB 拔插或换口后先执行：

python -X utf8 tools\diag_can_channels.py

确认主臂、从臂通道和固件后才能运行遥操作或部署。运动前确认两臂有支撑、夹爪为空、工作区清空、急停可触及。关节失能和夹爪失能是独立通道；没有新鲜反馈时间戳时，失能状态必须报告为 UNCONFIRMED。

## 遥操作命令模板

python -X utf8 -m lerobot.nero_teleoperate --leader-channel <主臂通道> --follower-channel <从臂通道> --leader-firmware v120 --follower-firmware v121 --duration 60

尖括号必须替换为现场诊断得到的通道，不能盲目使用历史的 0/1 映射。

## 训练和部署规则

最终版的干净代码副本不包含数据集。训练前必须由用户提供数据集路径，并核对 state 维度、state 顺序、图像 key、训练验证划分和视频后端。不要把 relation_v2、base14 或 eye-in-hand 数据集直接套入 22 维模型。

部署前先做相机快照和 dry-run，确认实时检测框、相机角色和 state 维度，再低速短时实机。nero_act22_50k 必须使用 merged22 state 布局。维度不匹配时停止排查，不能删除保护。

## 明确排除

code_clean 不包含 outputs、原始或转换后数据集、MP4、Parquet、数据转换脚本、测试目录、缓存、训练日志、历史实验输出或 PI0.5 大模型。转换脚本也被刻意排除；数据工作应在最终版之外进行。

## 后续 AI 工作规则

1. 先读本文件、AGENTS.md 和权重目录说明。
2. 不要假设数据集在最终版目录内。
3. 不要覆盖 weights_nero_act22_50k，除非用户明确要求并已备份。
4. 硬件动作前先确认 CAN 通道和物理安全条件。
5. 不把离线 MAE、训练 loss 或检测可见率称为抓取成功率。
6. 修改部署脚本后先 dry-run，再低速、短时、逐步延长。
7. 退出后分别确认关节和夹爪失能；没有新鲜反馈就报告 UNCONFIRMED。

## 交接结论

code_clean 是可迁移的 NERO 毕设代码副本，保留源码、运行入口和必要配置，排除数据集与转换代码；最终版根目录保留用户原有权重和其他材料，不做删除。
