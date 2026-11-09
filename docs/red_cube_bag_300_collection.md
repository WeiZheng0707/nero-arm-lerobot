# 红方块 → 袋子 300 条双机位采集手册

本轮数据集替代全部历史数据。三组需求：

| 组 | 条数 | 场景 | 目标 |
|---|---:|---|---|
| A | 220 | 桌面只有红方块 | 主几何能力 |
| B-远 | 40 | 蓝球距红方块 > 15 cm | 学会忽略无关物体 |
| B-近 | 40 | 蓝球距红方块 5–10 cm | 学会精确选择 |
| 合计 | **300** | 全部：**只夹红方块放进袋子** | |

**袋子位置会移动。** 这不是可选项，它决定了整个架构:

- 方块、球、袋子的位置**全部只在 base（固定第三视角）里检测**
- 腕部相机的像素坐标**没有任何位置含义**（臂一动就跳 30–60 px），禁止用于任何标签
- 袋子移动就体现在 `bag_x/bag_y` 随 episode 变化，这是模型学会"袋子在哪"的唯一途径

> 如果以后砍掉 base 只留腕部，这套数据的核心价值（袋子会动）就没了。采集器会打印明确警告。

## 0. 采前必做（不做完不许按 S）

```
[ ] 两臂支撑、夹爪空载、工作区清空、实体急停可触达
[ ] 只读确认两个 CAN 通道 + 固件 + 反馈时间戳推进（见下）
[ ] base 相机落地支架固定，胶带锁死对焦环
[ ] 两路曝光/白平衡/增益全手动固定并记录
[ ] 运动中测帧龄：p95 < 40 ms、max < 120 ms、掉帧 < 1%
[ ] 磁盘 > 60 GB
```

只读认通道（**不要套用文档里的通道号**，历史记录 leader=0 和 leader=1 互相矛盾）：

```bat
python -X utf8 -m lerobot.motors.nero.read_nero_firmware --interface agx_cando --channel 0 --firmware default --no-enable --no-disable --no-mode-change
python -X utf8 -m lerobot.motors.nero.read_nero_firmware --interface agx_cando --channel 1 --firmware default --no-enable --no-disable --no-mode-change
```

判据是**反馈时间戳持续推进、关节角持续更新**，不是"能打开"。靠固件号 1.20 / 1.21 区分主从。

CAN 适配器**直插主板、不要插 HUB**。上次真机失败就是 HUB 上那个能 open 但发不了帧。

## 1. 相机布局

两台都是 **Intel RealSense D435i**，但**链路不同**：base 走 USB 3.0，腕部走 USB 2.0。
这个不对称决定了分辨率分配，**不能两台都开 640×480@30**。

| 键 | 安装 | 分辨率 | 链路 | 职责 |
|---|---|---|---|---|
| `wrist` | 现有腕部位，支架必须刚性 | 640×480@30 | USB 2.0 | 指尖与方块相对关系、闭合瞬间 |
| `base` | **独立落地支架**，斜俯视 55–65°，距工作区 80–100 cm | 640×480@30 | **USB 3.0** | **方块/球/袋子的绝对位置** |

**为什么这样分：** USB 2.0 单路 640×480@30 RGB8 约 18.4 MB/s，已经接近该链路实用上限，
再叠第二路必然掉帧。而 USB 3.0 跑 640×480@30 余量充足。所以把高分辨率的
确定性留给 base（它提供所有位置标签，掉帧代价最大），腕部在 USB 2.0 上单独跑。

> 如果验收发现 USB 2.0 上的腕部仍掉帧，第一选择是**把腕部降到 640×480@20**，
> 不要动 base。腕部只做视觉输入，降帧率的代价远小于 base 掉帧导致的标签污染。

base 放在机械臂**对面偏 20–30°**，让末端朝远离相机的方向下降，减少手臂挡视线。
**不要夹在桌边**（会被撞、会抖）。

### 带宽验收（必做，先于一切采集）

```bat
set PYTHONPATH=%CD%\src
D:\miniconda\envs\lerobot\python.exe -X utf8 tools\realsense_bandwidth_check.py --list
```

确认两台的序列号和 `usb=` 字段（`3.2` / `2.0`）。然后指定角色实测：

```bat
D:\miniconda\envs\lerobot\python.exe -X utf8 tools\realsense_bandwidth_check.py ^
  --serial-base  <base的序列号>  --base-size 640x480@30 ^
  --serial-wrist <腕部的序列号>  --wrist-size 640x480@30 --seconds 15
```

门槛：**帧间隔 p95 < 40 ms、max < 120 ms、掉帧率 < 1%、实际帧率达到请求值**。
工具会打印 `GATE PASSED` / `GATE FAILED`。

**这个检查必须在机械臂运动中再做一次**——腕部线缆反复弯折才是掉帧重灾区，
静止时测好看没有意义。

### 遮挡检查（必做，30 分钟）

把红方块放到 12 个格点上，让从臂完整走一遍抓放轨迹，逐格点确认：

- 接近段至少有 **60% 的帧**方块被检出
- 抓取瞬间允许短暂遮挡，但闭合前 10 帧内至少一次可靠检出
- **袋子移到每个计划位置时都能被检出**

不达标 → 抬高俯角或侧移相机**重装**，不要靠调 HSV 阈值硬补。

## 2. 采集命令

先框好 base 的桌面 ROI 和袋子 ROI（单位：像素，相对 base 原始帧）。

```bat
set PYTHONPATH=%CD%\src
D:\miniconda\envs\lerobot\python.exe -X utf8 -m lerobot.nero_tools.episode_capture ^
  --camera-type realsense ^
  --camera-spec wrist=serial:<腕部序列号>,size:640x480,fps:30 ^
  --camera-spec base=serial:<base序列号>,size:640x480,fps:30 ^
  --base-roi 40,60,560,380 ^
  --base-bag-roi 40,60,560,380 ^
  --cube-color red --ball-color blue ^
  --leader-channel 1 --follower-channel 0 ^
  --leader-firmware v120 --follower-firmware v121 ^
  --leader-gripper-width 0.10 --follower-gripper-width 0.07 ^
  --gripper-force 1.0 --fps 30 --max-joint-speed 0.6 --max-gripper-speed 3.0 ^
  --max-control-dt 0.1 --episode-seconds 12 ^
  --pose-file examples\nero\saved_start_pose.json ^
  --output outputs\red_cube_300_20261027
```

**相机按序列号绑定，不要用索引。** `--camera-spec` 支持 `serial:` 关键字；
采集器启动时会打印每台的 `usb=` 链路和实际协商到的 profile，
USB 2.0 上的那台会额外打 `[WARN]`。序列号插拔后不会变，索引会变。

关键参数：

| 参数 | 说明 |
|---|---|
| `--camera-spec` | 可重复。**第一台是 wrist**，名字叫 `base` 的那台提供位置标签 |
| `--base-roi` | base 里检测方块/球的像素区域，写进 metadata |
| `--base-bag-roi` | base 里检测袋子的像素区域，建议与 `--base-roi` 一致 |
| `--base-exclude-top` | 默认 0.03。**腕部的 0.28 会把袋子顶部切掉**，base 必须用极小值 |
| `--cube-color red` | 目标方块颜色 |
| `--ball-color blue` | 干扰球颜色；`none` 关闭球检测 |
| `--episode-seconds 12` | 单条 10–12 s，不要 20 s（长尾犹豫帧是噪声） |

## 3. 采集按键

| 键 | 行为 |
|---|---|
| `S` | 开始运动并录制当前 episode |
| `E` | **成功** → 保存，`result=manual_end` |
| `D` | **失败** → 丢弃（不确定就按 D） |
| `R` | 从臂自动归位，然后**人工把主臂拖回起点** |
| `Q`/`Esc` | 退出并请求失能 |

成功判定（严格执行）：方块**确实在袋内**，未抓球、未把球碰进袋、无碰撞、无人工干预。
**只要不确定就按 D。宁可少一条，不可错一条。**

`R` 之后主臂必须手动拖回——主臂自动回位在这套 SDK 上一直没成功过。

## 4. 袋子移动策略

袋子每 **20–30 条**移动一次，覆盖约 5–8 个位置。每次移动后：

1. 确认 base 画面里袋子仍被框住（`BAG DETECTED`）
2. 在采集记录里写下当前袋子位置编号
3. **不要动 base 相机本身**——移动的是袋子，不是相机

bag 的位移会逐帧写进 `timestamps.jsonl` 的 `bag_x/bag_y`，转换后进入
`observation.environment_state` 的 idx 5–8，训练时模型直接看到"袋子在哪"。

## 5. 采集后转换

```bat
D:\miniconda\envs\lerobot\python.exe -X utf8 -m lerobot.nero_tools.convert_episodes ^
  --source outputs\red_cube_300_20261027 ^
  --output outputs\red_cube_300_lerobot ^
  --image-key wrist=observation.images.wrist ^
  --image-key base=observation.images.base ^
  --task "Pick up the red cube and put it into the bag." ^
  --dataset-name nero_red_cube_bag
```

输出 schema：

```
observation.images.wrist        (3, 480, 640)
observation.images.base         (3, 480, 640)
observation.state               (8,)
action                          (8,)
observation.environment_state   (14,)
```

14 维定义：

| idx | 名称 | 含义 |
|---:|---|---|
| 0–3 | cube_x/y/w/h | 红方块框（base 归一化） |
| 4 | cube_visible | 方块是否检出 |
| 5–8 | bag_x/y/w/h | **袋子框 —— 袋子移动就体现在这里** |
| 9–12 | ball_x/y/w/h | 蓝球框（A 组恒为 0） |
| 13 | ball_visible | 蓝球是否在场（A 组为 0，B 组为 1） |

球的坐标是必须的：B-近 组（球距方块 5–10 cm）的训练价值全在"选哪个"，只有
`ball_visible` 模型知道"有球"但不知道球在哪，选择能力学不到位。

转换器会拒绝 `quit_end` 和 `invalid` 的 episode，需要人工复核后才转。

## 6. 已知限制（别当成已解决）

- **frames 分辨率写死为 640×480** 是转换器的断言，改相机分辨率要同步改 `convert_episodes.py` 的 shape。
- **base 裁剪（`--base-crop`）当前只在 metadata 里记录，转换时尚未实际裁图**。若要裁到 320×240 提升有效分辨率，需在转换器里补一步。
- `--episode-seconds` 到点自动结束的 episode 记 `auto_end`，**不等于成功**，转换器会要求人工复核。
- 帧龄超过 200 ms 的 episode 自动标 `invalid`，不写 success。
- **球的坐标已进 env-state**，但训练时要考虑以 p≈0.15 的概率把整组 env-state 置零（模拟漏检），
  否则策略会退化成"完全信任检测器"，真机一漏检就崩。

## 7. 安全红线

- 任何运动前：两臂有可靠支撑、夹爪空载、工作区无障碍、**实体急停可触达**
- 退出时七关节和夹爪**分别**用推进的反馈时间戳确认失能，否则报 `UNCONFIRMED`
- 网页控制台和 Python 异常退出**不是急停系统**
- CAN 适配器分属两个独立设备，主从通道不能相同
- 不要执行 `move_js`；不要未经确认改夹具固件、零位、最大行程或夹持力
