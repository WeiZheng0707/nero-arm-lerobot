# NERO 双臂网页控制台

双击仓库外层目录的 `start_nero_control_center.cmd`，或在项目目录运行：

```powershell
conda run --no-capture-output -n lerobot python -X utf8 examples\nero\control_center.py
```

控制台只监听 `127.0.0.1`，自动打开网页，不向局域网暴露 CAN 或 SDK。页面只提供三类固定操作：只读身份检查、双臂遥操作、七个关节与 AgxGripper 显式失能。

推荐流程：

1. 重新确认主、从臂 CAN 通道与固件档位。
2. 勾选机械臂支撑、夹爪为空和工作区清空。
3. 点击“只读身份检查”；该操作不使能、不失能、不改模式、不发送目标。
4. 当前配置通过检查后，才能启动双臂遥操作。
5. 使用“停止并请求 AgxGripper 失能”，并确认页面出现更新反馈下的失能结果。
6. 任何 `UNCONFIRMED` 都必须继续支撑机械臂并现场核对。

网页不是实体急停。浏览器断开、控制台进程被强杀或 CAN 通信中断都不保证完成失能。若 AgxGripper 之外还有独立示教盒或其他末端，必须按各自反馈单独确认。

## 只读身份检查

只读检查不再使用固定通道，必须显式给出两台设备的身份：

```powershell
python -X utf8 examples/nero/inspect_effectors.py `
  --leader-channel 0 --leader-firmware v120 --leader-version 1.20 `
  --follower-channel 1 --follower-firmware v121 --follower-version 1.21
```

# Nero 双臂遥操作（含夹爪）

## 夹具实时采集

夹具测试先使用只读采集器，不发使能、失能、控制模式、标定或开合指令，主臂和从臂分开运行。
它实时打印当前开合距离、反馈频率、力、夹具使能、零位状态、故障位、控制模式、七关节使能位，
并在结束时保存 JSON 摘要和所有采样。70 mm 配置只作为反馈显示，不由采集器推断或修改。

从臂单独采集（当前通道 1 / 主控 1.21）：

```cmd
python -X utf8 examples/nero/gripper_capture.py --role follower --channel 1 --firmware v121 --expected-version 1.21 --seconds 30 --output examples/nero/gripper_capture_follower.json
```

主臂单独采集（当前通道 0 / 主控 1.20）：

```cmd
python -X utf8 examples/nero/gripper_capture.py --role leader --channel 0 --firmware v120 --expected-version 1.20 --seconds 30 --output examples/nero/gripper_capture_leader.json
```

不加 `--seconds` 会持续采集，按 Ctrl+C 结束。建议先采集 30 秒静止状态，再在厂商允许的方式下手动改变夹具开口，
采集结束后比较 JSON 的 `distance_min_mm`、`distance_max_mm`、`distance_span_mm` 和逐帧 `distance_mm`。
如果夹具当前使能，不要用手阻挡；采集器不会自动失能，结束后应按已验证的独立失能流程处理。
5 项夹具采集离线测试通过，未连接实机。

## 跟随速度参数

用户反馈上一档可用后，下一档候选为参考速度 15 度/秒、参考加速度 30 度/秒平方、
50 Hz、从臂控制器 20%；归位独立采用 2 度/秒、控制器 10%。本档未由助手运行实机。

```cmd
python -X utf8 examples/nero/test_pair_j1.py --leader-channel 0 --follower-channel 1 --all-joints --return-to-start --full-range --seconds 20 --speed-deg-s 15 --accel-deg-s2 30 --fps 50 --follower-speed-percent 20 --return-speed-deg-s 2 --return-speed-percent 10
```

`--return-speed-deg-s` 默认 1，允许 (0,5]；`--return-speed-percent` 默认 5，允许 [1,20]。
归位参数分别传给主、从臂，归位后从臂再切换到遥操作控制器速度。保存文件不变，
60 度归位距离、1.5 度跟踪误差、反馈超时及末端退出失能检查不变。
跟随参考制动现采用离散步长的停车距离计算，修正更高速度下固定目标的微小越过。
24 项跟随测试、14 项保存/归位测试通过，仅为离线验证。
回退上一档使用 10 / 20 / 50 / 15，归位使用 1 / 5（依次对应速度、加速度、频率、控制器百分比及两个归位参数）。

`test_pair_j1.py` 已将遥操作参考速度、发送频率、从臂控制器速度分别参数化。
旧命令默认不变。新的下一档候选命令（未实机验证，先限时运行）：

```cmd
python -X utf8 examples/nero/test_pair_j1.py --leader-channel 0 --follower-channel 1 --all-joints --return-to-start --full-range --seconds 20 --speed-deg-s 10 --accel-deg-s2 20 --fps 50 --follower-speed-percent 15
```

- `--speed-deg-s`：参考目标速度上限，默认 0.5，软件允许 (0,30] 度/秒；不是实际速度保证。
- `--accel-deg-s2`：可选离散参考加速度，(0,60] 度/秒平方，旧命令不指定则沿用纯限速。
  急剧反向时目标会先制动，硬关节/范围边界优先于加速度连续性；不是控制器硬件加速度配置。
- `--fps`：目标发送频率 [20,60]，默认 20；循环扣除处理耗时，停顿后仍限制单步增量。
- `--follower-speed-percent`：遥操作阶段的 move_j 控制器速度 [1,30]，默认 5；不是角速度单位。
  归位完成后才设置，归位仍采用 5% 控制器速度、1 度/秒参考速度和 60 度距离检查。
- 日志新增主臂/目标/从臂逐轴 `*_speed_deg_s`（相邻消费样本按本地时间差估算）、
  `send_hz` 和 `accel_limited_axes`。`tracking_error_deg` 对齐本拍测量与上一拍已发送目标。
- SDK move_j 规划、驱动响应、负载及主臂速度仍会产生延迟；主臂超过目标速度上限时不保证同步。
  不会自动加速超过参数值，跟踪误差 1.5 度及反馈超时仍中止，两侧关节和末端退出失能保持。

24 项相关离线测试通过，包括速度/加速度、反向、硬限位、采样率和异常清理。
数值上限是输入约束，不代表已在当前机械臂全部姿态和负载下验证。

## 保存并恢复准备姿态

当前新增完整关节范围：`--all-joints --full-range` 取消相对起点的小范围测试限幅，
采用从臂 SDK 关节限位裁剪目标，日志 `joint_limited_axes` 标出触及限位的轴。
仍是相对起始姿态映射，速度和跟踪误差保护不取消。`--continuous` 持续运行到 Ctrl+C，
届时请求两侧关节和末端失能并确认；进程被强杀或通信断开不保证清理。
示例（保留 2 度/秒，不自动提高到软件上限）：

```cmd
python -X utf8 examples/nero/test_pair_j1.py --leader-channel 0 --follower-channel 1 --all-joints --return-to-start --full-range --continuous --speed-deg-s 2
```

SDK 配置限位不是避障规划，也不代表所有姿态组合都适合现场；须核实实际机型/零位和工作空间。
新模式与 60 度自动归位均仅完成离线验证，19 项联调测试和 10 项姿态测试通过，未自动实机运行。
夹具开合仍未启用，不能称为已验证的“含夹具完整系统”。

联调参数现可显式设置 `--range-deg` 与 `--speed-deg-s`，分别控制每个活动关节相对
起始姿态的范围与参考目标速度，默认仍为 3 度和 0.5 度/秒。下一档候选为
`--range-deg 5 --speed-deg-s 2`，尚未实机验证；不改变归位速度（1 度/秒）、控制器
速度百分比（5%）、跟踪误差阈值（1.5 度）、反馈超时和末端失能流程。
活动关节实际偏离允许范围超过 0.5 度时中止。参数输入上限为 15 度与 5 度/秒，
仅为软件边界，不是已验证的机械安全范围；启动仍检查完整目标区间的 SDK 限位。
日志新增 `range_limited_axes`、`speed_limited_axes`（一基轴号）与逐轴
`tracking_error_deg`（当前反馈减本拍目标），结束汇总受限制的帧数。
15 项相关离线测试通过；保存姿态文件不变，不需要重新保存。

以下功能在 `examples/nero/test_pair_j1.py` 联调入口中提供，不是编码器零点标定。
仅记录两台臂七关节弧度、角色和固件版本，不记录或驱动夹具开合。

先按厂商允许的方法将两台臂摆在希望保存的姿态并保持静止；不要强行推动使能或抱闸的关节。
确认物理角色/通道后运行只读保存：

```cmd
python -X utf8 examples/nero/test_pair_j1.py --leader-channel 0 --follower-channel 1 --save-start
```

默认文件为脚本目录的 `saved_start_pose.json`，不会自动覆盖，重新保存需显式加
`--overwrite-start`。版本检查不能代替物理设备身份核实；更换设备、零点标定或固件后应重新核实保存姿态。

以后启用自动准备姿态再联调：

```cmd
python -X utf8 examples/nero/test_pair_j1.py --leader-channel 0 --follower-channel 1 --all-joints --return-to-start --seconds 20
```

启动显示当前与保存角度，输入 START 才授权实际运动。依次从臂归位、主臂归位，
每轴参考目标速度不超过 1 度/秒，任一轴距离目标超过 60 度拒绝启动。
归位确认后读取稳定反馈作为相对跟随基准，READY 后才手动拖动主臂。
遥操作仍限制每轴 +/-3 度、0.5 度/秒；20 秒从 READY 阶段开始计时，不包含归位时间。
这不是避障规划：操作人必须核实两侧整个归位路径、两臂互相碰撞风险，以及退出失能时的支撑。
不得在 RETURN 期间抓握或阻挡机械臂。归位失败不会进入跟随，退出仍分别请求关节与末端失能。
9 项姿态保存/归位离线测试与 9 项联调离线测试通过；未实机验证自动归位。

## 现场测试记录（2026-09-21）

- 从臂通道 1 / 1.21 的 J1 正向 3 度实机测试通过：起始 1.411 度，目标 4.411 度，约 6.12 秒时反馈增量 2.997 度；其余六关节该时刻增量均为 0，日志所列偏移很小。普通状态持续 NORMAL、err=0。无回零/往返、无末端开合目标；结束后七关节失能及末端持续 >=0.5 秒更新反馈失能均确认，连接已释放。日志 examples/nero/follower_j1_plus3_20260921.log。只证明本次单关节小幅控制，不代表双臂跟随或全工作空间已验证。

- 末端联动失能改进仅应用于当前诊断测试流程：`test_leader_drag.py` 在现场输入 START 后，若末端已使能且已确认 CAN 控制，先单独失能并验证，再允许关节使能；非 CAN 控制不会自动切换。结束和异常退出仍在关节失能之后执行末端失能。共用 `disable_effector` 要求至少 3 个推进时间戳、连续 >=0.5 秒失能反馈，缓存或再次使能不算成功。21 项离线测试通过。本次另行对主臂通道 0 请求末端失能并得到更新反馈确认；未运行新的实机拖拽。程序不能阻止上位机、固件重启或其他程序随后重新使能末端，也不能在进程被强杀后保证清理。

- 主臂后续日志确认：after_enable 的 0x06 很快转为 0x00；进入 leader 模式后普通状态时间戳停止且 hz 归零。核对本机 SDK，set_leader_mode 明确发送关闭普通 CAN 推送。此前 after_leader_mode 强制要求普通状态持续更新的判据不适用，已修正：切换前仍须确认正常状态；切换后仅以主臂专用角度反馈的时间戳推进作为数据链路检查，禁止回退普通角度缓存。新收到的普通状态故障仍中止；普通状态缺失不等于健康已验证，主臂模式下连续故障可见性有限。启动 3 秒未获得更新反馈或运行停更超过 1 秒均退出，分别确认关节/末端失能。16 项离线测试通过，修正后尚未实机验证。

- 主臂拖拽测试曾在七关节使能确认后读到 0x06 并退出，关节和末端退出失能均确认；用户报告现场一直能正常拖动，不能据单次状态认定机械抱闸故障。`test_leader_drag.py` 已加入 after_enable / after_leader_mode 两个状态阶段：各最多等待 3 秒，需要两个时间戳推进的 NORMAL 帧，不放行持续 0x06。运行中检查状态停更与新故障，输出时间戳、模式及状态码；不发送关节目标或夹具开合指令。12 项离线回归通过，新流程尚未实机验证。执行 `python examples/nero/test_leader_drag_checks.py` 可离线复测。

- 当前原因分析：升级页面确认末端节点 A 为 `JXB2_DRV V2.0.7`。升级后重启时从臂回到 ETHERNET_CONTROL_MODE(0x03)，CAN 夹具命令可能被忽略；只读反馈还显示 `homing_status=False`，而 pyAgxArm 官方 Nero 示例明确要求失能夹具、配置实际最大行程 0.07 m、手动闭合到机械零位、执行 calibrate_gripper，再发送位置指令。此前测试没有完成这两个前置条件。已新增 `gripper_direct_test.py`，按 inspect -> prepare -> calibrate -> open 分阶段执行；写阶段先确认七关节全失能，再切 CAN 控制，末端异常退出也显式失能。

- 用户明确最新刷入的是夹具固件，文件名 `JXB2_DRVV2.0.7-g56ed3c3_20260519_161455.bin`。尚未核实文件适配型号或刷写成功后的设备版本。随后只读检查从臂：主控 1.21、ETHERNET_CONTROL_MODE(0x03)、JOINT_COMMUNICATION_ERR(0x05)、err_code=0；夹具状态时间戳持续推进，开口 1.161 mm、失能，所读末端故障位均 False。不能用 err_code=0 或夹具故障位为 False 否定主控通信异常；停止动作测试，待厂商升级流程和设备报警确认。夹具测试已加入主控模式/状态保护，不自动切模式或清错。

- 用户再次升级后，仅复测从臂夹爪：通道 1 查询主控版本仍为 1.21（不能据此判断末端固件是否更新）。1 N，起始 1.161 mm，首步目标 1.661 mm；3.015 秒、30 次采样中开口不变，末端使能位始终 False，时间戳持续更新，七关节全程失能。中止后显式请求末端失能，并确认更新反馈为失能；连接已释放。不同于上一轮末端使能位曾为 True，本轮尚未确认夹爪使能或位置目标生效。日志：`examples/nero/gripper_retest_after_upgrade_20260921.log`。未重试、未加力、未操作主臂。

- 3 秒等待复测：通道 1 / 固件 1.21，起始反馈 1.154 mm，只下发首步 1.654 mm / 1 N。约 3.03 秒内 30 次采样的开口均为 1.154 mm、时间戳推进、末端使能位为 True、七关节使能位全 False；因此中止，未继续到 5 mm。末端失能经更新反馈确认，连接已释放。日志：`examples/nero/gripper_open_3s_20260921.log`。延长等待没有解决位置不变的问题，未证明原因是力不足，也未证明位置目标已被接受。

- 用户确认后续读到的 1.143 mm 是手动打开所得，不能归因于此前 CAN 指令。下一次小行程测试保持 1 N、每步最多增加 0.5 mm、最大 5 mm，仅将每步等待从 1 秒延长至 3 秒，逐次打印采样；不自动加力。

- 最近确认：主臂 1.20 / v120、从臂 1.21 / v121；双适配器时主臂通道 0、从臂通道 1。USB 插拔后必须重新识别。
- 两侧关节和夹爪原始反馈已验证持续更新。从臂已通过短时使能/失能回读测试。
- 曾发现两侧七关节均失能而末端仍使能，原因尚未确认；单独调用 disable_gripper 后，两侧更新反馈均确认末端失能。
- 用户明确要求：测试使能过的末端，结束及异常退出时必须单独请求失能并检查更新反馈；不能只看七关节状态。仅观测的诊断不改变使能状态。
- 独立示教器若不属于 AgxGripper 反馈范围，必须另行核实，不能声称已覆盖。
- 尚未完成主臂拖拽、从臂位置跟随和末端开合实机验证。
- 从臂夹爪首次小行程测试：1 N，计划从近闭合逐步张开到 5 mm；实际只下发 0.5 mm 和 1 mm 目标后超时，未完成 5 mm 测试。使能位有响应，但未确认开口跟随。finally 请求末端失能，更新反馈确认失能，七关节仍失能。原 0.5 mm 容差会把 0 mm 反馈视为达到首个 0.5 mm 目标，已收紧为 0.2 mm；修改后未再下发实机动作。不得据此判断夹爪故障或自动增大夹持力，需现场观察核对。

本实现以项目中的 Piper 分层结构为模板，使用 Nero 主臂控制 Nero 从臂。
夹具按松灵 AgxGripper 电动夹爪实现；其他品牌、气动夹具需要另写末端驱动。
这里不是 Piper 六轴主臂到 Nero 七轴从臂的跨机型映射。

## 差异与修改依据

| 项目 | 仓库中的 Piper | Nero 实现 |
| --- | --- | --- |
| SDK/通信 | piper_sdk / CAN | pyAgxArm 1.0.0 / CAN |
| 动作维度 | 6 关节 + 夹爪 | 7 关节 + 夹爪，共 8 维 |
| 关节单位 | SDK 原始 0.001 度，由上层换算 | 总线统一输出 rad；上层不再换算 |
| 主臂反馈 | Piper 接口 | <=1.11 旧通道按项目原有实测从度转弧度；>=1.12 SDK 已返回 rad |
| 从臂控制 | JointCtrl | move_j，控制器规划；本入口不使用 MIT 或 move_js |
| 夹爪控制 | GripperCtrl，米转原始整数 | move_gripper_m / move_gripper_deg，力单位 N |
| 夹爪动作值 | 米 | gripper.pos 为 0..1，0 闭合、1 最大开口 |
| 启动 | Piper 原实现会发初始姿态 | 不回零；默认以两臂起始关节姿态做相对映射 |
| 退出 | 原实现会发回位指令 | 不回位，退出主臂模式、请求失能、释放 CAN |

本机 SDK 源码核对发现：`nero/versions/v112/parser.py` 删除旧的
0x501..0x507 接收项，使用 0x155/0x156/0x157/0x170；解码已经包含 DEG2RAD。
旧代码无条件再次转换主臂角度，会将新固件动作缩小约 57.3 倍，现已按档位修正。
旧固件 float32 的单位仍依据本仓库已有实机记录，未在本次任务中实机复核。
`--leader-angle-unit deg|rad` 可覆盖自动判断，仅用于已核实单位的设备。

夹爪 `auto` 来源：主臂优先读示教控制反馈 0x159，再读状态反馈 0x2A8；从臂只读状态。
pyAgxArm 1.0.0 的 0x159 解码统一乘 1e-6，角度指令实际精度为 1e-3 度；
适配层根据 status_code=5 补偿该比例。状态反馈由 SDK 的 mode 区分宽度和角度。
失能/复位控制报文不作为有效开合目标。未收到反馈时默认报错，不补闭合值。

## 文件

- `src/lerobot/motors/nero/nero.py`：固件、CAN、单位、关节限位、夹爪、反馈检查。
- `src/lerobot/teleoperators/nero_leader/`：主臂配置与动作读取。
- `src/lerobot/robots/nero_follower/`：从臂配置、观测、关节及夹爪指令。
- `src/lerobot/nero_teleoperate.py`：双臂连接、启动检查、映射、限速、退出清理。
- `examples/nero/teleoperate.py`：推荐入口，强制使用当前仓库源码。
- `src/lerobot/motors/nero/run_teleop.py`：保留原来的单臂诊断脚本，不是双臂入口。

## 安装与启动

在项目根目录、已配置好的 LeRobot 环境运行。新环境可安装 `pip install -e ".[nero]"`。
此 extra 固定 `pyAgxArm==1.0.0`，其他版本的单位和报文需要重新核对。
本机已有 conda 环境 `lerobot`，无需为本次代码测试重新安装依赖。

Windows PowerShell 示例，假设主臂确实是设备 1、从臂是设备 0：

```powershell
conda activate lerobot
python examples/nero/teleoperate.py --interface agx_cando --leader-channel 1 --follower-channel 0 --fps 30 --gripper-force 1.0
```

Linux 示例，CAN 接口需事先配置为 1 Mbps：

```bash
python examples/nero/teleoperate.py --interface socketcan --leader-channel can1 --follower-channel can0 --fps 30 --gripper-force 1.0
```

Windows 的通道是设备枚举序号，插拔可能改变顺序；上述序号不是已验证的实机分配。
通道必须对应正确角色。固件默认自动探测，也可分别指定
`--leader-firmware default --follower-firmware v120` 等实际档位。

## 映射与夹爪参数

默认关节目标为 `从臂起始角 + (主臂当前角 - 主臂起始角)`，保持启动时两臂的姿态差。
方向与偏置先由主臂 `joint_signs/joint_offsets` 处理；相对模式会抵消固定偏置。
夹爪始终采用归一化绝对开合映射，并从从臂当前开口逐步靠近主臂目标。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--mapping` | relative | absolute 为绝对关节跟随 |
| `--max-start-delta` | 0.1 rad | 绝对模式启动时任一关节差值超过该值则拒绝启动 |
| `--max-joint-speed` | 0.3 rad/s | 软件参考目标的变化速度，不是实测速度保证 |
| `--max-relative-target` | 0.05 rad | 从臂指令相对当前反馈的最大单步偏差 |
| `--speed-percent` | 20 | 控制器 move_j 速度百分比 |
| `--feedback-timeout` | 0.5 s | 反馈停止更新及缓存容忍时间 |
| `--gripper-mode` | width | 从臂夹爪 width 米 / angle 度控制 |
| `--leader-gripper-source` | auto | auto / status / control |
| `--leader-gripper-width` | 0.07 m | 主臂实际最大开口 |
| `--follower-gripper-width` | 0.07 m | 从臂实际最大开口 |
| `--gripper-max-angle` | 100 deg | 角度模式归一化满量程 |
| `--gripper-force` | 1 N | 从臂夹持力指令 |
| `--max-gripper-speed` | 0.5 /s | 归一化开合目标的变化速度 |
| `--invert-gripper` | 关闭 | 使用 1-opening 反向映射 |
| `--no-gripper` | 关闭 | 同时关闭两侧夹爪，动作变成 7 维 |
| `--duration` | 持续运行 | 指定运行秒数；Ctrl+C 退出 |
| `--keep-gripper-enabled-on-exit` | 关闭 | 退出时不请求从臂夹爪失能 |

满量程和夹持力必须按实际夹爪设置。方向相反可用
`--joint-signs -1 1 1 1 1 1 1`；该选项不是机械零位标定。
机械臂和夹爪零位应先通过厂商流程校准，本入口不会重新写零点。

## 行为与验证范围

启动会使能两台臂并让主臂进入拖拽模式。退出、通信错误或异常会停止循环、请求失能并断开。
失能可能使机械臂下垂或物体释放，运行前应支撑好机械臂并准备实体急停。
SDK 失能返回值可能使用旧反馈；程序会报告未确认状态，不能将进程退出当作物理失能证明。

已有 `nero_leader` / `nero_follower` 注册保持可用，record/replay 的动作仍为
`joint_1.pos`..`joint_7.pos` 与 `gripper.pos`。旧 Piper 的七维数据不兼容。
相对起始映射和轨迹限速只由本次新增双臂入口提供，通用 `lerobot.teleoperate`
以及 `record` 不会自动获得这两项行为。

硬件无关测试：

```bash
python -m pytest -q src/lerobot/robots/nero_follower/test_nero_follower.py src/lerobot/teleoperators/nero_leader/test_nero_leader.py src/lerobot/motors/nero/test_teleoperation.py
python src/lerobot/motors/nero/run_teleop_dryrun.py
```

测试覆盖新旧固件单位、真实 SDK 第七轴解码、八维动作、夹爪宽度/角度比例、
反馈停止更新、非有限输入、限位返回值、连接失败清理以及双臂循环。
本次未连接真实 CAN 或机械臂；实际固件行为、夹爪可拖拽性、负载和跟随手感尚需实机验证。
SDK 聚合关节消息的 timestamp/hz 只代表部分报文，不能证明七个关节报文同时有效；
现有预热与稳定姿态检查也不是所有分帧故障的完整检测。Python 循环不是硬实时急停系统。
