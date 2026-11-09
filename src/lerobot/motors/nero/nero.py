# src/lerobot/motors/nero/nero.py

"""松灵 NERO 七轴机械臂的 LeRobot 硬件适配层。

与 Piper 版的三个本质差异（不是"写复杂了"，是 SDK 语义不同）：

1. 单位
   Piper 的关节值是 0.001 度（所以 piper.py 里有 joint_factor = 57324.84 做换算），
   NERO 官方 get_joint_angles() 直接返回 rad。**不要**再做一次换算。

2. 使能确认
   Piper 的 EnableArm() 只发不等，必须在外面 while 轮询 6 个
   driver_enable_status 才知道有没有成功。
   NERO 的 robot.enable() **同样只发不等**（fire-and-forget），它的返回值
   不代表实际状态。低速反馈帧只有 1 Hz 量级，发完立刻回读必然读到旧值。

   ⚠️ 2026-09-15 真机实测（曾在这里写错过）：使能后立即读 -> 0000000，
   十几秒后同一进程再读 -> 1111111。所以"Nero 是同步阻塞、不用轮询"这个
   说法是**错的**。做法与 Piper 一致：发指令 + 轮询回读实际状态直到确认。
   见 enable() / disable()。

3. 固件档位
   NERO 的通信行为随固件版本变化，建连前必须先确定驱动档位
   （DEFAULT / V111 / V112 / V120 / V121）。这是 NERO 独有的麻烦，
   Piper 没有对应概念。见 _probe_firmware()。

4. 角色（role）
   同一套 NERO 既可以当从臂（follower，被控）也可以当主臂（leader，拖拽示教），
   两者的**反馈通道和单位都不同**，所以角色必须显式声明：

     role="follower"：普通模式，读常规关节反馈（SDK 已解码成弧度）
     role="leader"  ：set_leader_mode() 零力拖拽，主臂姿态走 leader 反馈通道

   固件 <=1.11 的 leader 通道（CAN 0x501-0x507）沿用本项目实测的度单位。
   固件 >=1.12 改用 0x155/0x156/0x157/0x170，SDK 已转换为弧度。
   仅旧档位需要度转弧度；leader_angle_unit 可显式覆盖。
   实测读数形如 85.31，SDK 消息类的中文注释写"单位rad"是错的
   （英文注释写 0.001 degrees）。不转弧度的话主臂摆 1.5° 会被当成 1.5 rad
   下发，差 57.3 倍。见 _read_raw_angles()。

5. 残缺帧
   NERO 的关节反馈是**分帧**发的（每关节一帧），刚连上或推送刚被唤醒时会读到
   "部分关节已更新、其余还是上一帧残留"的混合帧。这种帧数值完全失真，一旦被
   当成"当前位姿"去下发保持指令，机械臂会朝错误目标猛甩
   （2026-09-13 实测：从臂 J7 因此转了约 195°）。
   所以读取一律经过 _read_raw_angles() 的完整性过滤，见那里的说明。
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from platform import system
from typing import Any

from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    NeroFW,
    create_agx_arm_config,
    resolve_firmware_profile,
)

logger = logging.getLogger(__name__)

ARM_JOINT_NAMES: list[str] = [f"joint_{i}" for i in range(1, 8)]
JOINT_NAMES: list[str] = [*ARM_JOINT_NAMES, "gripper"]
ARM_JOINT_COUNT = len(ARM_JOINT_NAMES)  # 7
GRIPPER_NAME = "gripper"

# 控制模式。**没有 move_js**：官方 docstring 标注 Risk level: EXTREMELY HIGH
# （无平滑、无轨迹规划、控制器尽力最快响应，会造成剧烈机械冲击），代码里硬拦。
CONTROL_MODES: tuple[str, ...] = ("move_j", "mit")
# 角色
ROLES: tuple[str, ...] = ("follower", "leader")

# 固件档位 -> 是否属于「旧固件」（≤1.11，需要 set_normal_mode() 才有 CAN 推送）。
# 取值来自官方 NeroFW（pyAgxArm/api/arm_options.py）：
#     DEFAULT <= 1.10 / V111 1.11 / V112 1.12 / V120 1.20 / V121 >= 1.21
# ⚠️ 键是**小写**：比较前统一 .lower()，因为 CLI 里写 "DEFAULT" / "V111" 很常见，
#    旧实现用 `in (NeroFW.DEFAULT, NeroFW.V111)` 直接比，大写就直接判错。
_LEGACY_PROFILE_FLAGS: dict[str, bool] = {
    NeroFW.DEFAULT: True,
    NeroFW.V111: True,
    NeroFW.V112: False,
    NeroFW.V120: False,
    NeroFW.V121: False,
}

# ---------------------------------------------------------------------------
# 机械臂状态字（arm_status）
#
# 取值来自 SDK 的 `ArmMsgFeedbackStatusEnum.ArmStatus`
# （pyAgxArm/protocols/can_protocol/msgs/nero/default/feedback/arm_feedback_status.py）。
# 这条报文就是常规关节反馈的第一帧（CAN ID 0x2A1）。
#
# ⚠️ 别把"能读数据"当成"状态正常"：2026-09-16 实测新臂（1.20）在**完全能用**
#    的情况下照样报 `OTHER_ERR(0x0A)` —— 关节角 280 Hz 正常读、使能正常、
#    motion_status 是"到达点位"。所以必须**显式检查**这个状态字。
# ---------------------------------------------------------------------------
ARM_STATUS_NAMES: dict[int, str] = {
    0x00: "NORMAL 正常",
    0x01: "EMERGENCY_STOP 急停",
    0x02: "NO_SOLUTION 无解",
    0x03: "SINGULARITY_POINT 奇异点",
    0x04: "TARGET_POS_EXCEEDS_LIMIT 目标角度超限",
    0x05: "JOINT_COMMUNICATION_ERR 关节通信异常",
    0x06: "JOINT_BRAKE_NOT_RELEASED 关节抱闸未打开",
    0x07: "COLLISION_OCCURRED 发生过碰撞",
    0x08: "OVERSPEED_DURING_TEACHING_DRAG 拖动示教超速",
    0x09: "JOINT_STATUS_ERR 关节状态异常",
    0x0A: "OTHER_ERR 其它异常",
    0x0B: "TEACHING_RECORD 示教记录中",
    0x0C: "TEACHING_EXECUTION 示教执行中",
    0x0D: "TEACHING_PAUSE 示教暂停",
    0x0E: "MAIN_CONTROLLER_NTC_OVER_TEMPERATURE 主控过温",
    0x0F: "RELEASE_RESISTOR_NTC_OVER_TEMPERATURE 释放电阻过温",
    0xFF: "UNKNOWN 未知",
}
# 哪些算"可接受"。0x00 是常态；0x0B~0x0D 是示教流程里的正常阶段，不算故障。
ARM_STATUS_OK: frozenset[int] = frozenset({0x00, 0x0B, 0x0C, 0x0D})


def _as_int(value: Any) -> int | None:
    """把 SDK 返回的枚举/整数统一成 int；拿不到就返回 None。

    SDK 这些字段是 `IntEnumBase` 的实例，`int()` 能转；但也可能直接是 int
    或 None，所以这里统一兜一层。
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def _default_can_backend(role: str = "follower") -> tuple[str, str]:
    """按操作系统 **+ 角色** 给出官方推荐的 CAN 后端与默认通道。

    ⚠️ 默认通道**必须按角色区分**，否则主从会抢同一条总线。

    【为什么】`interface` / `channel` 都留 None 时，主臂和从臂会各自来这里取默认值。
    如果两个角色拿到同一个通道，`connect()` 第二个必然失败（CAN 通道独占），
    而且报错长得像"硬件没插好"，很难定位。

    【对齐 piper 的做法】piper 干脆不给 channel 配置项，直接写死成不同的名字：
        piper_follower -> can_name="can_follower"
        piper_leader   -> can_name="can_master"
    这里保持"可配置"，但把默认值按角色错开：

        Windows : agx_cando + "0"(从) / "1"(主)
                  ※ channel 是 **cando 枚举到的设备序号**，不是固定的物理口。
                    插拔顺序不同可能互换，串不了就干脆两个都显式指定。
        Linux   : socketcan + "can0" / "can1"
        macOS   : slcan     + "/dev/ttyACM0" / "/dev/ttyACM1"

    需要两个适配器（主从各一个）。
    """
    if role not in ROLES:  # pragma: no cover - 由 config 校验兜住
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    is_leader = role == "leader"

    platform_system = system()
    if platform_system == "Windows":
        return "agx_cando", ("1" if is_leader else "0")
    if platform_system == "Linux":
        return "socketcan", ("can1" if is_leader else "can0")
    if platform_system == "Darwin":
        return "slcan", ("/dev/ttyACM1" if is_leader else "/dev/ttyACM0")
    raise RuntimeError(
        f"pyAgxArm 不支持当前操作系统：{platform_system}。"
        "仅支持 Windows(agx_cando) / Linux(socketcan) / macOS(slcan)。"
    )


def _parse_version(version: str) -> tuple[int, ...]:
    """把 '1.10' 解析成 (1, 10)，用于比较。解析不了就返回空元组。"""
    parts = []
    for piece in str(version).strip().split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            return ()
        parts.append(int(digits))
    return tuple(parts)


@dataclass
class NeroMotorsBusConfig:
    """NERO 机械臂硬件配置。"""

    # CAN 后端与通道。None 表示按操作系统自动选择。
    interface: str | None = None
    channel: str | None = None

    # 固件档位。None 表示连接后自动探测。
    # 注意：官方 SDK 的参数名是 `firmeware_version`（官方拼写少一个 a），
    # 这里刻意沿用该拼写以免下游传参时对不上。
    firmeware_version: Any = None

    # ---- 角色 ----
    # "follower" : 从臂/被控对象。普通模式 + 使能，读常规关节反馈（弧度）
    # "leader"   : 主臂/拖拽示教。set_leader_mode() 零力拖拽，读 leader 通道（度）
    role: str = "follower"

    # ---- 反馈来源 ----
    # None / "auto" : 按 role 自动决定（follower 读常规通道，leader 优先读主臂通道）
    # "leader"      : 强制读主臂通道（0x501-0x507，单位是度，内部转弧度）
    # "follower"    : 强制读常规通道（0x2A1-0x2A7，已是弧度）
    # 显式指定通常用于排查：leader 模式下若主臂通道异常，
    # 可强制 'follower' 对比两条通道的读数。
    angle_source: str | None = None

    # ---- 控制方式 ----
    # "move_j" : 供应商位置控制（控制器自带规划与重力补偿）
    # "mit"    : MIT 阻抗模式，逐关节 move_mit，流式下发天然柔顺
    control_mode: str = "move_j"

    # MIT 的 PD 增益与期望速度。官方范围 kp ∈ [0, 500]、kd ∈ [-5, 5]。
    # 【默认值取官方 SDK 的 10.0 / 0.8，不要随手往上调】曾把 kp 调到 60，
    # 结果从臂第一轴一上来就甩出去 90°（过增益失稳 + 撞限位后静默罢工）。
    mit_kp: float = 10.0
    mit_kd: float = 0.8
    mit_v_des: float = 0.0
    mit_t_ff: float = 0.0

    # 打开 SDK 的关节限位。驱动默认**不开**，不开时 move_mit 只在 ±12.5 rad 处
    # 兜底，等于没有任何机械行程保护。
    joint_limits_enabled: bool = True
    # 每次 move_mit 是否自动重发运动模式。开着的话 7 关节 × 60 Hz = 每秒 420 条
    # 模式切换报文白占总线；关掉后由 connect() 设一次即可。
    auto_set_motion_mode: bool = False

    # 使能超时（秒）
    #
    # ⚠️ 2026-09-16 真机实测：旧臂（1.10）上「发使能 -> 回读到 1111111」滞后
    # **超过 15 秒**（read_nero_firmware.py 轮询 15s 仍是 0000000，之后才刷成 1111111）。
    # 所以这个默认值给 5 秒**偏短**，真机上很容易误报"使能失败"。
    # 从臂/遥操作用 8~10 秒更稳。见 enable() 的轮询说明。
    enable_timeout: float = 10.0
    # enable() 轮询期间「重新唤醒 CAN 推送」的间隔（秒）。0 表示不重试。
    #
    # 【为什么必须有这一项】模式指令只有在【已使能】状态下才会生效，
    # 而 connect() 里那次唤醒是在**失能状态**下发的 -> 对「启动时本来就失能」
    # 的臂纯属空操作 -> 推送不开 -> 回读使能状态恒为 0000000 -> 超时失败。
    # 2026-09-16 真机上撞到的就是这个死锁（diag：119 帧全部 hz=0.0、
    # enable_flags=0000000、arm_status=JOINT_BRAKE_NOT_RELEASED）。
    # enable() 发完指令的那一瞬间臂才真正通电，所以**必须在这之后补发**。
    # read_nero_firmware.py 里"重开推送后重试"能救回来，正是这个道理。
    can_push_rewake_interval_s: float = 1.0
    # 探测固件的总超时（秒）
    probe_timeout: float = 15.0
    # 连接后等第一条关节反馈的最长时间（秒）
    read_timeout_s: float = 5.0
    # 收到第一帧反馈后的"预热窗口"（秒）。只有在这个窗口内才做残缺帧过滤，
    # 详见 _read_raw_angles：常开的过滤会把零位附近的合法姿态误判成残缺帧。
    feedback_warmup_s: float = 3.0

    # 运行速度百分比 [0, 100]
    speed_percent: int = 100

    # 单次 write 相对**当前位姿**允许的最大变化量（rad）。None 表示不钳制。
    # 注意：mit 模式下不要用这个做限幅 —— 详见 write() 的说明。
    max_relative_target: float | None = None

    # ---- 夹爪（第 8 轴）----
    # 是否启用 AgxGripper 夹爪
    has_gripper: bool = True
    # 夹爪最大行程（米）
    gripper_max_width_m: float = 0.07
    # 夹爪最大角度（度）。angle 模式下 SDK 返回 0..100，用它归一化。
    gripper_max_angle_deg: float = 100.0
    # 夹爪夹持力（N）
    gripper_force_n: float = 1.0
    # 夹爪只在开合度变化超过这个阈值时才下发（避免 60 Hz 空刷 CAN 总线）
    gripper_send_threshold: float = 0.005
    gripper_mode: str = "width"
    gripper_source: str = "auto"
    disable_gripper_on_disconnect: bool = True
    feedback_timeout_s: float = 0.5
    # auto: <=1.11 raw degrees; >=1.12 SDK-decoded radians.
    leader_angle_unit: str = "auto"

    def __post_init__(self) -> None:
        if self.gripper_mode not in ("width", "angle"):
            raise ValueError("gripper_mode must be 'width' or 'angle'")
        if self.gripper_source not in ("auto", "status", "control"):
            raise ValueError("gripper_source must be 'auto', 'status' or 'control'")
        if self.leader_angle_unit not in ("auto", "deg", "rad"):
            raise ValueError("leader_angle_unit must be 'auto', 'deg' or 'rad'")
        for name in ("feedback_timeout_s", "enable_timeout", "probe_timeout", "read_timeout_s",
                     "gripper_max_width_m", "gripper_max_angle_deg", "gripper_force_n"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.speed_percent <= 100:
            raise ValueError("speed_percent must be in [0, 100]")
        if self.max_relative_target is not None and not math.isfinite(self.max_relative_target):
            raise ValueError("max_relative_target must be finite")
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        if self.angle_source not in (None, "auto", "leader", "follower"):
            raise ValueError(
                f"angle_source 必须是 None/'auto'/'leader'/'follower'，得到 {self.angle_source!r}"
            )
        if self.control_mode == "move_js":
            raise ValueError(
                "control_mode='move_js' 被禁用：官方 docstring 标注 Risk level: EXTREMELY HIGH"
                "（无平滑、无轨迹规划、控制器尽力最快响应，会造成剧烈机械冲击）。"
                "请用 'move_j' 或 'mit'。"
            )
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}, got {self.control_mode!r}")
        if self.mit_kp < 0 or self.mit_kp > 500:
            raise ValueError(f"mit_kp 官方允许范围是 [0, 500]，得到 {self.mit_kp}")
        if self.mit_kd < -5 or self.mit_kd > 5:
            raise ValueError(f"mit_kd 官方允许范围是 [-5, 5]，得到 {self.mit_kd}")
        if self.max_relative_target is not None and self.max_relative_target < 0:
            raise ValueError("max_relative_target 必须 >= 0（或 None 表示不钳制）")
        if self.read_timeout_s <= 0:
            raise ValueError(f"read_timeout_s 必须为正数，得到 {self.read_timeout_s}")
        if self.can_push_rewake_interval_s < 0:
            raise ValueError(
                f"can_push_rewake_interval_s 必须 >= 0（0 表示不重试），得到 {self.can_push_rewake_interval_s}"
            )
        if self.feedback_warmup_s < 0:
            raise ValueError(f"feedback_warmup_s 必须 >= 0，得到 {self.feedback_warmup_s}")
        if self.gripper_max_width_m <= 0:
            raise ValueError(f"gripper_max_width_m 必须为正数，得到 {self.gripper_max_width_m}")
        if self.gripper_max_angle_deg <= 0:
            raise ValueError(f"gripper_max_angle_deg 必须为正数，得到 {self.gripper_max_angle_deg}")
        if self.gripper_force_n <= 0:
            raise ValueError(f"gripper_force_n 必须为正数（N），得到 {self.gripper_force_n}")
        if not (0 <= self.gripper_send_threshold <= 1):
            raise ValueError(
                f"gripper_send_threshold 必须在 [0, 1] 内，得到 {self.gripper_send_threshold}"
            )


class NeroMotorsBus:
    """LeRobot -> NERO 硬件适配层。

    对外提供 LeRobot 需要的接口：
        connect(enable=True) / disconnect() / read() / write()
        is_connected / is_calibrated / motor_names(=motors)

    上层 Robot / Teleoperator 不直接接触 pyAgxArm。

    角色（config.role）决定读哪条反馈通道、以及连上后进哪个模式：
        "follower" -> 普通模式，读常规关节反馈（弧度）
        "leader"   -> 主臂模式（零力拖拽），读 leader 反馈通道（**度**，内部转弧度）
    """

    def __init__(self, config: NeroMotorsBusConfig):
        self.config = config

        # 未显式指定的通道**按角色**取默认值：主臂和从臂刻意错开，
        # 免得两边都落到同一条 CAN 总线上（见 _default_can_backend）。
        if config.interface is None or config.channel is None:
            iface, ch = _default_can_backend(config.role)
            if config.interface is None:
                config.interface = iface
            if config.channel is None:
                config.channel = ch

        self.robot = None
        self.gripper = None

        self._is_connected = False
        self._is_enabled = False
        self._firmware_version: str | None = None
        # 是否进过需要显式退出的模式（主臂模式 / MIT）。
        # 只有进过才在 disconnect 时退 —— 普通从臂直接 disable 即可，
        # 滥调模式切换会把 CAN 推送关掉（见 disconnect 的说明）。
        self._entered_special_mode = False

        # ---- 残缺帧过滤状态 ----
        # 收到第一帧原始反馈的时刻。残缺帧过滤只在它之后的 feedback_warmup_s
        # 秒内生效（见 _read_raw_angles）。
        self._first_frame_at: float | None = None
        # 最近一次成功读到的有效关节角（读不到时供上层兜底）
        self._last_read: list[float] | None = None
        # 最近一次真正下发的目标角（限幅/插值的基准）
        self._last_sent: list[float] | None = None
        self._warned_no_feedback = False
        # "配置了夹爪但读不到"只告警一次（read() 是 60 Hz 调的，不能刷屏）
        self._warned_gripper_unreadable = False
        # 最近一次下发/读到的夹爪开合度
        self._last_gripper_sent: float | None = None
        self._last_gripper_read: float | None = None
        self._last_read_at: float | None = None
        self._last_gripper_read_at: float | None = None
        self._feedback_stamps: dict[str, tuple[float, float]] = {}
        # 读数冻结检测（通道停更但缓存帧仍返回同一个值，完全静默）
        self._last_raw_key: tuple[float, ...] | None = None
        self._frozen_since: float | None = None
        self._frozen_reported = False
        # 定期健康自检 + 读帧分类计数
        #
        # 【为什么要分类计数】`_read_raw_angles()` 有三道关，任一不过都返回 None，
        # 从外面只看得到「读不到」三个字，分不清是"推送关了"还是"采样太快、读到
        # 装配未完成的半成品帧"。2026-09-16 定位 run_arm.py 第 4 段那个 FAIL 时，
        # 正是靠这些计数才把原因区分开的。
        self.stats: dict[str, Any] = {
            "read_ok": 0,
            "reject_none": 0,  # get_joint_angles()/leader 通道都没数据
            "reject_unparsable": 0,  # 帧内容解析不了
            "reject_incomplete": 0,  # 长度不足 7（7 条分帧还没装配完）
            "reject_hz": 0,  # hz <= 0（帧没凑齐或已陈旧）
            "reject_partial": 0,  # 预热窗口内的残缺帧
            "last_raw_hz": None,  # 最近一次原始帧的 hz，最直接的现场证据
        }

    @property
    def role(self) -> str:
        return self.config.role

    @property
    def is_leader(self) -> bool:
        return self.config.role == "leader"

    # ==========================================================
    # 模式切换（透传给 SDK）
    # ==========================================================

    def set_normal_mode(self) -> None:
        """进普通模式。**唯一**能把 enable_can_push 置 ENABLE 的函数。

        不使能电机、不会动臂，只改推送/联动配置。
        ⚠️ `set_leader_mode()` / `set_follower_mode()` 反而把推送置 DISABLE ——
        想开推送只能用这个。
        """
        self._check_connected()
        try:
            self.robot.set_normal_mode()
        except Exception:
            logger.debug("set_normal_mode() 失败", exc_info=True)

    def set_leader_mode(self) -> None:
        """进主臂模式（零力拖拽/重力补偿）。人有在场要求，会关闭常规 CAN 推送。"""
        self._check_connected()
        try:
            self.robot.set_leader_mode()
            self._entered_special_mode = True
        except Exception:
            logger.warning("set_leader_mode() 失败", exc_info=True)

    def normal_feedback_hz(self) -> float | None:
        """NORMAL 关节通道（0x2A1-0x2A7）的 hz，不看 leader 通道。

        为什么需要它：`wait_for_angles()` 会优先解析 leader 通道，而零力拖拽
        模式下那条通道照样活着。于是臂还停在 drag mode、控制器根本不理会
        位置指令时，它仍然报告"反馈正常"。验证归位必须显式看 NORMAL 通道。
        """
        if not self._is_connected or self.robot is None:
            return None
        try:
            msg = self.robot.get_joint_angles()
        except Exception:
            return None
        if msg is None:
            return None
        hz = getattr(msg, "hz", None)
        try:
            return float(hz) if hz is not None else None
        except (TypeError, ValueError):
            return None

    def wait_for_normal_feedback(self, timeout_s: float = 3.0) -> bool:
        """等 NORMAL 通道真的恢复上报。"""
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            hz = self.normal_feedback_hz()
            if hz is not None and hz > 0.0:
                return True
            time.sleep(0.05)
        return False

    def prepare_for_control(self) -> dict[str, Any]:
        """下发前的驱动配置：关节限位 + 是否每帧重发运动模式。

        `joint_limits_enabled` 驱动默认**不开**，不开时 move_mit 只在 ±12.5 rad
        处兜底，等于没有任何机械行程保护。MIT 模式下这条尤其重要。
        """
        self._check_connected()
        report: dict[str, Any] = {
            "joint_limits_enabled": self.config.joint_limits_enabled,
            "auto_set_motion_mode": self.config.auto_set_motion_mode,
        }

        for attr, value in (
            ("_joint_limits_enabled", self.config.joint_limits_enabled),
            ("_auto_set_motion_mode", self.config.auto_set_motion_mode),
        ):
            try:
                if hasattr(self.robot, attr):
                    setattr(self.robot, attr, value)
                else:
                    report.setdefault("missing", []).append(attr)
            except Exception:
                logger.debug("设置 %s 失败", attr, exc_info=True)
                report.setdefault("failed", []).append(attr)

        return report

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """本次连接实际使用的关节限位（rad），形如 {'joint1': (lo, hi), ...}。

        ⚠️ 键名跟 SDK 一致是 **`joint1`**（不带下划线），LeRobot 侧叫 `joint_1`，
        转换见 NeroFollower 里的 `_sdk_joint_key()`。

        走公开 API `create_agx_arm_config()` 读 `config["joint_limits"]`，
        拿到的是**实际生效**的那份（含用户通过 joint_limits 覆盖的部分），
        比伸手去 import `pyAgxArm.api.agx_arm_factory` 的内部常量稳。

        限位预设只跟**机型**有关、跟固件档位无关，所以未连接时也能取
        （档位还没探测就用 DEFAULT 去问，结果一样）。
        """
        profile = self.config.firmeware_version or NeroFW.DEFAULT
        try:
            cfg = self._build_config(profile)
        except Exception:
            logger.debug("取关节限位失败", exc_info=True)
            return {}

        raw = cfg.get("joint_limits") or {}
        limits: dict[str, tuple[float, float]] = {}
        for name, pair in raw.items():
            try:
                low, high = float(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError):
                logger.debug("限位项 %s 无法解析：%r", name, pair)
                continue
            limits[str(name)] = (low, high)
        return limits

    # ==========================================================
    # 属性
    # ==========================================================

    @property
    def motor_names(self) -> list[str]:
        """LeRobot 生成 action / observation.state 用的维度名。"""
        return (
            list(JOINT_NAMES) if self.config.has_gripper else list(ARM_JOINT_NAMES)
        )

    @property
    def has_gripper(self) -> bool:
        """**配置里**是否要求接管夹爪（静态值，连接前后一致）。

        这是"要不要夹爪"的唯一真相，决定 `motor_names` / `read()` 的键集合，
        也是上层 features 维度的依据。

        ⚠️ 2026-09-19 起 `_init_gripper()` 失败会**直接抛错**，不再把它改成
        False，所以这个值**不随连接结果变化** —— 这样"连接前算好的 features"
        与"连接后的实际键集合"才不会错位（那正是录制时 KeyError 的根因）。

        想知道夹爪驱动**实际**有没有初始化成功，用 `gripper_available`。
        """
        return bool(self.config.has_gripper)

    @property
    def gripper_available(self) -> bool:
        """夹爪驱动**实际**是否可用（运行时状态）。

        `has_gripper` 回答"要不要夹爪"，这个回答"现在能不能用"。
        两者不一致只可能出现在"配置要夹爪、但硬件一路读不到"的场景，
        此时 `read_gripper()` 返回 None，`read()` 会告警一次。
        """
        return self.gripper is not None

    @property
    def motors(self) -> list[str]:
        """piper_follower 的 get_motor_names() 读的是 bus.motors"""
        return self.motor_names

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    @property
    def is_calibrated(self) -> bool:
        """piper_follower:111 读这个属性。NERO 无软件标定，连上即视为已标定。"""
        return self._is_connected

    @property
    def firmware_version(self) -> str | None:
        """主控固件版本号（如 '1.10'）。未连接时为 None。"""
        return self._firmware_version

    # ==========================================================
    # 连接
    # ==========================================================

    def connect(self, enable: bool = True, enter_leader_mode: bool = True) -> bool:
        if self._is_connected:
            raise RuntimeError("NERO is already connected.")
        try:
            return self._connect(enable, enter_leader_mode=enter_leader_mode)
        except BaseException:
            self.disconnect()
            raise

    def connect_feedback_only(self) -> bool:
        """Open the CAN feedback stream without changing actuator state.

        Unlike ``connect(enable=False)``, this path does not call the SDK's
        disable routines and does not initialize or command the gripper. It is
        intended for diagnostics and policy preview while the arm remains in
        the state selected by the operator or pendant.
        """
        if self._is_connected:
            raise RuntimeError("NERO is already connected.")
        if self.config.firmeware_version is None:
            self._probe_firmware()
        else:
            self._firmware_version = str(self.config.firmeware_version)
        self.robot = AgxArmFactory.create_arm(self._build_config(self.config.firmeware_version))
        if self.robot is None:
            raise RuntimeError("Failed to create NERO robot instance.")
        try:
            self.robot.connect()
            self._is_connected = True
            self._is_enabled = False
            if hasattr(self.robot, "get_config"):
                actual = self.robot.get_config().get("firmeware_version")
                if actual is not None:
                    self.config.firmeware_version = actual
            self._first_frame_at = None
            self._last_read = None
            self._last_sent = None
            self._last_read_at = None
            self._last_gripper_read = None
            self._last_gripper_read_at = None
            self._feedback_stamps.clear()
            if self._is_legacy_firmware():
                try:
                    self.robot.set_normal_mode()
                except Exception:
                    logger.debug("set_normal_mode() failed", exc_info=True)
                time.sleep(0.2)
            return self.wait_for_angles(timeout_s=self.config.read_timeout_s)
        except BaseException:
            # The SDK connection itself must still be released, but do not use
            # disconnect(), which would send disable commands on this path.
            self.close_keeping_enabled()
            raise

    def _connect(self, enable: bool = True, enter_leader_mode: bool = True) -> bool:
        """建立 CAN 通信，按角色切模式，并按需使能。

        参数 enable 与 PiperMotorsBus.connect(enable=...) 签名保持一致
        （piper_follower:128 写的是 self.bus.connect(enable=True)）。

        返回：是否处于期望的使能状态。

        流程（顺序有讲究，别调换）：
            1) 探测/确定固件档位 -> 2) 正式连接 -> 3) 开 CAN 推送
            -> 4) 初始化夹爪 -> 5) 使能 -> 6) 按角色切模式 -> 7) 等首帧反馈
        """
        if self._is_connected:
            raise RuntimeError("NERO is already connected.")

        # 1) 未指定档位就先探测固件，NERO 的通信行为随固件版本变化
        if self.config.firmeware_version is None:
            self._probe_firmware()
        else:
            self._firmware_version = str(self.config.firmeware_version)

        # 2) 用确定的档位建立正式连接
        self.robot = AgxArmFactory.create_arm(self._build_config(self.config.firmeware_version))
        if self.robot is None:
            raise RuntimeError("Failed to create NERO robot instance.")

        self.robot.connect()
        self._is_connected = True
        self._is_enabled = False
        # SDK may resolve the requested profile during connection.
        if hasattr(self.robot, "get_config"):
            actual = self.robot.get_config().get("firmeware_version")
            if actual is not None:
                self.config.firmeware_version = actual

        # 重新连接要重新走预热窗口（残缺帧过滤会在窗口内重新生效）
        self._first_frame_at = None
        self._last_read = None
        self._last_sent = None
        self._last_raw_key = None
        self._frozen_since = None
        self._frozen_reported = False
        self._warned_no_feedback = False
        self._warned_gripper_unreadable = False
        self._last_read_at = None
        self._last_gripper_read_at = None
        self._last_gripper_read = None
        self._last_gripper_sent = None
        self._feedback_stamps.clear()

        # 3) 开 CAN 推送。
        #    ⚠️ 必须**无条件**下发，不能只写在"使能轮询的循环体"里。
        #    NERO 的 CAN 推送默认是关的，而机械臂常常已经处于使能状态
        #    （上次没失能干净 / 刚被别的脚本或 Web 上位机使能过），
        #    这时 enable() 第一次就成功、循环体一次都不执行 -> 推送始终没开 ->
        #    读不到关节角，看起来像硬件坏了。
        #    副作用很隐蔽：行为会随"机械臂之前的状态"漂移，不可复现。
        #    固件 ≤1.11 需要 set_normal_mode()（唯一把 enable_can_push 置 ENABLE
        #    的函数）；set_leader_mode()/set_follower_mode() 反而置 DISABLE。
        #
        #    【为什么这里只发一次】实测反复调用会把固件的应答挤掉
        #    （见 _probe_firmware 的顺序安排）。一次足够开推送，
        #    "状态有没有刷新"交给后面的 enable() 轮询去等。
        #    对比：D:\arm_demo 那份实现发 3 次，是为了对付"驱动卡在 MIT 模式
        #    时忽略单次模式切换"——那是**退出模式**的场景，本文件放在
        #    disconnect() 里处理（那里确实发 3 次）。
        if self._is_legacy_firmware():
            try:
                self.robot.set_normal_mode()
            except Exception:
                logger.debug("set_normal_mode() failed", exc_info=True)
            time.sleep(0.2)

        # 4) 初始化夹爪（必须在 connect 之后，且每个实例只能调一次）
        if self.config.has_gripper:
            self._init_gripper()

        # 5) 使能 / 失能（内部会轮询回读确认，可能要等几秒）
        ok = self.enable() if enable else self.disable()
        if enable and not ok:
            raise TimeoutError(f"NERO enable not confirmed within {self.config.enable_timeout}s")

        # 6) 按角色切模式
        if enter_leader_mode or not self.is_leader:
            self._apply_role_mode()

        # 6b) 推送存活确认。
        #     enable() 报 True 只说明"使能位读到了 1"，**不代表 CAN 推送真的在发**。
        #     2026-09-16 实测过这种组合：connect() 返回 True、紧接着的采样循环
        #     却因为 hz<=0 一帧都过不了完整性过滤。这里读一次原始帧的 hz 当体检，
        #     不健康就再唤醒一次并明确告警 —— 免得又把"推送关了"误诊成硬件坏了。
        #     注意主臂模式（leader）下常规推送是**故意关掉**的，不能拿 hz 判它。
        if enable and self._is_enabled and not self.is_leader:
            hz = self._raw_frame_hz()
            if hz is not None and hz <= 0.0:
                logger.warning(
                    "CAN 推送疑似未生效（原始帧 hz=%s）。关节反馈会被完整性过滤全部拦下，"
                    "表现为「读得到连接、读不到关节角」。正在补发一次模式指令；"
                    "若后续仍报读不到，请先确认臂在启动时是**失能**状态（本机实测过该死锁）。",
                    hz,
                )
                if self._enable_can_push():
                    time.sleep(0.3)

        # 7) 设定速度（仅从臂需要；主臂是手动拖拽，没有速度概念）
        if self._is_enabled and not self.is_leader:
            self.set_speed_percent(self.config.speed_percent)

        logger.info(
            "NERO connected. role=%s firmware=%s enabled=%s mode=%s",
            self.config.role,
            self._firmware_version,
            self._is_enabled,
            self.config.control_mode,
        )
        return ok

    def _apply_role_mode(self) -> None:
        """按角色切换工作模式。

        - follower：留在**普通模式**。**绝不**调 set_follower_mode() ——
          它会把 enable_can_push 置 DISABLE（linkage 0xFC），那是给"两台臂挂在
          同一根 CAN 总线上做硬件主从配对"用的。主从臂插在**两个独立适配器**上时
          硬件联动根本建立不起来，只能软件桥接：读主臂 -> 写从臂。
          从臂必须保留普通模式：推送开着（能读自身反馈）+ 接受显式运动指令。
        - leader：set_leader_mode() 进零力拖拽（重力补偿），可用手拖动。
        """
        if not self.is_leader:
            # 从臂留在普通模式（见 docstring）。MIT 也是一种特殊模式，
            # 退出时要显式处理，这里先记下来。
            if self.config.control_mode == "mit":
                self._entered_special_mode = True
            return

        self._entered_special_mode = True
        try:
            self.robot.set_leader_mode()
        except Exception as exc:
            raise RuntimeError("NERO failed to enter leader mode") from exc
        time.sleep(0.3)

        # set_leader_mode() 会关掉常规 CAN 推送，此后真正反映主臂姿态的只有
        # leader 通道（0x501-0x507）。若该通道没上线，读到的会是"进 leader mode
        # 之前的最后一帧缓存"——主臂怎么拖、读数纹丝不动，从臂完全不跟随。
        # 所以这里等它上线，等到就早点返回，等不到给出可操作的告警。
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            try:
                if self.robot.get_leader_joint_angles() is not None:
                    logger.debug("leader 反馈通道已上线")
                    return
            except Exception:
                logger.debug("get_leader_joint_angles() 探测失败", exc_info=True)
            time.sleep(0.05)

        logger.warning(
            "leader 反馈通道 3s 内未上线。若启动后拖动主臂而从臂不跟随，先查这里"
            "（leader mode 下常规推送已关闭，常规通道读数可能停在旧值）。"
        )

    def _probe_firmware(self) -> None:
        """探测 NERO 固件版本，据此选定驱动档位。

        为什么必须探测（2026-09-15 在真机 1.10 固件上实测）：

          - 固件 ≤1.10 的臂在 default 档下能读关节角，但固件查询
            （CAN ID 0x4AF）拿不到应答，get_firmware() 返回 None。
          - 在 v112 / v121 档下固件查询能正常返回版本号。

        所以先用高阶档位问出版本号，再据此重建配置。
        实测：profile=v112 → {'software_version': '1.10'}
        """
        logger.info("Probing NERO firmware version ...")

        probe_order = [NeroFW.V121, NeroFW.V112, NeroFW.DEFAULT]
        software_version: str | None = None
        last_error: Exception | None = None

        for profile in probe_order:
            probe_robot = None
            try:
                probe_robot = AgxArmFactory.create_arm(self._build_config(profile))
                probe_robot.connect()

                deadline = time.time() + self.config.probe_timeout / len(probe_order)
                while time.time() < deadline:
                    fw_info = probe_robot.get_firmware(timeout=1.0, min_interval=0.5)
                    if fw_info is not None:
                        software_version = fw_info.get("software_version")
                        if software_version:
                            break
                    time.sleep(0.2)

                if software_version:
                    logger.info("NERO firmware detected via profile=%s: %s", profile, software_version)
                    break
            except Exception as exc:
                last_error = exc
                logger.debug("探测档位 %s 失败：%s", profile, exc, exc_info=True)
            finally:
                if probe_robot is not None:
                    try:
                        probe_robot.disconnect()
                    except Exception:
                        logger.debug("probe_robot.disconnect() failed", exc_info=True)
                # Windows 上 CAN 句柄释放有延迟，立刻重开会 Access denied
                time.sleep(1.0)

        if not software_version:
            raise TimeoutError(
                f"探测 NERO 固件版本失败（尝试档位 {probe_order}）。"
                f"请检查：CAN 通道 {self.config.channel} 是否被其他进程占用、"
                f"波特率是否为 1Mbps、主控是否上电。"
                + (f" 最后一次错误：{last_error}" if last_error else "")
            )

        self._firmware_version = software_version
        self.config.firmeware_version = resolve_firmware_profile("nero", software_version)
        logger.info(
            "NERO firmware detected: %s -> profile %s", software_version, self.config.firmeware_version
        )

    def _build_config(self, firmeware_version: Any) -> dict:
        return create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=firmeware_version,
            interface=self.config.interface,
            channel=self.config.channel,
        )

    def _init_gripper(self) -> None:
        """初始化 AgxGripper。**失败就抛**，不再静默降级。

        ⚠️ 2026-09-19 修（原实现是"失败则降级为无夹爪"，埋了个维度错位的雷）

        旧版失败时执行 `self.config.has_gripper = False`。看着友好，实际是：

        - `NeroFollower.action_features` 是**连接之前**就算好的 ——
          `record.py` 先读 features 建数据集（:274-301），再 `robot.connect()`（:307）；
        - 而 `bus.read()` 返回哪些键，是**连接之后**按 `config.has_gripper` 决定的；
        - 于是降级后：features 声明 **8** 键、`get_observation()` 只给 **7** 键
          -> `build_dataset_frame()` 直接 KeyError，而且是在录数据的时候才炸。

        而且"降级"本身语义就错了：用户把 `use_gripper` 设成 True 等于声明
        「这台臂有夹爪」。中途少一个轴属于**必须解决的配置/硬件不一致**，
        不该由底层代码悄悄替用户决定丢掉那根轴 —— 那样录出来的数据集
        会少一列，且没有任何报错。

        要走无夹爪的臂，请显式设 `NeroFollowerConfig.use_gripper=False`
        （维度变成 7；**主从两侧必须一致**）。
        """
        try:
            self.gripper = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        except Exception as exc:
            # 先把状态清干净再抛，避免留下半个初始化结果
            self.gripper = None
            raise RuntimeError(
                "初始化 AgxGripper 失败，但 config.has_gripper=True。\n"
                "这里**不能静默降级** —— 降级会让 read() 少返回 gripper 键，"
                "而 features 里声明了它，录制数据集时会 KeyError（且是在录的时候才炸）。\n"
                "二选一：\n"
                "  1) 把末端夹爪接好（上电、检查接线）后重连；\n"
                "  2) 这台臂确实不装夹爪 -> 把 **use_gripper 设为 False**"
                "（NeroFollowerConfig.use_gripper；主臂侧 NeroLeaderConfig 也要一致，"
                "否则遥操作时 action 键对不上）。\n"
                f"底层错误：{type(exc).__name__}: {exc}"
            ) from exc

    # ==========================================================
    # 使能 / 失能
    # ==========================================================

    def enable(self, timeout: float | None = None) -> bool:
        """使能全部关节，返回是否确认使能成功。

        ⚠️ 2026-09-15 真机实测修正（上一版写错了）

        上一版只看 robot.enable() 的返回值就下结论，理由是"官方 enable 是阻塞调用"。
        **这个前提是错的。** 实测证据（同一次运行）：
            使能后立即读  -> 0000000    (旧值)
            ... 中间做别的事约十几秒 ...
            最后再读      -> 1111111    (真值才刷新出来)
        说明 enable() 是**只发不等**（fire-and-forget），发完立刻回读必然读到旧值。
        低速反馈帧只有 1 Hz 量级，帧没刷新前读到的都是上一次的状态。

        正确做法 = 发指令 + **轮询回读实际状态**直到确认（或超时）：
        - 只认 get_joints_enable_status_list() 的真实读值，不认 enable() 的返回值；
        - 帧没刷新就等一会儿再读，避免把"还没刷到"误判成"使能失败"。

        这也是 Piper 的思路（piper.py 的 connect 里就是 while 轮询
        driver_enable_status + sleep(0.5)），只是它轮询的是 6 个独立电机字段。

        参数：
            timeout: 覆盖 config.enable_timeout（秒）。臂若已使能会自动快速返回。

        【2026-09-16 真机补充】轮询期间会**周期性补发模式指令**去唤醒 CAN 推送
        （间隔见 config.can_push_rewake_interval_s）。没有这一步，"启动时处于
        失能状态"的臂会陷入死锁：模式指令不生效 -> 推送不开 -> 使能位读不到
        -> 被误判成"使能失败"。详见循环内的注释。
        """
        self._check_connected()

        timeout = self.config.enable_timeout if timeout is None else timeout

        # 先发指令。enable() 的返回值不可信，但"抛异常"是有意义的。
        try:
            self.robot.enable()
        except Exception:
            logger.debug("enable() 调用异常", exc_info=True)

        # 轮询回读，直到确认全部关节使能
        #
        # 🔑 关键：轮询期间**必须补发模式指令**去唤醒 CAN 推送。
        #    2026-09-16 真机诊断出来的死锁：
        #        不 enable         -> 模式指令在失能态不生效（石沉大海）
        #        指令不生效         -> CAN 推送不开
        #        推送不开            -> 回读恒为 0000000 -> 超时失败
        #    connect() 里那次唤醒是发在**失能**状态下的，对「启动时本来就失能」
        #    的臂等于没发；而上面那句 robot.enable() 是只发不等，发完的**这一瞬间**
        #    臂才真正通电 —— 补发必须发生在它之后，否则永远解不开。
        #    （D:\arm_demo 那份实现也是把 set_normal_mode() 放在这个重试循环里。）
        #
        #    只在**尚未确认**时补发：确认成功就直接 return，不会多打报文。
        #    这样也避开了「反复调用会挤掉固件应答」那条约束 ——
        #    那条管的是 _probe_firmware() 的连接探测阶段，不是这里。
        deadline = time.time() + timeout
        flags: list[bool] | None = None
        rewake_interval = float(self.config.can_push_rewake_interval_s or 0.0)
        next_rewake = (
            time.time() + rewake_interval if rewake_interval > 0 else float("inf")
        )
        rewakes = 0
        while time.time() < deadline:
            flags = self._read_enable_flags()
            if flags is not None and all(flags):
                self._is_enabled = True
                logger.debug("NERO enable confirmed: %s", self._format_flags(flags))
                return True

            now = time.time()
            if now >= next_rewake:
                if self._enable_can_push():
                    rewakes += 1
                next_rewake = now + rewake_interval

            time.sleep(0.25)

        self._is_enabled = False
        if flags is None:
            logger.warning(
                "NERO 使能未确认：读不到使能状态（CAN 推送未开）。"
                "已补发 %d 次唤醒指令仍无效 —— 检查固件档位是否正确、"
                "以及 CAN 总线上是否有别的程序在抢。",
                rewakes,
            )
        else:
            logger.warning(
                "NERO 使能超时（%.1fs），实际状态=%s（%d/%d 个关节已使能）；"
                "期间补发 CAN 推送唤醒 %d 次。若状态恒为 %s 且 arm_status 报"
                "JOINT_BRAKE_NOT_RELEASED，多半是臂在启动时就处于失能态、"
                "模式指令始终没生效。",
                timeout,
                self._format_flags(flags),
                sum(flags),
                len(flags),
                rewakes,
                "0" * len(flags),
            )
        return False

    def disable(self, timeout: float | None = None) -> bool:
        """失能全部关节，返回是否确认失能成功。

        警告：臂若处于抬起姿态会立即掉落。

        与 enable() 同理：disable() 也是只发不等，必须轮询回读实际状态确认。
        """
        self._check_connected()

        timeout = self.config.enable_timeout if timeout is None else timeout

        try:
            self.robot.disable()
        except Exception:
            logger.debug("disable() 调用异常", exc_info=True)

        deadline = time.time() + timeout
        while time.time() < deadline:
            flags = self._read_enable_flags()
            if flags is not None and not any(flags):
                self._is_enabled = False
                return True
            time.sleep(0.25)

        # 失能确认不了也按"已失能"记状态：避免 disconnect() 反复重试失能
        self._is_enabled = False
        logger.warning("NERO 失能未确认（%.1fs 内仍读到使能状态）。", timeout)
        return False

    # ----------------------------------------------------------
    # 使能状态回读（仅供 enable/disable 内部使用）
    # ----------------------------------------------------------

    def _read_enable_flags(self) -> list[bool] | None:
        """读各关节使能状态。读不到返回 None。

        底层是 1 Hz 低速帧，连接后的第一帧是旧值，所以**不要**拿第一次读值下结论，
        由调用方轮询。
        """
        try:
            flags = self.robot.get_joints_enable_status_list()
        except Exception:
            logger.debug("get_joints_enable_status_list() 失败", exc_info=True)
            return None
        if flags is None:
            return None
        try:
            return [bool(x) for x in flags]
        except TypeError:
            return None

    @staticmethod
    def _format_flags(flags: list[bool]) -> str:
        """把 [True, False, ...] 显示成 1010111，便于日志里一眼看出哪个关节没使能。"""
        return "".join("1" if f else "0" for f in flags)

    def _is_legacy_firmware(self) -> bool:
        """固件 ≤1.11 需要 set_normal_mode() 才有 CAN 推送。

        判据分三层（2026-09-19 补全，旧实现只做了第一层的一半）：

          1) `config.firmeware_version` 是**已知档位** -> 查表直接定。
             大小写不敏感（`DEFAULT` / `V111` 也认）。
          2) 档位未知/未设 -> 退回 `_firmware_version` 版本号比较。
          3) 两者都判不出来 -> **返回 True（按旧固件处理）**。

        【第 3 条为什么这么定】两个方向的代价**完全不对称**：
          - 多发一次：在 ≥1.12 上是空操作，SDK 只会发一条 UserWarning。
          - 漏发  ：CAN 推送永远不开 -> 关节角一个都读不到，
                    看起来像"硬件坏了 / 线没接"，是本文档里最高频的误诊。
        所以判不出来时宁可多发。
        """
        profile = self.config.firmeware_version
        if isinstance(profile, str):
            known = _LEGACY_PROFILE_FLAGS.get(profile.strip().lower())
            if known is not None:
                return known

        parsed = _parse_version(self._firmware_version or "")
        if parsed:
            return parsed <= (1, 11)

        logger.debug(
            "固件档位判不出来（profile=%r, version=%r），按旧固件处理"
            "（多发一次 set_normal_mode 是安全方向）",
            profile,
            self._firmware_version,
        )
        return True

    def _enable_can_push(self) -> bool:
        """唤醒 CAN 推送（固件感知）。返回是否真的下发了指令。

        ⚠️ **调用时机有讲究**：模式指令只有在【已使能】状态下才会生效。
        在失能状态下发 = 石沉大海（2026-09-16 实测）。所以这个函数既要用在
        connect() 里（对付"臂已经被上次会话使能过"的情况），更要用在
        enable() 的轮询循环里（那时臂才刚通电）—— 后者才是关键，见 enable()。

        固件分支：
          - ≤1.11：`set_normal_mode()` 是唯一把 enable_can_push 置 ENABLE 的函数。
          - ≥1.12：`set_normal_mode()` 是**空操作**（SDK 自己会发 UserWarning），
                   上电默认就是 follower 且推送已开，所以这里**什么都不发**。
                   ⚠️ 不能改用 `set_follower_mode()` —— 它是给"两台臂挂同一根
                   CAN 总线做硬件主从联动"用的，会把 enable_can_push 置 DISABLE，
                   一发就把推送关掉，下一次读全是 hz=0。
        """
        if not self._is_legacy_firmware():
            return False
        try:
            self.robot.set_normal_mode()
            return True
        except Exception:
            logger.debug("set_normal_mode() failed", exc_info=True)
            return False

    def _raw_frame_hz(self) -> float | None:
        """读一张原始反馈帧的 hz 字段（**不经过滤**）。拿不到返回 None。

        hz 是判断「CAN 推送到底有没有在发」的唯一可靠信号：
        `get_joint_angles()` 在推送关闭时**照样返回缓存帧**（非 None），
        只有 hz 会掉到 0.0，而 `_read_raw_angles()` 的完整性过滤正是靠它拦帧。
        """
        if not self._is_connected or self.robot is None:
            return None
        try:
            msg = (self.robot.get_leader_joint_angles() if self.is_leader
                   else self.robot.get_joint_angles())
        except Exception:
            logger.debug("get_joint_angles() 失败", exc_info=True)
            return None
        if msg is None:
            return None
        hz = getattr(msg, "hz", None)
        if hz is None:
            return None
        try:
            return float(hz)
        except (TypeError, ValueError):
            return None

    # ==========================================================
    # 断开
    # ==========================================================

    def disconnect(self) -> None:
        """失能并断开 NERO。

        顺序讲究（吃过亏）：
          1) 主臂模式必须先退出 —— 主臂模式下 disable() **可能被忽略**
             （<=1.11 用 set_normal_mode()，>=1.12 用 set_follower_mode()）。
          2) 发失能 -> 等反馈刷新 -> 复查。失能状态读的是低速帧（1 Hz 量级），
             "发完立刻查"读到旧帧会误报失败（实测出现过 6 个已关、第 7 个还报
             True 的假象），所以只做有限次重试、失败也照常断开。
          3) 断开场景不无限等确认：用短超时，避免退出时卡好几秒。
        """
        if self.robot is not None:
            if self.gripper is not None and self.config.disable_gripper_on_disconnect:
                try:
                    if not self.gripper.disable_gripper():
                        logger.warning(
                            "UNCONFIRMED: gripper disable request was sent but feedback did not confirm disabled."
                        )
                except Exception:
                    logger.warning(
                        "UNCONFIRMED: gripper disable failed; feedback confirmation is unavailable.",
                        exc_info=True,
                    )
            # 1) 退出主臂模式（若进过）。
            #
            # ⚠️ 2026-09-16 真机固件 **1.20** 上实测到的重要修正：
            #    `set_normal_mode()` 在固件 ≥1.12 上是**空操作**，SDK 会直接发警告：
            #      "set_normal_mode() has no effect on Nero firmware >= 1.12
            #       (NeroFW.V112): only leader/follower modes (Piper-aligned);
            #       default is follower with CAN push on power-up."
            #    所以那条路径必须改用 `set_follower_mode()`。
            #
            #    而且**只在真的进过特殊模式时才需要退** —— 普通从臂模式下
            #    只是使能着，直接 disable 就行；滥调模式切换反而会把 CAN 推送关掉
            #    （`set_follower_mode()` 会把 enable_can_push 置 DISABLE），
            #    导致下一次 connect 读不到数据。
            #
            #    **发多次**是刻意的：机械臂若停在主臂/MIT 模式里，
            #    驱动会**忽略**单次模式切换和失能（实测：teleoperate 退出后
            #    从臂推不动、仍处使能）。
            if self._entered_special_mode:
                self._leave_special_mode()

            # 2) 失能
            if self._is_connected:
                try:
                    self.disable(timeout=1.0)
                except Exception:
                    logger.debug("disable() during disconnect failed", exc_info=True)

            try:
                self.robot.disconnect()
            except Exception:
                logger.debug("robot.disconnect() failed", exc_info=True)

        self.gripper = None
        self.robot = None
        self._is_connected = False
        self._is_enabled = False
        self._entered_special_mode = False
        self._first_frame_at = None
        self._last_raw_key = None
        self._last_read = None
        self._last_read_at = None
        self._last_sent = None
        self._last_gripper_read = None
        self._last_gripper_read_at = None
        self._last_gripper_sent = None
        self._feedback_stamps.clear()
        logger.info("NERO disconnected.")

    def _leave_special_mode(self) -> None:
        """退出主臂 / MIT 等特殊模式，回到能被正常失能的状态（发 3 次）。

        固件档位决定用哪个函数 —— 见 disconnect() 里的说明：
          - ≤1.11：`set_normal_mode()`（唯一把 enable_can_push 置 ENABLE 的那个）
          - ≥1.12：`set_follower_mode()`（此时 normal_mode 是空操作）
        """
        name = "set_normal_mode" if self._is_legacy_firmware() else "set_follower_mode"
        fn = getattr(self.robot, name, None)
        if fn is None:
            logger.debug("SDK 没有 %s()，跳过退模式", name)
            return

        for _ in range(3):
            try:
                fn()
            except Exception:
                logger.debug("%s() 失败", name, exc_info=True)
            time.sleep(0.2)

    # ==========================================================
    # 速度
    # ==========================================================

    def set_speed_percent(self, percent: int = 100) -> None:
        """设定运行速度百分比，范围 [0, 100]，作用于 move_j。"""
        self._check_connected()
        percent = max(0, min(int(percent), 100))
        try:
            self.robot.set_speed_percent(percent)
        except Exception as exc:
            raise RuntimeError(f"Failed to set NERO speed to {percent}%") from exc

    # ==========================================================
    # 读取
    # ==========================================================

    def _resolve_angle_source(self) -> str:
        """决定读哪条反馈通道：显式 angle_source 优先，否则按 role。"""
        if self.config.angle_source in ("leader", "follower", "auto"):
            return self.config.angle_source
        return self.config.role

    def _feedback_is_fresh(self, message: Any, source: str) -> bool:
        """Reject stopped streams without mistaking a stationary pose for packet loss."""
        hz = getattr(message, "hz", None)
        if hz is not None and (not math.isfinite(float(hz)) or float(hz) <= 0):
            return False
        stamp = getattr(message, "timestamp", None)
        if stamp is None:
            return True
        stamp = float(stamp)
        if not math.isfinite(stamp) or stamp <= 0:
            return False
        now = time.monotonic()
        previous = self._feedback_stamps.get(source)
        if previous is None or stamp > previous[0]:
            self._feedback_stamps[source] = (stamp, now)
            return True
        return stamp == previous[0] and now - previous[1] <= self.config.feedback_timeout_s

    def _read_raw_angles(self) -> list[float] | None:
        """读 7 个关节角，**统一返回弧度**，带完整性过滤。读不到返回 None。

        为什么必须有这层（而不是直接 get_joint_angles()）：

        1) 角色决定通道，两条通道**单位不同**
           - follower：普通模式，读常规反馈（0x2A1-0x2A7）。SDK 解码时已含
             DEG2RAD，出来就是弧度。
           - leader：<=1.11 使用旧 float32 通道，按本项目实测从度转弧度；
             >=1.12 使用 SDK 已解码成弧度的新通道，不能重复换算。
             auto 策略优先有效主臂帧，然后尝试常规反馈。

        2) **残缺帧过滤**（这条防的是最严重的事故）
           NERO 的关节反馈是**分帧**发的（每关节一帧），刚连上或推送刚被唤醒时
           会读到"部分关节已更新、其余还是上一轮残留"的混合帧。实测：
               真实 = [-88.59, +59.33, +1.46, +93.28, +0.87, -2.74, +97.28] deg
               读到 = [-88.11,  -0.35, +1.03, +13.78, +0.95, -2.76,  +3.85] deg
           这种帧一旦被当成"当前位姿"去下发保持指令，机械臂会朝错误目标猛甩
           —— 2026-09-13 实测：从臂 J7 因此转了约 195°。
           ⚠️ 过滤**只在预热窗口内生效**：常开会把零位附近的合法姿态误杀
           （编码器分辨率 0.001 度，静止时读到精确 0.0 完全正常），
           那样上层只能长期退回旧值，表现为"主臂怎么动从臂都不跟随"。
        """
        self._check_connected()

        source = self._resolve_angle_source()
        msg = None
        used = None

        if source in ("leader", "auto"):
            msg = self.robot.get_leader_joint_angles()
            if msg is not None and not self._feedback_is_fresh(msg, "leader"):
                msg = None
            if msg is not None:
                used = "leader"

        if msg is None and source in ("follower", "auto"):
            msg = self.robot.get_joint_angles()
            if msg is not None:
                used = "follower"

        if msg is None:
            self.stats["reject_none"] += 1
            return None

        if not self._feedback_is_fresh(msg, used):
            self.stats["reject_hz"] += 1
            return None

        try:
            values = [float(v) for v in list(msg.msg)]
        except (TypeError, ValueError):
            logger.debug("关节反馈内容不可解析", exc_info=True)
            self.stats["reject_unparsable"] += 1
            return None

        if len(values) < ARM_JOINT_COUNT:
            self.stats["reject_incomplete"] += 1
            raise RuntimeError(
                f"Expected {ARM_JOINT_COUNT} joint angles, got {len(values)}."
            )
        values = values[:ARM_JOINT_COUNT]
        if not all(math.isfinite(value) for value in values):
            self.stats["reject_unparsable"] += 1
            return None

        # Only legacy leader frames need conversion; >=1.12 is already radians.
        unit = self.config.leader_angle_unit
        if used == "leader" and (unit == "deg" or (unit == "auto" and self._is_legacy_firmware())):
            values = [v * math.pi / 180.0 for v in values]

        # 2) 帧是否还在推送（hz <= 0 说明这帧没凑齐或已陈旧）
        #
        # ⚠️ 2026-09-16 重要：这里的 hz 是**每次返回的那一帧自己带的**。
        #    而 NERO 的关节反馈是**7 条分帧**（每关节一条 CAN 报文，常规通道
        #    0x2A1-0x2A7）装配成的一条消息 —— 采样比推送还快时，大量调用会落在
        #    "7 条还没装配完"的瞬间，这时 hz 就是 0，于是被这里拦下。
        #    → **采得越密，被拒比例越高**：这是采样方式的问题，不是推送坏了。
        #    判据：`stats["reject_hz"]` 占绝大多数、且 `stats["last_raw_hz"]` 也接近 0
        #    时，先怀疑采样太快；而"稀疏读也失败"才是真推送问题。
        hz = getattr(msg, "hz", None)
        try:
            self.stats["last_raw_hz"] = None if hz is None else float(hz)
        except (TypeError, ValueError):
            self.stats["last_raw_hz"] = None
        if hz is not None and float(hz) <= 0.0:
            self.stats["reject_hz"] += 1
            return None

        # 3) 残缺帧过滤：只在预热窗口内生效。
        #    预热计时必须在残缺帧检查**之前**起算 —— 若放在成功返回时才记，
        #    一旦最初的帧全是残缺帧，计时器永远启动不了，窗口就永远过不去。
        if self._first_frame_at is None:
            self._first_frame_at = time.perf_counter()

        if self._in_warmup():
            zeros = sum(1 for v in values if v == 0.0)
            if 0 < zeros < ARM_JOINT_COUNT:
                logger.debug("丢弃残缺帧（%d/%d 个关节仍为 0）", zeros, ARM_JOINT_COUNT)
                self.stats["reject_partial"] += 1
                return None

        # 4) 读数冻结检测（通道停更但缓存帧仍在 -> 每帧返回同一个值，完全静默）
        self._check_frozen(values, used, hz)

        self.stats["angle_source"] = used
        self.stats["read_ok"] += 1
        return values

    def _in_warmup(self) -> bool:
        """是否还在"推送刚唤醒"的预热窗口内（此时才需要防残缺帧）。"""
        if self._first_frame_at is None:
            return True
        warmup = float(self.config.feedback_warmup_s or 0.0)
        return warmup > 0 and (time.perf_counter() - self._first_frame_at) < warmup

    def _check_frozen(self, values: list[float], used: str | None, hz: Any) -> None:
        """连续 5 秒逐位不变就提示一次（不阻塞、不抛错，只告警）。

        `_read_raw_angles` 返回 None 的情况能抓到"通道没数据"，但抓不到
        【通道停更、缓存帧仍在】—— 后者每帧都返回同一个合法旧值。
        """
        now = time.perf_counter()
        key = tuple(values)
        if key == self._last_raw_key:
            if self._frozen_since is None:
                self._frozen_since = now
            elif now - self._frozen_since >= 5.0 and not self._frozen_reported:
                logger.warning(
                    "读数已连续 %.0f 秒无变化（通道 %s，hz=%s）。静止时属正常，"
                    "拖一下就会变；若你正在拖动机械臂而读数纹丝不动，说明该通道停更。",
                    now - self._frozen_since,
                    used,
                    hz,
                )
                self._frozen_reported = True
        else:
            self._frozen_since = None
            self._frozen_reported = False
        self._last_raw_key = key

    def _read_angles_lenient(self) -> list[float] | None:
        """`_read_raw_angles()` 的宽松包装：**未连接抛，数据问题返回 None**。

        【为什么要这层】`_read_raw_angles()` 的 RuntimeError 有两个来源：
            1) 未连接（`_check_connected()`）
            2) **短帧**：`len(values) < 7`，即 7 条 CAN 反馈报文还没装配完
        两者的处置**必须相反**：
            - 未连接是编程错误 -> 抛，否则"忘了 connect"会被伪装成"读不到数据"
            - 短帧是瞬态数据问题 -> 返回 None，让上层沿用上一帧；
              越界抛出去会直接崩掉 60 Hz 遥操作循环

        实现上就是先单独做连接检查，之后的 RuntimeError 只可能是短帧。
        """
        self._check_connected()
        try:
            return self._read_raw_angles()
        except RuntimeError as exc:
            logger.debug("关节反馈帧不完整，已按拒收处理：%s", exc)
            return None
        except Exception:
            logger.debug("读关节角失败", exc_info=True)
            return None

    def try_read_joint_positions(self) -> list[float] | None:
        """读 7 个关节角（弧度）；帧无效/读不到时返回 None，**不抛异常**。

        上层（Robot / Teleoperator）的常规入口：遥操作循环里读不到反馈时应当
        沿用上一帧而不是崩掉，同时要能察觉"一直读不到"。

        ⚠️ 2026-09-19 修正（原实现与这段 docstring 自相矛盾）
            旧版是：

                try:    return self._read_raw_angles()
                except RuntimeError:  raise        # ← 直接抛出去

            而 `_read_raw_angles()` 在**短帧**（关节数不足 7，7 条 CAN 报文还没
            装配完）时也抛 RuntimeError。于是这个宣称"不抛异常"的 API 变成了会抛：
            一个短帧就能让 60 Hz 遥操作循环**整个崩掉**（`teleoperate()` 只 catch
            KeyboardInterrupt），录制时同理会毁掉当前 episode。

            判据很明显：短帧那条分支**先** `stats["reject_incomplete"] += 1`
            （当成一次拒收计数）、**然后才 raise**，而其它拒收分支全是 return None
            —— 说明本意就是当普通拒收处理。

        现在委托给 `_read_angles_lenient()`：未连接照旧抛，短帧返回 None。
        """
        return self._read_angles_lenient()

    def read_joint_positions(self) -> list[float]:
        """读取七个关节角，单位 rad。返回 [j1, ..., j7]。

        区别于 try_read_joint_positions()：这里是"我必须拿到有效数据"的语义，
        读不到就抛 RuntimeError。同样经过完整性过滤。
        """
        self._check_connected()
        values = self._read_raw_angles()
        if values is None:
            raise RuntimeError("Failed to read NERO joint angles (no valid frame).")
        self._last_read = list(values)
        self._last_read_at = time.monotonic()
        return values

    def present_positions(self) -> list[float] | None:
        """当前实测位姿；短暂丢帧时使用超时窗口内的反馈缓存。

        超时返回 None，不使用目标指令冒充实际位置。

        ⚠️ 2026-09-19：改用 `_read_angles_lenient()` —— 以前直接调
        `_read_raw_angles()`，**短帧会抛出去**，于是 `read()` ->
        `get_observation()` 一路把录制 episode 打崩。它的语义本来就该是
        "读不到就退回上一帧"，不该抛。
        """
        angles = self._read_angles_lenient()
        if angles is not None:
            self._last_read = list(angles)
            self._last_read_at = time.monotonic()
            return angles
        if (self._last_read is not None and self._last_read_at is not None
                and time.monotonic() - self._last_read_at <= self.config.feedback_timeout_s):
            return list(self._last_read)
        return None

    def wait_stable_pose(
        self, n: int = 3, tol_rad: float = 0.004, timeout_s: float = 20.0
    ) -> list[float] | None:
        """连续读到 n 帧互相一致（每关节差 < tol_rad）才返回，否则返回 None。

        任何**要以当前位姿为基准下发运动指令**的场景（点动 / 对齐 / 遥操作起步 /
        锚定起始姿态）都必须先过这一关。0.004 rad ≈ 0.23°。

        【为什么不能省】实测事故：点动 J1 5°，其余关节"保持不动"= 下发读到的
        当前角；而刚连上读到的是残缺帧 -> J7 实际转了约 195°。
        等不到稳定姿态时应当**终止，绝不猜**。
        """
        deadline = time.perf_counter() + timeout_s
        last: list[float] | None = None
        streak = 0
        while time.perf_counter() < deadline:
            cur = self._read_angles_lenient()
            if cur is None:
                streak = 0
                last = None
            elif last is not None and max(
                abs(cur[i] - last[i]) for i in range(ARM_JOINT_COUNT)
            ) < tol_rad:
                streak += 1
                if streak >= n:
                    self._last_read = list(cur)
                    self._last_read_at = time.monotonic()
                    return cur
            else:
                streak = 1
            last = cur
            time.sleep(0.05)
        return None

    def wait_for_angles(self, timeout_s: float | None = None) -> bool:
        """等第一条有效关节反馈，用于 connect 后确认数据链路真的通了。

        用宽松读取：短帧只算"还没等到"，不该把整个 connect 打崩。
        """
        timeout_s = self.config.read_timeout_s if timeout_s is None else timeout_s
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            angles = self._read_angles_lenient()
            if angles is not None:
                self._last_read = list(angles)
                return True
            time.sleep(0.05)
        return False

    def read_health(self) -> dict[str, Any]:
        """读机械臂健康状态（状态字、逐关节故障位、使能位串、CAN 推送），用于自检。

        【为什么必须有】MIT 是力矩控制，机械臂一旦触发碰撞保护/通讯错误就会
        "静默罢工"：上层照常下发指令，从臂纹丝不动，看起来像"跟随逻辑坏了"，
        其实是驱动已经不收命令了。看一眼状态字和故障位就能立刻区分。

        ⚠️ 2026-09-16 修正两处：
          1) 旧版读的是 `msg.comm_error` —— **SDK 这条报文里根本没有这个字段**
             （它的 `_fields_` 是 ctrl_mode / arm_status / mode_feedback /
             teach_status / motion_status / trajectory_num / err_code），
             于是那个键永远是 None，拿它当告警判据等于**死代码**。
             现在改成读真正存在的 `err_code`（16 位，逐关节）。
          2) 光打印状态字不够 —— 要给出 `arm_status_ok` 这种可直接判定的布尔值，
             否则调用方各写各的、总会漏。配套 `health_problems()` 给问题清单。

        返回的键（拿不到的都缺席，不塞假值）：
            arm_status / arm_status_code / arm_status_text / arm_status_ok
            motion_status / ctrl_mode / mode_feedback / teach_status
            err_code / err_joints_communication / err_joints_angle_limit
            enable_flags / can_push_hz / can_push_ok
        """
        health: dict[str, Any] = {}
        if not self._is_connected or self.robot is None:
            return health

        try:
            status = self.robot.get_arm_status()
            if status is not None:
                msg = getattr(status, "msg", None)
                raw = getattr(msg, "arm_status", None)
                code = _as_int(raw)
                health["arm_status"] = raw  # 保留枚举对象，日志里可读性更好
                if code is not None:
                    health["arm_status_code"] = code
                    health["arm_status_text"] = ARM_STATUS_NAMES.get(
                        code, f"未知状态 0x{code:02X}"
                    )
                    health["arm_status_ok"] = code in ARM_STATUS_OK

                for key in ("motion_status", "ctrl_mode", "mode_feedback", "teach_status"):
                    value = getattr(msg, key, None)
                    if value is not None:
                        health[key] = value

                # err_code 是 NERO 真正的故障位图：bit0-6 = 关节 1-7 通信异常，
                # bit8-14 = 关节 1-7 角度超限（映射见 SDK 的 err_code setter）。
                err = _as_int(getattr(msg, "err_code", None))
                if err is not None:
                    health["err_code"] = err
                    health["err_joints_communication"] = [
                        i + 1 for i in range(ARM_JOINT_COUNT) if err & (1 << i)
                    ]
                    health["err_joints_angle_limit"] = [
                        i + 1 for i in range(ARM_JOINT_COUNT) if err & (1 << (i + 8))
                    ]
        except Exception:
            logger.debug("get_arm_status() 失败", exc_info=True)

        flags = self._read_enable_flags()
        if flags is not None:
            health["enable_flags"] = self._format_flags(flags)

        # CAN 推送是否真的在发。这是「读不到关节角」最常见的真凶：
        # 推送关掉时 get_joint_angles() 照样返回缓存帧（非 None），
        # 只有 hz 会掉到 0.0，而 _read_raw_angles() 的完整性过滤正是靠它拦帧。
        hz = self._raw_frame_hz()
        if hz is not None:
            health["can_push_hz"] = hz
            health["can_push_ok"] = hz > 0.0
        return health

    def health_problems(self) -> list[str]:
        """把 `read_health()` 的结果翻译成"人能读的问题清单"。空列表 = 一切正常。

        【为什么要单独有这个方法】光把状态字打印出来，调用方还得自己记住
        哪些值算正常 —— 各写各的就总会漏。这里集中一处，上机脚本直接
        `ok = not bus.health_problems()`。
        """
        h = self.read_health()
        problems: list[str] = []

        if h.get("arm_status_ok") is False:
            problems.append(f"arm_status = {h.get('arm_status_text')}")
        for joint in h.get("err_joints_communication", []):
            problems.append(f"joint_{joint} 通信异常（err_code bit{joint - 1}）")
        for joint in h.get("err_joints_angle_limit", []):
            problems.append(f"joint_{joint} 角度超限（err_code bit{joint + 7}）")
        # In leader mode, set_leader_mode() intentionally stops the normal
        # feedback stream. The leader-specific 0x501-0x507 stream is checked
        # by the angle reader, so normal-stream hz=0 is expected here.
        if h.get("can_push_ok") is False and not self.is_leader:
            problems.append(f"CAN 推送没在发（hz={h.get('can_push_hz')}）")
        flags = h.get("enable_flags")
        if isinstance(flags, str) and flags and "0" in flags:
            problems.append(f"有未使能的关节（enable_flags={flags}）")
        return problems

    # ----------------------------------------------------------
    # 读帧计数（诊断"读不到关节角"到底卡在哪一关）
    # ----------------------------------------------------------

    _READ_STAT_KEYS = (
        "read_ok",
        "reject_none",
        "reject_unparsable",
        "reject_incomplete",
        "reject_hz",
        "reject_partial",
    )

    def read_stats(self) -> dict[str, Any]:
        """读帧计数快照，含总数与成功率（丢帧率 = 1 - ok_ratio）。

        用途：`reset_read_stats()` -> 采样一段 -> `read_stats()`，
        一眼看出"被拒的是哪一类"，而不是笼统地看到"读不到"。
        """
        snap: dict[str, Any] = {k: self.stats.get(k, 0) for k in self._READ_STAT_KEYS}
        snap["last_raw_hz"] = self.stats.get("last_raw_hz")
        snap["angle_source"] = self.stats.get("angle_source")
        total = sum(snap[k] for k in self._READ_STAT_KEYS)
        snap["total"] = total
        snap["ok_ratio"] = (snap["read_ok"] / total) if total else 0.0
        return snap

    def reset_read_stats(self) -> None:
        """清零读帧计数。采样前后各调一次，就只统计这一段。"""
        for key in self._READ_STAT_KEYS:
            self.stats[key] = 0

    def read_gripper(self) -> float | None:
        """读取夹爪开合，归一化到 [0, 1]；读不到返回 None（**不是 0**）。

        官方 get_gripper_status().msg.value：
          - width 模式单位是米 -> 除以最大行程归一化
          - angle 模式单位是度（0..100）-> 除以 gripper_max_angle_deg

        ⚠️ 读不到时返回 None 而不是 0.0：0.0 是"夹爪完全闭合"这个**有效值**，
        用 0.0 表示"读不到"会让上层无法区分，遥操作时可能把夹爪当成已闭合。
        未启用夹爪（has_gripper=False）时也返回 None。
        """
        if not self.config.has_gripper or self.gripper is None:
            return None

        sources = [self.config.gripper_source]
        if sources == ["auto"]:
            sources = ["control", "status"] if self.is_leader else ["status"]
        status = None
        used = None
        for source in sources:
            try:
                getter = (self.gripper.get_gripper_ctrl_states if source == "control"
                          else self.gripper.get_gripper_status)
                candidate = getter()
                if candidate is not None and self._feedback_is_fresh(candidate, f"gripper_{source}"):
                    status, used = candidate, source
                    break
            except Exception:
                logger.debug("Gripper %s feedback unavailable", source, exc_info=True)
        if status is None:
            return None

        try:
            value = float(status.msg.value)
            mode = getattr(status.msg, "mode", "width")
            if used == "control":
                code = int(status.msg.status_code)
                if code not in (1, 5):
                    return None
                mode = "angle" if code == 5 else "width"
                # pyAgxArm 1.0.0 decodes 0x159 as meters for both modes.
                if mode == "angle":
                    value *= 1000.0
        except (AttributeError, TypeError, ValueError):
            logger.debug("夹爪反馈不可解析", exc_info=True)
            return None

        if not math.isfinite(value):
            return None

        if mode == "angle":
            max_a = self.config.gripper_max_angle_deg
            if max_a <= 0:
                return None
            normalized = value / max_a
        elif mode == "width":
            max_w = self.config.gripper_max_width_m
            if max_w <= 0:
                return None
            normalized = value / max_w
        else:
            return None

        result = max(0.0, min(normalized, 1.0))
        self._last_gripper_read = result
        self._last_gripper_read_at = time.monotonic()
        return result

    def read_gripper_or_last(self) -> float | None:
        """读夹爪；仅允许短暂沿用反馈缓存，超时返回 None。"""
        value = self.read_gripper()
        if value is not None:
            return value
        if (self._last_gripper_read is not None and self._last_gripper_read_at is not None
                and time.monotonic() - self._last_gripper_read_at <= self.config.feedback_timeout_s):
            return self._last_gripper_read
        return None

    def read(self) -> dict[str, float]:
        """返回 {motor_name: value}，键序与 motor_names 一致。

        注意：走的是 present_positions()，读不到反馈时退回上一帧而非抛错，
        这样遥操作循环不会因为一帧丢包而中断。

        `has_gripper=False` 时不含 'gripper' 键；为 True 时**一定有**这个键
        —— 否则 `NeroFollower.action_features` 声明的维度就和实际观测对不上，
        录制时 `build_dataset_frame()` 会 KeyError。
        """
        joints = self.present_positions()
        if joints is None:
            raise RuntimeError("Failed to read NERO joint angles (no valid frame).")
        state = dict(zip(ARM_JOINT_NAMES, joints, strict=True))

        if self.config.has_gripper:
            grip = self.read_gripper_or_last()
            if grip is None:
                raise RuntimeError("NERO gripper feedback unavailable or stale; check wiring/use_gripper.")
            state[GRIPPER_NAME] = float(grip)

        return state

    # ==========================================================
    # 写入
    # ==========================================================

    def _send_joints(self, joints: list[float]) -> None:
        """把 7 个目标角（rad）下发。按 control_mode 走 move_j 或 move_mit。"""
        if len(joints) != ARM_JOINT_COUNT or not all(math.isfinite(v) for v in joints):
            raise ValueError("Expected exactly seven finite joint angles")
        if self.is_leader and self._entered_special_mode:
            raise RuntimeError("Joint writes are not allowed after entering leader mode")
        if not self._is_enabled:
            raise RuntimeError("NERO enable has not been confirmed")
        mode = self.config.control_mode
        if mode == "move_j":
            # move_j 会自动切到 MOVE J 模式（控制器自带规划与重力补偿）
            self.robot.move_j(joints)
        elif mode == "mit":
            for idx, p_des in zip(range(1, ARM_JOINT_COUNT + 1), joints, strict=True):
                # 注意 move_mit 内部有个固件 bug 兼容（joint != 6 时 v_des *= -1，
                # 1.21 才修）。默认 v_des=0 所以这条补丁不会咬人；
                # 一旦要下发非零 v_des，务必核对符号。
                self.robot.move_mit(
                    joint_index=idx,
                    p_des=p_des,
                    v_des=self.config.mit_v_des,
                    kp=self.config.mit_kp,
                    kd=self.config.mit_kd,
                    t_ff=self.config.mit_t_ff,
                )
        else:  # pragma: no cover - __post_init__ 已校验
            raise ValueError(f"unsupported control_mode: {mode!r}")

    def write(self, target_joint: list[float]) -> None:
        """控制七轴 + 夹爪。

        target_joint: [j1, ..., j7] 或 [j1, ..., j7, gripper]
            关节单位 rad；夹爪为 [0, 1] 归一化值。

        connect(enable=True) 必须先确认使能，主臂禁止下发关节目标。

        ⚠️ max_relative_target 的含义随 control_mode 变化，别改混
        （2026-09-13 实测教训）：
          - move_j：相对**当前实测位姿**钳制（_clamp_relative），位置控制下安全。
          - mit   ：**不做**这里钳制。MIT 的力矩 = kp × (p_des - 实测)，
            把 p_des 钳在实测 ±cap 会把力矩上限钉死在 kp × cap
            （kp=60、cap=0.05 时仅约 3 N·m），从臂连自重都抗不住，
            表现为"主臂动了从臂几乎不动"。mit 的限速应由上层对
            **参考轨迹**（上一拍指令值）限步来完成 —— 见 slow_move_to()，
            以及 NeroFollower.send_action() 的做法。
        """
        self._check_connected()

        target_joint = [float(x) for x in target_joint]
        if len(target_joint) not in (ARM_JOINT_COUNT, ARM_JOINT_COUNT + 1) or not all(
            math.isfinite(v) for v in target_joint
        ):
            raise ValueError(
                f"NERO requires at least {ARM_JOINT_COUNT} joints, got {len(target_joint)}."
            )

        if len(target_joint) == ARM_JOINT_COUNT + 1 and not self.config.has_gripper:
            raise ValueError("Gripper target supplied with has_gripper=False")
        self.send_joints(target_joint[:ARM_JOINT_COUNT])

        if self.config.has_gripper and len(target_joint) > ARM_JOINT_COUNT:
            self.write_gripper(target_joint[ARM_JOINT_COUNT])

    def send_joints(self, joints: list[float]) -> list[float]:
        """下发 7 个目标角（rad），返回**实际下发**的值（可能被钳制）。

        给上层（Robot.send_action）用的入口：返回值可以原样写进数据集，
        保证训练数据里的 action 就是机械臂真正收到的那一条。
        """
        self._check_connected()
        joints = [float(x) for x in joints]
        if len(joints) != ARM_JOINT_COUNT or not all(math.isfinite(v) for v in joints):
            raise ValueError(
                f"NERO requires at least {ARM_JOINT_COUNT} joints, got {len(joints)}."
            )
        joints = joints[:ARM_JOINT_COUNT]
        if self.config.control_mode == "move_j":
            joints = self._clamp_relative(joints)
        if self.config.joint_limits_enabled:
            limits = self.joint_limits()
            if any(f"joint{i}" not in limits for i in range(1, ARM_JOINT_COUNT + 1)):
                raise RuntimeError("NERO joint limits unavailable")
            joints = [max(limits[f"joint{i}"][0], min(value, limits[f"joint{i}"][1]))
                      for i, value in enumerate(joints, 1)]
        self._send_joints(joints)
        self._last_sent = list(joints)
        return list(joints)

    def write_gripper(self, value: float) -> float:
        """控制夹爪，value 为 [0, 1] 归一化值（0=闭合，1=最大开口）。

        夹爪是"到位置"型指令（不是力矩控制），每帧重复下发同一个值只会白占
        CAN 总线，所以只在变化超过 gripper_send_threshold 时才真正发送。
        设阈值为 0 可恢复"每帧都发"。
        """
        self._check_connected()
        if not self.config.has_gripper or self.gripper is None:
            raise RuntimeError("NERO gripper is not available")
        if self.is_leader:
            raise RuntimeError("Gripper writes are not allowed on a leader bus")
        if not self._is_enabled:
            raise RuntimeError("NERO enable has not been confirmed")
        if not math.isfinite(float(value)):
            raise ValueError("Gripper target must be finite")

        value = max(0.0, min(float(value), 1.0))
        prev = self._last_gripper_sent
        threshold = float(self.config.gripper_send_threshold or 0.0)
        if prev is not None and threshold > 0 and abs(value - prev) < threshold:
            return prev

        try:
            move = (self.gripper.move_gripper_deg if self.config.gripper_mode == "angle"
                    else self.gripper.move_gripper_m)
            scale = (self.config.gripper_max_angle_deg if self.config.gripper_mode == "angle"
                     else self.config.gripper_max_width_m)
            move(
                value=value * scale,
                force=self.config.gripper_force_n,
            )
            self._last_gripper_sent = value
        except Exception as exc:
            raise RuntimeError("NERO gripper command failed") from exc
        return value

    def slow_move_to(
        self,
        target: list[float],
        duration_s: float = 4.0,
        fps: int = 60,
        hold_after_s: float = 0.5,
    ) -> None:
        """把机械臂**平滑地**从当前位姿插值走到 target。

        target 长度 7（只要关节）或 8（7 关节 + 夹爪开合度 0..1）。
        夹爪不参与插值：它是"到位置"型指令，起步时直接对齐一次即可。

        用途：遥操作开始前把从臂"对齐"到主臂当前姿位。主从臂各自独立标零，
        静止读数可以差 100° 以上，直接开始遥操作会让从臂一路追过去，
        起步那一下容易吓人；先摆到同一姿态，之后的跟随就是小幅微调。

        注意：
          - 余弦缓动插值，两端速度为 0，起停都不突兀，因此**不再**套
            max_relative_target（每步增量本来就很小）；
          - 关节超限仍由 SDK 自行 clamp；
          - 起步前先确位姿稳定（wait_stable_pose），等不到就报错、绝不猜。

        用 `target` 时**不支持 mit 之外的限速**，因为它本身就是限速的。
        """
        self._check_connected()
        if len(target) not in (ARM_JOINT_COUNT, ARM_JOINT_COUNT + 1):
            raise ValueError(
                f"target must have {ARM_JOINT_COUNT} or {ARM_JOINT_COUNT + 1} elements "
                f"(7 joints [+ gripper]), got {len(target)}"
            )
        if len(target) == ARM_JOINT_COUNT + 1 and not self.config.has_gripper:
            raise ValueError(
                "target 有 8 个元素（含夹爪）但本臂 effector='none'；"
                "要么接上夹爪，要么只传 7 个关节值。"
            )

        start_pos = self.present_positions()
        if start_pos is None:
            # 刚连上时可能还在反馈预热窗口内（残缺帧被拦），等姿态读稳再取一次。
            # 直接抛错会让"起步对齐"在连接后头几秒必然失败。
            if self.wait_stable_pose(timeout_s=self.config.read_timeout_s) is not None:
                start_pos = self.present_positions()
        if start_pos is None:
            raise RuntimeError(
                "读不到当前位姿，无法做平滑对齐。请确认 CAN 推送已开启"
                "（固件 <= 1.11 需 firmware='default'/'v111'）。"
            )

        # 夹爪先对齐（行程短、动作慢，先走更省时间）
        if len(target) == ARM_JOINT_COUNT + 1:
            self.write_gripper(target[ARM_JOINT_COUNT])

        steps = max(1, int(duration_s * fps))
        period = 1.0 / fps
        for k in range(1, steps + 1):
            alpha = 0.5 - 0.5 * math.cos(math.pi * k / steps)  # 余弦缓动
            joints = [
                start_pos[i] + (target[i] - start_pos[i]) * alpha
                for i in range(ARM_JOINT_COUNT)
            ]
            self._send_joints(joints)
            self._last_sent = joints
            time.sleep(period)

        # 收尾精确落到目标并保持一下。
        # 【坑】_send_joints 只吃 7 个关节，第 8 位是夹爪、必须走 write_gripper ——
        # 直接把 8 元素的 target 传进去会在 zip(strict=True) 处报
        # "zip() argument 2 is longer than argument 1"（2026-09-13 实测）。
        joints = [float(x) for x in target[:ARM_JOINT_COUNT]]
        self._send_joints(joints)
        self._last_sent = joints
        if len(target) == ARM_JOINT_COUNT + 1:
            self.write_gripper(target[ARM_JOINT_COUNT])
        time.sleep(hold_after_s)

    def _clamp_relative(self, target: list[float]) -> list[float]:
        """相对当前位姿钳制单步变化量（**仅 move_j 模式使用**）。

        mit 模式不要走这里，原因见 write() 的 docstring：
        把 p_des 钳在实测附近会把 MIT 的力矩上限钉死。

        注意：LeRobot 的机器人侧已有 ensure_safe_goal_position()
        （robots/utils.py）做同类保护，这里是硬件层兜底，
        给直接使用 NeroMotorsBus 的场景留一道保险。
        """
        cap = self.config.max_relative_target
        if cap is None:
            return target

        current = self.present_positions()
        if current is None:
            raise RuntimeError("Cannot limit target without fresh NERO joint feedback")

        clamped = [cur + max(-cap, min(tgt - cur, cap)) for tgt, cur in zip(target, current, strict=True)]

        if clamped != target:
            logger.warning("目标关节角超出单步上限 %.4f rad，已钳制以避免危险动作。", cap)
        return clamped

    # ==========================================================
    # 兼容 piper_follower 的调用
    # ==========================================================

    def safe_disconnect(self) -> None:
        """Piper 版语义是"先移到安全位再失能"。

        NERO 官方没有"移动到安全位"的接口，贸然 move_j 反而危险，
        因此这里只做失能准备，不主动发运动指令。
        """

    def close_keeping_enabled(self) -> None:
        """Close the CAN session but leave motors and the effector powered.

        Why this exists: `disconnect()` deliberately disables first, which makes a
        raised arm collapse under gravity. That is the right default, but it is the
        wrong behaviour when the operator is holding or has just finished a task
        and the process is only being restarted to collect more episodes.

        No enable/disable command, no mode switch and no motion command is sent
        here. The motors keep whatever state the controller already has, so the
        arm stays where it is. The operator is responsible for the state left
        behind, and the caller is expected to say so in the output.
        """
        if self.robot is None:
            self.gripper = None
            self._is_connected = False
            return
        try:
            self.robot.disconnect()
        except Exception:
            logger.debug("robot.disconnect() failed", exc_info=True)
        self.gripper = None
        self.robot = None
        self._is_connected = False
        self._is_enabled = False
        self._entered_special_mode = False
        self._first_frame_at = None
        self._last_raw_key = None
        self._last_read = None
        self._last_read_at = None
        self._last_sent = None
        self._last_gripper_read = None
        self._last_gripper_read_at = None
        self._last_gripper_sent = None
        self._feedback_stamps.clear()
        logger.info("NERO CAN session closed; motors were left enabled.")

    def apply_calibration(self) -> None:
        """Piper 版语义是"移动到初始位置"。NERO 不做任何运动。"""

    def apply_calibration_master(self) -> None:
        """同上。"""

    def set_calibration(self) -> None:
        return

    def revert_calibration(self) -> None:
        return

    # ==========================================================
    # 工具
    # ==========================================================

    def _check_connected(self) -> None:
        if not self._is_connected or self.robot is None:
            raise RuntimeError("NERO is not connected. Call connect() first.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
