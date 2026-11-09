#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("nero_leader")
@dataclass
class NeroLeaderConfig(TeleoperatorConfig):
    """松灵 NERO 八轴主臂（leader）配置：7 关节 + 夹爪。

    主臂的职责只有"读"：把手臂切到零力拖拽（`set_leader_mode()`），
    人手拖到哪就报哪，由 `NeroFollower` 负责把这些值写进从臂。

    【与 piper_leader 的关键差异 —— 照抄会差 57.3 倍】
    piper_leader 的 `get_action()` 里有 `val / 57324.840764`，因为
    piper_sdk 给的关节值是 0.001 度。NERO **完全不同**：
      - 常规关节反馈（0x2A1-0x2A7）：SDK 解码时已含 DEG2RAD，出来就是**弧度**
      - <=1.11 主臂反馈（0x501-0x507）：旧 float32 通道，按本项目实测从度转弧度
      - >=1.12 主臂反馈：SDK 已转换成弧度，不能重复转换
    这两件事都由 `NeroMotorsBus._read_raw_angles()` 统一处理好了，
    本模块**不要**再做任何单位换算。

    连接参数（None = 按**角色**自动选，主臂 -> "1" / "can1"）：
      - Windows : interface="agx_cando", channel="1"
      - Linux   : interface="socketcan", channel="can1"
    ⚠️ 从臂侧默认是 **"0" / "can0"**，两边天然错开（对齐 piper 的
    can_follower / can_master）。但 Windows 上 channel 是 cando 的
    **设备序号**、会随插拔顺序变，**建议两边都显式指定**。
    """

    # ---- CAN 连接 ----
    interface: str | None = None
    # 留 None 时**按角色自动选**：主臂 -> "1"（Linux 是 "can1"）。
    # 从臂侧会自动选 "0" / "can0"，两边刻意错开，免得主从抢同一条总线
    # （对齐 piper：把 follower/master 写死成两个不同的名字）。
    # ⚠️ Windows 上 channel 是 cando 枚举到的**设备序号**，会随插拔顺序变；
    #    **建议显式指定**，并且接错时先看是不是两台臂换了序号。
    channel: str | None = None
    # None = 连接前自动探测固件版本（推荐）
    firmware: str | None = None

    # ---- 反馈来源 ----
    # "auto"     : 先试主臂通道（0x501-0x507），拿不到再退回常规通道。
    #              推荐 —— set_leader_mode() 会关掉常规推送，但不同固件行为有差异。
    # "leader"   : 只读主臂通道（按固件版本统一到弧度）
    # "follower" : 只读常规通道（已是弧度）。leader mode 下可能停在进模式前的
    #              最后一帧缓存 -> 表现为"主臂怎么拖、读数纹丝不动"。
    angle_source: str = "auto"
    leader_angle_unit: str = "auto"
    enable_timeout: float = 25.0
    feedback_timeout_s: float = 0.5

    # ---- 方向与偏置修正 ----
    # 两台臂各自独立标零，方向也可能相反。逐关节 `输出 = 读数 * sign + offset`。
    # 关节方向相反时**只改主臂侧**，不要去改从臂。
    # 长度 8 = 7 关节 + 夹爪；也可只给 7 个（夹爪按 +1/0 处理）。
    joint_signs: list[float] = field(default_factory=lambda: [1.0] * 8)
    joint_offsets: list[float] = field(default_factory=lambda: [0.0] * 8)

    # ---- 读取/时序 ----
    # 连接后等第一条关节反馈的最长时间（秒）
    read_timeout_s: float = 5.0
    # 收到第一帧反馈后的"预热窗口"（秒）。只有窗口内才做残缺帧过滤。
    feedback_warmup_s: float = 3.0
    # 连接时若始终读不到关节角是否直接报错
    require_angles_on_connect: bool = True

    # ---- 夹爪（第 8 轴）----
    # True = 读夹爪开合度，action 里带 `gripper.pos`。
    # ⚠️ 必须与从臂侧一致，否则从臂 send_action 会因为缺少 gripper 键而 KeyError。
    use_gripper: bool = True
    # 夹爪满量程，必须与从臂侧、上位机里配置的真实行程一致。
    gripper_max_width_m: float = 0.07
    gripper_max_angle_deg: float = 100.0
    gripper_source: str = "auto"
    # 主臂夹爪**从头到尾都读不到**时，action 里的 `gripper.pos` 取什么值。
    #   0.0 = 完整闭合，1.0 = 最大开口。
    # 【为什么需要这一项】主臂的 `read_gripper_or_last()` 给不出"上一帧指令值"
    # 这种兜底（主臂从不下发夹爪指令，`_last_gripper_sent` 恒为 None），
    # 所以"读不到"时必须由**配置**明确一个值，不能偷偷补 0.0
    # —— 那会让从臂夹爪被持续指令闭合，而且完全静默（2026-09-19 审计发现）。
    # 一旦真的从未读到，运行时会**告警**并说明当前取的是哪个值。
    # 默认 None：从未收到反馈则报错。显式数值只供兼容旧流程使用。
    gripper_fallback_when_unreadable: float | None = None

    def __post_init__(self):
        if self.gripper_source not in ("auto", "status", "control"):
            raise ValueError("gripper_source must be 'auto', 'status' or 'control'")
        if self.leader_angle_unit not in ("auto", "deg", "rad"):
            raise ValueError("leader_angle_unit must be 'auto', 'deg' or 'rad'")
        for name in ("enable_timeout", "feedback_timeout_s", "read_timeout_s",
                     "gripper_max_width_m", "gripper_max_angle_deg"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.angle_source not in ("auto", "leader", "follower"):
            raise ValueError(
                f"angle_source must be one of ('auto', 'leader', 'follower'), "
                f"got {self.angle_source!r}"
            )
        if self.read_timeout_s <= 0:
            raise ValueError(f"read_timeout_s 必须为正数，得到 {self.read_timeout_s}")
        if self.feedback_warmup_s < 0:
            raise ValueError(f"feedback_warmup_s 必须 >= 0，得到 {self.feedback_warmup_s}")
        if self.gripper_max_width_m <= 0:
            raise ValueError(f"gripper_max_width_m 必须为正数，得到 {self.gripper_max_width_m}")
        if self.gripper_max_angle_deg <= 0:
            raise ValueError(f"gripper_max_angle_deg 必须为正数，得到 {self.gripper_max_angle_deg}")
        if (self.gripper_fallback_when_unreadable is not None
                and not 0.0 <= self.gripper_fallback_when_unreadable <= 1.0):
            raise ValueError(
                "gripper_fallback_when_unreadable 必须在 [0, 1] 内"
                f"（0.0=闭合，1.0=最大开口），得到 {self.gripper_fallback_when_unreadable}"
            )

        for name in ("joint_signs", "joint_offsets"):
            values = list(getattr(self, name))
            if len(values) == 7:
                # 只给关节：夹爪按默认（+1 / 0）补齐
                values = [*values, 1.0 if name == "joint_signs" else 0.0]
            if len(values) != 8:
                raise ValueError(
                    f"{name} 必须是 7 或 8 个元素（7 关节 [+ 夹爪]），得到 {len(values)}"
                )
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} must contain finite values")
            setattr(self, name, values)
