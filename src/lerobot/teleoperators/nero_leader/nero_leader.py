#!/usr/bin/env python

"""松灵 NERO 八轴主臂（leader）遥操作器。

职责单一：把手臂切到零力拖拽，**读**出人手的姿态，转成 action 字典。
真正的写入由 `NeroFollower.send_action()` 完成
（LeRobot 的循环是 `action = teleop.get_action()` -> `robot.send_action(action)`）。

与 `piper_leader` 的差异（为什么不能复用）：

  1. SDK 不同：pyAgxArm vs piper_sdk，本项目两者都使用 CAN
  2. 自由度不同：NERO 7 关节 + 夹爪；PiPER 6 轴 + 夹爪
  3. **单位不同**：piper_leader 里有 `val / 57324.840764`（0.001 度 -> 弧度）。
     NERO 常规通道 SDK 已解码成弧度；主臂 <=1.11 按度换算，>=1.12 已是弧度。
     两者都由 `NeroMotorsBus._read_raw_angles()` 统一处理，
     本模块**不做任何单位换算**。
  4. 主臂模式不同：NERO 用 `set_leader_mode()`；PiPER 没有这个接口。

【旧固件单位记录】以下 0x501-0x507 说明仅适用于 <=1.11：
  - 常规关节反馈 (0x2A1-0x2A7)：解码含 ×1e-3×DEG2RAD，出来就是弧度
  - 主臂反馈 (0x501-0x507)：`decode_leader_joint_state_by_index` 是 float32
    **裸直出，固件发的是"度"**。SDK 消息类的中文注释写"单位rad"是**错的**
    （英文注释写 0.001 degrees，且实测读数形如 85.31，弧度不可能是这个数）。
  不转的话，主臂摆 1.5° 会被当成 1.5 rad 下发给从臂，**差 57.3 倍**。

【为什么 get_action() 必须做"读不到就用上一帧"】遥操作循环是 60 Hz，
丢一帧很常见（CAN 拥塞、推送短暂中断）。若直接返回 0 或抛错，
从臂会盯着一个错误的/陈旧的目标。但"一直读不到"是严重故障
（从臂不跟随且没有任何动作），所以连续读不到必须告警。

【夹爪要单独说一句 —— 2026-09-19 审计修掉的隐患】主臂的
`_last_gripper_sent` **恒为 None**（主臂只读不写、从不下发夹爪指令），
所以 `read_gripper_or_last()` 在主臂上**给不出兜底值**：一旦读不到就直接是 None。
旧代码在这里补了 `0.0`，而 0.0 是"**完整闭合**"这个有效值 ——
结果是主臂夹爪一读不到（没装/没上电/接线松），从臂夹爪就被**持续指令闭合**，
而且**全程静默、没有任何告警**。
现在：短暂丢帧沿用成功读数，超过 feedback_timeout_s 报错；从未读到默认报错。
旧流程可显式配置 gripper_fallback_when_unreadable，新双臂入口不使用该兜底。
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.nero import (
    ARM_JOINT_COUNT,
    ARM_JOINT_NAMES,
    GRIPPER_NAME,
    NeroMotorsBus,
    NeroMotorsBusConfig,
)

from ..teleoperator import Teleoperator
from .config_nero_leader import NeroLeaderConfig

logger = logging.getLogger(__name__)


class NeroLeader(Teleoperator):
    """LeRobot 主臂接口：读出人手拖动出的关节角与夹爪开合度。"""

    config_class = NeroLeaderConfig
    name = "nero_leader"

    def __init__(self, config: NeroLeaderConfig):
        super().__init__(config)
        self.config = config

        self.bus = NeroMotorsBus(
            NeroMotorsBusConfig(
                interface=config.interface,
                channel=config.channel,
                firmeware_version=config.firmware,
                role="leader",
                angle_source=config.angle_source,
                # 主臂只读不写，控制模式对它是无关项；保留默认以免自检报错
                control_mode="move_j",
                has_gripper=config.use_gripper,
                gripper_max_width_m=config.gripper_max_width_m,
                gripper_max_angle_deg=config.gripper_max_angle_deg,
                gripper_source=config.gripper_source,
                read_timeout_s=config.read_timeout_s,
                feedback_warmup_s=config.feedback_warmup_s,
                enable_timeout=config.enable_timeout,
                feedback_timeout_s=config.feedback_timeout_s,
                leader_angle_unit=config.leader_angle_unit,
            )
        )

        self.logs: dict[str, Any] = {}
        self._is_connected = False
        self._is_calibrated = False
        # 最近一次有效读数（读不到时兜底，避免遥操作循环抖动）
        self._last_angles: list[float] | None = None
        self._last_gripper: float | None = None
        self._last_source: str | None = None
        # 连续读不到反馈的帧数（诊断"读数被冻结/通道停更"）
        self._stale_streak = 0
        # 连续读不到**夹爪**的帧数。与关节分开统计 —— 夹爪没装/没上电时，
        # 关节反馈照样正常，只有夹爪这一路是空的。
        self._gripper_stale_streak = 0
        self._angles_at: float | None = None
        self._gripper_at: float | None = None

    # ------------------------------------------------------------------ 描述
    @property
    def action_features(self) -> dict[str, type]:
        """与目标从臂的 action 键保持完全一致（顺序也要一致：7 关节 -> 夹爪）。"""
        names = list(ARM_JOINT_NAMES)
        if self.config.use_gripper:
            names.append(GRIPPER_NAME)
        return {f"{n}.pos": float for n in names}

    @property
    def feedback_features(self) -> dict[str, type]:
        """NERO 主臂当前不做力反馈。"""
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    # -------------------------------------------------------------- 连接/断开
    def connect(self, calibrate: bool = True, enter_leader_mode: bool = True) -> None:
        """连接主臂并进入零力拖拽模式。

        ⚠️ 零力拖拽意味着**机械臂会失去位置保持**（只做重力补偿）。
        连接前请确认：人在现场、周围无障碍、急停在手边。
        """
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(
            "connecting NERO leader: interface=%s channel=%s firmware=%s source=%s gripper=%s",
            self.config.interface,
            self.config.channel,
            self.config.firmware,
            self.config.angle_source,
            self.config.use_gripper,
        )

        try:
            # role="leader" 时 bus.connect() 内部会：
            #   开推送 -> 初始化夹爪 -> 使能 -> set_leader_mode -> 等 leader 通道上线
            if enter_leader_mode:
                connected = self.bus.connect(enable=True)
            else:
                connected = self.bus.connect(enable=True, enter_leader_mode=False)
            if not connected:
                raise RuntimeError("Leader enable was not confirmed")
        except Exception as exc:
            self.disconnect()
            raise RuntimeError(
                f"{self} 连接失败：{exc}\n"
                "请依次检查：\n"
                "  1) 机械臂是否已上电、底座指示灯是否为绿色\n"
                "  2) CAN 线 CAN_H/CAN_L 是否接对、插紧（黄色 CAN_H / 蓝色 CAN_L）\n"
                f"  3) channel 是否选对（当前 channel={self.config.channel}）；"
                "主从两台臂必须用**不同**通道\n"
                "  4) 有没有别的程序（Web 上位机、另一个脚本）占用同一个 CAN 通道\n"
                "  5) firmware 档位是否与真实固件一致，或留 None 让程序自动探测"
            ) from exc

        self._is_connected = True

        if calibrate:
            self.calibrate()

        # 等第一条关节反馈，确认数据链路真的通了。
        # 【为什么必须等】leader 通道没上线时读到的是"进 leader mode 之前的
        # 最后一帧缓存"，之后每帧数值相同 -> 主臂怎么拖、action 纹丝不动，
        # 从臂完全不动，而且**没有任何报错**。
        if not enter_leader_mode:
            logger.info("%s connected in normal position mode for startup-pose return.", self)
            return

        if not self.bus.wait_for_angles(self.config.read_timeout_s):
            if self.config.require_angles_on_connect:
                self.disconnect()
                raise RuntimeError(
                    f"{self}: 已进入主臂模式，但 {self.config.read_timeout_s}s 内读不到关节角。\n"
                    "  1) 固件 <= 1.11 需要开 CAN 推送（connect 里会自动发 set_normal_mode）\n"
                    "  2) 可把 angle_source 设为 'follower' 强制读常规反馈试试\n"
                    "  3) 先用 run_arm.py 单独验证读取链路"
                )
            logger.warning("%s: no joint angles yet after connect.", self)

        if self.bus.has_gripper:
            self.bus.read_gripper_or_last()

        logger.info("%s connected (angle source = %s).", self, self._last_source)

    def enter_leader_mode(self) -> None:
        """Switch to zero-force leader mode after any position-mode setup."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self}: not connected")
        self.bus.set_leader_mode()
        if not self.bus.wait_for_angles(self.config.read_timeout_s):
            raise TimeoutError(f"{self}: leader feedback did not become fresh")

    def calibrate(self) -> None:
        """NERO 主臂的"标定" = 以当前姿态作为拖拽参考，**不做回零运动**。

        真正的编码器零点标定由松灵上位机完成（Web 端"设置零点"），本代码不做。
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._is_calibrated = True

    def configure(self, **kwargs) -> None:
        """无额外运行时配置。"""

    def send_feedback(self, *args, **kwargs) -> None:
        """NERO 主臂无反馈功能（no-op），保留接口以满足基类契约。"""

    def disconnect(self) -> None:
        """断开主臂。

        ⚠️ 不能照抄 piper_leader 的 `self.bus.safe_disconnect()` 后就不管了：
        主臂模式下 `disable()` **可能被忽略**，必须先退出主臂模式。
        这套顺序（退模式 -> 失能 -> 断开）已封在 bus.disconnect() 里。
        """
        try:
            self.bus.disconnect()
        except Exception as exc:
            logger.warning("%s: bus.disconnect() failed: %r", self, exc)

        self._is_connected = False
        self._is_calibrated = False
        # 缓存必须一起清掉：否则重连后第一帧还在读不到时，
        # 会拿**上一次会话**的夹爪读数/关节角当"上一帧"用。
        self._last_angles = None
        self._last_gripper = None
        self._last_source = None
        self._stale_streak = 0
        self._gripper_stale_streak = 0
        self._angles_at = None
        self._gripper_at = None
        logger.info("%s disconnected.", self)

    def close_keeping_enabled(self) -> None:
        """Close the CAN session but leave the leader's motors enabled.

        Sends no disable and no mode switch, so the arm keeps holding position.
        """
        try:
            self.bus.close_keeping_enabled()
        except Exception as exc:
            logger.warning("%s: 保持使能断开失败: %r", self, exc)
        self._is_connected = False
        self._is_calibrated = False
        self._last_angles = None
        self._last_gripper = None
        self._last_source = None
        self._stale_streak = 0
        self._gripper_stale_streak = 0
        self._angles_at = None
        self._gripper_at = None
        logger.info("%s CAN session closed; motors left enabled.", self)

    # ------------------------------------------------------------------ 读取
    def wait_stable_pose(
        self, n: int = 3, tol_rad: float = 0.004, timeout_s: float = 20.0
    ) -> list[float] | None:
        """连续读到 n 帧互相一致才返回，否则 None。

        与 `NeroFollower` 侧同签名：遥操作起步前两边各调一次，
        确认读到的都是**稳定真实**的主臂姿态，再拿去做对齐/起步基准，
        避免第一帧残缺数据把从臂带跑偏。
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        angles = self.bus.wait_stable_pose(n=n, tol_rad=tol_rad, timeout_s=timeout_s)
        if angles is not None:
            self._last_angles = list(angles)
        return angles

    def get_joint_positions(self) -> list[float] | None:
        """当前主臂 7 个关节角（弧度）；读不到返回 None。"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        angles = self.bus.try_read_joint_positions()
        if angles is not None:
            self._last_angles = list(angles)
        return angles

    def get_action(self) -> dict[str, float]:
        """读主臂 7 个关节角（弧度）+ 夹爪开合度，转成从臂可用的 action。

        关节：读不到反馈时沿用上一帧（避免遥操作循环里出现全 0 抖动），
        但连续读不到会告警 —— 那意味着**从臂不会跟随**，是必须立刻知道的故障。

        夹爪：读不到时**同样沿用上一帧成功读到的值**，绝不静默补 0.0
        （0.0 = 完整闭合，会让从臂夹爪被持续指令闭合）。
        若从头到尾都没读到过，用 `config.gripper_fallback_when_unreadable`，
        并明确告警说明当前取的是哪个值。
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        angles = self.bus.try_read_joint_positions()
        if angles is None:
            self._stale_streak += 1
            if self._stale_streak == 1 or self._stale_streak % 120 == 0:
                logger.warning(
                    "%s: 已连续 %d 帧读不到关节角，action 将沿用旧值 —— **从臂不会跟随！**"
                    " 检查 CAN 推送是否开启 / angle_source 是否选对（当前 %s）。",
                    self,
                    self._stale_streak,
                    self._last_source,
                )
            if self._last_angles is None:
                raise RuntimeError(
                    f"{self}: 还没有任何有效的关节反馈，无法生成 action。"
                )
            if self._angles_at is None or time.monotonic() - self._angles_at > self.config.feedback_timeout_s:
                raise TimeoutError("NERO leader joint feedback timed out")
            angles = list(self._last_angles)
        else:
            self._stale_streak = 0
            self._last_angles = list(angles)
            self._angles_at = time.monotonic()

        self._last_source = self.bus.stats.get("angle_source")

        cfg = self.config
        signs = list(cfg.joint_signs)
        offsets = list(cfg.joint_offsets)

        action: dict[str, float] = {}
        for i, name in enumerate(ARM_JOINT_NAMES):
            action[f"{name}.pos"] = float(angles[i] * signs[i] + offsets[i])

        if cfg.use_gripper:
            raw = self.bus.read_gripper()
            if raw is not None:
                self._last_gripper = float(raw)
                self._gripper_stale_streak = 0
                self._gripper_at = time.monotonic()
            else:
                if self._last_gripper is None and cfg.gripper_fallback_when_unreadable is None:
                    raise RuntimeError("No NERO leader gripper feedback; refusing to invent a target")
                if self._gripper_at is not None and time.monotonic() - self._gripper_at > cfg.feedback_timeout_s:
                    raise TimeoutError("NERO leader gripper feedback timed out")
                # ⚠️ 这里**绝不能静默补 0.0**（2026-09-19 审计发现的原隐患）：
                #    主臂的 `_last_gripper_sent` 恒为 None（主臂从不下发夹爪指令），
                #    所以 `read_gripper_or_last()` 在主臂上给不出"上一帧指令值"这种
                #    兜底 —— 只要读不到，旧代码就落到 0.0，而 0.0 = **完整闭合**，
                #    从臂夹爪会被持续指令闭合，且**全程没有任何告警**。
                #    正确做法：沿用上一帧**成功读到**的值；从未读到过就用配置里
                #    明确指定的 fallback，并把后果讲清楚。
                self._gripper_stale_streak += 1
                if self._gripper_stale_streak == 1 or self._gripper_stale_streak % 120 == 0:
                    if self._last_gripper is not None:
                        consequence = (
                            f"当前沿用上一帧读数 {self._last_gripper:.4f}。"
                        )
                    else:
                        consequence = (
                            "⚠️ 从未成功读到过，将从 "
                            "config.gripper_fallback_when_unreadable 取值 "
                            f"{cfg.gripper_fallback_when_unreadable:.4f}"
                            "（0.0 表示闭合，从臂夹爪会一直保持在这个位置）。"
                        )
                    logger.warning(
                        "%s: 已连续 %d 帧读不到主臂夹爪（约 %.1fs @60Hz）。%s"
                        " 检查顺序："
                        "① 主臂末端是否真的装了夹爪；"
                        "② 夹爪是否上电、接线是否松；"
                        "③ 若主臂本来就不接夹爪，请把**主从两侧**的 use_gripper "
                        "都设为 False（两边必须一致，否则从臂 send_action 会因为"
                        "缺 gripper.pos 而 KeyError）。",
                        self,
                        self._gripper_stale_streak,
                        self._gripper_stale_streak / 60.0,
                        consequence,
                    )
                if self._last_gripper is None:
                    self._last_gripper = float(cfg.gripper_fallback_when_unreadable)

            action[f"{GRIPPER_NAME}.pos"] = float(
                max(0.0, min(self._last_gripper * signs[7] + offsets[7], 1.0))
            )

        if not all(math.isfinite(value) for value in action.values()):
            raise ValueError("NERO leader produced a non-finite action")
        self.logs["read_ms"] = (time.perf_counter() - start) * 1e3
        self.logs["angle_source"] = self._last_source
        # >0 表示读数已冻结这么多个周期（正常应恒为 0）
        self.logs["stale_frames"] = self._stale_streak
        # >0 表示夹爪已经这么多帧读不到了（关节可能完全正常，要分开看）
        self.logs["gripper_stale_frames"] = self._gripper_stale_streak
        if self._last_gripper is not None:
            self.logs["gripper"] = self._last_gripper
        return action
