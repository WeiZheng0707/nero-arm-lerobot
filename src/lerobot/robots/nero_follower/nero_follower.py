#!/usr/bin/env python

"""松灵 NERO 从臂（follower）＝ 7 关节 + 夹爪。

结构和 `piper_follower` 一一对应，只是把 `PiperMotorsBus` 换成 `NeroMotorsBus`：

    piper_follower                      nero_follower
    ------------------------------      ------------------------------
    self.bus = PiperMotorsBus(...)      self.bus = NeroMotorsBus(...)
    bus.connect(enable=True)            bus.connect(enable=True)
    bus.read()                          bus.read()
    bus.write(target_joints)            bus.send_joints(target_joints)
    bus.safe_disconnect()               （空桩，NERO 没有"先移到安全位"这回事）
    bus.apply_calibration()             （空桩，NERO 零点在控制器里）

**与 piper_follower 的三处故意不同**（照抄会出事）：

1. **`send_action()` 返回"实际下发值"，不是原样返回 action。**
   基类 `Robot.send_action` 的 docstring 写得很明确：返回值应当是
   "The action actually sent to the motors potentially clipped or modified"。
   piper 那边是 `return action` —— 一旦限幅生效，录进数据集的 action 就和
   机械臂真正收到的不是一回事，训练出来的策略会系统性偏移。
   这里返回 `bus.write()` 钳制后的值。

2. **`is_calibrated` 恒为 True，`calibrate()` 是空操作。**
   基类 docstring："Should be always `True` if not applicable"。
   NERO 的编码器零点由松灵 Web 上位机写进控制器，**软件层没有标定这回事**。
   往 `homing_offset` 里塞偏移等于在一个已经正确的值上再错一次 ——
   而控制器内部（重力补偿、力矩控制）用的还是它自己的零点，
   软件层歪掉会导致"指令和实际运动对不上"，比读数错更难查。

3. **维度由 `config.use_gripper` 决定**：True -> **8**（joint_1..7 + gripper），
   False -> **7**。这是**静态决定**，连接前后不允许变 ——
   因为 `record.py` 是「先读 features 建数据集」再 `connect()` 的，
   维度若在连接时再变，就会出现"声明 8 列、实际 7 列"的 KeyError。
   单位**已经是弧度**：piper 那边要 `/57324.84`（0.001 度 -> rad），
   NERO 的 SDK 直接给 rad，再做一次换算会让目标角小 5.7 万倍（臂几乎不动）。

⚠️ **别照抄 `piper_follower.disconnect()`** —— 它最后一句是
   `self.bus.connect(enable=False)`，那是**第二次 connect**。
   `NeroBus.connect()` 开头就有"已连接就报错"的检查，照抄必然炸。
   正确做法是 `bus.disconnect()`（内部负责失能 + 断开）。

安全说明（**别说成"不会动"**）：
  - `connect()` 会让电机通电锁住当前位姿（位置不变，但从"抱闸夹住"变"电机顶住"）；
  - `send_action()` 每调一次就下发一次 `move_j`；
  - `disconnect()` 会失能 —— **臂若处于抬起姿态会因重力下垂**。
"""

import logging
import math
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.nero import (
    ARM_JOINT_COUNT,
    ARM_JOINT_NAMES,
    GRIPPER_NAME,
    JOINT_NAMES,
    NeroMotorsBus,
    NeroMotorsBusConfig,
)

from ..robot import Robot
from .config_nero_follower import NeroFollowerConfig

logger = logging.getLogger(__name__)


class NeroFollower(Robot):
    """LeRobot 从臂接口：把 action 字典下发给 NERO，并读回观测。"""

    config_class = NeroFollowerConfig
    name = "nero_follower"

    def __init__(self, config: NeroFollowerConfig):
        super().__init__(config)
        self.config = config

        self.bus = NeroMotorsBus(
            NeroMotorsBusConfig(
                interface=config.interface,
                channel=config.channel,
                # ⚠️ 字段名是 firmeware_version（官方 SDK 拼写少一个 a），别写成 firmware
                firmeware_version=config.firmware,
                # 从臂：普通位置模式 + 使能，读常规关节反馈（SDK 已解码成弧度）
                role="follower",
                control_mode="move_j",
                # 维度来源：config.use_gripper（静态）-> bus 的 has_gripper。
                # 两处必须同源，否则 features 与 read() 的键集合会错位。
                has_gripper=config.use_gripper,
                enable_timeout=config.enable_timeout,
                max_relative_target=config.max_relative_target,
                speed_percent=config.speed_percent,
                gripper_mode=config.gripper_mode,
                gripper_max_width_m=config.gripper_max_width_m,
                gripper_max_angle_deg=config.gripper_max_angle_deg,
                gripper_force_n=config.gripper_force_n,
                gripper_send_threshold=config.gripper_send_threshold,
                disable_gripper_on_disconnect=config.disable_gripper_on_disconnect,
                feedback_timeout_s=config.feedback_timeout_s,
            )
        )

        self.cameras = make_cameras_from_configs(config.cameras)
        self.logs: dict[str, Any] = {}
        self._is_connected = False

    # ------------------------------------------------------------------ 特征
    @property
    def _motor_names(self) -> list[str]:
        """本从臂的轴名。**维度是静态的**，只由 `config.use_gripper` 决定。

        【为什么必须静态】`record.py` 的次序是
        「读 `action_features` / `observation_features` 建数据集」-> `robot.connect()`。
        也就是说 features 在**连接之前**就被拿去建数据集了。
        旧版 `_init_gripper()` 失败会把 `bus.config.has_gripper` 改成 False，
        导致连接后 `read()` 只返回 7 键、而数据集声明了 8 列
        -> 录制时 `build_dataset_frame()` KeyError。
        现在维度只认 config，且那条静默降级已改成**直接报错**。
        """
        return list(JOINT_NAMES) if self.config.use_gripper else list(ARM_JOINT_NAMES)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """电机特征：joint_1..joint_7 [+ gripper]，键一律带 `.pos`。"""
        return {f"{name}.pos": float for name in self._motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征。键用**裸相机名**（上游会补 observation.images. 前缀）。

        shape 固定 `(height, width, 3)`：`Camera` 基类**没有** `channels`
        属性，写 `cam.channels` 会 AttributeError（piper 那边就是这么写的，
        只是因为默认没配相机才没暴露）。
        """
        return {
            key: (cam.height, cam.width, 3) for key, cam in self.cameras.items()
        }

    @property
    def camera_features(self) -> dict[str, dict]:
        """给 `get_features_from_robot()` 用的相机描述。

        ⚠️ 这里的键**必须带** `observation.images.` 前缀 ——
        `get_features_from_robot()` 是原样并入 dataset features 的，不再补前缀。
        和 `_cameras_ft`（要裸名）约定相反，别搞混。
        """
        return {
            f"observation.images.{key}": {
                "shape": (cam.height, cam.width, 3),
                "names": ["height", "width", "channels"],
                "info": None,
            }
            for key, cam in self.cameras.items()
        }

    @property
    def motor_features(self) -> dict[str, dict]:
        keys = list(self._motors_ft)
        return {
            "action": {"dtype": "float32", "shape": (len(keys),), "names": keys},
            "observation.state": {"dtype": "float32", "shape": (len(keys),), "names": keys},
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    # ------------------------------------------------------------------ 状态
    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """恒为 True：NERO 的零点在控制器里，软件层**不适用**标定。

        见模块 docstring 第 2 条。基类 docstring 明确说"不适用时应恒为 True"。
        """
        return True

    @property
    def has_camera(self) -> bool:
        return len(self.cameras) > 0

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    # ------------------------------------------------------------ 连接/断开
    def connect(self, calibrate: bool = True) -> None:
        """连接从臂与相机。

        `calibrate` 参数是为了兼容基类签名（`Robot.connect(calibrate=True)`），
        对 NERO 来说它不产生任何动作 —— `calibrate()` 是空操作。
        """
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected. 不要重复调 connect()。"
            )

        logger.info(
            "连接 NERO 从臂：interface=%s channel=%s firmware=%s",
            self.config.interface,
            self.config.channel,
            self.config.firmware,
        )

        try:
            # 内部会：初始化夹爪 -> 唤醒 CAN 推送 -> 使能（轮询 + 补发唤醒）-> 设速度
            if not self.bus.connect(enable=True):
                raise RuntimeError("Follower enable was not confirmed")
        except Exception as exc:
            self.disconnect()
            raise RuntimeError(
                f"{self} 连接失败：{exc}\n"
                "逐条检查：\n"
                "  1) 臂是否上电、底座指示灯是否为绿\n"
                "  2) CAN_H(黄)/CAN_L(蓝) 是否接对、插紧\n"
                "  3) channel 是否选对；两台臂必须用**不同**通道\n"
                "  4) 有没有别的程序（Web 上位机、另一个脚本）占着同一个通道\n"
                "  5) firmware 留 None 让 SDK 自动解析，或填正确档位"
            ) from exc

        try:
            for name in self.cameras:
                self.cameras[name].connect()
                logger.info("%s: 相机 %s 已连接", self, name)
        except BaseException:
            self.disconnect()
            raise

        self._is_connected = True
        if calibrate:
            self.calibrate()

    def disconnect(self) -> None:
        """断开相机与机械臂。**失能后臂若抬起会下垂。**

        ⚠️ 不要照抄 `piper_follower` 的 `bus.connect(enable=False)`（第二次
        connect，必然报错）。这里的 `bus.disconnect()` 内部负责失能 + 断开。
        """
        for name, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception as exc:
                logger.warning("%s: 相机 %s 断开失败：%r", self, name, exc)

        try:
            self.bus.disconnect()
        except Exception as exc:
            logger.warning("%s: bus.disconnect() 失败：%r", self, exc)

        self._is_connected = False
        logger.info("%s 已断开。", self)

    def close_keeping_enabled(self) -> None:
        """Close cameras and the CAN session while leaving motors enabled.

        Unlike `disconnect()` this sends no disable, no mode switch and no motion
        command, so the arm keeps holding its current pose instead of sagging.
        """
        for name, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception as exc:
                logger.warning("%s: 相机 %s 断开失败：%r", self, name, exc)
        try:
            self.bus.close_keeping_enabled()
        except Exception as exc:
            logger.warning("%s: 保持使能断开失败：%r", self, exc)
        self._is_connected = False
        logger.info("%s CAN 会话已关闭，电机保持使能。", self)

    def calibrate(self) -> None:
        """空操作 —— NERO 没有软件层标定。

        零点由松灵 Web 上位机的"设置零点"写进控制器，SDK 返回的已经是标零后的
        弧度值。这里刻意不 `move_j`：贸然发运动指令比"不标定"危险得多。
        保留这个方法是因为基类把它声明为抽象方法。
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        logger.debug("%s: NERO 不需要软件标定，跳过。", self)

    def configure(self, **kwargs) -> None:
        """无额外运行时配置（速度/限幅都在 config 里）。"""
        pass

    # ------------------------------------------------------------------ 观测
    def get_observation(self) -> dict[str, Any]:
        """读 7 个关节角（rad）[+ 夹爪开合度（0..1）] + 相机图像。"""
        if not self._is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. Run `connect()` first."
            )

        state = self.bus.read()
        obs: dict[str, Any] = {
            f"{name}.pos": float(value) for name, value in state.items()
        }

        # 相机键用**裸名**：build_dataset_frame() 会用
        # values[key.removeprefix("observation.images.")] 来取值。
        for name, cam in self.cameras.items():
            obs[name] = cam.async_read()

        # 自检：键集合必须与 observation_features 一致。
        # 【为什么值得每次查】维度错位会一路潜伏到**录制时**才 KeyError，
        # 那时已经跑了一半 episode。旧的"夹爪初始化失败就静默降级"正是这样漏出去的
        # （features 8 键、实际观测 7 键）。这里当场炸，错误信息直接指向配置。
        expected_keys = set(self.observation_features)
        got_keys = set(obs)
        if got_keys != expected_keys:
            missing = sorted(expected_keys - got_keys)
            extra = sorted(got_keys - expected_keys)
            raise RuntimeError(
                f"{self}: 观测键与 observation_features 不一致。"
                f" 缺少 {missing}，多出 {extra}。"
                " 最常见原因：use_gripper 与实际接的夹爪不一致"
                "（改了 use_gripper 却没重启进程时，features 是 cached_property，"
                "会保持旧值）。"
            )

        return obs

    # ------------------------------------------------------------------ 控制
    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """收 action -> 限幅 -> 下发 -> 返回**实际下发**的值。"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        motor_names = self._motor_names
        expected = list(self._motors_ft)
        missing = [key for key in expected if key not in action]
        if missing:
            raise KeyError(
                f"{self} 收到的 action 缺少 {missing}。\n"
                f"需要 {expected}（7 关节{'+ 夹爪' if self.config.use_gripper else ''}），"
                f"实际收到 {sorted(action)}。\n"
                "若主臂侧没接夹爪，请把**主从两侧**的 use_gripper 都设为 False"
                "（两边必须一致，否则键对不上）。"
            )

        target = [float(action[f"{name}.pos"]) for name in motor_names]
        if not all(math.isfinite(value) for value in target):
            raise ValueError("All NERO action values must be finite")

        # ⚠️ 用 send_joints() 而不是 write()：
        #    - write(target)       会下发，但**返回值是 None**，且会连带处理夹爪
        #    - send_joints(joints) 内部做"相对当前位姿限幅"，**返回实际下发的关节角**
        #    基类 Robot.send_action 的契约要求返回 "actually sent … potentially
        #    clipped"，所以这里必须走能拿到返回值的那条路径。
        sent_joints = self.bus.send_joints(target[:ARM_JOINT_COUNT])

        sent: dict[str, float] = {
            f"{name}.pos": value
            for name, value in zip(ARM_JOINT_NAMES, sent_joints, strict=True)
        }

        if self.config.use_gripper:
            # 夹爪单独下发（send_joints 只管 7 个关节），最终落在 [0, 1]
            gripper = max(0.0, min(target[ARM_JOINT_COUNT], 1.0))
            sent[f"{GRIPPER_NAME}.pos"] = self.bus.write_gripper(gripper)

        return sent
