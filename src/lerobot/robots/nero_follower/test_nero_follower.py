"""NeroFollower 与 NeroMotorsBus 的**纯逻辑测试**（不连硬件，臂断电也能跑）。

用假的 bus / 假 robot 顶替硬件，覆盖：
  - 配置校验、特征键与顺序、7 轴还是 8 轴
  - 未连接守卫、connect/disconnect 的调用契约
  - `send_action` 返回的是**限幅后**的值（不是原样返回 action）
  - **三条保命机制的回归**：hz<=0 丢帧、使能轮询里补发唤醒、相对限幅
    （每一条都有真机事故依据，见 nero_bus.py 的模块 docstring）

支持两种跑法：
    pytest test_nero_follower.py
    python test_nero_follower.py        # 内置 _MiniRunner，不需要 pytest
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import time
from unittest import TestCase
from pathlib import Path
from typing import Any

# 把本仓库的 src 插到 sys.path 最前。
# ⚠️ 本机 conda env `lerobot` 的 editable 安装指向**另一份副本**，
#    不插的话这里会测到旧代码，结论无效。
_REPO_SRC = Path(__file__).resolve().parents[3]  # .../lerobot_piper2-main/src
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError  # noqa: E402
from lerobot.motors.nero import (  # noqa: E402
    ARM_JOINT_COUNT,
    ARM_JOINT_NAMES,
    GRIPPER_NAME,
    JOINT_NAMES,
    NeroMotorsBus,
    NeroMotorsBusConfig,
)
from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig  # noqa: E402

# --------------------------------------------------------------------------
# 假硬件
# --------------------------------------------------------------------------


class _FakeMsg:
    """模拟 pyAgxArm 的消息对象：正文装在 .msg，另外带一个 hz。"""

    def __init__(self, msg: Any, hz: float | None = 140.0):
        self.msg = msg
        if hz is not None:
            self.hz = hz


class _FakeGripper:
    def __init__(self, value: float = 0.035, mode: str = "width", fail: bool = False):
        self.value = value
        self.mode = mode
        self.fail = fail
        self.last_move: tuple[float, float] | None = None

    def get_gripper_status(self):
        if self.fail:
            raise OSError("没有夹爪硬件")
        return _FakeMsg(type("M", (), {"value": self.value, "mode": self.mode})())

    def move_gripper_m(self, value, force):
        self.last_move = (value, force)


class _FakeRobot:
    """假的 arm 实例（NeroMotorsBus.robot）。"""

    def __init__(self, joints=None, hz: float | None = 140.0):
        self.joint_nums = ARM_JOINT_COUNT
        self._joints = (
            list(joints) if joints is not None else [0.1 * i for i in range(ARM_JOINT_COUNT)]
        )
        self.hz = hz
        self.calls: list[str] = []
        self.flags = [True] * ARM_JOINT_COUNT
        self.moved_to: list[float] | None = None
        self.speed_percent: int | None = None
        self._normal_mode_sent = 0
        self.OPTIONS = type("O", (), {"EFFECTOR": type("E", (), {"AGX_GRIPPER": "agx_gripper"})})

    def connect(self):
        self.calls.append("connect")

    def disconnect(self):
        self.calls.append("disconnect")

    def enable(self, joint_index: int = 255):
        self.calls.append("enable")
        self.flags = [True] * ARM_JOINT_COUNT
        return True

    def disable(self, joint_index: int = 255):
        self.calls.append("disable")
        self.flags = [False] * ARM_JOINT_COUNT
        return True

    def get_joints_enable_status_list(self):
        self.calls.append("get_flags")
        return list(self.flags)

    def set_normal_mode(self):
        self.calls.append("set_normal_mode")
        self._normal_mode_sent += 1

    def get_firmware(self, timeout: float = 1.0, min_interval: float = 0.0):
        self.calls.append("get_firmware")
        return {"software_version": "1.20"}

    def get_joint_angles(self):
        self.calls.append("get_joint_angles")
        if self._joints is None:
            return None
        return _FakeMsg(list(self._joints), hz=self.hz)

    def move_j(self, joints):
        self.calls.append("move_j")
        self.moved_to = list(joints)

    def set_speed_percent(self, percent):
        self.calls.append("set_speed_percent")
        self.speed_percent = percent

    def init_effector(self, effector):
        self.calls.append("init_effector")
        return _FakeGripper()


def _make_bus(**overrides) -> tuple[NeroMotorsBus, _FakeRobot]:
    """造一个"已连接"的 NeroMotorsBus，底层是 _FakeRobot。"""
    overrides.setdefault("enable_timeout", 1.0)  # 别让测试真等 25 秒
    # 预热窗口关掉：完整版在窗口内会把「有关节恰好为 0」的帧判成残缺帧
    # （假 robot 的默认姿态第 1 个关节就是 0.0），开着会误伤所有 read 测试。
    overrides.setdefault("feedback_warmup_s", 0.0)
    cfg = NeroMotorsBusConfig(interface="fake", channel="0", **overrides)
    bus = NeroMotorsBus(cfg)
    fake = _FakeRobot()
    bus.robot = fake
    bus._is_connected = True
    bus._is_enabled = True
    bus.gripper = _FakeGripper() if cfg.has_gripper else None
    # 完整版没有 `_profile`，档位在 config 字段 `firmeware_version` 上
    bus.config.firmeware_version = "v120"  # 新固件：不发 set_normal_mode
    return bus, fake


class _FakeBus:
    """假的 NeroMotorsBus（NeroFollower.bus），用来测上层。"""

    def __init__(self, cap: float = 0.05, gripper: float = 0.3):
        self.cap = cap
        self.calls: list[str] = []
        self.written: list[list[float]] = []
        self._state = {name: 0.0 for name in JOINT_NAMES}
        self._state[GRIPPER_NAME] = gripper
        self.gripper = _FakeGripper()

    def connect(self, enable: bool = True):
        self.calls.append(f"connect(enable={enable})")
        return True

    def disconnect(self):
        self.calls.append("disconnect")

    def read(self) -> dict[str, float]:
        return dict(self._state)

    def send_joints(self, joints: list[float]) -> list[float]:
        """记录原始目标，返回**限幅后**的关节角（对应真实 bus 的 send_joints）。"""
        self.written.append(list(joints))
        current = [self._state[name] for name in ARM_JOINT_NAMES]
        return [
            cur + max(-self.cap, min(tgt - cur, self.cap))
            for cur, tgt in zip(current, joints[:ARM_JOINT_COUNT], strict=True)
        ]

    def write(self, target: list[float]) -> None:
        """完整版的 write() **返回值是 None**（不给钳制后的值）—— 保留以兼容旧断言。"""
        self.send_joints(target)

    def write_gripper(self, value: float) -> None:
        self.calls.append("write_gripper")
        self._last_gripper = float(value)
        return self._last_gripper

    @property
    def motor_names(self) -> list[str]:
        return list(JOINT_NAMES)

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def safe_disconnect(self) -> None:
        return


# ⚠️ 构造 NeroFollower 时 `Robot.__init__` 会 mkdir calibration_dir。
#    不指定就会往用户 home（~/.cache/huggingface/lerobot/calibration/）里建目录。
#    用一次性的临时目录，别弄脏用户目录（与 test_nero_leader.py 同样的做法）。
import tempfile  # noqa: E402

_TMP_CALIB = Path(tempfile.mkdtemp(prefix="nero_follower_test_"))


def _make_follower(**overrides) -> tuple[NeroFollower, _FakeBus]:
    overrides.setdefault("calibration_dir", _TMP_CALIB)
    cfg = NeroFollowerConfig(id="test", **overrides)
    follower = NeroFollower(cfg)
    fake = _FakeBus()
    follower.bus = fake
    follower._is_connected = True
    return follower, fake


class _FakeCam:
    is_connected = True
    height = 480
    width = 640

    def async_read(self):
        return "IMG"

    def connect(self):
        return

    def disconnect(self):
        return


class _FakeCamCfg:
    height = 480
    width = 640


def _full_action(value: float = 0.0, gripper: float = 0.5) -> dict[str, float]:
    act = {f"{name}.pos": value for name in ARM_JOINT_NAMES}
    act[f"{GRIPPER_NAME}.pos"] = gripper
    return act


# --------------------------------------------------------------------------
# 1. 配置
# --------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        cfg = NeroFollowerConfig(id="x")
        assert cfg.interface is None
        assert cfg.channel is None
        assert cfg.firmware is None
        assert cfg.max_relative_target == 0.05
        assert cfg.enable_timeout == 25.0
        assert cfg.speed_percent == 100

    def test_backend_auto_resolved(self):
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        if sys.platform == "win32":
            assert follower.bus.config.interface == "agx_cando"
            assert follower.bus.config.channel == "0"
        else:
            assert follower.bus.config.interface is not None

    def test_explicit_channel_preserved(self):
        follower = NeroFollower(NeroFollowerConfig(id="x", interface="agx_cando", channel="1"))
        assert follower.bus.config.channel == "1"

    def test_negative_max_relative_target_rejected(self):
        try:
            NeroFollowerConfig(id="x", max_relative_target=-1.0)
        except ValueError:
            return
        raise AssertionError("应该抛 ValueError")

    def test_bad_enable_timeout_rejected(self):
        try:
            NeroFollowerConfig(id="x", enable_timeout=0)
        except ValueError:
            return
        raise AssertionError("应该抛 ValueError")

    def test_bad_speed_rejected(self):
        try:
            NeroFollowerConfig(id="x", speed_percent=101)
        except ValueError:
            return
        raise AssertionError("应该抛 ValueError")


# --------------------------------------------------------------------------
# 2. 特征
# --------------------------------------------------------------------------


class TestFeatures:
    def test_action_keys_are_8_and_ordered(self):
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        want = [f"joint_{i}.pos" for i in range(1, 8)] + [f"{GRIPPER_NAME}.pos"]
        assert list(follower.action_features) == want

    def test_action_features_all_float(self):
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        assert set(follower.action_features.values()) == {float}

    def test_observation_covers_motor_keys(self):
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        assert set(follower.action_features) <= set(follower.observation_features)

    def test_motor_features_names_match_action(self):
        """record.py 的 build_dataset_frame 按这些名字取值，对不上就 KeyError。"""
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        mf = follower.motor_features
        assert mf["action"]["names"] == list(follower.action_features)
        assert mf["observation.state"]["names"] == list(follower.action_features)
        assert mf["action"]["shape"] == (8,)

    def test_camera_ft_uses_bare_name(self):
        """★ observation_features 里的相机键必须是**裸名**：
        hw_to_dataset_features() 会自己补 observation.images. 前缀，
        自己再拼一次会得到 observation.images.observation.images.xxx。"""
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        follower.cameras = {"front": _FakeCam()}
        follower.config.cameras = {"front": _FakeCamCfg()}
        follower.__dict__.pop("observation_features", None)
        assert follower.observation_features["front"] == (480, 640, 3)

    def test_camera_features_uses_prefix(self):
        """★ 反过来，camera_features 的键**必须带**前缀：
        get_features_from_robot() 是原样并入的，不再补前缀。"""
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        follower.cameras = {"front": _FakeCam()}
        assert "observation.images.front" in follower.camera_features
        assert follower.camera_features["observation.images.front"]["shape"] == (480, 640, 3)


# --------------------------------------------------------------------------
# 3. 连接契约
# --------------------------------------------------------------------------


class TestConnection:
    def test_not_connected_initially(self):
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        assert follower.is_connected is False

    def test_is_calibrated_always_true(self):
        """基类：Should be always True if not applicable。
        NERO 的零点在控制器里，软件层不适用标定。"""
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        assert follower.is_calibrated is True

    def test_get_observation_raises_when_not_connected(self):
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        try:
            follower.get_observation()
        except DeviceNotConnectedError:
            return
        raise AssertionError("应该抛 DeviceNotConnectedError")

    def test_send_action_raises_when_not_connected(self):
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        try:
            follower.send_action(_full_action())
        except DeviceNotConnectedError:
            return
        raise AssertionError("应该抛 DeviceNotConnectedError")

    def test_calibrate_raises_when_not_connected(self):
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        try:
            follower.calibrate()
        except DeviceNotConnectedError:
            return
        raise AssertionError("应该抛 DeviceNotConnectedError")

    def test_connect_uses_enable_true(self):
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        fake = _FakeBus()
        follower.bus = fake
        follower.connect()
        assert "connect(enable=True)" in fake.calls
        assert follower.is_connected is True

    def test_connect_signature_matches_base_class(self):
        """基类声明 connect(self, calibrate: bool = True)，别把参数漏了。"""
        sig = inspect.signature(NeroFollower.connect)
        assert "calibrate" in sig.parameters

    def test_double_connect_raises(self):
        follower, _ = _make_follower()
        try:
            follower.connect()
        except DeviceAlreadyConnectedError:
            return
        raise AssertionError("应该抛 DeviceAlreadyConnectedError")

    def test_disconnect_never_calls_connect_again(self):
        """⚠️ piper_follower.disconnect() 最后一句是 bus.connect(enable=False)
        —— 那是**第二次** connect，NeroMotorsBus 会直接报错。这里锁死不这么干。"""
        follower, fake = _make_follower()
        follower.disconnect()
        assert "disconnect" in fake.calls
        assert not any(c.startswith("connect") for c in fake.calls)

    def test_disconnect_is_idempotent(self):
        follower, _ = _make_follower()
        follower.disconnect()
        follower.disconnect()  # 不该炸
        assert follower.is_connected is False

    def test_configure_is_noop(self):
        follower, _ = _make_follower()
        assert follower.configure() is None

    def test_has_camera_and_num_cameras(self):
        follower = NeroFollower(NeroFollowerConfig(id="x"))
        assert follower.has_camera is False
        assert follower.num_cameras == 0


# --------------------------------------------------------------------------
# 4. 观测
# --------------------------------------------------------------------------


class TestGetObservation:
    def test_keys_match_observation_features(self):
        follower, _ = _make_follower()
        obs = follower.get_observation()
        assert set(obs) == set(follower.observation_features)

    def test_values_are_float(self):
        follower, _ = _make_follower()
        obs = follower.get_observation()
        assert all(isinstance(v, float) for v in obs.values())

    def test_camera_uses_bare_name(self):
        """build_dataset_frame 用 values[key.removeprefix("observation.images.")]
        取值 —— 所以 get_observation() 必须用裸名。"""
        follower, _ = _make_follower()
        follower.cameras = {"front": _FakeCam()}
        obs = follower.get_observation()
        assert obs["front"] == "IMG"


# --------------------------------------------------------------------------
# 5. send_action
# --------------------------------------------------------------------------


class TestSendAction:
    def test_returns_all_keys_with_pos_suffix(self):
        follower, _ = _make_follower()
        sent = follower.send_action(_full_action())
        assert list(sent) == list(follower.action_features)

    def test_missing_key_raises(self):
        follower, _ = _make_follower()
        try:
            follower.send_action({"joint_1.pos": 0.0})
        except KeyError:
            return
        raise AssertionError("应该抛 KeyError")

    def test_returns_clamped_value_not_original(self):
        """★ 基类契约：返回值是 actually sent … potentially clipped or modified。
        piper 那边 return action 是违反契约的 —— 限幅一生效，
        录进数据集的 action 就和机械臂真正收到的不是一回事。"""
        follower, fake = _make_follower(max_relative_target=0.05)
        sent = follower.send_action(_full_action(value=1.0))  # 远超单步上限
        assert fake.written[0][0] == 1.0, "原始目标应当被记下来（证明确实限幅了）"
        assert abs(sent["joint_1.pos"]) < 0.06, f"返回值应当是钳制后的 {sent['joint_1.pos']}"
        assert sent["joint_1.pos"] != 1.0

    def test_gripper_clamped_to_unit_range(self):
        follower, _ = _make_follower()
        sent = follower.send_action(_full_action(gripper=3.0))
        assert sent[f"{GRIPPER_NAME}.pos"] == 1.0
        sent = follower.send_action(_full_action(gripper=-2.0))
        assert sent[f"{GRIPPER_NAME}.pos"] == 0.0


# --------------------------------------------------------------------------
# 6. NeroMotorsBus 的三条保命机制
# --------------------------------------------------------------------------


class TestBusGuards:
    def test_partial_frame_rejected(self):
        """★ 保命机制 1：hz<=0 的帧必须丢。
        2026-09-13 就是这种帧被当成"当前位姿"下发，从臂 J7 被甩了约 195°。"""
        bus, fake = _make_bus()
        fake.hz = 0.0
        try:
            bus.read_joint_positions()
        except RuntimeError as exc:
            # 完整版在 read_joint_positions() 里抛的是**笼统消息**
            # （逐帧被拒的具体原因在 try_read_joint_positions() 的返回值
            #   和 read_stats() 的分类计数里），所以别断言消息里含 "hz"。
            assert "no valid frame" in str(exc) or "hz" in str(exc)
            return
        raise AssertionError("hz<=0 的帧必须被拒绝")

    def test_missing_hz_attribute_is_tolerated(self):
        """SDK 万一没给 hz 字段，不能因此把所有帧都丢掉。"""
        bus, fake = _make_bus()
        fake.hz = None  # _FakeMsg 不带 hz
        assert len(bus.read_joint_positions()) == ARM_JOINT_COUNT

    def test_none_frame_rejected(self):
        bus, fake = _make_bus()
        fake._joints = None
        try:
            bus.read_joint_positions()
        except RuntimeError:
            return
        raise AssertionError("拿不到帧必须抛错")

    def test_short_frame_rejected(self):
        bus, fake = _make_bus()
        fake._joints = [0.1, 0.2, 0.3]  # 帧还在装配
        try:
            bus.read_joint_positions()
        except RuntimeError as exc:
            # 完整版的消息是英文：Expected 7 joint angles, got 3.
            assert "got 3" in str(exc) or "只拿到" in str(exc)
            return
        raise AssertionError("长度不足必须抛错")

    def test_read_returns_8_keys(self):
        bus, _ = _make_bus()
        assert list(bus.read()) == JOINT_NAMES

    def test_relative_clamp_caps_single_step(self):
        """★ 保命机制 3：相对当前位姿限幅。

        ⚠️ 用 `send_joints()` 而不是 `write()`：完整版的 `write()` 一样会限幅，
        但**返回值是 None**（只有 `send_joints()` 返回实际下发值），
        所以想观察到钳制结果就必须走这条路 —— `NeroFollower.send_action()`
        也正是因为这个才选 `send_joints()`。
        """
        bus, fake = _make_bus(max_relative_target=0.05)
        current = bus.read_joint_positions()
        sent = bus.send_joints([c + 1.0 for c in current])
        for got, cur in zip(sent, current, strict=True):
            assert abs(got - cur) <= 0.05 + 1e-9
        assert fake.moved_to == sent, "下发的必须是钳制后的值"

    def test_relative_clamp_rejects_when_read_fails(self):
        bus, fake = _make_bus(max_relative_target=0.05)
        fake._joints = None  # 读不到当前位姿
        target = [1.0] * ARM_JOINT_COUNT
        with TestCase().assertRaises(RuntimeError):
            bus.send_joints(target)
        assert fake.moved_to is None

    def test_no_cap_when_none(self):
        bus, _ = _make_bus(max_relative_target=None, joint_limits_enabled=False)
        target = [1.0] * ARM_JOINT_COUNT
        assert bus.send_joints(target) == target

    def test_write_rejects_short_target(self):
        bus, _ = _make_bus()
        try:
            bus.write([0.1, 0.2])
        except ValueError:
            return
        raise AssertionError("关节数不足应该抛 ValueError")

    def test_write_writes_gripper_too(self):
        bus, _ = _make_bus()
        bus.gripper = _FakeGripper()
        bus.write([0.0] * ARM_JOINT_COUNT + [0.5])
        assert bus.gripper.last_move is not None
        assert abs(bus.gripper.last_move[0] - 0.5 * bus.config.gripper_max_width_m) < 1e-9

    def test_gripper_read_normalized_width_mode(self):
        bus, _ = _make_bus()
        bus.gripper = _FakeGripper(value=0.035)  # 米
        assert abs(bus.read()[GRIPPER_NAME] - 0.5) < 1e-6

    def test_gripper_read_normalized_angle_mode(self):
        bus, _ = _make_bus()
        bus.gripper = _FakeGripper(value=50.0, mode="angle")
        assert abs(bus.read()[GRIPPER_NAME] - 0.5) < 1e-6

    def test_gripper_read_failure_raises(self):
        bus, _ = _make_bus()
        bus.gripper = _FakeGripper(fail=True)
        with TestCase().assertRaises(RuntimeError):
            bus.read()

    def test_gripper_absent_raises(self):
        bus, _ = _make_bus()
        bus.gripper = None
        with TestCase().assertRaises(RuntimeError):
            bus.read()


class TestLenientReadContract:
    """★ bug 修复回归：`try_read_joint_positions()` 说了"不抛异常"就必须真的不抛。

    旧实现是 `except RuntimeError: raise`，而 `_read_raw_angles()` 在**短帧**
    （关节数不足 7，7 条 CAN 报文还没装配完）时**也**抛 RuntimeError。
    于是这个"宽松 API"变成了会抛 —— 一个短帧就能让 60 Hz 遥操作循环整个崩掉
    （`teleoperate()` 只 catch KeyboardInterrupt），录制时同理会毁掉当前 episode。
    """

    def test_short_frame_returns_none(self):
        bus, fake = _make_bus()
        fake._joints = [0.1, 0.2, 0.3]  # 帧还在装配
        assert bus.try_read_joint_positions() is None

    def test_short_frame_is_counted_as_reject(self):
        bus, fake = _make_bus()
        fake._joints = [0.1, 0.2, 0.3]
        bus.try_read_joint_positions()
        assert bus.read_stats()["reject_incomplete"] >= 1, \
            "短帧应当计入 reject_incomplete（与其它拒收分支一致）"

    def test_strict_api_still_raises_on_short_frame(self):
        """严格 API 的契约不能被顺手改掉。"""
        bus, fake = _make_bus()
        fake._joints = [0.1, 0.2, 0.3]
        try:
            bus.read_joint_positions()
        except RuntimeError:
            return
        raise AssertionError("read_joint_positions() 必须保持抛错语义")

    def test_hz_zero_returns_none(self):
        bus, fake = _make_bus()
        fake.hz = 0.0
        assert bus.try_read_joint_positions() is None

    def test_none_frame_returns_none(self):
        bus, fake = _make_bus()
        fake._joints = None
        assert bus.try_read_joint_positions() is None

    def test_not_connected_still_raises(self):
        """未连接是**编程错误**，不能被伪装成"读不到数据" —— 否则极难定位。"""
        cfg = NeroMotorsBusConfig(interface="fake", channel="0", feedback_warmup_s=0.0)
        bus = NeroMotorsBus(cfg)
        try:
            bus.try_read_joint_positions()
        except RuntimeError:
            return
        raise AssertionError("未连接必须抛 RuntimeError")

    def test_present_positions_lenient_on_short_frame(self):
        """`read()` -> `present_positions()` 是录制路径，同样不能被短帧打崩。"""
        bus, fake = _make_bus()
        assert bus.read_joint_positions() is not None  # 先存下 _last_read
        fake._joints = [0.1, 0.2, 0.3]
        got = bus.present_positions()  # 不能抛
        assert got is not None, "短帧时应退回上一帧读数"

    def test_wait_stable_pose_lenient_on_short_frame(self):
        bus, fake = _make_bus()
        fake._joints = [0.1, 0.2, 0.3]
        assert bus.wait_stable_pose(n=1, timeout_s=0.2) is None

    def test_wait_for_angles_lenient_on_short_frame(self):
        bus, fake = _make_bus()
        fake._joints = [0.1, 0.2, 0.3]
        assert bus.wait_for_angles(0.2) is False


class TestGripperDimensions:
    """★ bug 修复回归：维度必须是**静态**的，且不能静默降级。

    旧行为：`_init_gripper()` 失败 -> `config.has_gripper = False`（静默降级）
    -> `read()` 从 8 键变 7 键；而 features 是**连接之前**算好的 8 键
    -> 录制时 `build_dataset_frame()` KeyError（跑到一半才炸）。
    """

    def test_init_gripper_failure_raises_instead_of_degrading(self):
        bus, fake = _make_bus()

        def boom(effector):
            raise RuntimeError("no gripper hardware")

        fake.init_effector = boom
        try:
            bus._init_gripper()
        except RuntimeError as exc:
            assert "use_gripper" in str(exc), f"报错要给出处置办法，实际：{exc}"
        else:
            raise AssertionError("夹爪初始化失败必须抛错，不能静默降级")

        assert bus.gripper is None
        assert bus.config.has_gripper is True, (
            "config.has_gripper 不能再被运行时改写 —— 它是维度的唯一真相"
        )

    def test_has_gripper_is_static_and_available_is_runtime(self):
        bus, _ = _make_bus(has_gripper=True)
        bus.gripper = None
        assert bus.has_gripper is True
        assert bus.gripper_available is False, "还没初始化，运行时不可用"
        bus._init_gripper()
        assert bus.gripper_available is True
        assert bus.has_gripper is True, "has_gripper 不随初始化结果变化"

    def test_motor_names_7_when_no_gripper(self):
        bus, _ = _make_bus(has_gripper=False)
        assert bus.motor_names == ARM_JOINT_NAMES
        assert list(bus.read()) == ARM_JOINT_NAMES

    def test_read_rejects_unreadable_gripper_without_changing_dimensions(self):
        bus, _ = _make_bus()
        bus.gripper = None  # 配置了夹爪，但驱动不可用
        with TestCase().assertRaises(RuntimeError):
            bus.read()
        assert bus.motor_names == JOINT_NAMES


class TestFeatureObservationConsistency:
    """★ bug 修复回归：features 的键集合必须与实际观测**完全一致**。

    这类错位会一路潜伏到录制时才 KeyError，所以 `get_observation()` 现在自带自检。
    """

    def test_default_is_8d(self):
        follower, _ = _make_follower()
        want = [f"{n}.pos" for n in ARM_JOINT_NAMES] + [f"{GRIPPER_NAME}.pos"]
        assert list(follower.action_features) == want
        assert list(follower.observation_features) == want

    def test_use_gripper_false_gives_7d(self):
        follower, _ = _make_follower(use_gripper=False)
        want = [f"{n}.pos" for n in ARM_JOINT_NAMES]
        assert list(follower.action_features) == want
        assert list(follower.observation_features) == want

    def test_usegripper_is_passed_to_bus(self):
        """`use_gripper` 必须传到 bus 的 `has_gripper` —— 两处不同源就会维度错位。"""
        # 这里**不替换** bus，直接看构造结果
        off = NeroFollower(
            NeroFollowerConfig(id="t", use_gripper=False, calibration_dir=_TMP_CALIB)
        )
        assert off.bus.has_gripper is False
        on = NeroFollower(
            NeroFollowerConfig(id="t", use_gripper=True, calibration_dir=_TMP_CALIB)
        )
        assert on.bus.has_gripper is True

    def test_send_action_7d_does_not_touch_gripper(self):
        follower, fake = _make_follower(use_gripper=False)
        act = {f"{n}.pos": 0.0 for n in ARM_JOINT_NAMES}
        sent = follower.send_action(act)
        assert list(sent) == list(act)
        assert "write_gripper" not in fake.calls

    def test_missing_gripper_key_message_points_to_use_gripper(self):
        follower, _ = _make_follower()
        act = {f"{n}.pos": 0.0 for n in ARM_JOINT_NAMES}  # 故意少 gripper
        try:
            follower.send_action(act)
        except KeyError as exc:
            assert "use_gripper" in str(exc), f"报错要给出处置办法，实际：{exc}"
            return
        raise AssertionError("缺键必须 KeyError")

    def test_observation_mismatch_is_caught(self):
        """模拟旧版的"声明 8 键、实际只给 7 键"，自检必须当场炸。"""
        follower, fake = _make_follower()
        del fake._state[GRIPPER_NAME]
        try:
            follower.get_observation()
        except RuntimeError as exc:
            assert "observation_features" in str(exc)
            return
        raise AssertionError("键集合不一致必须报错，不能放过")


class TestLegacyFirmwareDetection:
    """判据决定要不要发 set_normal_mode（≤1.11 才有推送）。"""

    def test_default_and_v111_are_legacy(self):
        for profile in ("default", "v111", "DEFAULT"):
            bus, _ = _make_bus()
            bus.config.firmeware_version = profile
            assert bus._is_legacy_firmware() is True

    def test_new_profiles_are_not_legacy(self):
        for profile in ("v112", "v120", "v121"):
            bus, _ = _make_bus()
            bus.config.firmeware_version = profile
            assert bus._is_legacy_firmware() is False

    def test_unknown_profile_defaults_to_legacy(self):
        """判不出来时宁可多发一次：≥1.12 上它只是空操作 + 一条警告，
        而 ≤1.11 上漏发就意味着推送永远不开、什么都读不到。"""
        bus, _ = _make_bus()
        bus.config.firmeware_version = None
        bus._firmware_version = None
        assert bus._is_legacy_firmware() is True

    def test_falls_back_to_version_string(self):
        bus, _ = _make_bus()
        bus.config.firmeware_version = None
        bus._firmware_version = "1.10"
        assert bus._is_legacy_firmware() is True
        bus._firmware_version = "1.20"
        assert bus._is_legacy_firmware() is False

    def test_wake_is_noop_on_new_firmware(self):
        bus, fake = _make_bus()
        bus.config.firmeware_version = "v120"
        assert bus._enable_can_push() is False
        assert fake._normal_mode_sent == 0
        assert "set_follower_mode" not in fake.calls, (
            "绝不能改用 set_follower_mode —— 它会把 CAN 推送置 DISABLE"
        )


class TestEnableFlow:
    """★ 保命机制 2：使能轮询里必须补发唤醒。

    死锁链条：不 enable -> 模式指令在失能态不生效 -> 推送不开
              -> 使能位读不到 -> 超时失败。
    2026-09-16 真机实测：轮询 15 秒全程 0000000，重开推送后才刷成 1111111。
    """

    def test_enable_confirms_when_flags_are_ready(self):
        bus, fake = _make_bus(enable_timeout=1.0)
        bus._is_enabled = False
        fake.flags = [True] * ARM_JOINT_COUNT
        assert bus.enable() is True
        assert bus._is_enabled is True

    def test_enable_rewakes_to_break_deadlock(self):
        bus, fake = _make_bus(enable_timeout=3.0)
        bus.config.firmeware_version = "default"  # 旧固件才需要唤醒
        bus._is_enabled = False
        fake.push_on = False

        def flags_gate():
            fake.calls.append("get_flags")
            if not fake.push_on:
                return [False] * ARM_JOINT_COUNT
            return [True] * ARM_JOINT_COUNT

        def normal_mode():
            fake.calls.append("set_normal_mode")
            fake._normal_mode_sent += 1
            # 补发发生在 robot.enable() 之后 —— 那时臂才通电，指令才生效
            fake.push_on = True

        fake.get_joints_enable_status_list = flags_gate
        fake.set_normal_mode = normal_mode

        assert bus.enable() is True, "补发唤醒后应该能解开死锁"
        assert fake._normal_mode_sent >= 1

    def test_enable_returns_false_on_timeout(self):
        bus, fake = _make_bus(enable_timeout=0.6)
        bus._is_enabled = False
        fake.enable = lambda joint_index=255: True  # 不让 flags 被翻成 True
        fake.flags = [False] * ARM_JOINT_COUNT
        assert bus.enable() is False
        assert bus._is_enabled is False

    def test_disable_confirms(self):
        bus, fake = _make_bus(enable_timeout=1.0)
        fake.flags = [False] * ARM_JOINT_COUNT
        assert bus.disable() is True
        assert bus._is_enabled is False

    def test_enable_raises_when_not_connected(self):
        bus = NeroMotorsBus(NeroMotorsBusConfig(interface="fake", channel="0"))
        try:
            bus.enable()
        except RuntimeError:
            return
        raise AssertionError("未连接时 enable() 应该抛 RuntimeError")


class TestBusBasics:
    def test_motor_names_is_8(self):
        bus = NeroMotorsBus(NeroMotorsBusConfig(interface="fake", channel="0"))
        assert bus.motor_names == JOINT_NAMES

    def test_is_calibrated_follows_connection(self):
        bus = NeroMotorsBus(NeroMotorsBusConfig(interface="fake", channel="0"))
        assert bus.is_calibrated is False
        bus._is_connected = True
        assert bus.is_calibrated is True

    def test_piper_compat_stubs_do_not_move_arm(self):
        """这几个桩绝不能偷偷发运动指令（piper 的 apply_calibration 会 move）。"""
        bus, fake = _make_bus()
        bus.safe_disconnect()
        bus.apply_calibration()
        bus.apply_calibration_master()
        bus.set_calibration()
        bus.revert_calibration()
        assert fake.moved_to is None
        assert "move_j" not in fake.calls

    def test_double_connect_raises(self):
        bus, _ = _make_bus()
        try:
            bus.connect()
        except RuntimeError:
            return
        raise AssertionError("已连接时再 connect 应该抛 RuntimeError")


# --------------------------------------------------------------------------
# 不装 pytest 时的裸跑入口
# --------------------------------------------------------------------------


class _MiniRunner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[tuple[str, str]] = []

    def run(self) -> int:
        classes = [
            obj
            for name, obj in sorted(globals().items())
            if name.startswith("Test") and isinstance(obj, type)
        ]
        total = sum(len([m for m in dir(c) if m.startswith("test_")]) for c in classes)
        print(f"收集到 {total} 个测试")
        print("=" * 68)
        for cls in classes:
            for name in sorted(m for m in dir(cls) if m.startswith("test_")):
                label = f"{cls.__name__}::{name}"
                try:
                    getattr(cls(), name)()
                    self.passed += 1
                    print(f"  ok    {label}")
                except Exception as exc:  # noqa: BLE001
                    self.failed.append((label, f"{type(exc).__name__}: {exc}"))
                    print(f"  FAIL  {label}")
                    print(f"        {type(exc).__name__}: {exc}")
        print("=" * 68)
        print(f"通过 {self.passed}  失败 {len(self.failed)}   (共 {total})")
        if self.failed:
            print("\n失败明细：")
            for label, detail in self.failed:
                print(f"  - {label}: {detail}")
        return 1 if self.failed else 0


if __name__ == "__main__":
    start = time.time()
    code = _MiniRunner().run()
    print(f"\n耗时 {time.time() - start:.2f}s")
    sys.exit(code)
