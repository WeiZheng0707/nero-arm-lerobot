"""NeroLeader 功能测试（纯逻辑，不连接硬件、不动臂）。

用假的 `NeroMotorsBus` 替身驱动。重点覆盖：
  - 单位：主臂通道是**度**，bus 已转弧度，leader **不能**再转一次
    （piper_leader 里有 /57324.84，照抄会差 57.3 倍）
  - 读不到反馈时必须沿用上一帧，且要能察觉"一直读不到"
  - joint_signs / joint_offsets 方向与偏置修正
  - action 键与从臂完全一致（顺序也要一致）

用法：
    python test_nero_leader.py    # 不需要 pytest
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 让 lerobot 解析到**本仓库**的 src
#
# ⚠️ 本机 conda env `lerobot` 装的是 editable 版，指向 **另一份副本**
#   （D:\arm_demo\project\lerobot_piper2-main）。不显式插到 sys.path 最前面，
#   `import lerobot` 会在另一份代码上跑测试，结论无效。
# ---------------------------------------------------------------------------

_REPO_SRC = Path(__file__).resolve().parents[3]  # .../lerobot_piper2-main/src
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError  # noqa: E402
from lerobot.motors.nero import ARM_JOINT_COUNT, ARM_JOINT_NAMES, GRIPPER_NAME  # noqa: E402
from lerobot.teleoperators.nero_leader import NeroLeader, NeroLeaderConfig  # noqa: E402

# ===========================================================================
# 假 bus
# ===========================================================================


class _FakeBus:
    """模拟 NeroMotorsBus 的对外接口。"""

    def __init__(self, has_gripper: bool = True):
        self.config = type(
            "C", (), {"has_gripper": has_gripper, "read_timeout_s": 0.2}
        )()
        self._is_connected = False
        self.gripper = object() if has_gripper else None
        self.calls: list[str] = []

        # 关节读数（**弧度**：bus 已经转好了）
        self.angles: list[float] | None = [0.0] * ARM_JOINT_COUNT
        self.gripper_value: float | None = 0.5
        self.stats: dict[str, Any] = {"angle_source": "leader"}
        self._last_sent: list[float] | None = None
        self._last_gripper_sent: float | None = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_connected

    @property
    def has_gripper(self) -> bool:
        return self.config.has_gripper

    def connect(self, enable: bool = True) -> bool:
        self.calls.append(f"connect(enable={enable})")
        self._is_connected = True
        return True

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self._is_connected = False

    def wait_for_angles(self, timeout_s: float | None = None) -> bool:
        self.calls.append("wait_for_angles")
        return self.angles is not None

    def try_read_joint_positions(self) -> list[float] | None:
        self.calls.append("try_read_joint_positions")
        return None if self.angles is None else list(self.angles)

    def wait_stable_pose(self, n: int = 3, tol_rad: float = 0.004, timeout_s: float = 20.0):
        self.calls.append("wait_stable_pose")
        return None if self.angles is None else list(self.angles)

    def read_gripper(self) -> float | None:
        return self.gripper_value

    def read_gripper_or_last(self) -> float | None:
        if self.gripper_value is not None:
            return self.gripper_value
        return self._last_gripper_sent


# ⚠️ 构造 NeroLeader 时 `Teleoperator.__init__` 会 mkdir calibration_dir。
#    用临时目录，别往用户 home（~/.cache/huggingface/lerobot/calibration）里塞。
import tempfile  # noqa: E402


def _make_config(**overrides) -> NeroLeaderConfig:
    """构造配置，**强制**使用一次性的临时 calibration 目录。"""
    overrides.setdefault("id", "test")
    overrides.setdefault(
        "calibration_dir", Path(tempfile.mkdtemp(prefix="nero_leader_test_"))
    )
    return NeroLeaderConfig(**overrides)


def _make_leader(**overrides) -> tuple[NeroLeader, _FakeBus]:
    cfg = _make_config(**overrides)
    leader = NeroLeader(cfg)
    fake = _FakeBus(has_gripper=cfg.use_gripper)
    leader.bus = fake
    leader._is_connected = True
    leader._is_calibrated = True
    fake._is_connected = True
    return leader, fake


# ===========================================================================
# 特征描述
# ===========================================================================


class TestFeatures:
    def test_action_keys_8d(self):
        leader, _ = _make_leader()
        keys = list(leader.action_features)
        assert keys == [f"{n}.pos" for n in ARM_JOINT_NAMES] + [f"{GRIPPER_NAME}.pos"]
        assert len(keys) == 8

    def test_action_keys_7d_without_gripper(self):
        leader, _ = _make_leader(use_gripper=False)
        assert len(list(leader.action_features)) == 7
        assert f"{GRIPPER_NAME}.pos" not in leader.action_features

    def test_all_floats(self):
        leader, _ = _make_leader()
        assert all(t is float for t in leader.action_features.values())

    def test_feedback_features_empty(self):
        """NERO 主臂不做力反馈。"""
        leader, _ = _make_leader()
        assert leader.feedback_features == {}

    def test_angle_source_defaults_to_auto(self):
        assert NeroLeaderConfig(id="t").angle_source == "auto"

    def test_bad_angle_source_rejected(self):
        try:
            NeroLeaderConfig(id="t", angle_source="bogus")
        except ValueError as exc:
            assert "angle_source" in str(exc)
            return
        raise AssertionError("应该抛 ValueError")


# ===========================================================================
# 连接
# ===========================================================================


class TestConnection:
    def test_not_connected_initially(self):
        leader = NeroLeader(_make_config())
        assert leader.is_connected is False
        assert leader.is_calibrated is False

    def test_get_action_raises_when_not_connected(self):
        leader = NeroLeader(_make_config())
        try:
            leader.get_action()
        except DeviceNotConnectedError:
            return
        raise AssertionError("应该抛 DeviceNotConnectedError")

    def test_double_connect_raises(self):
        leader, _ = _make_leader()
        try:
            leader.connect()
        except DeviceAlreadyConnectedError:
            return
        raise AssertionError("应该抛 DeviceAlreadyConnectedError")

    def test_connect_uses_enable_true(self):
        """主臂也要使能 —— 零力拖拽是"使能着却能拖"，不是失能。"""
        leader = NeroLeader(_make_config())
        fake = _FakeBus()
        leader.bus = fake
        leader.connect()
        assert "connect(enable=True)" in fake.calls

    def test_disconnect_is_idempotent(self):
        leader, _ = _make_leader()
        leader.disconnect()
        leader.disconnect()
        assert leader.is_connected is False

    def test_disconnect_clears_last_angles(self):
        leader, _ = _make_leader()
        leader.get_action()
        assert leader._last_angles is not None
        leader.disconnect()
        assert leader._last_angles is None


# ===========================================================================
# 单位（本模块最大的坑）
# ===========================================================================


class TestUnits:
    def test_radians_are_not_converted_again(self):
        """bus 给的已经是弧度，leader **不能**再除 57324.84。

        piper_leader 里有 `val / 57324.840764`（因为 piper_sdk 给 0.001 度）。
        照抄到 NERO 会让目标角小 57324 倍 —— 从臂几乎不动。
        """
        leader, fake = _make_leader()
        fake.angles = [math.pi / 2] * ARM_JOINT_COUNT  # 90°
        action = leader.get_action()
        assert abs(action["joint_1.pos"] - math.pi / 2) < 1e-12

    def test_small_angle_preserved(self):
        """1.5° 就是 0.0262 rad，不能被缩成 4e-7。"""
        leader, fake = _make_leader()
        rad = math.radians(1.5)
        fake.angles = [rad] * ARM_JOINT_COUNT
        action = leader.get_action()
        assert abs(action["joint_1.pos"] - rad) < 1e-12
        assert action["joint_1.pos"] > 0.02


# ===========================================================================
# get_action 的健壮性
# ===========================================================================


class TestGetAction:
    def test_returns_all_keys(self):
        leader, _ = _make_leader()
        action = leader.get_action()
        assert list(action) == list(leader.action_features)

    def test_gripper_included(self):
        leader, fake = _make_leader()
        fake.gripper_value = 0.25
        action = leader.get_action()
        assert abs(action[f"{GRIPPER_NAME}.pos"] - 0.25) < 1e-12

    def test_gripper_absent_when_disabled(self):
        leader, _ = _make_leader(use_gripper=False)
        assert f"{GRIPPER_NAME}.pos" not in leader.get_action()

    def test_falls_back_to_last_frame(self):
        """丢一帧时沿用上一帧，避免遥操作循环里出现全 0 抖动。"""
        leader, fake = _make_leader()
        fake.angles = [0.4] * ARM_JOINT_COUNT
        first = leader.get_action()
        assert abs(first["joint_1.pos"] - 0.4) < 1e-12

        fake.angles = None  # 反馈断了
        second = leader.get_action()
        assert abs(second["joint_1.pos"] - 0.4) < 1e-12

    def test_stale_streak_counts(self):
        leader, fake = _make_leader()
        fake.angles = [0.1] * ARM_JOINT_COUNT
        leader.get_action()
        assert leader._stale_streak == 0

        fake.angles = None
        leader.get_action()
        leader.get_action()
        assert leader._stale_streak == 2

        fake.angles = [0.2] * ARM_JOINT_COUNT
        leader.get_action()
        assert leader._stale_streak == 0

    def test_raises_when_no_feedback_ever(self):
        """一帧都没读到 -> 必须报错（不能编一个全 0 的 action 出来）。"""
        leader, fake = _make_leader()
        fake.angles = None
        try:
            leader.get_action()
        except RuntimeError:
            return
        raise AssertionError("应该抛 RuntimeError")

    def test_logs_stale_frames(self):
        leader, fake = _make_leader()
        fake.angles = [0.1] * ARM_JOINT_COUNT
        leader.get_action()
        assert leader.logs["stale_frames"] == 0
        assert "read_ms" in leader.logs

    def test_logs_angle_source(self):
        leader, fake = _make_leader()
        fake.stats = {"angle_source": "leader"}
        leader.get_action()
        assert leader.logs["angle_source"] == "leader"


# ===========================================================================
# joint_signs / joint_offsets
# ===========================================================================


class TestSignsAndOffsets:
    def test_default_is_identity(self):
        leader, fake = _make_leader()
        fake.angles = [0.3] * ARM_JOINT_COUNT
        action = leader.get_action()
        assert abs(action["joint_1.pos"] - 0.3) < 1e-12

    def test_sign_flips_direction(self):
        leader, fake = _make_leader(
            joint_signs=[-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        fake.angles = [0.3] * ARM_JOINT_COUNT
        action = leader.get_action()
        assert abs(action["joint_1.pos"] - (-0.3)) < 1e-12
        assert abs(action["joint_2.pos"] - 0.3) < 1e-12

    def test_offset_applied(self):
        leader, fake = _make_leader(
            joint_offsets=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        fake.angles = [0.3] * ARM_JOINT_COUNT
        action = leader.get_action()
        assert abs(action["joint_1.pos"] - 0.4) < 1e-12

    def test_sign_and_offset_combined(self):
        leader, fake = _make_leader(
            joint_signs=[-1.0] * 8, joint_offsets=[0.5] * 8
        )
        fake.angles = [0.3] * ARM_JOINT_COUNT
        action = leader.get_action()
        assert abs(action["joint_1.pos"] - (-0.3 + 0.5)) < 1e-12

    def test_seven_element_lists_are_padded(self):
        """只给 7 个元素时，夹爪按默认（+1 / 0）补齐。"""
        cfg = NeroLeaderConfig(id="t", joint_signs=[-1.0] * 7)
        assert len(cfg.joint_signs) == 8
        assert cfg.joint_signs[7] == 1.0
        assert len(cfg.joint_offsets) == 8

    def test_wrong_length_rejected(self):
        try:
            NeroLeaderConfig(id="t", joint_signs=[1.0] * 5)
        except ValueError as exc:
            assert "joint_signs" in str(exc)
            return
        raise AssertionError("应该抛 ValueError")

    def test_gripper_sign_applied(self):
        leader, fake = _make_leader(
            joint_signs=[1.0] * 7 + [-1.0], joint_offsets=[0.0] * 7 + [1.0]
        )
        fake.gripper_value = 0.8
        action = leader.get_action()
        assert abs(action[f"{GRIPPER_NAME}.pos"] - 0.2) < 1e-12


# ===========================================================================
# ★ 隐患修复回归：主臂夹爪读不到时不能静默补 0.0
# ===========================================================================


class TestGripperUnreadable:
    """旧的隐患：主臂夹爪读不到 -> action 里悄悄补 0.0（0.0 = **完整闭合**）
    -> 从臂夹爪被持续指令闭合，而且全程没有任何告警。

    根因是主臂的 `_last_gripper_sent` **恒为 None**（主臂只读不写），
    所以 `read_gripper_or_last()` 在主臂上给不出兜底值，一路落到 None。

    新行为：读不到 -> 沿用上一帧**成功读到**的值；
            从未读到过 -> 用 `config.gripper_fallback_when_unreadable`（显式可配）。
    """

    def test_holds_last_readable_value_instead_of_zero(self):
        leader, fake = _make_leader()
        fake.gripper_value = 0.42
        assert abs(leader.get_action()[f"{GRIPPER_NAME}.pos"] - 0.42) < 1e-12

        # 之后读不到了：必须沿用 0.42，**不能**掉到 0.0
        fake.gripper_value = None
        for _ in range(5):
            got = leader.get_action()[f"{GRIPPER_NAME}.pos"]
            assert abs(got - 0.42) < 1e-12, f"读不到时应沿用 0.42，实际 {got}"

    def test_never_readable_uses_config_fallback(self):
        """从未读到过 -> 用**显式配置**的兜底值，不再是硬编码的 0.0。"""
        leader, fake = _make_leader(gripper_fallback_when_unreadable=0.0)
        fake.gripper_value = None
        fake._last_gripper_sent = None  # 主臂永远不下发夹爪指令
        assert leader.get_action()[f"{GRIPPER_NAME}.pos"] == 0.0

    def test_never_readable_respects_custom_fallback(self):
        """常开型夹爪：可以配成「读不到就保持张开」。"""
        leader, fake = _make_leader(gripper_fallback_when_unreadable=1.0)
        fake.gripper_value = None
        fake._last_gripper_sent = None
        got = leader.get_action()[f"{GRIPPER_NAME}.pos"]
        assert abs(got - 1.0) < 1e-12, f"应取配置的 1.0，实际 {got}"

    def test_fallback_out_of_range_rejected(self):
        for bad in (-0.1, 1.1):
            try:
                NeroLeaderConfig(id="t", gripper_fallback_when_unreadable=bad)
            except ValueError as exc:
                assert "gripper_fallback_when_unreadable" in str(exc)
            else:
                raise AssertionError(f"{bad} 应该被拒")

    def test_stale_streak_tracked_in_logs(self):
        leader, fake = _make_leader()
        fake.gripper_value = 0.3
        leader.get_action()
        assert leader.logs["gripper_stale_frames"] == 0

        fake.gripper_value = None
        leader.get_action()
        leader.get_action()
        assert leader.logs["gripper_stale_frames"] == 2

    def test_recovers_after_reading_again(self):
        leader, fake = _make_leader()
        fake.gripper_value = 0.7
        leader.get_action()

        fake.gripper_value = None
        leader.get_action()
        assert leader.logs["gripper_stale_frames"] == 1

        fake.gripper_value = 0.2
        got = leader.get_action()[f"{GRIPPER_NAME}.pos"]
        assert abs(got - 0.2) < 1e-12
        assert leader.logs["gripper_stale_frames"] == 0

    def test_no_gripper_key_when_disabled(self):
        """use_gripper=False 时不该有这条路径。"""
        leader, fake = _make_leader(use_gripper=False)
        fake.gripper_value = None
        assert f"{GRIPPER_NAME}.pos" not in leader.get_action()

    def test_disconnect_clears_gripper_cache(self):
        """重连后不能拿**上一次会话**的夹爪读数当「上一帧」。"""
        leader, fake = _make_leader()
        fake.gripper_value = 0.9
        leader.get_action()
        assert leader._last_gripper == 0.9

        leader.disconnect()
        assert leader._last_gripper is None
        assert leader._gripper_stale_streak == 0
        assert leader._last_angles is None
        assert leader._last_source is None


# ===========================================================================
# ★ 隐患修复回归：主从的**默认通道**必须错开
# ===========================================================================


class TestChannelDefaults:
    """旧的隐患：主臂和从臂的 config 里 `channel` 都默认 None，
    `NeroMotorsBus.__init__` 各自兜底 -> **两边都得到同一个 "0"**
    -> 抢同一条 CAN 总线，第二个 connect() 必然失败，报错还像「硬件没插好」。

    对齐 piper 的思路（它干脆不给配置项，写死成 can_follower / can_master）：
    这里保留可配置，但默认值**按角色错开**。
    """

    def test_roles_get_different_default_channels(self):
        from lerobot.motors.nero.nero import _default_can_backend

        follower = _default_can_backend("follower")
        leader = _default_can_backend("leader")
        assert follower[0] == leader[0], "后端类型应一致（同一平台同一后端）"
        assert follower[1] != leader[1], (
            f"两个角色的默认通道必须不同，否则主从抢总线：{follower} vs {leader}"
        )

    def test_bus_resolves_channel_by_role(self):
        from lerobot.motors.nero import NeroMotorsBus, NeroMotorsBusConfig
        from lerobot.motors.nero.nero import _default_can_backend

        f_bus = NeroMotorsBus(NeroMotorsBusConfig(role="follower"))
        l_bus = NeroMotorsBus(NeroMotorsBusConfig(role="leader"))
        assert f_bus.config.channel == _default_can_backend("follower")[1]
        assert l_bus.config.channel == _default_can_backend("leader")[1]
        assert f_bus.config.channel != l_bus.config.channel

    def test_explicit_channel_is_not_overridden(self):
        from lerobot.motors.nero import NeroMotorsBus, NeroMotorsBusConfig

        bus = NeroMotorsBus(NeroMotorsBusConfig(role="leader", channel="7"))
        assert bus.config.channel == "7"

    def test_explicit_interface_is_not_overridden(self):
        from lerobot.motors.nero import NeroMotorsBus, NeroMotorsBusConfig

        bus = NeroMotorsBus(NeroMotorsBusConfig(role="follower", interface="socketcan"))
        assert bus.config.interface == "socketcan"

    def test_bad_role_rejected(self):
        from lerobot.motors.nero.nero import _default_can_backend

        try:
            _default_can_backend("nope")
        except ValueError as exc:
            assert "role" in str(exc)
            return
        raise AssertionError("应该抛 ValueError")


# ===========================================================================
# 其它接口
# ===========================================================================


class TestMisc:
    def test_send_feedback_is_noop(self):
        leader, _ = _make_leader()
        assert leader.send_feedback({"x": 1.0}) is None

    def test_configure_is_noop(self):
        leader, _ = _make_leader()
        assert leader.configure() is None

    def test_calibrate_sets_flag(self):
        leader, _ = _make_leader()
        leader._is_calibrated = False
        leader.calibrate()
        assert leader.is_calibrated is True

    def test_calibrate_raises_when_not_connected(self):
        leader = NeroLeader(_make_config())
        try:
            leader.calibrate()
        except DeviceNotConnectedError:
            return
        raise AssertionError("应该抛 DeviceNotConnectedError")

    def test_wait_stable_pose_delegates(self):
        leader, fake = _make_leader()
        result = leader.wait_stable_pose(n=2, timeout_s=0.1)
        assert result is not None
        assert "wait_stable_pose" in fake.calls

    def test_get_joint_positions(self):
        leader, fake = _make_leader()
        fake.angles = [0.2] * ARM_JOINT_COUNT
        assert leader.get_joint_positions() == [0.2] * ARM_JOINT_COUNT


# ===========================================================================
# 与从臂的键兼容性
# ===========================================================================


class TestFollowerCompatibility:
    """从臂侧现在是**精简版**：固定 8 轴（7 关节 + 夹爪），没有 `use_gripper` 开关。

    → 遥操作时**主臂也必须 8 轴**，否则 action 键对不上，teleop 循环直接 KeyError。
    """

    def test_keys_match_nero_follower_8d(self):
        from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig

        leader, _ = _make_leader()
        follower = NeroFollower(NeroFollowerConfig(id="f"))
        assert list(leader.action_features) == list(follower.action_features)

    def test_follower_is_always_8d(self):
        """精简版从臂没有 `use_gripper` 开关，固定 8 轴。"""
        from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig

        follower = NeroFollower(NeroFollowerConfig(id="f"))
        want = [f"joint_{i}.pos" for i in range(1, 8)] + ["gripper.pos"]
        assert list(follower.action_features) == want

    def test_7d_leader_would_not_match_follower(self):
        """锁死这个约束：7 轴主臂配 8 轴从臂 = 键不一致，遥操作必然 KeyError。"""
        from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig

        leader, _ = _make_leader(use_gripper=False)
        follower = NeroFollower(NeroFollowerConfig(id="f"))
        assert list(leader.action_features) != list(follower.action_features)


# ===========================================================================
# 裸跑入口
# ===========================================================================


def _collect() -> list[tuple[str, Any]]:
    import inspect as _inspect

    found: list[tuple[str, Any]] = []
    module = sys.modules[__name__]
    for name, obj in vars(module).items():
        if not _inspect.isclass(obj) or not name.startswith("Test"):
            continue
        instance = obj()
        for meth_name, meth in vars(obj).items():
            if meth_name.startswith("test_") and callable(meth):
                found.append((f"{name}::{meth_name}", getattr(instance, meth_name)))
    return found


def _main() -> int:
    collected = _collect()
    print(f"\n收集到 {len(collected)} 个测试")
    print("=" * 72)
    passed = failed = 0
    failures: list[tuple[str, str]] = []
    for name, func in collected:
        try:
            func()
        except Exception as exc:
            failed += 1
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name}")
            print(f"        {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ok    {name}")
    print("=" * 72)
    print(f"通过 {passed}  失败 {failed}")
    if failures:
        print("\n失败明细：")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    _launched_by_pytest = any("pytest" in arg for arg in sys.argv)
    if _launched_by_pytest:
        raise SystemExit(0)
    try:
        sys.exit(_main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
