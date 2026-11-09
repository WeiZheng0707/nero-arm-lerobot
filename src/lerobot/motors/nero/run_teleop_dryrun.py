#!/usr/bin/env python

"""`python -m lerobot.teleoperate` 的**干跑**：不连硬件，验命令到底能不能跑。

【为什么需要它】改完 teleoperate 相关代码（注册点 / config 字段 / 主从键 / 遥操作循环），
最省事的验证不是接上两台臂试，而是**用同一条命令的原始参数**跑一遍：

    1) `draccus.parse` —— 与真实 CLI 走**完全相同**的解析路径
       （字段名拼错、类型不对、注册缺失，都在这一步暴露）
    2) `make_robot_from_config` / `make_teleoperator_from_config`
    3) 灌入假总线，真调一遍 `teleop_loop()` —— 循环里任何异常都会当场炸

秒级出结论，而且**臂不会动、不会使能、不占 CAN**。

【它验不了的（别把干跑通过当成真机能跑）】
  - 主臂通道 0x501-0x507 是否真的在上报
    （不上报会退回常规通道的**冻结缓存帧** → 从臂不跟随且**不报错**）
  - 从臂 `send_joints → move_j` 写通路（要跑 `run_follower.py --test-write`）
  - 夹爪 `move_gripper_m`
  - **channel 与哪台臂的对应关系**（Windows 上它是 cando 的设备序号，随插拔顺序变）

用法：
    python run_teleop_dryrun.py
        # 用下面 DEFAULT_TELEOP_ARGS 里那条 NERO 命令

    python run_teleop_dryrun.py -- --robot.channel=0 --teleop.channel=1 --fps=30
        # `--` 之后的内容**追加**在默认命令后面（重复的以后者为准），
        # 所以既能只改一个字段，也能整条换掉（如 --robot.type=so101_follower）

    python run_teleop_dryrun.py -- --robot.use_gripper=false --teleop.use_gripper=false
        # 不接夹爪时**两侧都要关**（只关一侧会因为键对不上直接 KeyError）

会先打一行 `lerobot 解析到本仓库 src`：本机 editable 安装指向**另一份副本**，
这行确认到底 import 的是哪份代码 —— 不对就直接拒绝往下跑。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# 把本仓库的 src 插到 sys.path 最前。
# ⚠️ 本机 conda env `lerobot` 的 editable 安装指向**另一份副本**（D:\arm_demo\...），
#    不插的话这里会测到那份旧代码，结论完全无效。
#
# ⚠️⚠️ parents 的层数必须数准（2026-09-19 在这个文件上真踩过）：
#     本文件在 <repo>/src/lerobot/motors/nero/ 下
#         parents[0] = motors/nero   parents[1] = motors
#         parents[2] = lerobot       parents[3] = src   ← 要的是这个
#     写成 parents[2] 会把 `src/lerobot` 插进 sys.path —— 那对 import 毫无意义，
#     `import lerobot` 于是静默落到 D 盘那份副本上。表现是：
#     报 `unrecognized arguments: --robot.use_gripper`，
#     而 usage 里冒出一堆本仓库根本没有的字段（effector / anchor_start_pose …）。
_REPO_SRC = Path(__file__).resolve().parents[3]  # .../lerobot_piper2-main/src
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

# 内置的默认命令（与手敲 `python -m lerobot.teleoperate ...` 等价）
DEFAULT_TELEOP_ARGS: list[str] = [
    "--robot.type=nero_follower",
    "--robot.id=nero_follower",
    "--robot.interface=agx_cando",
    "--robot.channel=1",
    "--teleop.type=nero_leader",
    "--teleop.id=nero_leader",
    "--teleop.interface=agx_cando",
    "--teleop.channel=0",
    "--robot.cameras={}",
    "--fps=60",
]

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


class FakeBus:
    """顶替 NeroMotorsBus：只实现 nero_follower / nero_leader 真正调用的那些方法。"""

    def __init__(
        self,
        joints: list[float],
        has_gripper: bool = True,
        cap: float = 0.05,
        gripper: float = 0.3,
        source: str = "follower",
    ):
        self._joints = list(joints)
        self._has_gripper = has_gripper
        self._cap = cap
        self._grip = gripper
        self.calls: list[str] = []
        # `role` 决定默认通道，这里只用来通过构造，不影响假数据
        self.config = type("C", (), {"has_gripper": has_gripper, "channel": "fake"})()
        # nero_leader 会读它取 angle_source
        self.stats: dict[str, object] = {"angle_source": source}

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def has_gripper(self) -> bool:
        return self._has_gripper

    @property
    def gripper_available(self) -> bool:
        return self._has_gripper

    def connect(self, enable: bool = True) -> bool:
        self.calls.append("connect")
        return True

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def read(self) -> dict[str, float]:
        state = {f"joint_{i}": v for i, v in enumerate(self._joints, 1)}
        if self._has_gripper:
            state["gripper"] = self._grip
        return state

    def try_read_joint_positions(self) -> list[float]:
        return list(self._joints)

    def wait_for_angles(self, timeout_s: float | None = None) -> bool:
        return True

    def wait_stable_pose(self, n: int = 3, tol_rad: float = 0.004, timeout_s: float = 20.0):
        return list(self._joints)

    def send_joints(self, joints: list[float]) -> list[float]:
        """返回**限幅后**的关节角（模拟真实 bus 的相对限幅）。"""
        self.calls.append("send_joints")
        return [
            cur + max(-self._cap, min(tgt - cur, self._cap))
            for cur, tgt in zip(self._joints, joints, strict=True)
        ]

    def write_gripper(self, value: float) -> None:
        self.calls.append("write_gripper")
        self._grip = float(value)
        return self._grip

    def read_gripper(self) -> float:
        return self._grip

    def read_gripper_or_last(self) -> float:
        return self._grip


def main(argv: list[str]) -> int:
    teleop_args = list(DEFAULT_TELEOP_ARGS)
    if "--" in argv:
        # `--` 之后的参数**追加**在默认命令后面。
        # argparse 对重复选项是"后者覆盖前者"，所以既能改单个字段，
        # 也能整条换掉（例如换个 --robot.type）。
        teleop_args += argv[argv.index("--") + 1 :]

    print(f"待验命令参数（{len(teleop_args)} 项）：")
    for a in teleop_args:
        print(f"  {a}")

    import draccus

    # ---- 硬闸门：确认 import 到的就是本仓库这份代码 ----
    # 本机 editable 安装指向**另一份副本**，路径只要插错一格就会静默测到旧代码，
    # 而结论看起来完全正常（字段名、键数都对得上），极难发现。
    section("0) 确认 import 到的是本仓库（不是那份 editable 指向的副本）")
    import lerobot

    actual = Path(lerobot.__file__).resolve()
    try:
        actual.relative_to(_REPO_SRC)
        in_repo = True
    except ValueError:
        in_repo = False
    check("lerobot 解析到本仓库 src", in_repo, str(actual))
    if not in_repo:
        print(f"        期望在 {_REPO_SRC} 之下。")
        print("        → 后面所有结论都无效，先修 sys.path 的层数或 PYTHONPATH。")
        return _summary()

    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperate import TeleoperateConfig, teleop_loop

    # ---------------------------------------------------------------- 1) 解析
    section("1) draccus 解析（与 `python -m lerobot.teleoperate` 同一条路径）")
    sys.argv = ["teleoperate", *teleop_args]
    try:
        cfg = draccus.parse(config_class=TeleoperateConfig)
        check("解析成功", True)
    except Exception as exc:
        check("解析成功", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return _summary()

    print(f"        robot.type       = {cfg.robot.type!r}")
    print(f"        robot.id         = {cfg.robot.id!r}")
    print(f"        robot.interface  = {getattr(cfg.robot, 'interface', '<无>')!r}")
    print(f"        robot.channel    = {getattr(cfg.robot, 'channel', '<无>')!r}")
    print(f"        robot.cameras    = {getattr(cfg.robot, 'cameras', '<无>')!r}")
    print(f"        robot.use_gripper= {getattr(cfg.robot, 'use_gripper', '<无>')!r}")
    print(f"        teleop.type      = {cfg.teleop.type!r}")
    print(f"        teleop.id        = {cfg.teleop.id!r}")
    print(f"        teleop.channel   = {getattr(cfg.teleop, 'channel', '<无>')!r}")
    print(f"        teleop.angle_src = {getattr(cfg.teleop, 'angle_source', '<无>')!r}")
    print(f"        teleop.use_gripper= {getattr(cfg.teleop, 'use_gripper', '<无>')!r}")
    print(f"        fps              = {cfg.fps}")
    print(f"        teleop_time_s    = {cfg.teleop_time_s}")

    r_ch = getattr(cfg.robot, "channel", None)
    t_ch = getattr(cfg.teleop, "channel", None)
    check(
        "主从通道不同（相同会抢同一条 CAN 总线）",
        r_ch != t_ch,
        f"robot={r_ch} teleop={t_ch}",
    )
    if hasattr(cfg.robot, "use_gripper") and hasattr(cfg.teleop, "use_gripper"):
        check(
            "主从 use_gripper 一致（不一致 → 键对不上直接 KeyError）",
            cfg.robot.use_gripper == cfg.teleop.use_gripper,
            f"robot={cfg.robot.use_gripper} teleop={cfg.teleop.use_gripper}",
        )

    # -------------------------------------------------------------- 2) 装配
    section("2) 工厂装配")
    try:
        robot = make_robot_from_config(cfg.robot)
        check("make_robot_from_config", True, type(robot).__name__)
    except Exception as exc:
        check("make_robot_from_config", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return _summary()

    try:
        teleop = make_teleoperator_from_config(cfg.teleop)
        check("make_teleoperator_from_config", True, type(teleop).__name__)
    except Exception as exc:
        check("make_teleoperator_from_config", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return _summary()

    # ------------------------------------------------------ 3) 键一致性
    section("3) 主从 action 键（遥操作的硬要求）")
    robot_keys = list(robot.action_features)
    teleop_keys = list(teleop.action_features)
    check("键与顺序完全一致", robot_keys == teleop_keys, f"{len(robot_keys)} 键")
    if robot_keys != teleop_keys:
        print(f"        从臂 {robot_keys}")
        print(f"        主臂 {teleop_keys}")
    check(
        "observation_features 覆盖 action 键",
        all(k in robot.observation_features for k in robot_keys),
    )
    check(
        "motor_features['action'].names 与 action_features 一致（record.py 依赖）",
        robot.motor_features["action"]["names"] == robot_keys,
    )
    check(
        "display_len 可算（teleop_loop 用它排版）",
        max(len(k) for k in robot.action_features) > 0,
        f"display_len={max(len(k) for k in robot.action_features)}",
    )

    # --------------------------------------------------------- 4) 跑循环
    section("4) 真跑一遍 teleop_loop（假总线，限时 0.15s）")
    # 主臂给一个和从臂**不同**的姿态，好确认"读主臂 -> 写从臂"这条链真的通了
    robot.bus = FakeBus([0.0] * 7, has_gripper=cfg.robot.use_gripper, source="follower")
    robot._is_connected = True
    teleop.bus = FakeBus(
        [0.5] * 7, has_gripper=cfg.teleop.use_gripper, gripper=0.6, source="leader"
    )
    teleop._is_connected = True

    try:
        teleop_loop(teleop, robot, cfg.fps, display_data=False, duration=0.15)
        check("teleop_loop 跑完且未抛异常", True)
    except Exception as exc:
        check("teleop_loop 跑完且未抛异常", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    check("从臂收到运动指令 send_joints", "send_joints" in robot.bus.calls)
    if cfg.robot.use_gripper:
        check("从臂收到夹爪指令 write_gripper", "write_gripper" in robot.bus.calls)
    check("从臂没有收到任何运动指令以外的写入",
          set(robot.bus.calls) <= {"send_joints", "write_gripper", "connect", "disconnect"},
          str(sorted(set(robot.bus.calls))))

    return _summary()


def _summary() -> int:
    print()
    print("=" * 70)
    if FAILS:
        print(f"未通过 {len(FAILS)} 项：")
        for name in FAILS:
            print(f"  - {name}")
        print()
        print("干跑没过就不要上真机。")
        return 1
    print("干跑全部通过 —— 命令的解析 / 装配 / 循环都可跑。")
    print()
    print("⚠️ 但干跑**覆盖不到**这 4 件事，真机前仍要自己确认：")
    print("  1) channel 与哪台臂对应（Windows 上是 cando 设备序号，随插拔顺序变）")
    print("     —— 搞反的话：被判为 leader 的那台其实是**从臂**，会被 set_leader_mode()")
    print("        变成零力状态、可能下坠；同时真从臂会收到 move_j")
    print("  2) 主臂通道 0x501-0x507 是否真的在上报（不上报会退回冻结缓存帧，")
    print("     表现为「拖主臂、从臂不动，且不报错」）")
    print("  3) 从臂写通路 send_joints -> move_j（跑 run_follower.py --test-write）")
    print("  4) 夹爪 move_gripper_m")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
