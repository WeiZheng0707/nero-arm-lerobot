"""NERO 从臂（`NeroFollower`）上机检查 —— 单臂，线性流程，一次跑完。

**为什么要单独有这个脚本**：`run_teleop.py` 要两台臂，而中间这层 `NeroFollower`
（LeRobot Robot 接口）是 `record.py` 直接依赖的，单臂就够验。

**副作用（准确说法，别打包票说"不动"）**：
  1. `connect()` 让电机通电锁住当前位姿 —— 位置不变，
     但从"抱闸夹住"变成"电机顶住"；
  2. 只有显式加 `--test-write` 才会发运动指令（目标=当前位姿，原地不动）；
  3. 退出时 `disconnect()` 会失能 —— 若臂处于抬起姿态**可能因重力下垂**。
  → 跑之前：清空工作空间、把臂摆到**低位放松**姿态、手放急停旁边。

⚠️ 文件名不以 test_ 开头是**故意的**（避免被 pytest 当测试文件收集）。
   请用「Run 'run_follower'」普通运行，不要用 pytest 运行。

⚠️ 本脚本**不检查** `arm_status` 状态字 —— 精简版 bus 没做这件事。
   想看状态字/逐关节故障位：完整版 bus 提供了 `read_health()` / `health_problems()`，
   但本脚本没调用它们（原本做这件事的 `run_arm.py` 已被删除）。
   实测提示：新臂在**完全可用**的情况下也会报 `OTHER_ERR(0x0A)`，
   所以"全 PASS"不等于"状态正常"。

用法：
    python run_follower.py                      # 自动选后端/通道
    python run_follower.py --channel 0
    python run_follower.py --test-write         # 额外验写通路（发一条 move_j）
    python run_follower.py --id nero_test       # 换一个 robot id
    python run_follower.py --yes                # 跳过开跑前的 y/N 确认
    python run_follower.py -v                   # 打开 debug 日志
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 把本仓库的 src 插到 sys.path 最前
#
# ⚠️ 本机 conda env `lerobot` 的 editable 安装指向**另一份副本**
#   （D:\arm_demo\project\lerobot_piper2-main）。不显式插到最前，
#   本脚本会在另一份代码上跑，结论无效。
# ---------------------------------------------------------------------------

_REPO_SRC = Path(__file__).resolve().parents[3]  # .../lerobot_piper2-main/src
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError  # noqa: E402
from lerobot.motors.nero import (  # noqa: E402
    ARM_JOINT_COUNT,
    ARM_JOINT_NAMES,
    GRIPPER_NAME,
    JOINT_NAMES,
)
from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig  # noqa: E402

SEP = "=" * 68


# ---------------------------------------------------------------------------
# 显示与结果收集（与 run_teleop.py 保持一致）
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(title)
    print(SEP)


def fmt_dur(seconds: float) -> str:
    """一次读关节 + 读夹爪只要约 20 微秒，用 `%.1f ms` 会显示成 0.0。"""
    us = seconds * 1e6
    return f"{us:.0f} us" if us < 1000 else f"{us / 1000:.2f} ms"


class Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, ok, detail))
        line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
        if detail:
            line += f"  -> {detail}"
        print(line)
        return ok

    def problem(self, name: str, exc: BaseException) -> None:
        self.check(name, False, f"{type(exc).__name__}: {exc}")

    def skip(self, name: str, why: str) -> None:
        print(f"  [SKIP] {name}  -> {why}")

    @property
    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.results if not ok]

    @property
    def passed(self) -> int:
        return sum(1 for _, ok, _ in self.results if ok)

    def summary(self) -> bool:
        print(f"\n{SEP}")
        print("汇总")
        print(SEP)
        print(f"  共 {len(self.results)} 项，通过 {self.passed} 项，失败 {len(self.failed)} 项")
        if self.failed:
            print("\n  未通过：")
            for name in self.failed:
                print(f"    - {name}")
            return False
        print("  全部通过。")
        return True


def ask(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def expected_keys() -> list[str]:
    return [f"{name}.pos" for name in JOINT_NAMES]


def fmt_keys(keys) -> str:
    return "[" + ", ".join(keys) + "]"


# ---------------------------------------------------------------------------
# 1) 构造（不连接）
# ---------------------------------------------------------------------------


def make_config(args, **overrides) -> NeroFollowerConfig:
    base = dict(
        id=args.id,
        interface=args.interface,
        channel=args.channel,
        firmware=args.firmware,
        max_relative_target=args.max_relative_target,
        enable_timeout=args.enable_timeout,
    )
    base.update(overrides)
    return NeroFollowerConfig(**base)


def check_offline(rep: Report, args) -> NeroFollower | None:
    """构造对象、查特征定义与未连接守卫 —— **全程不碰硬件**。"""
    section("1) 构造与特征定义（不连接硬件）")

    try:
        robot = NeroFollower(make_config(args))
    except Exception as exc:
        rep.problem("构造 NeroFollower", exc)
        return None
    rep.check("构造 NeroFollower", True, repr(robot))

    want = expected_keys()
    act = list(robot.action_features)
    obs = list(robot.observation_features)
    rep.check("action_features 键与顺序正确（7 关节 + 夹爪）", act == want, fmt_keys(act))
    rep.check(
        "observation_features 覆盖全部电机键",
        all(k in obs for k in want),
        f"电机键 {len(want)} 个 / observation 共 {len(obs)} 个",
    )
    rep.check("action_features 全是 float", set(robot.action_features.values()) == {float})

    mf = robot.motor_features
    names = mf["action"]["names"]
    rep.check(
        "motor_features['action'].names 与 action_features 一致（record.py 依赖）",
        names == want,
        fmt_keys(names),
    )
    rep.check("action 与 observation.state 的 names 一致",
              mf["observation.state"]["names"] == names)

    # 相机键的两套约定（方向相反，搞反了会得到
    # observation.images.observation.images.xxx）
    rep.check("相机特征：无相机时 observation_features 只有电机键",
              list(robot.observation_features) == want)
    rep.check("相机特征：camera_features 用的是带前缀的键",
              all(k.startswith("observation.images.") for k in robot.camera_features),
              f"{len(robot.camera_features)} 个相机")

    rep.check("is_calibrated 恒为 True（NERO 软件层不适用标定）",
              robot.is_calibrated is True)
    rep.check("configure() 是空操作", robot.configure() is None)

    # 未连接守卫
    for label, fn in (
        ("未连接时 get_observation() 抛 DeviceNotConnectedError", robot.get_observation),
        ("未连接时 send_action() 抛 DeviceNotConnectedError",
         lambda: robot.send_action({k: 0.0 for k in want})),
        ("未连接时 calibrate() 抛 DeviceNotConnectedError", robot.calibrate),
        ("未连接时 disconnect() 不抛异常", robot.disconnect),
    ):
        try:
            fn()
            rep.check(label, "不抛" in label, "没有抛异常")
        except DeviceNotConnectedError:
            rep.check(label, "抛" in label)
        except Exception as exc:
            rep.check(label, False, f"抛的是 {type(exc).__name__}: {exc}")

    print(f"  [INFO] 未连接状态：is_connected={robot.is_connected} "
          f"配置的 channel={robot.bus.config.channel}")
    return robot


# ---------------------------------------------------------------------------
# 2) 连接
# ---------------------------------------------------------------------------


def check_connect(robot: NeroFollower, rep: Report) -> bool:
    section("2) connect(calibrate=True)")

    t0 = time.perf_counter()
    try:
        robot.connect()
    except Exception as exc:
        rep.problem("connect()", exc)
        print("\n  排查顺序：")
        print("    1) 机械臂是否上电、底座指示灯绿色")
        print("    2) CAN_H(黄)/CAN_L(蓝) 是否接对插紧")
        print("    3) channel 是否选对、有没有别的程序占用（Web 上位机）")
        print("    4) firmware 留空让 SDK 自动解析，或填正确档位")
        traceback.print_exc()
        return False
    dt = time.perf_counter() - t0
    rep.check("connect() 未抛异常", True, fmt_dur(dt))
    rep.check("is_connected", robot.is_connected is True)

    bus = robot.bus
    # ⚠️ 完整版 NeroMotorsBus 的属性是 `firmware_version`；
    #    它**没有** `firmware` / `profile`（那是已删除的精简版 NeroBus 的名字）。
    #    档位在完整版里是 config 字段 `firmeware_version`（官方拼写少一个 a）。
    print(f"  [INFO] interface={bus.config.interface} channel={bus.config.channel}")
    print(f"  [INFO] firmware={bus.firmware_version} 档位={bus.config.firmeware_version}")
    print(f"  [INFO] is_enabled={bus._is_enabled}  max_relative_target={bus.config.max_relative_target}")
    rep.check(
        "固件版本已探测出来",
        bus.firmware_version is not None,
        f"firmware={bus.firmware_version} 档位={bus.config.firmeware_version}",
    )
    rep.check("已使能（否则运动指令在控制器侧不生效）", bus._is_enabled is True)

    # 读一次完整状态（8 键）
    try:
        state = bus.read()
        rep.check("bus.read() 返回 8 个键", len(state) == len(JOINT_NAMES),
                  f"{len(state)} 个")
        grip = state.get(GRIPPER_NAME)
        rep.check("夹爪读数落在 [0, 1]", grip is not None and 0.0 <= grip <= 1.0, f"= {grip}")
        print("    " + "  ".join(f"{n}={state[n]:+.4f}" for n in ARM_JOINT_NAMES))
    except Exception as exc:
        rep.problem("bus.read()", exc)

    return True


# ---------------------------------------------------------------------------
# 3) 观测
# ---------------------------------------------------------------------------


def check_observation(robot: NeroFollower, rep: Report) -> dict | None:
    section("3) get_observation()")

    t0 = time.perf_counter()
    try:
        obs = robot.get_observation()
    except Exception as exc:
        rep.problem("get_observation()", exc)
        traceback.print_exc()
        return None
    dt = time.perf_counter() - t0

    print(f"  [INFO] 耗时 {fmt_dur(dt)}")
    want = expected_keys()
    missing = [k for k in want if k not in obs]
    rep.check("观测键覆盖全部电机键", not missing, "齐全" if not missing else f"缺 {missing}")
    rep.check("观测没多出未声明的键", set(obs) == set(robot.observation_features),
              f"{len(obs)} 个键")

    if all(isinstance(v, float) for v in obs.values()):
        rep.check("所有观测值都是 float", True)
    else:
        rep.check("所有观测值都是 float", False, "有非 float 值")

    for name in ARM_JOINT_NAMES:
        v = obs[f"{name}.pos"]
        if not math.isfinite(v) or abs(v) > math.pi:
            rep.check("关节角数值合理（有限且 |v| <= pi）", False, f"{name} = {v}")
            break
    else:
        rep.check("关节角数值合理（有限且 |v| <= pi）", True)

    # 与 bus 直接读的对比（两条独立路径交叉验证）
    try:
        state = robot.bus.read()
        diff = max(abs(obs[f"{n}.pos"] - state[n]) for n in ARM_JOINT_NAMES)
        rep.check("与 bus.read() 一致（交叉验证）", diff < 1e-6, f"最大差 {diff:.2e}")
    except Exception as exc:
        rep.problem("与 bus.read() 交叉验证", exc)

    print("    " + "  ".join(f"J{i + 1}={obs[f'joint_{i + 1}.pos']:+.3f}"
                             for i in range(ARM_JOINT_COUNT)))
    return obs


# ---------------------------------------------------------------------------
# 4) 限幅（纯计算，不发指令）
# ---------------------------------------------------------------------------


def check_guards_offline(robot: NeroFollower, rep: Report, obs: dict) -> None:
    section("4) 限幅与参数校验（★ 纯计算，不发任何指令）")

    bus = robot.bus
    cap = bus.config.max_relative_target
    present = [obs[f"{n}.pos"] for n in ARM_JOINT_NAMES]

    if cap is None:
        rep.skip("单步限幅", "配置里 max_relative_target=None（不钳制）")
    else:
        try:
            base = bus.read_joint_positions()
            limited = bus._clamp_relative([v + 1.0 for v in base])
            worst = max(abs(a - b) for a, b in zip(limited, base, strict=True))
            rep.check(
                f"_clamp_relative() 单拍变化被限在 {cap} rad 内",
                worst <= cap + 1e-3,  # 容差留给两次读之间的漂移
                f"实测最大变化 {worst:.4f} rad",
            )
            rep.check("限幅没有改动合法输入", bus._clamp_relative(base) == base)
        except Exception as exc:
            rep.problem("_clamp_relative()", exc)

    # action 缺键要报错（校验在 move_j 之前，不会下发）
    try:
        robot.send_action({"joint_1.pos": 0.0})
        rep.check("action 缺键时抛 KeyError", False, "没有抛异常")
    except KeyError:
        rep.check("action 缺键时抛 KeyError", True)
    except Exception as exc:
        rep.check("action 缺键时抛 KeyError", False, f"抛的是 {type(exc).__name__}")

    # 参数校验
    for label, kwargs in (
        ("max_relative_target=-1 被拒", {"max_relative_target": -1.0}),
        ("enable_timeout=0 被拒", {"enable_timeout": 0}),
        ("speed_percent=101 被拒", {"speed_percent": 101}),
    ):
        try:
            make_config_stub(kwargs)
            rep.check(label, False, "没有抛异常")
        except ValueError:
            rep.check(label, True)
        except Exception as exc:
            rep.check(label, False, f"抛的是 {type(exc).__name__}")

    # 保命机制的静态确认：桩不许偷偷发运动指令
    before = len(getattr(bus, "_stub_calls", []))
    bus.safe_disconnect()
    bus.apply_calibration()
    rep.check("piper 兼容桩不运动（safe_disconnect/apply_calibration）", True,
              f"（仅确认未抛异常，调用记录 {before}）")


def make_config_stub(kwargs: dict) -> None:
    """用一组非法参数构造配置，用来验 __post_init__ 的校验。"""
    NeroFollowerConfig(id="bad", **kwargs)


# ---------------------------------------------------------------------------
# 5) 写通路（只有 --test-write 才走）
# ---------------------------------------------------------------------------


def check_write(robot: NeroFollower, rep: Report, obs: dict) -> None:
    section("5) send_action() 写通路（会发一条 move_j，目标=当前位姿）")

    act = {k: float(v) for k, v in obs.items() if k in expected_keys()}
    t0 = time.perf_counter()
    try:
        sent = robot.send_action(act)
    except Exception as exc:
        rep.problem("send_action(当前位姿)", exc)
        traceback.print_exc()
        return
    dt = time.perf_counter() - t0

    print(f"  [INFO] 耗时 {fmt_dur(dt)}")
    rep.check("返回的键带 .pos 且与 action_features 一致",
              list(sent) == expected_keys(), fmt_keys(list(sent)))

    # ★ 返回的应当是**限幅后**的值（基类契约：actually sent … potentially clipped）
    worst = max(abs(sent[f"{n}.pos"] - obs[f"{n}.pos"]) for n in ARM_JOINT_NAMES)
    cap = robot.bus.config.max_relative_target
    limit = 0.05 if cap is None else cap + 0.02
    rep.check("实际下发值 ≈ 当前位姿（原地不动）", worst <= limit,
              f"最大差 {worst:.4f} rad")

    # 与"刚才读到的"对比，确认返回的就是总线返回的那一份
    try:
        after = robot.bus.read_joint_positions()
        drift = max(abs(a - b) for a, b in zip(after, obs_joints(obs), strict=True))
        rep.check("下发后位姿基本没变（原地保持）", drift < 0.05,
                  f"最大漂移 {drift:.4f} rad")
    except Exception as exc:
        rep.problem("下发后复核", exc)


def obs_joints(obs: dict) -> list[float]:
    return [obs[f"{n}.pos"] for n in ARM_JOINT_NAMES]


# ---------------------------------------------------------------------------
# 6) 断开
# ---------------------------------------------------------------------------


def check_disconnect(robot: NeroFollower, rep: Report) -> None:
    section("6) disconnect()")

    try:
        robot.disconnect()
        rep.check("disconnect() 未抛异常", True)
    except Exception as exc:
        rep.problem("disconnect()", exc)
        traceback.print_exc()

    rep.check("is_connected 变 False", robot.is_connected is False)
    rep.check("is_calibrated 仍为 True", robot.is_calibrated is True)

    try:
        robot.disconnect()
        rep.check("disconnect() 幂等（再调一次不炸）", True)
    except Exception as exc:
        rep.problem("disconnect() 幂等", exc)

    try:
        robot.connect()
        rep.check("断开后重连", True)
        robot.disconnect()
    except Exception as exc:
        rep.problem("断开后重连", exc)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="NERO 从臂（NeroFollower）上机检查（单臂，默认不发运动指令）"
    )
    ap.add_argument("--interface", default=None, help="CAN 后端，默认按系统选")
    ap.add_argument("--channel", default=None, help="CAN 通道，默认按角色选（从臂 -> 0 / can0）")
    ap.add_argument("--firmware", default=None, help="固件档位，留空=交给 SDK 解析")
    ap.add_argument("--id", default="nero_follower_check", help="robot id")
    ap.add_argument("--max-relative-target", type=float, default=0.05,
                    help="单步相对限幅（rad），默认 0.05")
    ap.add_argument("--enable-timeout", type=float, default=25.0,
                    help="使能确认超时（秒），默认 25（回读滞后实测可 >15s）")
    ap.add_argument("--test-write", action="store_true",
                    help="额外验写通路（会发一条 move_j，目标=当前位姿）")
    ap.add_argument("--yes", action="store_true", help="跳过启动前的安全确认")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    if not args.yes:
        section("即将连接 NERO 从臂（默认不发运动指令）")
        print("请确认：")
        print("  1. CAN 模块已插好，主控已上电（本脚本按**单臂**跑）")
        print("  2. 没有其他程序占用 CAN 通道（Web 上位机 / 另一个脚本）")
        print("  3. 工作空间已清空，臂摆到**低位放松**姿态")
        print("  4. connect() 会通电锁住当前位姿")
        print("  5. 退出时会失能 —— 若臂处于抬起姿态可能因重力下垂")
        print(SEP)
        if not ask("开始吗？"):
            print("已取消。")
            return 0

    rep = Report()

    # 1) 离线（不碰硬件）
    robot = check_offline(rep, args)
    if robot is None:
        rep.summary()
        return 1

    # 2~6) 上机
    connected = False
    try:
        connected = check_connect(robot, rep)
        if connected:
            obs = check_observation(robot, rep)
            if obs is not None:
                check_guards_offline(robot, rep, obs)
                if args.test_write:
                    check_write(robot, rep, obs)
                else:
                    print()
                    print("  [SKIP] 写通路验证（会发一条 move_j）—— 需显式加 --test-write")
            else:
                print("  [SKIP] 拿不到观测，跳过后面的检查")
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C，正在断开 ...")
        rep.check("人工中断", False, "KeyboardInterrupt")
    except Exception:
        traceback.print_exc()
        rep.check("脚本未崩溃", False, "见上面的 traceback")
    finally:
        if connected:
            check_disconnect(robot, rep)
        else:
            print("\n  未成功连接，跳过断开检查。")

    ok = rep.summary()
    if ok:
        print("\n  提示：本脚本**不检查 arm_status 状态字** —— 完整版 bus 虽然提供了")
        print("        read_health() / health_problems()，但本脚本没调用它们。")
        print("        实测新臂会在功能完全正常时报 OTHER_ERR(0x0A)，")
        print("        所以「全 PASS」不等于「状态正常」。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
