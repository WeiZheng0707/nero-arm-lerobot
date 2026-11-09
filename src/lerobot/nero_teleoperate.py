"""NERO-to-NERO joint teleoperation, including the AgxGripper end effector."""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

try:  # 仅 Windows：让"在控制台里按 Q"也能退出，不要求预览窗口有焦点
    import msvcrt
except ImportError:  # pragma: no cover - 非 Windows 平台
    msvcrt = None

from lerobot.motors.nero import ARM_JOINT_NAMES
from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig
from lerobot.teleoperators.nero_leader import NeroLeader, NeroLeaderConfig

logger = logging.getLogger(__name__)
JOINT_KEYS = tuple(f"{name}.pos" for name in ARM_JOINT_NAMES)
DEFAULT_START_POSE = Path(__file__).resolve().parents[2] / "examples" / "nero" / "saved_start_pose.json"
PREVIEW_WINDOW = "NERO teleoperation"
# 单轴归位上限，与 examples/nero/start_pose.py 的 MAX_RETURN_DEG 保持一致。
# 现场姿态和恢复点差太远时，归位会变成一次大范围搬运，必须先显式确认。
MAX_RETURN_DEG_PER_AXIS = 60.0


class LatestCamera:
    """后台线程持续读帧，主线程只取最新帧。

    ⚠️ DSHOW 的坑：`isOpened()` 返回 True **不代表会出帧**。设备被别的进程
    半占用、USB 带宽不足、分辨率不被支持、驱动进入坏状态时，`read()` 会一直
    返回 False 但**不抛异常**。原实现把这种失败静默吞掉，上层只能看到
    `latest()` 永远返回 None —— 表现就是"窗口建好了但一片白，也没有任何报错"，
    完全无法判断是相机坏了、线松了还是程序卡了。

    所以这里显式记账：帧计数、连续失败数、最后一次异常，并对外提供
    `status_text()` 供占位图直接显示。
    """

    def __init__(self, index: int, backend: str, width: int, height: int, fps: int):
        api = cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_MSMF
        self.index = index
        self.size = (width, height)
        self.capture = cv2.VideoCapture(index, api)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera index={index} backend={backend}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = None
        self.updated_at = 0.0
        self.frame_count = 0
        self.read_failures = 0      # 连续失败次数
        self.total_failures = 0
        self.last_error = ""
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.stop_event.is_set():
            try:
                ok, frame = self.capture.read()
            except Exception as exc:  # noqa: BLE001 - 读线程里任何异常都不能让它静默死掉
                ok, frame = False, None
                with self.lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
            if ok and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.updated_at = time.monotonic()
                    self.frame_count += 1
                    self.read_failures = 0
            else:
                with self.lock:
                    self.read_failures += 1
                    self.total_failures += 1
                self.stop_event.wait(0.01)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def age_ms(self) -> float:
        """最新帧的年龄（毫秒）。这是预览延迟的主要来源，相机 30fps 时约 33ms 一轮。"""
        with self.lock:
            stamp = self.updated_at
        return 0.0 if stamp <= 0.0 else (time.monotonic() - stamp) * 1000.0

    def is_live(self, threshold_ms: float = 1000.0) -> bool:
        """是否正在出帧。相机 30fps 时 age 约 33ms；超过阈值就是掉线或卡住。"""
        with self.lock:
            frames = self.frame_count
        return frames > 0 and self.age_ms() <= threshold_ms

    def wait_first_frame(self, timeout_s: float = 4.0) -> bool:
        """等首帧。打开成功但迟迟不出帧时，必须尽早暴露，不能拖到白屏。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.latest() is not None:
                return True
            time.sleep(0.02)
        return False

    def status_text(self) -> str:
        """一行状态，直接画到预览图上，让"白屏"变成可读的诊断信息。"""
        with self.lock:
            frames = self.frame_count
            fails = self.read_failures
            total = self.total_failures
            error = self.last_error
        if frames == 0:
            state = "NO FRAME YET"
        elif fails > 0:
            state = f"STALLED (consecutive fail {fails})"
        else:
            state = "OK"
        text = f"idx{self.index} {state} frames={frames} fail={total} age={self.age_ms():.0f}ms"
        return f"{text} | {error}" if error else text

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.5)
        try:
            self.capture.release()
        except Exception:  # noqa: BLE001 - 退出路径不能被清理失败打断
            logger.debug("camera release failed", exc_info=True)


def format_positions(values: dict, width_m: float, mode: str = "width",
                     max_angle_deg: float = 100.0) -> str:
    joints = " ".join(f"J{index}={math.degrees(values[key]):.2f}"
                      for index, key in enumerate(JOINT_KEYS, 1))
    gripper = "disabled"
    if "gripper.pos" in values:
        scale, unit = (width_m * 1000, "mm") if mode == "width" else (max_angle_deg, "deg")
        gripper = f"{values['gripper.pos'] * scale:.2f} {unit}"
    return f"joints(deg): {joints} | gripper={gripper}"


@dataclass
class NeroTeleopConfig:
    fps: int = 120
    duration_s: float | None = None
    mapping: str = "relative"
    # 关节参考速度上限。0.3 rad/s 只是"能跟上慢速摆臂"，手稍微快一点从臂就追不上，
    # 体感上会被误判成"延迟大"。1.2 rad/s 约 69 deg/s，接近舒适手动速度。
    max_joint_speed: float = 1.2  # rad/s, reference trajectory limit
    max_gripper_speed: float = 2.5  # normalized opening/s; 70 mm follower ~= 175 mm/s
    max_control_dt_s: float = 0.1
    max_start_delta: float = 0.1  # rad, absolute mode startup gate
    startup_timeout_s: float = 10.0
    # 预览刷新率与控制回路解耦：相机只有 30fps，让 imshow 跟着 120Hz 控制回路跑
    # 只会白占主线程时间，反过来拖慢控制。
    preview_fps: float = 30.0

    def __post_init__(self):
        if self.mapping not in ("relative", "absolute"):
            raise ValueError("mapping must be 'relative' or 'absolute'")
        for name in ("fps", "max_joint_speed", "max_gripper_speed", "max_control_dt_s", "max_start_delta",
                     "startup_timeout_s", "preview_fps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.duration_s is not None and (
            not math.isfinite(self.duration_s) or self.duration_s <= 0
        ):
            raise ValueError("duration_s must be finite and positive")


class TargetMapper:
    """Map leader motion into the follower frame and bound reference velocity."""

    def __init__(self, config: NeroTeleopConfig, leader_start: dict, follower_start: dict):
        self.config = config
        self.keys = tuple(leader_start)
        self.leader_start = dict(leader_start)
        self.follower_start = {key: follower_start[key] for key in self.keys}
        self.previous = dict(self.follower_start)
        if not all(math.isfinite(v) for v in (*leader_start.values(), *self.previous.values())):
            raise ValueError("Startup positions must be finite")
        if config.mapping == "absolute":
            error = max(abs(leader_start[key] - follower_start[key]) for key in JOINT_KEYS)
            if error > config.max_start_delta:
                raise ValueError(
                    f"Absolute startup mismatch {error:.3f} rad exceeds {config.max_start_delta:.3f}; "
                    "align the arms manually or use relative mapping"
                )

    def target(self, action: dict[str, float], dt: float) -> dict[str, float]:
        if set(action) != set(self.keys) or not all(math.isfinite(v) for v in action.values()):
            raise ValueError("Leader action keys/values are invalid")
        # A delayed loop must not turn into one large position step.
        dt = max(0.0, min(dt, self.config.max_control_dt_s))
        result = {}
        for key in self.keys:
            value = action[key]
            if key in JOINT_KEYS:
                if self.config.mapping == "relative":
                    value += self.follower_start[key] - self.leader_start[key]
                cap = self.config.max_joint_speed * dt
            else:
                value = max(0.0, min(value, 1.0))
                cap = self.config.max_gripper_speed * dt
            previous = self.previous[key]
            result[key] = previous + max(-cap, min(value - previous, cap))
        return result

    def accept(self, sent: dict[str, float]) -> None:
        self.previous = dict(sent)


def load_start_pose(path: str | Path) -> dict[str, list[float]]:
    pose_path = Path(path).expanduser().resolve()
    with pose_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema") != 1 or data.get("unit") != "rad":
        raise ValueError(f"Unsupported start-pose file: {pose_path}")
    result: dict[str, list[float]] = {}
    for role in ("leader", "follower"):
        entry = data.get(role, {})
        values = entry.get("joints")
        if (entry.get("firmware") not in (None, "1.20", "1.21")
                or not isinstance(values, list) or len(values) != 7
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)):
            raise ValueError(f"Invalid {role} start pose in {pose_path}")
        result[role] = [float(value) for value in values]
    return result


def return_to_start_pose(leader: NeroLeader, follower: NeroFollower,
                         pose: dict[str, list[float]], duration_s: float,
                         allow_large_return: bool = False,
                         pose_path: str = "start pose",
                         leader_drag_timeout_s: float = 45.0) -> None:
    """使能后把两臂带回恢复点。**两臂都能自动归位**，前提是调用方按模式连接。

    核心前提（踩了的表现是"主臂一动不动，程序却说成功"）：

      主臂必须以 `enter_leader_mode=False` 连接，停在固件上电默认的**普通位置
      模式**。一旦进了零力拖拽（linkage 0xFA），控制器就不再接受位置指令：
      `slow_move_to()` 只是把整条轨迹发完就返回，而 `wait_for_angles()` 优先解析
      leader 通道、drag 模式下那条通道照样在推数据 —— 它会报"反馈正常"，把
      "根本没动"彻底掩盖。

      固件 1.20/1.21 上 `set_normal_mode()` 是官方标注的空操作（v112 里只抛
      UserWarning 然后 return None），进去 0xFA 就回不来。所以**只能一开始就
      别进去**：先归位，再由调用方 `enter_leader_mode()` 进零力拖拽。

    这里仍用 NORMAL 通道（0x2A1-0x2A7）有无推送来判断主臂是否真的在位置模式，
    判据不成立就降级为人工拖动，绝不盲发指令。
    """
    # ---- 1) 两臂都先量差值，把风险先摆到屏幕上 ----
    for role, device in (("follower", follower), ("leader", leader)):
        current = device.bus.wait_stable_pose(timeout_s=3.0)
        if current is None:
            raise TimeoutError(f"{role}: 归位前读不到稳定当前位姿，拒绝盲动")
        deltas = [math.degrees(goal - now)
                  for now, goal in zip(current, pose[role], strict=True)]
        worst = max(abs(value) for value in deltas)
        print(f"START POSE | {role} 归位预览: 单轴最大 {worst:.1f} deg | "
              f"各轴(deg)={[round(value, 1) for value in deltas]}", flush=True)
        if worst > MAX_RETURN_DEG_PER_AXIS and not allow_large_return:
            raise RuntimeError(
                f"{role} 需要单轴移动 {worst:.1f} deg，"
                f"超过 {MAX_RETURN_DEG_PER_AXIS:.0f} deg 安全上限。\n"
                f"  说明现场姿态与恢复点 {pose_path} 不一致。两条路选一条：\n"
                "  1) 先把两臂摆到想要的位姿，重新保存恢复点（推荐）\n"
                "  2) 确认这次大范围运动安全后，显式加 --allow-large-return"
            )

    # ---- 2) 从臂归位 ----
    print("START POSE: 从臂归位中 ...", flush=True)
    follower.bus.slow_move_to(pose["follower"], duration_s=duration_s, fps=50, hold_after_s=0.5)
    settled = follower.bus.wait_stable_pose(timeout_s=5.0)
    if settled is None:
        raise TimeoutError("Follower did not provide stable feedback after start-pose return")
    residual = max(abs(math.degrees(goal - now))
                   for now, goal in zip(settled, pose["follower"], strict=True))
    print(f"START POSE: follower 到位，残差 {residual:.2f} deg", flush=True)
    if residual > 5.0:
        print(f"  [WARN] 从臂残差 {residual:.1f} deg 偏大："
              "可能被限位、被线缆拉住或负载挡了一下。", flush=True)

    # ---- 3) 主臂归位：确认在普通位置模式才驱动 ----
    normal_hz = leader.bus.normal_feedback_hz()
    if normal_hz:
        print(f"START POSE: 主臂普通位置模式 {normal_hz:.0f} Hz，归位中 ...", flush=True)
        leader.bus.slow_move_to(pose["leader"], duration_s=duration_s, fps=50, hold_after_s=0.5)
        moved = leader.bus.wait_stable_pose(timeout_s=5.0)
        if moved is not None:
            moved_residual = max(abs(math.degrees(goal - now))
                                 for now, goal in zip(moved, pose["leader"], strict=True))
            if moved_residual <= 5.0:
                print(f"START POSE: 主臂到位，残差 {moved_residual:.1f} deg", flush=True)
                print("START POSE: 两臂都在恢复点，即将进入零力拖拽并开始遥操作。", flush=True)
                return
            print(f"  [WARN] 主臂归位后仍差 {moved_residual:.1f} deg，转人工拖动", flush=True)
    else:
        print("START POSE: 主臂不在普通位置模式（NORMAL 通道无推送），无法自动归位。",
              flush=True)

    if leader_drag_timeout_s <= 0:
        print("START POSE: 主臂归位由调用方自行处理（本函数跳过等待）。", flush=True)
        return

    print(
        "START POSE: 主臂需要人工拖到恢复点附近。\n"
        "  ==> 请现在用手拖主臂（不必很准，相对映射会以实际姿态锚定）。\n"
        f"  等待上限 {leader_drag_timeout_s:.0f} 秒，超时不会中止。",
        flush=True,
    )
    deadline = time.monotonic() + leader_drag_timeout_s
    target = pose["leader"]
    best_error = float("inf")
    announced = False
    while time.monotonic() < deadline:
        try:
            angles = leader.bus.try_read_joint_positions()
        except (RuntimeError, TimeoutError):
            angles = None
        if angles is not None:
            error = max(abs(a - b) for a, b in zip(angles, target, strict=True))
            best_error = min(best_error, error)
            if error <= math.radians(10.0):
                print(f"START POSE: 主臂已在目标附近（差 {math.degrees(error):.1f} deg），"
                      "开始遥操作。", flush=True)
                return
            if not announced:
                print(f"  当前主臂距目标 {math.degrees(error):.1f} deg，继续等待拖动…",
                      flush=True)
                announced = True
        time.sleep(0.1)
    print(f"  [WARN] {leader_drag_timeout_s:.0f}s 内未把主臂拖到目标附近（最近差 "
          f"{math.degrees(best_error):.1f} deg）。按当前姿态继续：相对映射会以实际姿态重新锚定，"
          "从臂不会跳。", flush=True)


def console_quit_requested() -> bool:
    """控制台里按 Q / Esc 是否等于请求退出。

    为什么需要：`cv2.waitKey` 只在预览窗口**拥有焦点**时才收得到按键。用户
    在控制台窗口敲 Q 是完全没反应的，很容易以为程序卡死。这里读控制台缓冲区
    作为兜底，两个窗口按哪个都算。
    """
    if msvcrt is None:
        return False
    try:
        while msvcrt.kbhit():
            if msvcrt.getwch() in ("q", "Q", "\x1b"):
                return True
    except Exception:  # 非交互式控制台（重定向）下 kbhit 可能抛错
        return False
    return False


def _placeholder_preview(cameras: list[tuple[str, "LatestCamera"]], count: int,
                        measured_hz: float):
    """没有任何可用帧时画的诊断占位图。

    存在的意义：白屏是最糟糕的故障表现 —— 用户无法区分"相机没出图"和
    "程序卡死"，也没法决定该等还是该断电。把每路相机的状态直接写在画面上，
    故障就从"一片白"变成"idx1 STALLED consecutive fail 214"。
    """
    import numpy as np

    width = sum(camera.size[0] for _, camera in cameras) or 640
    height = max((camera.size[1] for _, camera in cameras), default=480)
    # 深灰底而不是纯黑：纯黑和"窗口没画东西"在当前截图上很难区分
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.putText(canvas, "NO CAMERA FRAME", (28, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 2, cv2.LINE_AA)
    for position, (name, camera) in enumerate(cameras):
        cv2.putText(canvas, f"{name}: {camera.status_text()}", (28, 110 + position * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    hint_y = height - 56
    cv2.putText(canvas, "check: camera in use by another app / USB cable / index changed",
                (28, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Q / Esc = QUIT      count={count}  loop={measured_hz:.0f}Hz",
                (28, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


def compose_preview(cameras: list[tuple[str, "LatestCamera"]], count: int,
                    measured_hz: float = 0.0):
    """把多路相机横向拼成一张预览图。**任何情况下都返回一张图，永不返回 None。**

    以前没有可用帧时返回 None，调用方 `if canvas is not None` 就跳过 imshow，
    窗口一直空白 —— 那就是用户看到的"白屏"，而且全程零报错。现在没帧会返回
    诊断占位图，`imshow` 无条件执行，所以任何时刻窗口内容都是可解释的。

    帧龄是判断"看起来延迟"的直接依据：相机 30fps 时它稳定在 0–33ms；
    一旦持续 >100ms，说明读取线程被阻塞或 USB 带宽不够。
    """
    tiles = []
    missing = []
    for name, camera in cameras:
        frame = camera.latest()
        if frame is None:
            missing.append(name)
            continue
        age = camera.age_ms()
        cv2.putText(frame, f"{name}  {frame.shape[1]}x{frame.shape[0]}  age {age:.0f}ms",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(frame)
    if not tiles:
        return _placeholder_preview(cameras, count, measured_hz)
    height = min(tile.shape[0] for tile in tiles)
    scaled = [
        tile if tile.shape[0] == height
        else cv2.resize(tile, (int(tile.shape[1] * height / tile.shape[0]), height))
        for tile in tiles
    ]
    canvas = scaled[0] if len(scaled) == 1 else cv2.hconcat(scaled)
    # 退出提示必须显眼：这个窗口一开，控制台的 Ctrl+C 就不再可靠，
    # 用户唯一的退出手段是在这个窗口里按键（窗口需要有焦点）。
    height = canvas.shape[0]
    cv2.putText(canvas, f"Q / Esc = QUIT      count={count}  loop={measured_hz:.0f}Hz",
                (8, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, f"Q / Esc = QUIT      count={count}  loop={measured_hz:.0f}Hz",
                (8, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2, cv2.LINE_AA)
    # 有帧但某一路掉线时，把掉线的那路状态补在角上，避免"半边画面不动"没人发现
    if missing:
        cv2.putText(canvas, f"MISSING: {', '.join(missing)}", (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


_preview_ready = False


def ensure_preview_window(canvas) -> None:
    """第一次真正要出图时才建窗。

    原实现在 `main()` 里连臂、归位之前就把窗口建好，结果使能 + 归位这十几秒里
    窗口一直没有 imshow，用户去点它（想按 Q）就会看到 Windows 的"未响应"
    白屏。改成延迟到有画面可显示时再建，启动阶段根本不存在那个空窗口。
    """
    global _preview_ready
    if _preview_ready:
        return
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    # 加余量：Windows 上 resizeWindow 用的是**含标题栏和边框**的窗口尺寸，
    # 客户区会矮掉约 30px。按图像尺寸原样设，画面底部那行 "Q / Esc = QUIT"
    # 就会被裁掉一半 —— 观众以为提示不存在。
    cv2.resizeWindow(PREVIEW_WINDOW,
                     int(canvas.shape[1]) + 20, int(canvas.shape[0]) + 60)
    cv2.moveWindow(PREVIEW_WINDOW, 40, 40)
    _preview_ready = True


def run(leader: NeroLeader, follower: NeroFollower, config: NeroTeleopConfig,
        start_pose: dict[str, list[float]] | None = None,
        return_duration_s: float = 5.0,
        cameras: list[tuple[str, LatestCamera]] | None = None,
        allow_large_return: bool = False,
        pose_path: str = "start pose") -> int:
    """Run a session. Both devices are cleaned up even after a partial connection."""
    if leader.action_features != follower.action_features:
        raise ValueError("Leader/follower features differ; use_gripper must match on both sides")
    left, right = leader.bus.config, follower.bus.config
    if (left.interface, str(left.channel)) == (right.interface, str(right.channel)):
        raise ValueError("Leader and follower must use different CAN channels")
    try:
        # ⚠️ 有恢复点时主臂必须**先进普通位置模式**（enter_leader_mode=False）。
        # 原因：主臂一旦进了零力拖拽（linkage 0xFA），控制器就不接受位置指令，
        # slow_move_to 只会把轨迹发完就返回，而 wait_for_angles() 会误报"反馈正常"，
        # 把"根本没动"彻底掩盖。固件 1.20/1.21 上 set_normal_mode() 是官方空操作，
        # 进去就回不来 —— 所以只能一开始就别进去。
        # 归位完成后再调 enter_leader_mode() 进零力拖拽，见下方。
        leader.connect(enter_leader_mode=start_pose is None)
        follower.connect()
        if start_pose is None:
            for device in (leader, follower):
                if device.bus.wait_stable_pose(timeout_s=config.startup_timeout_s) is None:
                    raise TimeoutError(f"No stable startup pose from {device}")
        else:
            if follower.bus.wait_stable_pose(timeout_s=config.startup_timeout_s) is None:
                raise TimeoutError(f"No stable startup pose from {follower}")
            if leader.bus.present_positions() is None:
                raise TimeoutError(f"No stable startup pose from {leader}")

        if start_pose is not None:
            return_to_start_pose(leader, follower, start_pose, return_duration_s,
                                 allow_large_return=allow_large_return, pose_path=pose_path)
            leader.enter_leader_mode()
            leader_start = leader.get_action()
            follower_start = follower.get_observation()
        else:
            deadline = time.monotonic() + config.startup_timeout_s
            while True:
                try:
                    leader_start = leader.get_action()
                    follower_start = follower.get_observation()
                    break
                except (RuntimeError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)

        mapper = TargetMapper(config, leader_start, follower_start)
        started = previous = time.monotonic()
        next_health = started
        next_preview = started
        preview_period = 1.0 / config.preview_fps
        # 每次运行都从"未建窗"开始：上一次运行结束时 destroyAllWindows 过，
        # 全局标记必须跟着复位，否则第二次运行会 imshow 到一个不存在的窗口。
        global _preview_ready
        _preview_ready = False
        next_camera_warn = started
        window_start = started
        count = 0
        window_count = 0
        window_cost = 0.0
        window_worst = 0.0
        measured_hz = 0.0
        while config.duration_s is None or time.monotonic() - started < config.duration_s:
            tick = time.monotonic()
            action = leader.get_action()
            # Always read feedback, even without camera visualization.
            observation = follower.get_observation()
            if tick >= next_health:
                for device in (leader, follower):
                    problems = device.bus.health_problems()
                    if problems:
                        raise RuntimeError(f"{device}: {'; '.join(problems)}")
                span = max(1e-6, tick - window_start)
                measured_hz = window_count / span
                logger.info(
                    "LOOP | 实测 %.1f Hz（目标 %d）| 单步平均 %.2f ms 最差 %.2f ms",
                    measured_hz, config.fps,
                    window_cost / max(1, window_count) * 1000.0, window_worst * 1000.0,
                )
                logger.info("LEADER mapped | %s", format_positions(
                    action, leader.config.gripper_max_width_m,
                    max_angle_deg=leader.config.gripper_max_angle_deg))
                logger.info("FOLLOWER feedback | %s", format_positions(
                    observation, follower.config.gripper_max_width_m,
                    follower.config.gripper_mode, follower.config.gripper_max_angle_deg))
                next_health = tick + 1.0
                window_start = tick
                window_count = 0
                window_cost = 0.0
                window_worst = 0.0
            target = mapper.target(action, tick - previous)
            mapper.accept(follower.send_action(target))
            previous = tick
            count += 1
            window_count += 1
            step_cost = time.monotonic() - tick
            window_cost += step_cost
            window_worst = max(window_worst, step_cost)
            # 预览与控制解耦：相机只有 30fps，imshow/putText 若跟着 120Hz 控制回路跑，
            # 只会白占主线程时间，反过来把控制拖慢。
            if tick >= next_preview:
                if cameras:
                    # compose_preview 现在保证返回一张图（没帧就是诊断占位图），
                    # 所以 imshow 无条件执行 —— 不存在"这一帧跳过、窗口留白"的路径。
                    canvas = compose_preview(cameras, count, measured_hz)
                    ensure_preview_window(canvas)
                    cv2.imshow(PREVIEW_WINDOW, canvas)
                    # 相机掉线时控制台也要说一声（每 5 秒最多一次），
                    # 否则用户只盯着窗口，容易以为是自己没对准。
                    if tick >= next_camera_warn:
                        dead = [name for name, camera in cameras if not camera.is_live()]
                        if dead:
                            logger.warning(
                                "CAMERA | %s 没有新帧：%s（画面上的诊断图会写明原因）",
                                ", ".join(dead),
                                "; ".join(camera.status_text()
                                          for name, camera in cameras if name in dead))
                            next_camera_warn = tick + 5.0
                        else:
                            next_camera_warn = tick + 1.0
                    # waitKey 无条件调用：即使相机掉线、首帧未就绪、USB 掉帧，
                    # 也必须能按 Q 退出。
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        return count
                # 控制台里按 Q 也算（预览窗口没焦点时的兜底，无相机时同样有效）
                if console_quit_requested():
                    return count
                next_preview = tick + preview_period
            time.sleep(max(0.0, 1.0 / config.fps - (time.monotonic() - tick)))
        return count
    finally:
        # Independent cleanup: a failure on one side must not skip the other.
        for device in (follower, leader):
            try:
                device.disconnect()
            except Exception:
                logger.exception("Failed to disconnect %s", device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--leader-channel", required=True)
    parser.add_argument("--follower-channel", required=True)
    parser.add_argument("--leader-firmware", default=None)
    parser.add_argument("--follower-firmware", default=None)
    parser.add_argument("--leader-angle-unit", choices=("auto", "deg", "rad"), default="auto")
    parser.add_argument("--angle-source", choices=("auto", "leader", "follower"), default="auto")
    parser.add_argument("--mapping", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--joint-signs", type=float, nargs=7, default=[1.0] * 7)
    parser.add_argument("--joint-offsets", type=float, nargs=7, default=[0.0] * 7)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-joint-speed", type=float, default=1.2,
                        help="关节参考速度上限 rad/s。0.3 只够慢速摆臂，手快点从臂就追不上")
    parser.add_argument("--max-gripper-speed", type=float, default=2.5)
    parser.add_argument("--max-control-dt", type=float, default=0.1)
    parser.add_argument("--max-start-delta", type=float, default=0.1)
    parser.add_argument("--speed-percent", type=int, default=60,
                        help="从臂固件速度百分比 [0,100]；太低会成为跟随延迟的另一来源")
    parser.add_argument("--max-relative-target", type=float, default=None,
                        help="Optional target-to-feedback joint limit in radians; disabled when omitted")
    parser.add_argument("--enable-timeout", type=float, default=25.0)
    parser.add_argument("--feedback-timeout", type=float, default=0.5)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--gripper-mode", choices=("width", "angle"), default="width")
    parser.add_argument("--leader-gripper-source", choices=("auto", "status", "control"), default="auto")
    parser.add_argument("--leader-gripper-width", type=float, default=0.07)
    parser.add_argument("--follower-gripper-width", type=float, default=0.07)
    parser.add_argument("--gripper-max-angle", type=float, default=100.0)
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--return-to-start", action="store_true",
                        help="使能后把从臂自动移回恢复点，并提示手动拖回主臂"
                             "（固件 1.20/1.21 上主臂无法自动归位）")
    parser.add_argument("--pose-file", default=str(DEFAULT_START_POSE))
    parser.add_argument("--return-duration", type=float, default=6.0,
                        help="归位插值时长（秒）。越短归位越快，但也越猛")
    parser.add_argument("--allow-large-return", action="store_true",
                        help="允许单轴归位超过 60 deg 的安全上限（大范围搬运动作，慎用）")
    parser.add_argument("--camera-index", type=int, nargs="+", default=None,
                        help="相机索引，可给多个；本机实测是 1 3（双摄）。第一个作为主预览")
    parser.add_argument("--camera-labels", nargs="+", default=None,
                        help="与 --camera-index 对应的名字，例如 wrist base")
    parser.add_argument("--camera-backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30,
                        help="D435i 在 640x480 下 UVC 最多 30fps，写 60 也只会拿到 30")
    parser.add_argument("--preview-fps", type=float, default=30.0,
                        help="预览刷新率，与 --fps 控制频率解耦")
    parser.add_argument("--camera-warmup", type=float, default=4.0,
                        help="打开相机后等待首帧的秒数；超时直接报错退出，"
                             "避免出现「窗口白屏但零报错」")
    args = parser.parse_args(argv)
    if args.return_duration <= 0 or not math.isfinite(args.return_duration):
        parser.error("--return-duration must be finite and positive")
    if args.camera_index and args.camera_labels and len(args.camera_labels) != len(args.camera_index):
        parser.error("--camera-labels 数量必须和 --camera-index 一致")
    start_pose = load_start_pose(args.pose_file) if args.return_to_start else None
    config = NeroTeleopConfig(
        fps=args.fps, duration_s=args.duration, mapping=args.mapping,
        max_joint_speed=args.max_joint_speed, max_gripper_speed=args.max_gripper_speed,
        max_control_dt_s=args.max_control_dt,
        max_start_delta=args.max_start_delta, preview_fps=args.preview_fps,
    )
    leader = NeroLeader(NeroLeaderConfig(
        id="nero_leader", interface=args.interface, channel=args.leader_channel,
        firmware=args.leader_firmware, angle_source=args.angle_source,
        leader_angle_unit=args.leader_angle_unit, enable_timeout=args.enable_timeout,
        feedback_timeout_s=args.feedback_timeout, use_gripper=not args.no_gripper,
        gripper_max_width_m=args.leader_gripper_width, gripper_max_angle_deg=args.gripper_max_angle,
        gripper_source=args.leader_gripper_source,
        joint_signs=[*args.joint_signs, -1.0 if args.invert_gripper else 1.0],
        joint_offsets=[*args.joint_offsets, 1.0 if args.invert_gripper else 0.0],
    ))
    follower = NeroFollower(NeroFollowerConfig(
        id="nero_follower", interface=args.interface, channel=args.follower_channel,
        firmware=args.follower_firmware, enable_timeout=args.enable_timeout,
        feedback_timeout_s=args.feedback_timeout, use_gripper=not args.no_gripper,
        speed_percent=args.speed_percent, max_relative_target=args.max_relative_target,
        gripper_mode=args.gripper_mode, gripper_max_width_m=args.follower_gripper_width,
        gripper_max_angle_deg=args.gripper_max_angle, gripper_force_n=args.gripper_force,
        # End effectors must always be explicitly disabled during disconnect,
        # including normal exit, exceptions, and KeyboardInterrupt cleanup.
        disable_gripper_on_disconnect=True,
    ))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.warning("Connecting enables both arms. Exit disables arms; support the arms and any held object.")
    cameras: list[tuple[str, LatestCamera]] = []
    try:
        if args.camera_index:
            labels = args.camera_labels or []
            for position, index in enumerate(args.camera_index):
                name = labels[position] if position < len(labels) else f"cam{index}"
                camera = LatestCamera(index, args.camera_backend, args.camera_width,
                                      args.camera_height, args.camera_fps)
                cameras.append((name, camera))
                print(f"CAMERA: {name} = index {index} "
                      f"({args.camera_width}x{args.camera_height}@{args.camera_fps})", flush=True)
                # 打开成功 != 会出帧。DSHOW 下 read() 会静默失败，若不在这里拦住，
                # 后面就是"窗口白屏且零报错"。宁可启动时明确失败，也不要白屏。
                if not camera.wait_first_frame(args.camera_warmup):
                    print(f"[ERROR] {name} (index {index}) 已打开但 {args.camera_warmup:g}s 内"
                          f"没有任何帧。", flush=True)
                    print(f"        状态: {camera.status_text()}", flush=True)
                    print("        常见原因: 相机被别的程序占用 / USB 线松动或带宽不足 / "
                          "index 变了（先跑 tools/diag_dual_camera.py 看真实 index）",
                          flush=True)
                    raise RuntimeError(
                        f"Camera '{name}' (index {index}) opened but produced no frames")
                print(f"        首帧 OK  age={camera.age_ms():.0f}ms  "
                      f"请求={camera.size[0]}x{camera.size[1]}", flush=True)
            print(f"PREVIEW: {len(cameras)} 路并排显示，{args.preview_fps:g}fps 刷新；"
                  "按 Q 或 Esc 退出。", flush=True)
            # 窗口不在这里建。原实现在连臂/归位之前就建好窗口，那十几秒里窗口
            # 一帧图都没有，用户去点它（想按 Q）就会撞上 Windows 的"未响应"白屏。
            # 现在改为首次出图时由 ensure_preview_window() 创建。
        run(leader, follower, config, start_pose=start_pose,
            return_duration_s=args.return_duration, cameras=cameras,
            allow_large_return=args.allow_large_return, pose_path=args.pose_file)
    except KeyboardInterrupt:
        logger.info("Teleoperation interrupted")
    finally:
        for _name, camera in cameras:
            camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
