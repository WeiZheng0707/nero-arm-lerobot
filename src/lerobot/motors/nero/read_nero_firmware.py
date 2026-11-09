r"""松灵 NERO 固件版本读取（修正版）

原脚本的三个错误：
  1) interface 用了 "gs_usb"        -> 应该用 "agx_cando"（官方 Windows 唯一支持）
  2) baudrate=1000000               -> 参数名是 bitrate=（baudrate 会被静默丢弃）
  3) get_controller_firmware_version() -> 不存在，正确是 get_firmware()["software_version"]

连接后会按这个顺序做事：
  0) 打开 CAN 推送（set_normal_mode）—— 固件 ≤1.11 默认关闭，不开后面什么都读不到
  1) 检查使能状态，未使能则询问是否使能
  2) 读固件版本
  3) 读关节角（用来交叉判断臂是否真的活着）

为什么会需要第 0 步（2026-09-15 实测）：
  固件 1.10 的臂连上后 CAN 推送是关的，症状是"使能状态恒为 0000000、
  固件恒为 None"，看起来像硬件坏了，其实只是推送没开。

用法：
    python read_nero_firmware.py            # 自动选后端/通道，未使能时询问
    python read_nero_firmware.py --channel 0
    python read_nero_firmware.py --enable   # 未使能时直接使能，不询问
    python read_nero_firmware.py --no-enable  # 完全不碰使能状态
    python read_nero_firmware.py --enable --disable   # 使能 -> 读完 -> 自动失能（无人值守）
    python read_nero_firmware.py --no-disable         # 全程不碰失能

失能控制（2026-09-16 补，为什么之前不够）：
    原来的失能询问只在「使能**确认成功**」时才触发。真机上撞到过这种组合 ——
    enable 指令确实发出去了、臂也确实通电了，但低速帧回读 15 秒内一直是 0000000
    （之后才刷成 1111111）。于是 finally 里那个询问被跳过，臂被留在通电状态，
    而你以为程序会还回去。
    现在改成：**只要本脚本发过使能指令**，退出前一定走失能流程
    （`--disable` 直接失能、`--no-disable` 完全不碰，默认询问）。

注意：使能状态不会随进程退出保持（断电/重连后读到的是未使能），
      但**只要不断电、不失能，电机就一直通电顶着当前位姿** —— 抬起姿态下会一直
      靠电机顶着，既费电也不安全。

----------------------------------------------------------------------
2026-09-19：进程被系统直接杀掉（退出码 -1073740940 = 0xC0000374）
----------------------------------------------------------------------
现象：跑到"正在通过 CAN 连接机械臂 ..."之后进程凭空消失，
      退出码 0xC0000374 = STATUS_HEAP_CORRUPTION（堆损坏）——
      这是**原生层**崩溃，Python 的 try/except 抓不到。

定位（分层探针，见 probe_cando_crash.py）：

    cando_list_malloc            -> True
    cando_list_scan              -> True
    cando_list_num               -> True   count=1
    cando_malloc                 -> True
    cando_open                   -> False        ← 设备打不开
    cando_get_serial_number_str  -> '\\?\usb#vid_1d50&pid_606f&mi_00#...'  ← 但能看到设备
    cando_close                  -> ★ 崩（堆损坏，进程立刻消失）

→ 根因是 **SDK 的清理路径 bug**：`cando_open()` 失败后，它仍然对那个
  **没打开的句柄**调用 `cando_close()`，触发堆损坏。
  对照：`scan()`（只做 list_* 不做 open/close）连跑 3 次都正常。

已排除「设备被占用」：对同一设备路径直接 `CreateFileW`（**独占**模式）
  能成功打开（err=0），而 `cando_open` 仍返回 False → 问题在 cando.dll 内部。

对策：`preflight_can_device()` 在调用 pyAgxArm 之前先自己探一次 open。
  打不开就直接报人话原因并退出（退出码 3），**绝不走到 SDK 的崩溃路径**。
  这样至少能看到"设备打不开"，而不是"脚本莫名消失"。
"""

import argparse
import os
import sys
import time
import traceback
from platform import system

# ------------------------------------------------------------------
# Windows 上 pyAgxArm 依赖 agx_cando 插件（已装 python-can-agx-cando）
# 这里只做后端选择，不做 libusb 注入 —— agx_cando 走的是松灵自己的 cando.dll
# ------------------------------------------------------------------


def resolve_backend() -> tuple[str, str]:
    s = system()
    if s == "Windows":
        return "agx_cando", "0"
    if s == "Linux":
        return "socketcan", "can0"
    if s == "Darwin":
        return "slcan", "/dev/ttyACM0"
    raise RuntimeError(f"不支持的系统: {s}")


# ------------------------------------------------------------------
# 设备预检（2026-09-19 排障后新增）
#
# ⚠️ 为什么必须有这一步：
#   `cando_open()` **失败**时，SDK 的清理路径会在一个**没打开的句柄**上调用
#   `cando_close()`，触发 STATUS_HEAP_CORRUPTION (0xC0000374) —— 这是原生层
#   崩溃，进程被系统**直接杀掉**，Python 的 try/except 完全抓不到。
#   外观就是"脚本跑到一半突然消失，退出码 -1073740940"。
#
#   2026-09-19 实测到的完整崩溃链（同一台机器、可重复）：
#       cando_list_malloc -> True
#       cando_list_scan   -> True
#       cando_list_num    -> True   count=1
#       cando_malloc      -> True
#       cando_open        -> False        ← 设备打不开
#       cando_get_serial_number_str -> '\\?\usb#vid_1d50&pid_606f&mi_00#...'  ← 但能看到设备
#       cando_close       -> ★ 崩在这里，进程立刻消失
#
#   对照组：`_CandoApi.scan()`（只做 list_malloc/scan/num/free，**不碰
#   open/close**）连跑 3 次都正常 —— 印证「崩在 open 失败后的 close」。
#
#   顺带确认过不是"设备被占用"：对同一个设备路径直接 CreateFileW（独占模式）
#   能成功打开（err=0），而 cando_open 仍返回 False —— 问题在 cando.dll 内部。
#
#   所以这里在调用 pyAgxArm 之前**先自己探一次**：
#     - 打得开 -> 关掉、放行（此时 close 走正常路径，安全）
#     - 打不开 -> 报人话原因并退出，**绝不走到 SDK 的崩溃路径**
# ------------------------------------------------------------------

def preflight_can_device(interface: str, channel: str) -> bool:
    """确认设备真的打得开。返回 True 表示可以继续。

    ⚠️ open 失败时**故意不调用任何 close/free** —— 那正是会崩的一步。
       泄漏几个句柄无所谓：本函数返回 False 后 main() 会直接退出。

    拿不到 agx_cando 的内部 API 时**跳过预检**（返回 True）—— 预检是保险，
    不该因为它自己出问题而挡住正常流程。
    """
    if interface != "agx_cando":
        return True  # Linux/macOS 后端没有这个问题

    try:
        import ctypes
        from agx_cando.bus import _CandoApi, _U8, _VOID_P
    except Exception:
        print("  [跳过预检] 拿不到 agx_cando 内部 API（版本差异），交给 SDK 自己处理。")
        return True

    print("\n设备预检（只读，不使能、不发运动指令）...")

    try:
        api = _CandoApi()
        count = api.scan()
    except Exception as exc:
        print(f"  [WARN] 枚举设备失败：{type(exc).__name__}: {exc}")
        return True  # 枚举都失败，就让主流程去报更完整的错

    print(f"  检测到 {count} 个 CAN 设备")
    if count == 0:
        print("  [FAIL] 一个设备都没检测到。")
        print("         检查：① USB 是否插好 ② 设备管理器中该设备是否正常（无黄色感叹号）")
        return False

    if int(channel) >= int(count):
        print(f"  [FAIL] channel={channel} 超出范围（只检测到 {count} 个设备）。")
        print("         多台臂时每台必须用不同的 channel；单台用 0。")
        return False

    dll = api.dll
    list_handle = _VOID_P()
    dev_handle = _VOID_P()

    if not dll.cando_list_malloc(ctypes.byref(list_handle)):
        print("  [WARN] cando_list_malloc 失败，跳过预检。")
        return True
    if not dll.cando_list_scan(list_handle):
        print("  [WARN] cando_list_scan 失败，跳过预检。")
        return True
    if not dll.cando_malloc(list_handle, _U8(int(channel)), ctypes.byref(dev_handle)):
        print("  [WARN] cando_malloc 失败，跳过预检。")
        return True

    # 多试几次：偶发的第一次打开失败是见过的
    opened = False
    for attempt in range(5):
        if dll.cando_open(dev_handle):
            opened = True
            break
        time.sleep(0.2)

    if opened:
        # open 成功了，close 走的是正常路径 —— 安全
        dll.cando_close(dev_handle)
        dll.cando_free(dev_handle)
        dll.cando_list_free(list_handle)
        print("  设备可以打开。")
        return True

    # ---------------- 打不开：只报告，绝不 close ----------------
    print()
    print("=" * 62)
    print("  [FAIL] 设备枚举到了，但 cando_open 打不开（已重试 5 次）。")
    print("=" * 62)
    print("  ⚠️ 现在**必须停在这里**：如果继续让 SDK 去打开，它会在")
    print("     失败后的清理路径上调用 cando_close，触发堆损坏")
    print("     (0xC0000374)，把整个进程杀掉 —— 就是你看到的现象。")
    print()
    print("  已经确认过的（可排除）：")
    print("    - 设备没被别的程序独占：同一个设备路径用 CreateFileW")
    print("      独占模式能成功打开（Windows 层是通的）")
    print("    - 设备能被枚举、序列号能读出来")
    print("    → 问题在 cando.dll 内部，不在你的代码/接线")
    print()
    print("  按顺序试这几步：")
    print("    1) 【最有效】**拔掉 CAN 适配器，等 5 秒，重新插上**")
    print("       适配器固件在异常断开后可能卡住，重插会复位它")
    print("    2) 关掉可能碰这个适配器的程序（网页上位机、其他 CAN 工具），")
    print("       或者干脆重启电脑，清掉残留的驱动状态")
    print("    3) 换一个 USB 口（直连主板，别用 hub / 延长线）")
    print("    4) 看看适配器固件要不要升级（设备有独立的")
    print("       'candleLight firmware upgrade interface' 接口）")
    print("    5) 都不行 → cando.dll 与适配器固件可能不匹配，")
    print("       需要换一版 python-can-agx-cando")
    print()
    print("  注：以上都不影响机械臂本体 —— 用网页上位机仍可正常控制。")
    return False


# ------------------------------------------------------------------
# 使能状态检查与交互
#
# 为什么需要：未使能时关节角不推送，读出来是 None，看起来像"连接失败"，
# 其实臂是好的。实测（2026-09-13）使能报文只含"关节号 + 使能标志"，
# 不含任何目标位置 -> 使能本身不会让臂运动，实测保持 10s 漂移 0.008 度。
# ------------------------------------------------------------------

def ensure_can_push(arm) -> bool:
    """打开 CAN 推送（固件 ≤1.11 默认是关的）。

    【为什么必须有这一步 —— 2026-09-15 实测踩到】
    固件 1.10 的臂连上后 CAN 推送是**关闭**的，此时：
      - get_joints_enable_status_list() 读不到低速帧 -> 只能返回初始的 0000000
      - get_firmware() 收不到 0x4AF 应答 -> 返回 None
    表现就是"程序说未使能、使能了也还是 0000000、固件也读不到"，
    看起来像硬件坏了，其实只是推送没开。

    set_normal_mode() 是 SDK 里**唯一**把 enable_can_push 置 ENABLE 的函数
    （set_leader_mode / set_follower_mode 反而是 DISABLE）。
    它不使能电机、不会让机械臂运动，只改推送/联动配置。

    返回是否调用成功。注意：该方法没有返回值，这里只表示"没抛异常"。
    """
    try:
        arm.set_normal_mode()
        return True
    except Exception as exc:
        print(f"  [WARN] set_normal_mode() 失败：{type(exc).__name__}: {exc}")
        return False


def get_enable_flags(arm) -> list[bool] | None:
    """读取各关节使能状态。读不到返回 None。

    注意：底层是 1 Hz 低速反馈帧，连接后的第一帧不可信，
    调用方应间隔后再读一次（见 read_enable_flags_stable）。
    """
    try:
        flags = arm.get_joints_enable_status_list()
    except Exception:
        return None
    if flags is None:
        return None
    try:
        return [bool(x) for x in flags]
    except TypeError:
        return None


def read_enable_flags_stable(arm, settle: float = 1.0, stable_times: int = 2) -> list[bool] | None:
    """读到一个"稳定"的使能状态再返回。

    低速帧只有 1 Hz 量级，刚连上会读到旧值（实测 0000000 -> 1111111 跳变）。
    这里不做固定 sleep，而是**连续读到相同值 stable_times 次**才认为稳定，
    这样既不误判、又不会白白等太久。

    若一直读不到（返回 None）也按失败返回，由调用方决定怎么处理。
    """
    deadline = time.time() + max(settle * 4, 4.0)
    previous: list[bool] | None = None
    same_count = 0

    while time.time() < deadline:
        flags = get_enable_flags(arm)
        if flags is not None:
            if previous is not None and flags == previous:
                same_count += 1
                if same_count >= stable_times - 1:
                    return flags
            else:
                same_count = 0
            previous = list(flags)
        time.sleep(0.25)

    return previous


def format_flags(flags: list[bool]) -> str:
    """把 [True, False, ...] 显示成 1010111 这种，一眼能看懂。"""
    return "".join("1" if f else "0" for f in flags)


def ask_enable(flags: list[bool]) -> bool:
    """询问用户是否要使能。默认否。"""
    n = len(flags)
    print("\n" + "!" * 62)
    print(f"  机械臂当前【未使能】：{format_flags(flags)}")
    print("!" * 62)
    print("  未使能时关节角不推送（读出来是 None），所以要读关节数据需要先使能。")
    print("  使能只是给电机通电锁住当前位姿，**不会让机械臂运动**。")
    print("  注意：臂若处于抬起姿态，之后失能时会掉落。")
    try:
        answer = input("\n  现在使能吗？[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def ask_disable(flags: list[bool] | None, note: str | None = None) -> bool:
    """询问用户是否要失能。

    参数：
        flags: 当前使能状态；None 表示读不到（低速帧可能还没刷新）。
        note:  补充说明"为什么还要问你"（比如回读显示未使能但其实发过使能指令）。

    ⚠️ 为什么允许 flags=None 还要问：低速帧滞后可达十几秒，回读"未使能"
    **不代表臂真的没通电**。只按回读值决定要不要失能，会把臂留在通电状态。
    """
    if flags is None:
        print("\n  [注意] 读不到使能状态（低速帧可能尚未刷新）。")
    else:
        print(f"\n  当前使能状态：{format_flags(flags)}")
    if note:
        print(f"  {note}")
    try:
        answer = input("  要失能吗？（臂若抬起会掉落）[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def do_enable(arm, timeout: float = 25.0, rewake_interval: float = 2.0) -> list[bool] | None:
    """使能全部关节，返回使能后的实际状态读值（读不到返回 None）。

    ⚠️ 2026-09-15 实测修正：原来只 sleep 固定 2.5 秒就读一次，**不够**。
       使能后立即读 -> 0000000；中间隔十几秒 -> 1111111。
       所以改成轮询：每 0.25 秒读一次，读到全部使能立刻返回。

    ⚠️ 2026-09-16 再补一条（真机日志）：
       仅仅"发一次 enable + 事后补一次 set_normal_mode"仍然不够。实测时序是 ——
           发出使能        -> 回读 0000000
           持续轮询 15 秒  -> 仍是 0000000
           **重开了推送之后** -> 1111111，同时关节角能以 140 Hz 读出
       原因：模式指令（set_normal_mode）只有在【已使能】状态下才生效，而 enable()
       是只发不等 —— 必须在轮询过程中**反复补发**，才能把回读通道刷出来。
       这就是为什么这里加了 rewake_interval，默认 25 秒超时。
    """
    print("\n  正在使能 ...")
    try:
        arm.enable()
    except Exception as exc:
        print(f"  [FAIL] enable() 调用出错：{type(exc).__name__}: {exc}")
        return None

    # 使能刚发出、臂这时才真正通电 —— 立刻补开一次推送
    # （失能状态下发它是空操作，不使能电机、不会让臂运动）
    ensure_can_push(arm)

    # 轮询回读，直到确认全部使能；期间周期性重开推送
    deadline = time.time() + timeout
    next_rewake = time.time() + rewake_interval
    rewakes = 0
    flags = None
    last_reported = None
    while time.time() < deadline:
        flags = get_enable_flags(arm)
        if flags is not None:
            if flags != last_reported:
                print(f"    ... 状态 {format_flags(flags)} "
                      f"({sum(flags)}/{len(flags)})")
                last_reported = list(flags)
            if all(flags):
                return flags

        now = time.time()
        if now >= next_rewake:
            ensure_can_push(arm)
            rewakes += 1
            next_rewake = now + rewake_interval

        time.sleep(0.25)

    print(f"  [WARN] {timeout:.0f}s 内未读到全部使能（期间重开推送 {rewakes} 次）。")
    return flags


def do_disable(arm, timeout: float = 10.0, rewake_interval: float = 2.0) -> list[bool] | None:
    """失能全部关节，轮询确认。读不到状态返回 None。

    与 do_enable 同理：回读的是 1 Hz 低速帧，发完指令立刻查会读到旧值
    （实测出现过"6 个已关、第 7 个还报 True"的假象），所以必须轮询。
    期间也周期性重开推送，避免回读通道自己关掉导致「失能未确认」误报。
    """
    print("\n  正在失能 ...")
    try:
        arm.disable()
    except Exception as exc:
        print(f"  [FAIL] disable() 调用出错：{type(exc).__name__}: {exc}")
        return None

    ensure_can_push(arm)

    deadline = time.time() + timeout
    next_rewake = time.time() + rewake_interval
    flags = None
    while time.time() < deadline:
        flags = get_enable_flags(arm)
        if flags is not None and not any(flags):
            return flags
        now = time.time()
        if now >= next_rewake:
            ensure_can_push(arm)
            next_rewake = now + rewake_interval
        time.sleep(0.25)
    return flags



def sample_joint_feedback(arm, duration_s: float = 5.0) -> bool:
    """Observe timestamp progress, rather than treating a cached pose as live feedback."""
    deadline = time.monotonic() + duration_s
    last_stamp = None
    updates = 0
    positive_hz = False
    while time.monotonic() < deadline:
        message = arm.get_joint_angles()
        if message is None:
            print("  joints=None")
        else:
            stamp = getattr(message, "timestamp", None)
            hz = float(getattr(message, "hz", 0.0) or 0.0)
            if stamp is not None and stamp > 0:
                if last_stamp is not None and stamp > last_stamp:
                    updates += 1
                last_stamp = stamp
            positive_hz = positive_hz or hz > 0
            print(f"  timestamp={stamp}, hz={hz:.1f}, joints(rad)={[round(x, 4) for x in message.msg]}")
        time.sleep(0.5)
    live = updates >= 2 and positive_hz
    print(f"  timestamp advances={updates}; positive Hz observed={positive_hz}")
    print("  [PASS] Observed updating aggregate joint feedback." if live else
          "  [WARN] Live joint feedback not confirmed; cached values do not prove streaming or enable.")
    return live


def main() -> int:
    default_iface, default_channel = resolve_backend()

    ap = argparse.ArgumentParser(description="读取松灵 NERO 主控固件版本")
    ap.add_argument("--interface", default=default_iface,
                    help=f"CAN 后端，Windows 用 agx_cando（默认 {default_iface}）")
    ap.add_argument("--channel", default=default_channel,
                    help=f"CAN 通道/设备索引（默认 {default_channel}）")
    ap.add_argument("--bitrate", type=int, default=1_000_000,
                    help="CAN 波特率，松灵默认 1Mbps")
    ap.add_argument("--firmware", choices=("default", "v111", "v112", "v120", "v121"),
                    default="default", help="SDK query profile; does not flash or upgrade firmware")
    ap.add_argument("--no-mode-change", action="store_true",
                    help="Do not send mode/CAN-push commands; combine with --no-enable --no-disable")
    ap.add_argument("--enable", action="store_true",
                    help="直接使能，不询问（默认是询问）")
    ap.add_argument("--no-enable", action="store_true",
                    help="不主动使能，只读固件版本。注意：SDK 的 connect() 本身会唤醒"
                         "电机，因此退出时会主动失能并回读确认（除非同时给 --no-disable）")
    ap.add_argument("--disable", action="store_true",
                    help="退出前自动失能、不询问（配 --enable 可无人值守）")
    ap.add_argument("--no-disable", action="store_true",
                    help="完全不碰失能状态（默认行为是：本脚本发过使能或用了 --no-enable 就失能）")
    ap.add_argument("--enable-timeout", type=float, default=25.0,
                    help="使能/失能后等待状态确认的最长秒数（默认 25；实测回读可滞后 15s+）")
    args = ap.parse_args()

    if args.enable and args.no_enable:
        ap.error("--enable and --no-enable cannot be combined")
    if args.no_mode_change and not (args.no_enable and args.no_disable):
        ap.error("--no-mode-change requires --no-enable --no-disable")

    if args.disable and args.no_disable:
        print("[FATAL] --disable 与 --no-disable 不能同时用。")
        return 2

    print(f"interface = {args.interface}")
    print(f"channel   = {args.channel}")
    print(f"bitrate   = {args.bitrate}")

    try:
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
    except ImportError:
        print("[FATAL] 未安装 pyAgxArm。")
        print("        pip install \"git+https://github.com/agilexrobotics/pyAgxArm.git\"")
        return 2

    # ★ 设备预检：必须在 create_arm / connect **之前**。
    #   设备打不开时，SDK 会在失败清理路径上崩溃（0xC0000374 把进程杀掉），
    #   详见 preflight_can_device 的注释。
    if not preflight_can_device(args.interface, args.channel):
        return 3

    # 官方语义：comm 是"通讯类型"(can)，通道通过 channel 传
    cfg = create_agx_arm_config(
        robot="nero",
        firmeware_version=args.firmware,
        comm="can",
        interface=args.interface,
        channel=str(args.channel),
        bitrate=args.bitrate,
    )

    arm = AgxArmFactory.create_arm(cfg)

    print("正在通过 CAN 连接机械臂 ...")
    try:
        arm.connect()
        print("连接成功")
    except Exception as e:
        print(f"\n[FAIL] 连接失败: {type(e).__name__}: {e}")
        print("\n排查清单：")
        print("  1) 是否已装 python-can-agx-cando：")
        print("     pip install \"git+https://github.com/agilexrobotics/python-can-agx-cando.git\"")
        print("  2) interface 必须是 agx_cando（不是 gs_usb / socketcan）")
        print("  3) 确认没有别的程序（另一个 python、上位机）正占用该 CAN 模块")
        print("  4) 该 USB 口的驱动是否绑定成 WinUSB —— 官方 cando.dll 需要特定接口 GUID")
        print("     agx_cando 与 gs_usb 不能同时打开同一个设备，会互相抢")
        return 1

    # ★ 2026-10-27 修正：SDK 的 connect() 本身就会让机械臂通电。
    #
    #   实测：本脚本带 --no-enable --no-disable --no-mode-change 运行时，
    #   connect() 返回成功、脚本停在"等待固件回推"，而主臂已经绷紧（使能了）。
    #   原因：pyAgxArm 的 connect() 内部走 enable()，而 enable() 是"只发不等"
    #   的 fire-and-forget 指令 —— 指令一发出电机就通电，与脚本是否再读回执无关。
    #
    #   所以 --no-enable 只能"不主动使能"，**挡不住 connect 自身带来的通电**。
    #   真正的只读诊断必须承认这一点，并在退出时显式失能，否则臂会被留在
    #   通电顶着位姿的状态（AGENTS.md 明确禁止把这种情况报成成功）。
    if args.no_enable:
        print()
        print("  [WARN] 注意：SDK 的 connect() 本身会唤醒电机，--no-enable 挡不住它。")
        print("         本次运行可能已让机械臂短暂通电。若臂现在绷紧，属正常现象；")
        print("         退出时本脚本会主动失能并回读确认（除非另有 --no-disable）。")
        print("         请在整个过程中保持机械臂有支撑、夹爪空载、急停可触达。")

    # 是否**发过**使能指令（注意：不等于「已确认使能成功」）
    #
    # ⚠️ 为什么不能只记"确认成功"：真机实测过这种组合 ——
    #    enable 指令发出去了、臂确实通电了（后来能读到 1111111 + 140 Hz 关节角），
    #    但低速帧回读 15 秒内一直是 0000000。
    #    如果按"未确认"跳过退出时的失能询问，臂就被留在通电状态了。
    we_sent_enable = False

    try:
        # ---------------- 0) 先开 CAN 推送 ----------------
        # 【关键】固件 ≤1.11 连上后推送默认关闭，不开的话后面什么都读不到：
        # 使能状态恒为 0000000、固件恒为 None，看起来像硬件坏了。
        print(f"\nRequested SDK profile = {args.firmware}")
        print("No mode changes requested." if args.no_mode_change else "打开 CAN 推送（set_normal_mode）...")
        push_ok = False if args.no_mode_change else ensure_can_push(arm)
        if push_ok:
            print("  已发送。等待低速帧开始刷新（2 秒）...")
            time.sleep(2.0)
        elif not args.no_mode_change:
            print("  开启失败，继续尝试（可能档位 >=1.12，本来就不需要）。")

        # ---------------- 1) 先看使能状态 ----------------
        if args.no_enable:
            print("\n[--no-enable] 跳过使能检查。")
        else:
            print("\n检查使能状态（等待低速帧刷新 ...）")
            flags = read_enable_flags_stable(arm)

            if flags is None:
                print("  [WARN] 读不到使能状态。")
                print("         可能原因：档位不对 / 推送未开 / CAN 线接触不良。")
                print("         继续往下走，但关节角可能读不到。")
            else:
                print(f"  当前使能状态 = {format_flags(flags)}  "
                      f"({sum(flags)}/{len(flags)} 个关节已使能)")

                all_enabled = all(flags)
                if all_enabled:
                    print("  -> 已使能，可以直接读关节数据。")
                else:
                    if args.enable:
                        should_enable = True
                        print("  [--enable] 自动使能。")
                    else:
                        should_enable = ask_enable(flags)

                    if should_enable:
                        # 先记「发过指令」，再看结果 —— 顺序不能反
                        we_sent_enable = True
                        after = do_enable(arm, timeout=args.enable_timeout)
                        if after is None:
                            print("  使能后读不到状态，无法确认结果。")
                        elif all(after):
                            print(f"  使能后状态 = {format_flags(after)}")
                            print("  -> 使能成功。")
                        else:
                            print(f"  使能后状态 = {format_flags(after)}")
                            print(f"  -> 轮询 {args.enable_timeout:.0f}s 未确认全部使能。")
                            print("     注意：这只说明**回读**没刷出来，不等于臂没通电。")
                            print("           （本轮已每 0.25s 回读一次，并周期性重开 CAN 推送）")
                            print("     下一步的关节角能读出来就说明臂是活的 —— 以那个为准。")
                            print("     仍不行再按这个顺序查：")
                            print("     1) 臂本体电源开关是否打开、绿灯是否亮")
                            print("     2) 外部稳压电源 >= DC24V 10A")
                            print("     3) CAN 推送又自行关闭了（实测维持约 1 分钟）")
                            print("        -> 脚本已自动补发 set_normal_mode，可重连再试")
                            print("     4) 加大 --enable-timeout 再跑一次（实测回读可滞后 15s+）")
                    else:
                        print("  跳过使能。关节角可能读不到。")

        # ---------------- 2) 读固件版本 ----------------
        print("\n等待固件回推 ...")
        fw = None
        deadline = time.time() + 15.0
        while time.time() < deadline:
            fw = arm.get_firmware(timeout=0.5)   # 官方接口：返回 dict | None
            if fw is not None:
                break
            time.sleep(0.2)

        if fw is None:
            # 重试一轮：重新开推送 + 给更长时间。
            # 实测 set_normal_mode() 之后固件应答会短暂被挤掉，
            # 但等几秒、重新问一次通常能拿到。
            print("  第一轮没读到，重试固件查询 ...")
            if not args.no_mode_change:
                ensure_can_push(arm)
            time.sleep(3.0)
            deadline = time.time() + 15.0
            while time.time() < deadline:
                fw = arm.get_firmware(timeout=0.5)
                if fw is not None:
                    break
                time.sleep(0.2)

        if fw is None:
            print("[FAIL] 30s 内未读到固件信息。")
            print("\n排查顺序（按命中率从高到低）：")
            print("  1) CAN 推送未开启，或当前查询档位与设备不匹配")
            print("  2) 主控未上电 / 电源不足（需要外部稳压 DC24V >= 10A）")
            print(f"  3) 当前查询档位为 {args.firmware}；设备实际固件尚未确认")
            print("  4) CAN 线接触不良 / CAN_H、CAN_L 接反")
            print("  5) 被 Web 上位机占用（通常表现为打不开通道，而不是打开了读不到）")
            print("\n注意：读不到固件 ≠ 连接失败。继续往下试读关节角，")
            print("      还需检查时间戳是否持续更新，缓存角度不代表实时反馈。")

            # 不直接 return，继续尝试读关节角做交叉判断
        else:
            sv = fw.get("software_version")
            print(f"\n主控固件版本 software_version = {sv}")
            print(f"完整固件信息 = {fw}")

            # 映射到 SDK 驱动档位
            try:
                from pyAgxArm import resolve_firmware_profile
                profile = resolve_firmware_profile("nero", sv)
                print(f"\n对应 SDK 驱动档位 firmeware_version = {profile}")
                mapping = {
                    "default": "NeroFW.DEFAULT（固件 ≤1.10，需 set_normal_mode 唤醒 CAN 推送）",
                    "v111": "NeroFW.V111（固件 1.11，同 DEFAULT）",
                    "v112": "NeroFW.V112（固件 1.12，set_normal_mode 已 no-op，上电即推送）",
                    "v120": "NeroFW.V120（固件 1.20）",
                    "v121": "NeroFW.V121（固件 ≥1.21）",
                }
                print(f"说明: {mapping.get(str(profile), '未知档位')}")
            except Exception:
                print("[WARN] 档位解析失败：")
                traceback.print_exc()

        # ---------------- 3) 读关节数据（交叉判断臂是否活着）----------------
        print(f"\njoint_nums = {arm.joint_nums}")

        ja = arm.get_joint_angles()
        if ja is None:
            # 等一会儿 + 重开推送再试一次
            print("关节角暂未回推，重试一次 ...")
            if not args.no_mode_change:
                ensure_can_push(arm)
            time.sleep(2.0)
            ja = arm.get_joint_angles()

        if ja is not None:
            print(f"关节角(rad) = {[round(x, 4) for x in ja.msg]}")
            print(f"  数据频率 {ja.hz:.1f} Hz")
            print("  -> 已获取关节消息，下面检查是否持续更新。")
        else:
            print("关节角读不到。")
            print("  说明推送仍未打开，或臂没有上电。")
            print("  检查顺序：")
            print("    1) 臂本体电源开关是否打开、绿灯是否亮")
            print("    2) 外部稳压电源是否 >= DC24V 10A")
            print("    3) CAN 线两端是否插紧、CAN_H(黄)/CAN_L(蓝) 是否接反")
            print("    4) 重新跑一次本脚本（set_normal_mode 有时要第二次才生效）")

        # 把使能状态再报一次，和关节角对照
        final_flags = get_enable_flags(arm)
        if final_flags is not None:
            print(f"\n  当前使能状态 = {format_flags(final_flags)}")
            if ja is not None and not any(final_flags):
                print("  [注意] 使能状态未确认。可能确实未使能，也可能反馈陈旧；")
                print("         不能根据能读到关节角判断电机已经使能。")

        print("\n连续采样关节反馈（5 秒，不发送运动指令）...")
        live = sample_joint_feedback(arm)
        return 0 if fw is not None and live else 1

    finally:
        # ---------------- 失能控制 ----------------
        #
        # ⚠️ 2026-09-16 修正：原来这里只看 we_enabled（= 使能**确认成功**）。
        #    真机上出现过"指令发出去了、臂也通电了，但回读一直 0000000"，
        #    于是整段被跳过，臂被留在通电状态。现在只看「有没有发过使能指令」。
        #
        #    而且**不能拿回读值当作"臂没通电"的证据** —— 低速帧滞后可达十几秒，
        #    读到的 0000000 可能只是还没刷新。所以三种情况（已使能 / 未使能 /
        #    读不到）都会走失能流程，只是提示语不同。
        if args.no_disable:
            print("\n[--no-disable] 不碰失能状态，保持原样退出。")
            if args.no_enable:
                print("  [WARN] 但请注意：connect() 已唤醒过电机，臂可能仍在通电顶着。")
                print("         建议另跑 disable_pair.py 显式失能并确认，或断电。")
        elif we_sent_enable or args.disable or args.no_enable:
            # args.no_enable 也走这里：connect() 自身会通电，只读诊断同样必须收尾。
            try:
                flags_now = get_enable_flags(arm)

                if flags_now is not None and any(flags_now):
                    note = None
                elif flags_now is None:
                    note = ("[注意] 读不到状态，但 connect() 会唤醒电机 —— "
                            "建议失能，避免电机一直在通电顶着。")
                else:
                    note = ("[注意] 回读显示未使能，但低速帧滞后可达十几秒；"
                            "connect() 会唤醒电机，臂可能其实通着电。")

                if args.disable or args.no_enable:
                    if args.no_enable and not args.disable:
                        print("\n  只读诊断收尾：connect() 会通电，主动失能并确认。")
                    else:
                        print("\n  [--disable] 自动失能。")
                    should_disable = True
                else:
                    should_disable = ask_disable(flags_now, note)

                if should_disable:
                    after = do_disable(arm, timeout=args.enable_timeout)
                    if after is None:
                        print("  [注意] 失能后读不到状态，无法确认。请目视确认臂已松。")
                    elif not any(after):
                        print(f"  失能后状态 = {format_flags(after)}")
                        print("  -> 已确认失能，电机不再通电。")
                    else:
                        print(f"  失能后状态 = {format_flags(after)}")
                        print("  -> 未确认全部失能（回读可能滞后）。")
                        print("     若臂仍绷着劲，可重跑本脚本用 --disable 再失能一次。")
                else:
                    print("  保持使能状态。注意：进程退出**不会**自动失能，")
                    print("  只要不断电，电机就一直通电顶着当前位姿。")
            except Exception:
                traceback.print_exc()
        else:
            print("\n  本次没有发过使能指令，不碰臂的使能状态。")

        try:
            arm.disconnect()
            print("\n已断开连接。")
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
