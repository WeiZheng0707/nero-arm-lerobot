"""诊断 OpenCV 预览窗口到底是「白屏」还是「没拿到帧」。

背景：`diag_dual_camera.py` 只测了读帧速率，从未验证 imshow 显示层。
所以「相机卡白屏」存在两种完全不同的成因，必须区分开：

  A. 窗口内容真的是白的 / 未响应  -> HighGUI 或窗口消息泵的问题
  B. 窗口正常但一张图都没送进去    -> 应用层 compose_preview 返回 None

本脚本用 Windows API 直接给窗口拍快照，不看程序自己的说法。

用法（父进程自动派生子进程，无需手动分两步）：

    python -X utf8 tools/diag_preview_render.py --mode nofeed   # 只建窗不喂图
    python -X utf8 tools/diag_preview_render.py --mode late --delay 8
    python -X utf8 tools/diag_preview_render.py --mode color   # 立即喂纯色图

每条都会输出：窗口是否创建、是否「未响应」、截图平均色、判定的白屏像素占比。
截图存到 tools/_preview_shots/ 便于肉眼复核。
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

WINDOW_TITLE = "NERO preview diag"
SHOT_DIR = Path(__file__).resolve().parent / "_preview_shots"

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

PW_RENDERFULLCONTENT = 0x00000002
BI_RGB = 0
DIB_RGB_COLORS = 0
SRCCOPY = 0x00CC0020


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def grab_window(hwnd: int) -> "object":
    """用 PrintWindow 把窗口内容抓成 numpy BGRA 数组。"""
    import numpy as np

    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    hdc_window = user32.GetDC(hwnd)
    if not hdc_window:
        return None
    hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
    hbmp = gdi32.CreateCompatibleBitmap(hdc_window, width, height)
    gdi32.SelectObject(hdc_mem, hbmp)
    try:
        # 先填成洋红，这样"什么都没画"和"画了白色"能区分开
        brush = gdi32.CreateSolidBrush(0x00FF00FF)
        user32.FillRect(hdc_mem, ctypes.byref(rect), brush)
        gdi32.DeleteObject(brush)

        ok = user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)

        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height  # 负数 = 自上而下
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB

        buffer = ctypes.create_string_buffer(width * height * 4)
        lines = gdi32.GetDIBits(hdc_mem, hbmp, 0, height, buffer,
                                ctypes.byref(info), DIB_RGB_COLORS)
        if not lines:
            return None
        array = np.frombuffer(buffer.raw, dtype=np.uint8).reshape(height, width, 4)
        return {"image": array[:, :, :3].copy(), "printwindow_ok": bool(ok)}
    finally:
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_window)


def analyze(image) -> dict:
    """统计白屏 / 洋红（未绘制）/ 有效内容各自占比。"""
    import numpy as np

    flat = image.reshape(-1, 3).astype(np.int16)
    white = np.all(flat >= 235, axis=1)
    magenta = (flat[:, 2] >= 200) & (flat[:, 0] >= 200) & (flat[:, 1] <= 60)
    # "有内容"= 既不是纯白也不是洋红占位
    content = ~(white | magenta)
    total = flat.shape[0]
    return {
        "white_ratio": white.mean(),
        "magenta_ratio": magenta.mean(),
        "content_ratio": content.mean(),
        "mean_rgb": tuple(int(v) for v in flat.mean(axis=0)[::-1]),
        "size": (image.shape[1], image.shape[0]),
    }


def find_window(title: str, timeout_s: float = 10.0) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
        time.sleep(0.05)
    return 0


# --------------------------------------------------------------------------
# 子进程：模拟 nero_teleoperate 的窗口调用顺序
# --------------------------------------------------------------------------
def child_main(argv: list[str]) -> int:
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--hold", type=float, default=6.0)
    parser.add_argument("--window-mode", choices=("normal", "autosize"), default="normal")
    args = parser.parse_args(argv)

    flag = cv2.WINDOW_NORMAL if args.window_mode == "normal" else cv2.WINDOW_AUTOSIZE
    print(f"[child] namedWindow mode={args.window_mode} flag={flag}", flush=True)
    cv2.namedWindow(WINDOW_TITLE, flag)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 480)
    cv2.moveWindow(WINDOW_TITLE, 40, 40)
    print("[child] window created", flush=True)

    if args.mode == "nofeed":
        # 完全不 imshow，只泵消息 —— 模拟「建了窗但一张图没送」
        deadline = time.monotonic() + args.hold
        while time.monotonic() < deadline:
            cv2.waitKey(30)
        return 0

    if args.mode == "late":
        # 模拟实机场景：归位阶段 8 秒内一次 imshow 都没有，也不泵消息
        time.sleep(args.delay)
        print(f"[child] slept {args.delay}s before first imshow", flush=True)

    if args.mode in ("color", "late"):
        canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
        canvas[:, :640] = (60, 170, 60)     # 绿
        canvas[:, 640:] = (180, 120, 40)    # 蓝
        cv2.putText(canvas, "IF YOU SEE THIS, imshow WORKS", (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        deadline = time.monotonic() + args.hold
        while time.monotonic() < deadline:
            cv2.imshow(WINDOW_TITLE, canvas)
            if cv2.waitKey(30) in (ord("q"), 27):
                break
        return 0
    return 1


# --------------------------------------------------------------------------
# 父进程：派生子进程并给窗口拍快照
# --------------------------------------------------------------------------
def parent_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("nofeed", "late", "color"), default="late")
    parser.add_argument("--delay", type=float, default=8.0)
    parser.add_argument("--hold", type=float, default=6.0)
    parser.add_argument("--window-mode", choices=("normal", "autosize"), default="normal")
    parser.add_argument("--shots", type=float, nargs="*", default=None,
                        help="在第几秒拍照，默认按模式自动决定")
    args = parser.parse_args(argv)

    if args.shots is None:
        args.shots = [1.0, 3.0, 5.5] if args.mode == "nofeed" else (
            [2.0, args.delay + 1.5, args.delay + 4.0])

    child_args = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                  "--child", "--mode", args.mode, "--delay", str(args.delay),
                  "--hold", str(args.hold), "--window-mode", args.window_mode]
    child = subprocess.Popen(child_args)
    started = time.monotonic()

    hwnd = find_window(WINDOW_TITLE, timeout_s=10.0)
    if not hwnd:
        print("[FAIL] 窗口根本没创建出来")
        child.terminate()
        return 2

    print(f"[OK] 找到窗口 hwnd=0x{hwnd:X}")
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    verdicts = []
    for at in args.shots:
        wait = at - (time.monotonic() - started)
        if wait > 0:
            time.sleep(wait)
        if not user32.IsWindow(hwnd):
            print(f"  t={at:.1f}s 窗口已销毁")
            continue
        hung = bool(user32.IsHungAppWindow(hwnd))
        grabbed = grab_window(hwnd)
        if not grabbed:
            print(f"  t={at:.1f}s 截图失败")
            continue
        stats = analyze(grabbed["image"])
        path = SHOT_DIR / f"{args.mode}_{at:.1f}s.png"
        try:
            import cv2
            cv2.imwrite(str(path), grabbed["image"])
        except Exception:
            path = None
        line = (f"  t={at:.1f}s  未响应={hung}  "
                f"白={stats['white_ratio']*100:5.1f}%  "
                f"洋红(未绘制)={stats['magenta_ratio']*100:5.1f}%  "
                f"有内容={stats['content_ratio']*100:5.1f}%  "
                f"平均RGB={stats['mean_rgb']}  size={stats['size']}")
        print(line)
        verdicts.append((at, hung, stats))

    child.terminate()
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        child.kill()

    print("\n=== 判定 ===")
    if not verdicts:
        print("没有取到有效截图")
        return 2
    any_hung = any(h for _, h, _ in verdicts)
    worst_white = max(s["white_ratio"] for _, _, s in verdicts)
    best_content = max(s["content_ratio"] for _, _, s in verdicts)
    if any_hung and args.mode == "late":
        print(">>> 窗口在等待 imshow 期间变为「未响应」——这正是 Windows 把窗口涂白的机制。")
        print(">>> 修法：启动/归位阶段必须周期性泵消息，或延后建窗。")
    elif worst_white > 0.85:
        print(">>> 窗口内容几乎全白，即使喂了图 —— HighGUI 渲染层异常（opencv 双包混装是首要嫌疑）。")
    elif args.mode == "nofeed" and best_content < 0.05:
        print(">>> 没喂图时窗口是空的（预期）。若实机也这样，说明应用层没把帧送进去。")
    else:
        print(">>> 窗口能正常显示内容。")
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        index = sys.argv.index("--child")
        sys.exit(child_main(sys.argv[index + 1:]))
    sys.exit(parent_main(sys.argv[1:]))
