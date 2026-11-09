"""用真实双摄跑完整的生产预览链路，并给窗口拍快照，定位「白屏」到底出在哪一层。

它复用 `lerobot.nero_teleoperate` 里的 LatestCamera 和 compose_preview —— 不是
另写一套，所以复现的就是遥操作真实走的那条路径。

不碰机械臂、不碰 CAN，只开相机。

用法：
    python -X utf8 tools/diag_preview_live.py --indices 1 3 --seconds 15
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
PW_RENDERFULLCONTENT = 0x00000002
BI_RGB = 0
DIB_RGB_COLORS = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def grab(hwnd: int):
    import numpy as np

    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None
    hdc = user32.GetDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, width, height)
    gdi32.SelectObject(mem, bmp)
    try:
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        user32.PrintWindow(hwnd, mem, PW_RENDERFULLCONTENT)
        buf = ctypes.create_string_buffer(width * height * 4)
        if not gdi32.GetDIBits(mem, bmp, 0, height, buf, ctypes.byref(info),
                               DIB_RGB_COLORS):
            return None
        return np.frombuffer(buf.raw, dtype=np.uint8).reshape(
            height, width, 4)[:, :, :3].copy()
    finally:
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(hwnd, hdc)


def stats(image) -> str:
    import numpy as np

    flat = image.reshape(-1, 3).astype(np.int16)
    white = np.all(flat >= 235, axis=1)
    black = np.all(flat <= 20, axis=1)
    flat_std = float(flat.std())
    return (f"白={white.mean()*100:5.1f}% 黑={black.mean()*100:5.1f}% "
            f"标准差={flat_std:5.1f} 平均RGB={tuple(int(v) for v in flat.mean(axis=0)[::-1])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--preview-fps", type=float, default=30.0)
    parser.add_argument("--sleep-before", type=float, default=0.0,
                        help="建窗后先干等这么久再开始 imshow，模拟归位阶段")
    args = parser.parse_args(argv)

    import cv2
    from lerobot.nero_teleoperate import LatestCamera, compose_preview

    window = "NERO preview live diag"
    labels = args.labels or [f"cam{i}" for i in args.indices]

    print(f"[1] 打开相机 {args.indices} ...", flush=True)
    cameras: list[tuple[str, LatestCamera]] = []
    try:
        for position, index in enumerate(args.indices):
            name = labels[position] if position < len(labels) else f"cam{index}"
            camera = LatestCamera(index, args.backend, args.width, args.height, args.fps)
            cameras.append((name, camera))
            print(f"    {name} = index {index} opened", flush=True)
            # 生产代码是顺序打开、中间没有停顿；这里保持一样
        print("[2] 等首帧（最多 5s）", flush=True)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if all(c.latest() is not None for _, c in cameras):
                break
            time.sleep(0.05)
        for name, camera in cameras:
            frame = camera.latest()
            print(f"    {name}: first_frame={'None' if frame is None else frame.shape} "
                  f"age={camera.age_ms():.0f}ms", flush=True)

        print("[3] 建窗（与生产代码相同：WINDOW_NORMAL + resizeWindow）", flush=True)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, args.width * min(len(cameras), 2), args.height)
        cv2.moveWindow(window, 40, 40)

        if args.sleep_before > 0:
            print(f"[3.5] 故意干等 {args.sleep_before:g}s（模拟归位阶段不刷新）", flush=True)
            time.sleep(args.sleep_before)

        hwnd = user32.FindWindowW(None, window)
        print(f"[4] hwnd={hwnd}  开始 {args.seconds:g}s 预览", flush=True)
        shot_dir = Path(__file__).resolve().parent / "_preview_shots"
        shot_dir.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        next_preview = started
        next_shot = started + 2.0
        period = 1.0 / args.preview_fps
        count = 0
        none_canvas = 0
        while time.monotonic() - started < args.seconds:
            tick = time.monotonic()
            if tick >= next_preview:
                canvas = compose_preview(cameras, count, 0.0)
                if canvas is None:
                    none_canvas += 1
                    print(f"    [WARN] t={tick-started:4.1f}s compose_preview -> None "
                          f"（没有可用帧，imshow 被跳过）", flush=True)
                else:
                    cv2.imshow(window, canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                next_preview = tick + period
            if tick >= next_shot:
                image = grab(hwnd)
                if image is not None:
                    cv2.imwrite(str(shot_dir / f"live_{tick-started:.0f}s.png"), image)
                    ages = " ".join(f"{n}={c.age_ms():.0f}ms" for n, c in cameras)
                    print(f"    t={tick-started:4.1f}s 窗口: {stats(image)} | 帧龄 {ages}",
                          flush=True)
                next_shot = tick + 3.0
            count += 1
            time.sleep(max(0.0, 1.0 / 120.0 - (time.monotonic() - tick)))

        print(f"[5] 结束。循环 {count} 次，compose_preview 返回 None {none_canvas} 次",
              flush=True)
        if none_canvas > 0:
            print(">>> 有 None 帧：窗口会保持上一次内容或空白 —— 这就是白屏的直接来源。",
                  flush=True)
        return 0
    finally:
        for _, camera in cameras:
            try:
                camera.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
