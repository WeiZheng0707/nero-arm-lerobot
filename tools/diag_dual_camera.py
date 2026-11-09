"""双摄同时采集诊断：验证两路能否并行、实测帧率与帧龄。

这是纯相机测试，不碰机械臂、不发任何 CAN 指令。

测三件事：
  1) 枚举哪些 index 真的能出图（含实际分辨率 / FOURCC）
  2) 单路基准帧率（基线，判断相机自身能力）
  3) 双路同时采集的帧率与帧龄 —— 帧龄决定预览延迟，双路掉到 15fps 或帧龄 >100ms
     就不能用于遥操作采集

用法：
    python -X utf8 tools/diag_dual_camera.py --indices 1 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time

import cv2

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}


class Reader:
    """持续读取一个相机，记录每帧到达时间，供外部统计帧龄。"""

    def __init__(self, index: int, backend: str, width: int, height: int, fps: int,
                 fourcc: str | None = None):
        self.index = index
        self.capture = cv2.VideoCapture(index, BACKENDS[backend])
        if not self.capture.isOpened():
            raise RuntimeError(f"打不开 index={index} backend={backend}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        if fourcc:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.actual_fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stamps: list[float] = []
        self._frame = None
        self._updated_at = 0.0
        self.failures = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            now = time.monotonic()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._updated_at = now
                    self._stamps.append(now)
            else:
                with self._lock:
                    self.failures += 1
                self._stop.wait(0.005)

    def snapshot(self) -> tuple[float, int]:
        """返回 (最新帧时间戳, 累计帧数)，不加锁拷贝图像。"""
        with self._lock:
            return self._updated_at, len(self._stamps)

    def reset(self) -> None:
        with self._lock:
            self._stamps.clear()
            self.failures = 0

    def stats(self, window: float) -> dict:
        with self._lock:
            stamps = list(self._stamps)
        if len(stamps) < 2:
            return {"fps": 0.0, "n": len(stamps), "failures": self.failures}
        span = stamps[-1] - stamps[0]
        fps = (len(stamps) - 1) / span if span > 0 else 0.0
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        gaps_sorted = sorted(gaps)
        return {
            "fps": fps,
            "n": len(stamps),
            "failures": self.failures,
            "gap_p50": statistics.median(gaps),
            "gap_p95": gaps_sorted[int(len(gaps_sorted) * 0.95) - 1] if len(gaps_sorted) > 1 else 0.0,
            "gap_max": gaps_sorted[-1],
            "span": span,
            "window": window,
        }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.capture.release()


def fourcc_to_str(code: int) -> str:
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")


def measure(readers: list[Reader], seconds: float, label: str) -> list[dict]:
    """在指定时长内高频采样每路的帧时间戳，统计 fps 与帧龄。"""
    print(f"\n--- {label}（{seconds:.0f}s）---", flush=True)
    for r in readers:
        r.reset()
    ages: dict[int, list[float]] = {r.index: [] for r in readers}
    last_seen = {r.index: 0.0 for r in readers}

    started = time.monotonic()
    while time.monotonic() - started < seconds:
        now = time.monotonic()
        for r in readers:
            stamp, _ = r.snapshot()
            if stamp > last_seen[r.index]:
                last_seen[r.index] = stamp
            if stamp > 0:
                ages[r.index].append((now - stamp) * 1000.0)
        time.sleep(0.01)
    elapsed = time.monotonic() - started

    results = []
    for r in readers:
        s = r.stats(elapsed)
        a = sorted(ages[r.index])
        age_p50 = statistics.median(a) if a else float("nan")
        age_p95 = a[int(len(a) * 0.95) - 1] if len(a) > 1 else float("nan")
        results.append({**s, "index": r.index, "age_p50": age_p50, "age_p95": age_p95})
        print(
            f"  index={r.index}  实际 {r.actual_w}x{r.actual_h} "
            f"fourcc={fourcc_to_str(r.actual_fourcc)}  "
            f"fps={s['fps']:5.1f}  帧数={s['n']:4d}  读失败={s['failures']:3d}  "
            f"帧间隔 p50={s.get('gap_p50', 0) * 1000:5.1f}ms "
            f"p95={s.get('gap_p95', 0) * 1000:5.1f}ms  "
            f"帧龄 p50={age_p50:5.1f}ms p95={age_p95:5.1f}ms",
            flush=True,
        )
    return results


def probe(max_index: int, backend: str) -> list[int]:
    print(f"枚举相机 index 0..{max_index - 1}（backend={backend}）", flush=True)
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, BACKENDS[backend])
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            print(
                f"  index={i}  OK  {frame.shape[1]}x{frame.shape[0]}  "
                f"fourcc={fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))}",
                flush=True,
            )
            found.append(i)
        else:
            print(f"  index={i}  打开但读不到帧", flush=True)
        cap.release()
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="要测的相机 index；不给则自动枚举")
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="dshow")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default=None, help="强制 FOURCC，例如 MJPG；不给则用驱动默认（多为 YUY2）")
    parser.add_argument("--single-seconds", type=float, default=3.0)
    parser.add_argument("--dual-seconds", type=float, default=6.0)
    args = parser.parse_args(argv)

    indices = args.indices or probe(8, args.backend)
    if len(indices) < 2:
        print(f"\n[FAIL] 只找到 {len(indices)} 个可用相机：{indices}；双摄需要 2 个。", flush=True)
        return 1
    indices = indices[:2]
    print(f"\n将测试这两路：{indices}", flush=True)

    # 1) 单路基准
    single_results = []
    for index in indices:
        reader = Reader(index, args.backend, args.width, args.height, args.fps, args.fourcc)
        try:
            measure([reader], args.single_seconds, f"单路基准 index={index}")
            single_results.append(reader.stats(args.single_seconds))
        finally:
            reader.close()
        time.sleep(0.5)

    # 2) 双路并行
    readers = []
    try:
        for index in indices:
            readers.append(Reader(index, args.backend, args.width, args.height, args.fps, args.fourcc))
            time.sleep(0.3)
        dual = measure(readers, args.dual_seconds, "双路同时采集")
    finally:
        for r in readers:
            r.close()

    # 3) 结论
    print("\n=== 结论 ===", flush=True)
    worst_fps = min((r["fps"] for r in dual), default=0.0)
    worst_age = max((r["age_p95"] for r in dual), default=float("nan"))
    base_fps = min((r["fps"] for r in single_results), default=0.0)
    print(f"单路最低 fps : {base_fps:.1f}", flush=True)
    print(f"双路最低 fps : {worst_fps:.1f}", flush=True)
    print(f"双路最大帧龄 p95: {worst_age:.1f} ms", flush=True)

    ok = True
    if worst_fps < base_fps * 0.8:
        print(f"[WARN] 双路帧率比单路掉了 {(1 - worst_fps / max(base_fps, 1e-6)) * 100:.0f}%，"
              "疑似 USB 带宽或控制器共享问题。", flush=True)
        ok = False
    if worst_age > 100.0:
        print(f"[WARN] 帧龄 {worst_age:.0f}ms 偏大，预览会明显滞后。", flush=True)
        ok = False
    if worst_fps < 20.0:
        print(f"[WARN] 双路帧率只有 {worst_fps:.1f}fps，低于 20fps 不适合采集。", flush=True)
        ok = False
    print("[PASS] 双摄可用" if ok else "[FAIL] 双摄当前不满足采集要求", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
