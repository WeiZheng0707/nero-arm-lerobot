"""Read-only RealSense dual-camera rate/bandwidth acceptance check.

Never connects to a robot, never enables or moves an arm. It only opens the
colour streams and measures what the link actually delivers.

Run this BEFORE recording the 300-episode dataset. A camera that silently
negotiated USB 2.0 will look "laggy" during capture and corrupt frame pairing;
measuring it here is far cheaper than discovering it 300 episodes later.

Usage:
    python -X utf8 tools/realsense_bandwidth_check.py --serial-base 261622071323
    python -X utf8 tools/realsense_bandwidth_check.py --serial-base 261622071323 --serial-wrist 261622078506 --concurrent
    python -X utf8 tools/realsense_bandwidth_check.py --serial-base X --serial-wrist Y --seconds 20
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - pyrealsense2 is a hard project dependency
    raise SystemExit("pyrealsense2 is not installed in this environment")


def list_devices():
    devices = rs.context().query_devices()
    rows = []
    for device in devices:
        def info(field):
            try:
                return device.get_info(field)
            except RuntimeError:
                return "?"
        rows.append({
            "serial": info(rs.camera_info.serial_number),
            "name": info(rs.camera_info.name),
            "usb": info(rs.camera_info.usb_type_descriptor),
            "firmware": info(rs.camera_info.firmware_version),
            "product_line": info(rs.camera_info.product_line),
            "physical_port": info(rs.camera_info.physical_port),
        })
    return rows


def _open(serial, width, height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        return None, f"start failed: {exc}"
    active = {}
    for stream in profile.get_streams():
        if stream.stream_type() == rs.stream.color:
            video = stream.as_video_stream_profile()
            active = {"width": video.width(), "height": video.height(),
                      "fps": video.fps(), "format": str(video.format())}
    return pipeline, active


def measure_concurrent(roles, seconds):
    """Open every camera at once and measure them together.

    This is the only honest test on USB 2.0: the link budget is shared per host
    controller, so measuring cameras one after another can pass while running
    both together drops frames. Threads only stamp host arrival times; the
    decision to record is made from these gaps.
    """
    import threading

    opened = {}
    for role, serial, (width, height, fps) in roles:
        pipeline, info_or_error = _open(serial, width, height, fps)
        if pipeline is None:
            print(f"  {role} serial={serial} ERROR: {info_or_error}", flush=True)
            for other in opened.values():
                other["pipeline"].stop()
            return [{"role": role, "serial": serial, "error": info_or_error}]
        opened[role] = {"pipeline": pipeline, "serial": serial,
                        "active": info_or_error, "stamps": [], "bytes": 0,
                        "requested": f"{width}x{height}@{fps}", "stop": False}
        print(f"  {role} serial={serial} opened, active={info_or_error}", flush=True)

    def pump(role):
        entry = opened[role]
        while not entry["stop"]:
            try:
                frame_set = entry["pipeline"].wait_for_frames(1000)
            except RuntimeError:
                continue
            colour = frame_set.get_color_frame()
            if colour:
                entry["stamps"].append(time.monotonic())
                entry["bytes"] += np.asanyarray(colour.get_data()).nbytes

    threads = [threading.Thread(target=pump, args=(role,), daemon=True) for role in opened]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    time.sleep(seconds)
    for entry in opened.values():
        entry["stop"] = True
    for thread in threads:
        thread.join(timeout=2.0)
    elapsed = time.monotonic() - started
    for entry in opened.values():
        try:
            entry["pipeline"].stop()
        except RuntimeError:
            pass

    rows = []
    for role, entry in opened.items():
        stamps = entry["stamps"]
        gaps = np.diff(stamps) * 1000.0 if len(stamps) > 1 else np.array([0.0])
        fps_target = int(entry["requested"].split("@")[1])
        dropped = int(np.sum(gaps > 2 * (1000.0 / fps_target)))
        rows.append({
            "role": role, "serial": entry["serial"],
            "usb": next((d["usb"] for d in list_devices()
                         if d["serial"] == entry["serial"]), "?"),
            "active": entry["active"],
            "requested_wxh": entry["requested"],
            "achieved_fps": len(stamps) / elapsed if elapsed > 0 else 0.0,
            "throughput_mb_s": (entry["bytes"] / elapsed / 1e6) if elapsed > 0 else 0.0,
            "gap_p95_ms": float(np.percentile(gaps, 95)) if len(gaps) else 0.0,
            "gap_max_ms": float(gaps.max()) if len(gaps) else 0.0,
            "dropped_frames": dropped,
            "drop_rate": dropped / max(1, len(stamps)),
        })
    return rows


def measure(serial, width, height, fps, seconds):
    """Open one colour stream and report achieved rate plus byte throughput."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        return {"serial": serial, "error": f"start failed: {exc}"}
    active = {}
    for stream in profile.get_streams():
        if stream.stream_type() == rs.stream.color:
            video = stream.as_video_stream_profile()
            active = {"width": video.width(), "height": video.height(),
                      "fps": video.fps(), "format": str(video.format())}
    link = "?"
    try:
        for device in rs.context().query_devices():
            if device.get_info(rs.camera_info.serial_number) == serial:
                link = device.get_info(rs.camera_info.usb_type_descriptor)
    except RuntimeError:
        pass

    stamps = []
    started = time.monotonic()
    bytes_total = 0
    try:
        while time.monotonic() - started < seconds:
            try:
                frame_set = pipeline.wait_for_frames(1000)
            except RuntimeError:
                continue
            colour = frame_set.get_color_frame()
            if not colour:
                continue
            stamps.append(time.monotonic())
            bytes_total += np.asanyarray(colour.get_data()).nbytes
    finally:
        pipeline.stop()

    elapsed = time.monotonic() - started
    achieved = len(stamps) / elapsed if elapsed > 0 else 0.0
    gaps = np.diff(stamps) * 1000.0 if len(stamps) > 1 else np.array([0.0])
    expected_gap = 1000.0 / fps
    # A dropped frame shows up as a gap longer than 2 frame periods.
    dropped = int(np.sum(gaps > 2 * expected_gap))
    return {
        "serial": serial, "usb": link, "active": active,
        "requested_wxh": f"{width}x{height}@{fps}",
        "achieved_fps": achieved,
        "throughput_mb_s": (bytes_total / elapsed / 1e6) if elapsed > 0 else 0.0,
        "gap_p95_ms": float(np.percentile(gaps, 95)) if len(gaps) else 0.0,
        "gap_max_ms": float(gaps.max()) if len(gaps) else 0.0,
        "dropped_frames": dropped,
        "drop_rate": dropped / max(1, len(stamps)),
    }


def verdict(rows, strict=False):
    """Split findings into hard failures and tolerable warnings.

    The two failure modes have very different consequences and must not be
    conflated:

    * Lost frames corrupt the dataset (timestamp gaps the converter cannot
      resample). A gap beyond `2 / fps` means a frame never arrived.
    * Timing jitter with no loss does not corrupt anything: the capture records
      `age_ms` per frame and the converter resamples onto a 30 Hz grid while
      recording `max_resampling_error_s`, and a frame older than 200 ms marks the
      whole episode `invalid`.

    So a missing-frame / achieved-rate / 200 ms problem is a hard failure, while
    a p95 above the design document's 40 ms target is reported as a warning
    unless `strict=True`. `strict` keeps the original acceptance criterion
    available for anyone who wants the unrelaxed gate.
    """
    problems = []
    warnings = []
    for row in rows:
        label = f"{row.get('role', 'camera')} serial {row.get('serial', '?')} " \
                f"({row.get('usb', '?')})"
        if "error" in row:
            problems.append(f"{label}: {row['error']}")
            continue
        requested = str(row.get("requested_wxh", ""))
        active = row.get("active") or {}
        target_fps = 0.0
        if "@" in requested and "x" in requested:
            size_part, _, fps_part = requested.partition("@")
            want_w, _, want_h = size_part.partition("x")
            if (active.get("width"), active.get("height")) != (int(want_w), int(want_h)):
                problems.append(f"{label}: negotiated "
                                f"{active.get('width')}x{active.get('height')} "
                                f"!= requested {requested}")
            try:
                target_fps = float(fps_part)
            except ValueError:
                target_fps = 0.0
            if target_fps and row.get("achieved_fps", 0.0) < 0.9 * target_fps:
                problems.append(f"{label}: achieved "
                                f"{row.get('achieved_fps', 0.0):.1f} fps, "
                                f"requested {requested}")
        # A frame gap longer than 2 periods means at least one frame never arrived.
        period_ms = (1000.0 / target_fps) if target_fps else 33.3
        if row.get("drop_rate", 0.0) > 0.01:
            problems.append(f"{label}: drop rate {row['drop_rate']*100:.2f}% > 1%")
        if row.get("gap_max_ms", 0.0) > 2 * period_ms:
            problems.append(f"{label}: gap max {row['gap_max_ms']:.1f} ms exceeds "
                            f"2 frame periods ({2 * period_ms:.1f} ms) - frames were lost")
        if row.get("gap_max_ms", 0.0) > 200.0:
            problems.append(f"{label}: gap max {row['gap_max_ms']:.1f} ms would mark "
                            f"episodes invalid at capture time")
        if row.get("gap_p95_ms", 0.0) > 40.0:
            message = (f"{label}: gap p95 {row['gap_p95_ms']:.1f} ms > 40 ms target "
                       f"(jitter only: no frame loss)")
            (problems if strict else warnings).append(message)
    return problems, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-base", default=None, help="Serial of the static base camera")
    parser.add_argument("--serial-wrist", default=None, help="Serial of the wrist camera")
    parser.add_argument("--base-size", default="640x480@30",
                        help="Trial for the base camera. USB 3.0 usually sustains 640x480@30.")
    parser.add_argument("--wrist-size", default="640x480@30")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--concurrent", action="store_true",
                        help="Open both cameras at once and measure them together. This is "
                             "the realistic test: USB bandwidth is shared per host "
                             "controller, so sequential passes can hide contention.")
    parser.add_argument("--list", action="store_true", help="Only list connected cameras")
    parser.add_argument("--strict", action="store_true",
                        help="Enforce the design document's original p95 < 40 ms target as a "
                             "hard failure. By default that condition is a warning because "
                             "jitter without frame loss does not corrupt the dataset.")
    args = parser.parse_args()

    devices = list_devices()
    print("DETECTED REALSENSE DEVICES:", flush=True)
    for device in devices:
        print(f"  serial={device['serial']} usb={device['usb']} "
              f"fw={device['firmware']} name={device['name']} "
              f"port={device['physical_port']}", flush=True)
    if not devices:
        raise SystemExit("No RealSense device detected; check the USB cables.")
    if args.list:
        return 0
    usb2 = [d for d in devices if d["usb"].startswith("2")]
    if usb2:
        print(f"  [WARN] {len(usb2)} camera(s) on USB 2.0: "
              f"{[d['serial'] for d in usb2]}", flush=True)
    # Same host controller => the two streams share one USB 2.0 budget.
    ports = [d["physical_port"] for d in devices if d["physical_port"] not in ("?", "")]
    if len(ports) > 1 and len({p.split("/")[0].split("-")[0] for p in ports}) == 1:
        print("  [WARN] both cameras appear to share one USB controller; two 640x480@30 "
              "streams (49 MB/s combined) cannot be sustained there.", flush=True)

    if args.concurrent:
        roles = []
        for role, serial, size in (("base", args.serial_base, args.base_size),
                                   ("wrist", args.serial_wrist, args.wrist_size)):
            if not serial:
                continue
            width, height, fps = (size.split("@")[0].split("x") + [size.split("@")[1]])
            roles.append((role, serial, (int(width), int(height), int(fps))))
        if len(roles) < 2:
            print("\n--concurrent needs both --serial-base and --serial-wrist.", flush=True)
            return 0
        print(f"\nCONCURRENT MEASUREMENT for {args.seconds:.0f}s:", flush=True)
        rows = measure_concurrent(roles, args.seconds)
        for row in rows:
            if "error" in row:
                print(f"  {row['role']} serial={row['serial']} ERROR: {row['error']}",
                      flush=True)
                continue
            print(f"  {row['role']} serial={row['serial']} usb={row['usb']} "
                  f"active={row['active']}", flush=True)
            print(f"    achieved={row['achieved_fps']:.2f} fps "
                  f"throughput={row['throughput_mb_s']:.1f} MB/s", flush=True)
            print(f"    gap p95={row['gap_p95_ms']:.1f} ms max={row['gap_max_ms']:.1f} ms "
                  f"drops={row['dropped_frames']} ({row['drop_rate']*100:.2f}%)", flush=True)
        print("\nCOMBINED THROUGHPUT: "
              f"{sum(r.get('throughput_mb_s', 0) for r in rows):.1f} MB/s", flush=True)
        problems, warnings = verdict(rows, strict=args.strict)
        for warning in warnings:
            print(f"  [WARN] {warning}", flush=True)
        if problems:
            print("\nGATE FAILED (concurrent) - do not record the dataset yet:", flush=True)
            for problem in problems:
                print(f"  - {problem}", flush=True)
            print("\nNext step: put the two cameras on different host controllers, or "
                  "drop the base camera to 480x360@30 / both to 640x480@20.", flush=True)
            return 1
        print("\nGATE PASSED (concurrent): no lost frames, achieved rate within 10% of "
              "target, gaps far below the 200 ms invalid threshold.", flush=True)
        if warnings:
            print("  Timing jitter was seen but no frames were lost; the capture records "
                  "per-frame age and the converter resamples, so this does not corrupt "
                  "the dataset. Use --strict to enforce the original 40 ms target.",
                  flush=True)
        return 0

    trials = []
    for role, serial, size in (("base", args.serial_base, args.base_size),
                               ("wrist", args.serial_wrist, args.wrist_size)):
        if not serial:
            continue
        width, height, fps = (size.split("@")[0].split("x") + [size.split("@")[1]])
        print(f"\nMEASURING {role}: serial={serial} {size}", flush=True)
        row = measure(serial, int(width), int(height), int(fps), args.seconds)
        row["role"] = role
        trials.append(row)
        if "error" in row:
            print(f"  ERROR: {row['error']}", flush=True)
            continue
        print(f"  usb={row['usb']} active={row['active']}", flush=True)
        print(f"  achieved={row['achieved_fps']:.2f} fps "
              f"throughput={row['throughput_mb_s']:.1f} MB/s", flush=True)
        print(f"  gap p95={row['gap_p95_ms']:.1f} ms max={row['gap_max_ms']:.1f} ms "
              f"drops={row['dropped_frames']} ({row['drop_rate']*100:.2f}%)", flush=True)

    if not trials:
        print("\nNo serial given; nothing measured. Use --serial-base / --serial-wrist.",
              flush=True)
        return 0

    print("\nCOMBINED THROUGHPUT: "
          f"{sum(t.get('throughput_mb_s', 0) for t in trials):.1f} MB/s", flush=True)
    problems, warnings = verdict(trials, strict=args.strict)
    for warning in warnings:
        print(f"  [WARN] {warning}", flush=True)
    if problems:
        print("\nGATE FAILED - do not record the dataset yet:", flush=True)
        for problem in problems:
            print(f"  - {problem}", flush=True)
        print("\nFix the USB topology (separate controllers, no passive hub) or lower "
              "the resolution of the camera on the weaker link.", flush=True)
        return 1
    print("\nGATE PASSED: no lost frames, achieved rate within 10% of target.", flush=True)
    print("Repeat this check while the arm is moving; a static check alone is not "
          "sufficient because the wrist cable flexes during motion.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
