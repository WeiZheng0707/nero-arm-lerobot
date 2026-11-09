"""Inspect received CAN frames without sending application-level commands."""

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path


def collect_frames(bus, seconds, timeline=None):
    counts = Counter()
    latest = {}
    started = time.monotonic()
    deadline = started + seconds
    while (remaining := deadline - time.monotonic()) > 0:
        frame = bus.recv(timeout=min(0.2, remaining))
        if frame is None:
            continue
        key = (frame.arbitration_id, frame.is_extended_id, frame.is_error_frame, frame.is_remote_frame)
        counts[key] += 1
        latest[key] = bytes(frame.data).hex(" ")
        if timeline is not None:
            bucket = min(int(time.monotonic() - started), math.ceil(seconds) - 1)
            timeline.setdefault(bucket, Counter())[key] += 1
    return counts, latest, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--interface", default="agx_cando")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")

    import can

    # Reuse the existing open-failure preflight for this Windows driver.
    if args.interface == "agx_cando":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/lerobot/motors/nero"))
        from read_nero_firmware import preflight_can_device

        if not preflight_can_device(args.interface, args.channel):
            return 3

    print(f"Receive only: interface={args.interface}, channel={args.channel}, bitrate=1000000", flush=True)
    print(f"Collecting for {args.seconds:g}s. No enable, disable, mode or motion commands.", flush=True)
    options = {"local_loopback": False} if args.interface == "agx_cando" else {}
    timeline = {}
    with can.Bus(interface=args.interface, channel=args.channel, bitrate=1_000_000,
                 receive_own_messages=False, **options) as bus:
        counts, latest, elapsed = collect_frames(bus, args.seconds, timeline)
    print("Per-second received counts (local receive time, not controller timestamps):")
    print("SECONDS     TOTAL   2A5   2A6   2A7   2A9  GRIP_2A8")
    for second in range(math.ceil(args.seconds)):
        sample = timeline.get(second, Counter())
        values = [sample[(identifier, False, False, False)]
                  for identifier in (0x2A5, 0x2A6, 0x2A7, 0x2A9, 0x2A8)]
        print(f"{second:>3}-{min(second + 1, args.seconds):<5g} {sum(sample.values()):7}"
              + "".join(f"{value:6}" for value in values))
    print(f"Total received frames: {sum(counts.values())}; elapsed: {elapsed:.2f}s")
    print("CAN_ID    FORMAT  ERROR RTR  COUNT    AVG_HZ  LAST_DATA")
    for key, count in sorted(counts.items()):
        identifier, extended, error, remote = key
        print(f"0x{identifier:03X}  {'EXT' if extended else 'STD':>6}  {int(error):5} {int(remote):3}"
              f"  {count:5}  {count / elapsed:8.1f}  {latest[key]}")
    if not counts:
        print("No frames received in this window. This alone does not identify the cause.")
    print("Adapter released.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
