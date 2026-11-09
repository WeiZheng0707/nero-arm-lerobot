"""List agx_cando adapters per channel index without moving the robot.

The channel number in this backend is the position returned by ``cando_list``,
so it is NOT a stable identity: unplugging or swapping ports renumbers it.
Use this tool to pair each channel index with the adapter's USB serial number
before running teleoperation.

Each channel is probed in a separate child process on purpose. When
``cando_open`` fails, the cleanup path inside ``cando.dll`` can corrupt the
heap (0xC0000374) and kill the interpreter, so the probe must not share a
process with the rest of the tool.
"""

import argparse
import json
import subprocess
import sys


def probe(channel: int) -> dict:
    """Run in the child process: open one adapter and report its identity."""
    from agx_cando.bus import _CandoApi

    api = _CandoApi()
    try:
        info = api.describe(channel)
    except Exception as exc:  # noqa: BLE001 - report any backend failure verbatim
        return {"channel": channel, "opened": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "channel": channel,
        "opened": True,
        "serial": info.serial,
        "fw_version": info.fw_version,
        "hw_version": info.hw_version,
    }


def count_devices() -> int:
    import can

    return len(can.detect_available_configs(interfaces=["agx_cando"]))


def probe_isolated(channel: int) -> dict:
    """Probe one channel in a child process so a DLL crash cannot abort the scan."""
    cmd = [sys.executable, __file__, "--_child", str(channel)]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"channel": channel, "opened": False, "error": "timeout after 60s"}
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {completed.returncode}"
        return {"channel": channel, "opened": False, "error": f"child crashed: {tail}"}
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"channel": channel, "opened": False, "error": "unparsable child output"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--_child", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = parser.parse_args()

    if args._child is not None:
        print(json.dumps(probe(args._child)))
        return 0

    total = count_devices()
    if not args.json:
        print(f"agx_cando adapters detected: {total}")
    if total == 0:
        if args.json:
            print(json.dumps({"detected": 0, "channels": []}))
        else:
            print("No adapter found. Check the cable and the USB port.")
        return 1

    results = [probe_isolated(index) for index in range(total)]

    if args.json:
        print(json.dumps({"detected": total, "channels": results}, ensure_ascii=False))
        return 0

    print("\n  channel | status | serial                | fw  | hw")
    print("  --------+--------+-----------------------+-----+----")
    for row in results:
        if row.get("opened"):
            print(f"  {row['channel']:>7} | OPEN   | {row['serial']:<21} | {row['fw_version']:<3} | {row['hw_version']}")
        else:
            print(f"  {row['channel']:>7} | FAIL   | {row.get('error', '')}")
    print(
        "\nReminder: channel numbers are enumeration order, not identity.\n"
        "Record the serial of each arm once, then match by serial after every replug."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
