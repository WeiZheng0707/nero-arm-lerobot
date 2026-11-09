"""把两臂当前姿态保存为遥操作 / 采集的恢复点，或只对比不写入。

只读：用 `connect_feedback_only()` 打开反馈流，**不使能、不改模式、不动臂**。
（对比 `read_nero_firmware` 走的是 `connect()`，那条路径会真的给电机通电。）

两种用法：

1) 保存（用手把两臂摆好后）：
    python -X utf8 tools/save_start_pose.py ^
        --leader-channel 1 --follower-channel 0 ^
        --leader-firmware v120 --follower-firmware v121 --overwrite

2) 只对比当前姿态与已有恢复点，不写任何文件（不安全运动前先跑这个）：
    python -X utf8 tools/save_start_pose.py ^
        --leader-channel 1 --follower-channel 0 ^
        --leader-firmware v120 --follower-firmware v121 --compare

`--compare` 会算出从臂到恢复点每轴要走多少度，并直接告诉你会不会被
`nero_teleoperate` 的 60 deg 单轴安全闸拦下。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from lerobot.motors.nero import NeroMotorsBus, NeroMotorsBusConfig

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "examples" / "nero" / "saved_start_pose.json"
SW_TO_FW = {"v120": "1.20", "v121": "1.21", "v112": "1.12", "v111": "1.11"}


def read_pose(interface: str, channel: str, firmware: str, role: str,
              timeout_s: float) -> list[float]:
    bus = NeroMotorsBus(NeroMotorsBusConfig(
        interface=interface,
        channel=str(channel),
        firmeware_version=firmware,
        role=role,
    ))
    bus.connect_feedback_only()
    try:
        pose = bus.wait_stable_pose(timeout_s=timeout_s)
        if pose is None:
            raise RuntimeError(f"{role}（channel {channel}）读不到稳定位姿")
        return [float(value) for value in pose]
    finally:
        # 只读路径必须用这个收尾：普通 disconnect() 会发失能指令，
        # 而这里本来就从未使能过。
        bus.close_keeping_enabled()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="agx_cando")
    parser.add_argument("--leader-channel", required=True)
    parser.add_argument("--follower-channel", required=True)
    parser.add_argument("--leader-firmware", choices=tuple(SW_TO_FW), required=True)
    parser.add_argument("--follower-firmware", choices=tuple(SW_TO_FW), required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true",
                        help="目标文件已存在时允许覆盖")
    parser.add_argument("--compare", action="store_true",
                        help="只对比当前姿态与已有恢复点，不写文件")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    if args.leader_channel == args.follower_channel:
        parser.error("主臂和从臂不能占用同一个通道")
    if args.compare and args.overwrite:
        parser.error("--compare 只读，不能和 --overwrite 同时使用")

    out = Path(args.output).expanduser().resolve()
    if not args.compare and out.exists() and not args.overwrite:
        print(f"[ABORT] {out} 已存在。确认要替换请加 --overwrite。", flush=True)
        return 1

    reference = None
    if args.compare:
        if not out.exists():
            print(f"[ABORT] {out} 不存在，没有可对比的恢复点。", flush=True)
            return 1
        with out.open(encoding="utf-8") as handle:
            reference = json.load(handle)
        print(f"恢复点文件: {out}", flush=True)
        print(f"  保存于 {reference.get('saved_at', '未知')}", flush=True)

    print("只读读取两臂当前姿态（不会使能、不会动臂）...", flush=True)
    leader = read_pose(args.interface, args.leader_channel, args.leader_firmware,
                       "leader", args.timeout)
    print(f"  leader   (channel {args.leader_channel}) 读取完成", flush=True)
    follower = read_pose(args.interface, args.follower_channel, args.follower_firmware,
                         "follower", args.timeout)
    print(f"  follower (channel {args.follower_channel}) 读取完成", flush=True)

    print("\n各关节角度（度）：", flush=True)
    print("  关节   leader    follower   差值", flush=True)
    worst = 0.0
    for index in range(7):
        deg_l = math.degrees(leader[index])
        deg_f = math.degrees(follower[index])
        worst = max(worst, abs(deg_l - deg_f))
        print(f"  J{index + 1}   {deg_l:8.2f}  {deg_f:8.2f}  {deg_l - deg_f:8.2f}", flush=True)
    if worst > 10.0:
        print(f"  [提示] 主从姿态最大相差 {worst:.1f} deg。恢复点不要求两臂姿态相同"
              "（映射是相对的），但姿态接近时起步更平顺。", flush=True)

    if args.compare:
        print("\n=== 从臂到恢复点要走多少（这是 --return-to-start 真正会动的部分） ===", flush=True)
        goal_f = reference["follower"]["joints"]
        goal_l = reference["leader"]["joints"]
        print("  关节   从臂当前   从臂目标   要走(deg) | 主臂当前   主臂目标   差(deg)", flush=True)
        worst_f = 0.0
        worst_l = 0.0
        for index in range(7):
            now_f = math.degrees(follower[index])
            tgt_f = math.degrees(goal_f[index])
            now_l = math.degrees(leader[index])
            tgt_l = math.degrees(goal_l[index])
            worst_f = max(worst_f, abs(tgt_f - now_f))
            worst_l = max(worst_l, abs(tgt_l - now_l))
            print(f"  J{index + 1}  {now_f:9.2f}  {tgt_f:9.2f}  {tgt_f - now_f:9.2f}"
                  f"  | {now_l:9.2f}  {tgt_l:9.2f}  {now_l - tgt_l:8.2f}", flush=True)

        print("", flush=True)
        if worst_f > 60.0:
            print(f"  [会拦截] 从臂单轴最大 {worst_f:.1f} deg > 60 deg 安全闸。"
                  "直接跑会被拒绝，要么先把臂摆过去，要么加 --allow-large-return。", flush=True)
        else:
            print(f"  [可以通过] 从臂单轴最大 {worst_f:.1f} deg <= 60 deg，"
                  "--return-to-start 会正常执行。", flush=True)
        print(f"  [主臂] 单轴差 {worst_l:.1f} deg —— 但主臂在 leader 零力拖拽模式下"
              "不接受位置指令，这个数字只是记录，不会被执行；实际要你手动拖。", flush=True)
        return 0

    data = {
        "schema": 1,
        "unit": "rad",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "leader": {"firmware": SW_TO_FW[args.leader_firmware], "joints": leader},
        "follower": {"firmware": SW_TO_FW[args.follower_firmware], "joints": follower},
    }
    if not all(math.isfinite(v) for v in (*leader, *follower)):
        print("[ABORT] 读到的角度里有非有限值，拒绝写入。", flush=True)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"\n[OK] 恢复点已写入 {out}", flush=True)
    print("之后用 --return-to-start --pose-file <该文件> 即可使能后回到这个姿态。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
