"""训练前置检查：30 秒内告诉你还缺什么，不用等模型加载完才报错。

检查四项：
  1. 本地 pi0.5 权重目录完整
  2. 分位数统计文件存在且形状对得上
  3. PaliGemma 分词器能拿到（gated repo，需要 HF 登录 + 接受许可）
  4. GPU 可用且显存够

退出码非 0 表示有阻塞项。用法：
    python tools/pi05_check_env.py --dataset-root outputs/eye_in_hand_red_cube_v2/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model/pi_05_base"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    blocking: list[str] = []
    warnings: list[str] = []

    print("[1/4] 模型权重")
    weights = args.model / "model.safetensors"
    cfg = args.model / "config.json"
    if weights.exists() and cfg.exists():
        size = weights.stat().st_size / 1e9
        print(f"  OK  {weights}  {size:.2f} GB")
    else:
        blocking.append(f"缺少 {weights} 或 {cfg}")
        print(f"  缺失 {weights}")

    print("[2/4] 分位数统计")
    qpath = args.dataset_root / "meta" / "quantile_stats.json"
    if not qpath.exists():
        blocking.append(f"缺少 {qpath}")
        print(f"  缺失 {qpath}")
        print(f"  先跑: python tools\\pi05_quantile_stats.py --root {args.dataset_root}")
    else:
        try:
            payload = json.loads(qpath.read_text(encoding="utf-8"))
            dims = payload["observation.state"]["dims"]
            if payload["action"]["dims"] != dims:
                warnings.append("state 与 action 维度不一致")
            print(f"  OK  dims={dims} frames={payload['observation.state']['count']}")
        except Exception as exc:
            blocking.append(f"{qpath} 解析失败: {exc}")
            print(f"  解析失败: {exc}")

    print("[3/4] PaliGemma 分词器（gated）")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", local_files_only=True)
        print(f"  OK  词表大小 {len(tok)}")
    except Exception as exc:
        blocking.append("PaliGemma 分词器不可用")
        print(f"  失败: {type(exc).__name__}")
        print("  这个仓库是 gated：先去 huggingface.co/google/paligemma-3b-pt-224")
        print("  点 Agree、完成 Hugging Face 登录并下载 tokenizer 文件，然后重跑检查。")
        print(
            '  下载命令: python -c "from huggingface_hub import snapshot_download; '
            "snapshot_download('google/paligemma-3b-pt-224', allow_patterns=['tokenizer*','special_tokens_map.json','config.json'])\""
        )

    print("[4/4] GPU")
    try:
        import torch

        if not torch.cuda.is_available():
            blocking.append("CUDA 不可用")
            print("  CUDA 不可用")
        else:
            free, total = torch.cuda.mem_get_info()
            print(f"  OK  {torch.cuda.get_device_name(0)}  空闲 {free / 1e9:.1f} / {total / 1e9:.1f} GB")
            if free / 1e9 < 11:
                warnings.append("显存空闲不足 11 GB，训练可能 OOM；先关掉占显存的程序")
    except Exception as exc:
        blocking.append(f"GPU 检查失败: {exc}")
        print(f"  失败: {exc}")

    print()
    for warning in warnings:
        print(f"警告: {warning}")
    if blocking:
        for item in blocking:
            print(f"阻塞: {item}")
        raise SystemExit(1)
    print("全部就绪，可以训练。")


if __name__ == "__main__":
    main()
