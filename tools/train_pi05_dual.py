"""Local PI05 LoRA short training entry. Requires verified local tokenizer."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from transformers import AutoTokenizer
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts.train import train
    cfg = TrainPipelineConfig.from_pretrained(ROOT / 'configs/train_pi05_dual_lora_smoke.json')
    AutoTokenizer.from_pretrained(cfg.policy.tokenizer_name_or_path, local_files_only=True)
    cfg.policy.pretrained_path = str(ROOT / 'outputs/model')
    train(cfg)


if __name__ == '__main__':
    main()
