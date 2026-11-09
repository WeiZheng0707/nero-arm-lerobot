# Blue Cube ACT Training

This run uses only the blue-cube demonstrations. The green-cube dataset and its checkpoints remain untouched. The training commands below are prepared for the local RTX 4090 environment; they do not connect to robot hardware.

## Before training

Manually review the episodes in `outputs/blue_cube` first. The converter rejects malformed rows and `quit_end` episodes, but `auto_end` does not prove that a demonstration succeeded. Exclude failures, occlusions, stale robot states, and episodes with the wrong task before conversion. Keep the remaining episode directories under `outputs/blue_cube`; the converter assigns train/validation by whole episode with seed 42.

The current source directory contains 31 episode folders, with an intentional numbering gap at `episode_0017`. The converter preserves the source names in `conversion_manifest.json` and does not require contiguous numbering.

## Convert the raw captures

Run from an Anaconda Prompt. This uses the known environment directly through its activation script:

```bat
call D:\Miniconda\Scripts\activate.bat lerobot
cd /d C:\Users\Administrator\Desktop\test\lerobot_nero_standalone
set PYTHONPATH=%CD%\src

python -X utf8 -m lerobot.nero_tools.convert_episodes --source outputs\blue_cube --output outputs\blue_cube_lerobot --task "Pick up the blue cube and put it into the red bag." --dataset-name nero_blue_cube
```

The converter refuses to write into an existing destination. It checks image dimensions and decodability, finite state/action values, timestamps, gripper range, and row alignment, then writes independent `train` and `val` LeRobot datasets. Do not add `--resume` for a fresh conversion. If the destination already exists, inspect it before choosing a new output path; do not delete it just to rerun conversion.

## Train ACT on the RTX 4090

After conversion completes successfully, run:

```bat
python -X utf8 -m lerobot.scripts.train --policy.type=act --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --policy.chunk_size=50 --policy.n_action_steps=50 --dataset.repo_id=local/nero_blue_cube_train --dataset.root=outputs\blue_cube_lerobot\train --batch_size=8 --num_workers=4 --steps=30000 --log_freq=100 --save_freq=5000 --eval_freq=0 --output_dir=outputs\train\nero_blue_cube_act_v1 --wandb.enable=false --seed=42
```

The default ACT ResNet-18 ImageNet backbone is retained. Its weights may need to download the first time if they are not already cached. Mixed precision is enabled for the CUDA GPU. Batch size 8 is the conservative first run for the 640x480 images; after confirming memory usage, a second run can try `--batch_size=16` with a new output directory such as `outputs\train\nero_blue_cube_act_v1_bs16`. Training refuses to reuse an existing output directory unless explicitly resuming.

The training process writes periodic checkpoints and a final checkpoint under `outputs\train\nero_blue_cube_act_v1\checkpoints`. It uses train-set statistics. The `val` split is held out by complete episode and is not used by this training command; review and evaluate against it separately after training. Do not use validation statistics for normalization.

## Resume an interrupted run

Only use this if the run directory contains a saved checkpoint. Replace the original training command with:

```bat
python -X utf8 -m lerobot.scripts.train --config_path=outputs\train\nero_blue_cube_act_v1\checkpoints\last\pretrained_model --resume=true
```

## Important limits

- Training completion is not evidence of successful manipulation. Review validation episodes and inspect predicted actions offline before any robot test.
- No command in this guide performs inference on, enables, or moves either arm.
- The dataset is a single-camera, single-task policy target: blue cube into the red bag. Do not combine green- and blue-cube episodes under one task label.
- Start with checkpoint inspection and offline prediction. Any later real-arm evaluation needs a separate low-speed, supported, supervised procedure with the physical emergency stop ready.
