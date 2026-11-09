# Green + Blue Cube ACT Training

This procedure combines the existing green-cube and blue-cube demonstrations into one local LeRobot dataset, then trains one ACT policy on an RTX 4090. It does not start training, perform inference on a robot, or connect to CAN hardware.

## Task definition

Both colors use the same task text: `Pick up the cube and put it into the red bag.` The current ACT policy consumes the camera image and robot state; it does not consume language instructions. It can learn the shared behavior for cubes in either color, but this setup cannot be prompted to select one color while ignoring the other.

## Review the demonstrations

Review both raw directories before conversion:

- `outputs/green_cube`
- `outputs/blue_cube`

Both currently contain 31 episode folders. Green has three `manual_end` episodes; blue currently has only `auto_end` episodes. The converter accepts those two result labels, but an `auto_end` or `manual_end` label does not mean a successful demonstration. Remove or move unsuccessful, occluded, stale-state, wrong-object, or otherwise unsuitable episodes out of the source directories before conversion. The converter will reject `quit_end` episodes and malformed rows.

The conversion is stratified by source directory: with all 31 episodes retained in each directory and seed 42, it assigns 25 episodes of each color to train and 6 of each color to validation. Splitting whole episodes avoids placing frames from one trajectory in both sets. The original source data is read-only to the converter and is not modified.

## Convert both sources

In Anaconda Prompt:

```bat
call D:\Miniconda\Scripts\activate.bat lerobot
cd /d C:\Users\Administrator\Desktop\test\lerobot_nero_standalone
set PYTHONPATH=%CD%\src

python -X utf8 -m lerobot.nero_tools.convert_episodes --source outputs\green_cube --source outputs\blue_cube --output outputs\green_blue_cube_lerobot --task "Pick up the cube and put it into the red bag." --dataset-name nero_green_blue_cube
```

The destination is new and separate from both raw datasets and the existing green-only dataset. The converter refuses to write if `outputs\green_blue_cube_lerobot` already exists. If a previous attempt created that directory, inspect it and choose a new output name such as `outputs\green_blue_cube_lerobot_v2`; do not delete it blindly.

On success, the combined dataset is written to:

```text
outputs\green_blue_cube_lerobot\train
outputs\green_blue_cube_lerobot\val
outputs\green_blue_cube_lerobot\conversion_manifest.json
```

The manifest records the source directory and source episode for every converted episode, making the color mix auditable. Confirm that both splits contain both `green_cube` and `blue_cube` entries before training.

## Train on the RTX 4090

After conversion completes successfully, run from the same activated Anaconda Prompt and project directory:

```bat
python -X utf8 -m lerobot.scripts.train --policy.type=act --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --policy.chunk_size=50 --policy.n_action_steps=50 --dataset.repo_id=local/nero_green_blue_cube_train --dataset.root=outputs\green_blue_cube_lerobot\train --batch_size=8 --num_workers=4 --steps=30000 --log_freq=100 --save_freq=5000 --eval_freq=0 --output_dir=outputs\train\nero_green_blue_cube_act_v1 --wandb.enable=false --seed=42
```

The environment was checked to have an NVIDIA GeForce RTX 4090 with CUDA 12.8. The command uses the pretrained ResNet-18 ImageNet vision backbone and mixed precision. The backbone weights may need to download once if not cached. Batch size 8 is the conservative first run for 640x480 images. After checking GPU memory and training stability, a separate run can try batch size 16 with a new output directory, for example `outputs\train\nero_green_blue_cube_act_v1_bs16`.

Check the printed configuration before the run begins. It should show the combined train root, `device: cuda`, `use_amp: true`, and the intended new output directory. Training outputs and checkpoints go under `outputs\train\nero_green_blue_cube_act_v1`; the existing green-only model and data are not overwritten.

The built-in training command trains on `train` only. `--eval_freq=0` prevents it from trying to evaluate in a simulation environment; it does not evaluate the held-out `val` dataset. Do not treat training loss as validation performance. Use the manifest and val split for a separate offline evaluation before considering deployment. Reuse normalization statistics from the training dataset, not validation statistics.

## Controlled optimization experiment

If a run needs an explicit gradient clip, disable the policy preset. In this repository, the ACT preset otherwise overwrites the optimizer configuration during validation. The scheduler is optional for ACT; omitting it keeps the learning rate constant. After the configuration fix in `src/lerobot/configs/train.py`, a real clip-1.0 comparison can be run with:

```bat
python -X utf8 -m lerobot.scripts.train --policy.type=act --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --policy.chunk_size=30 --policy.n_action_steps=10 --dataset.repo_id=local/nero_green_blue_cube_train --dataset.root=outputs\green_blue_cube_lerobot\train --batch_size=8 --num_workers=4 --steps=20000 --log_freq=100 --save_freq=5000 --eval_freq=0 --use_policy_training_preset=false --optimizer.type=adamw --optimizer.lr=1e-5 --optimizer.weight_decay=1e-4 --optimizer.grad_clip_norm=1.0 --output_dir=outputs\train\nero_green_blue_cube_act_clip1_chunk30_20k --wandb.enable=false --seed=42
```

Do not compare this run using training loss alone. Reuse the same 588-window offline evaluation protocol and compare first-step MAE, p95, gripper out-of-range rate, per-colour gap, and action-jump metrics. Keep the existing 20k checkpoint as the current model candidate until this controlled run demonstrates a validation improvement.

## Resume an interrupted run

Use this only if the run has already produced a checkpoint under `checkpoints\last`:

```bat
python -X utf8 -m lerobot.scripts.train --config_path=outputs\train\nero_green_blue_cube_act_v1\checkpoints\last\pretrained_model --resume=true
```

## Before any robot test

Training success does not establish task success or safe actions. First inspect predictions offline: all eight outputs must be finite, joint values must be radians, and gripper values must remain within 0..1. Then review successful and failed validation episodes. Any later real-arm evaluation requires a separate low-speed supervised procedure, supported arms, a clear workspace, and a working physical emergency stop. No command in this document enables or moves an arm.
