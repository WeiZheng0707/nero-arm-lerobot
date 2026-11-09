@echo off
setlocal
cd /d "%~dp0"

set DATASET_ROOT=outputs/eye_in_hand_pick_place_lerobot\train
set OUTPUT_DIR=outputs/train/eye_in_hand_act
set DATASET_ID=local/nero_eye_in_hand_train

call D:\Miniconda\Scripts\activate.bat lerobot
if errorlevel 1 exit /b 1
set "PATH=D:\Miniconda\envs\lerobot;D:\Miniconda\envs\lerobot\Scripts;%PATH%"
set PYTHONPATH=%CD%\src

D:\Miniconda\envs\lerobot\python.exe -X utf8 -m lerobot.scripts.train ^
  --dataset.repo_id=%DATASET_ID% ^
  --dataset.root=%DATASET_ROOT% ^
  --policy.type=act ^
  --policy.device=cuda ^
  --output_dir=%OUTPUT_DIR% ^
  --job_name=eye_in_hand_act ^
  --steps=50000 ^
  --batch_size=8 ^
  --wandb.enable=false

endlocal
