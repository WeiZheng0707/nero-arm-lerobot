@echo off
setlocal
cd /d "%~dp0"

call D:\Miniconda\Scripts\activate.bat lerobot
if errorlevel 1 exit /b 1
set "PATH=D:\Miniconda\envs\lerobot;D:\Miniconda\envs\lerobot\Scripts;%PATH%"
set PYTHONPATH=%CD%\src;%CD%\tools
set PYTHONIOENCODING=utf-8

echo [info] PI0 red training uses the uncleaned eye-in-hand dataset.
echo [info] Dataset: outputs\eye_in_hand_red_cube_v2\train
echo [info] Output:  outputs\train\eye_in_hand_red_cube_pi0_v1
echo [info] This script starts training. Press Ctrl+C to stop safely.

"D:\Miniconda\envs\lerobot\python.exe" -X utf8 -m lerobot.scripts.train ^
  --policy.path=lerobot/pi0 ^
  --dataset.repo_id=local/nero_eye_in_hand_red_cube_train ^
  --dataset.root=outputs\eye_in_hand_red_cube_v2\train ^
  --policy.device=cuda ^
  --policy.push_to_hub=false ^
  --output_dir=outputs\train\eye_in_hand_red_cube_pi0_v1 ^
  --job_name=eye_in_hand_red_cube_pi0 ^
  --steps=30000 ^
  --batch_size=1 ^
  --num_workers=2 ^
  --save_freq=5000 ^
  --wandb.enable=false ^
  --seed=42

set EXIT_CODE=%ERRORLEVEL%
echo [done] exit=%EXIT_CODE%
endlocal & exit /b %EXIT_CODE%
