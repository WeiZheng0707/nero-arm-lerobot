@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "COLOUR=%~1"
if not defined COLOUR set "COLOUR=red"
set "STEPS=%~2"
if not defined STEPS set "STEPS=30000"
set "RUN_KIND=full"
if not "%STEPS%"=="30000" set "RUN_KIND=smoke_%STEPS%"

call D:\Miniconda\Scripts\activate.bat lerobot
if errorlevel 1 exit /b 1
set "PATH=D:\Miniconda\envs\lerobot;D:\Miniconda\envs\lerobot\Scripts;%PATH%"
set "PYTHONPATH=%CD%\src;%CD%\tools"
set "PYTHONIOENCODING=utf-8"

if /i "%COLOUR%"=="blue" goto blue
if /i "%COLOUR%"=="red" goto red
echo ERROR: colour must be red or blue.
exit /b 2

:red
set "DS_REPO=local/nero_eye_in_hand_red_cube_train"
set "DS_ROOT=outputs\eye_in_hand_red_cube_v2\train"
set "OUTDIR=outputs\train\eye_in_hand_red_pi05_lora_%RUN_KIND%"
set "JOB=eye_in_hand_red_pi05_lora_%RUN_KIND%"
goto check

:blue
set "DS_REPO=local/nero_eye_in_hand_blue_cube_train"
set "DS_ROOT=outputs\eye_in_hand_blue_cube_v1\train"
set "OUTDIR=outputs\train\eye_in_hand_blue_pi05_lora_%RUN_KIND%"
set "JOB=eye_in_hand_blue_pi05_lora_%RUN_KIND%"
goto check

:check
if exist "%OUTDIR%" (
  echo ERROR: output directory already exists: %OUTDIR%
  echo Choose another run name or inspect the existing output before proceeding.
  exit /b 3
)

D:\Miniconda\envs\lerobot\python.exe -X utf8 tools\pi05_check_env.py --dataset-root "%DS_ROOT%"
if errorlevel 1 exit /b 4

echo Starting PI05 LoRA run: %COLOUR%, steps=%STEPS%, output=%OUTDIR%
D:\Miniconda\envs\lerobot\python.exe -u -X utf8 -m lerobot.scripts.train ^
  --policy.path=model/pi_05_base ^
  --policy.device=cuda ^
  --policy.dtype=bfloat16 ^
  --policy.push_to_hub=false ^
  --policy.train_expert_only=true ^
  --policy.freeze_vision_encoder=true ^
  --policy.lora_enabled=true ^
  --policy.lora_rank=8 ^
  --policy.lora_alpha=16 ^
  --policy.lora_dropout=0.05 ^
  --policy.empty_cameras=2 ^
  --policy.quantile_stats_path=%DS_ROOT%\meta\quantile_stats.json ^
  --policy.chunk_size=50 ^
  --policy.n_action_steps=10 ^
  --policy.optimizer_lr=1e-4 ^
  --policy.optimizer_weight_decay=1e-4 ^
  --policy.scheduler_decay_lr=1e-5 ^
  --policy.gradient_checkpointing=true ^
  --dataset.repo_id=%DS_REPO% ^
  --dataset.root=%DS_ROOT% ^
  --output_dir=%OUTDIR% ^
  --job_name=%JOB% ^
  --steps=%STEPS% ^
  --batch_size=1 ^
  --num_workers=0 ^
  --save_freq=1000 ^
  --log_freq=20 ^
  --seed=42 ^
  --wandb.enable=false

set "EXIT_CODE=%ERRORLEVEL%"
echo Training exit code: %EXIT_CODE%
endlocal & exit /b %EXIT_CODE%
