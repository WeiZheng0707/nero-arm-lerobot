@echo off
setlocal
cd /d "%~dp0"
if not "%~1"=="a" if not "%~1"=="b" (
  echo Usage: train_base14_30k.cmd a OR train_base14_30k.cmd b
  exit /b 2
)
set "PYTHONPATH=%~dp0src;%~dp0tools"
set "PYTHONIOENCODING=utf-8"
set "HF_DATASETS_CACHE=D:/NERO/cache/datasets"
set "TRAIN_CONFIG=D:/NERO/outputs/train/base14_%~1_2k_v1/checkpoints/last/pretrained_model/train_config.json"
if not exist "%TRAIN_CONFIG%" (
  echo Missing checkpoint config: %TRAIN_CONFIG%
  exit /b 1
)
python -X utf8 -m lerobot.scripts.train --config_path="%TRAIN_CONFIG%" --resume=true --steps=30000 --save_freq=5000 --log_freq=100
exit /b %errorlevel%
