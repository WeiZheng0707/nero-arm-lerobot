@echo off
setlocal
cd /d "%~dp0"
if /I not "%~1"=="a" if /I not "%~1"=="b" (
  echo Usage: train_base14_short.cmd a OR train_base14_short.cmd b
  exit /b 2
)
set "PYTHONPATH=%~dp0src;%~dp0tools"
set "PYTHONIOENCODING=utf-8"
set "HF_DATASETS_CACHE=D:/NERO/cache/datasets"
python -X utf8 -m lerobot.scripts.train --config_path=configs/train_base14_%~1_2k.json
exit /b %errorlevel%
