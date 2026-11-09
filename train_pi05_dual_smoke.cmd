@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%~dp0tools"
set "PYTHONIOENCODING=utf-8"
set "HF_DATASETS_CACHE=D:/NERO/cache/datasets"
D:\NERO\envs\pi05_lora\Scripts\python.exe -u -X utf8 tools\train_pi05_dual.py
exit /b %errorlevel%
