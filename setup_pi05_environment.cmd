@echo off
setlocal
cd /d "%~dp0"
set "PI05_PY=D:/NERO/envs/pi05_lora/Scripts/python.exe"
if not exist "%PI05_PY%" (
  D:\Anaconda3\envs\lerobot\python.exe -m venv --system-site-packages D:\NERO\envs\pi05_lora
  if errorlevel 1 exit /b 1
)
"%PI05_PY%" -m pip install "transformers==4.57.1" "sentencepiece>=0.2,<0.3"
if errorlevel 1 exit /b 1
set "PYTHONPATH=%~dp0src;%~dp0tools"
"%PI05_PY%" -c "import torch, transformers; from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print('PI05 imports OK', transformers.__version__, torch.cuda.get_device_name(0))"
exit /b %errorlevel%
