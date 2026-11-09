@echo off
chcp 65001 > nul
REM ============================================================================
REM  只读核对：当前两臂姿态 vs examples\nero\saved_start_pose.json
REM
REM  - 不使能、不改模式、不动臂，只读反馈
REM  - 会告诉你从臂到恢复点每轴要走多少度，以及会不会被 60deg 安全闸拦下
REM  - 正式跑遥操作之前，先双击这个看一眼
REM ============================================================================

call "%~dp0_nero_env.bat"

REM --- 兜底：万一环境探测的变量没传出来，直接用已验证存在的路径
if not defined NERO_PYTHON (
  if exist "D:\Anaconda3\envs\lerobot\python.exe" set "NERO_PYTHON=D:\Anaconda3\envs\lerobot\python.exe"
)
if not defined NERO_PROJECT set "NERO_PROJECT=%~dp0"
if "%NERO_PROJECT:~-1%"=="\" set "NERO_PROJECT=%NERO_PROJECT:~0,-1%"

if not defined NERO_PYTHON (
  echo.
  echo [ABORT] 没找到 lerobot 环境的 python.exe，见上面提示。
  pause
  exit /b 1
)
if not exist "%NERO_PYTHON%" (
  echo.
  echo [ABORT] NERO_PYTHON 指向的路径不存在: %NERO_PYTHON%
  pause
  exit /b 1
)

REM --- 统一设置（无论环境探测是否成功都覆盖一遍，避免继承到旧值）
set "PYTHONPATH=%NERO_PROJECT%\src;%NERO_PROJECT%\tools"
set "PYTHONIOENCODING=utf-8"
REM --- 切到项目根目录，让后面的相对路径（tools\...）始终指向本项目
cd /d "%NERO_PROJECT%"

echo.
echo ============================================================
echo  当前解释器: %NERO_PYTHON%
echo  仅读取姿态，不会使能、不会动臂
echo ============================================================
echo.

"%NERO_PYTHON%" -X utf8 tools\save_start_pose.py ^
  --leader-channel 1 --follower-channel 0 ^
  --leader-firmware v120 --follower-firmware v121 ^
  --compare

echo.
pause
