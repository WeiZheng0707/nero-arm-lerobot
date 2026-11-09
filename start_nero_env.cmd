@echo off
REM ============================================================================
REM  NERO 一键启动环境 - 直接双击运行
REM
REM  效果：打开一个已经激活 lerobot 环境、已设好 PYTHONPATH 的 CMD 窗口，
REM        在里面直接可以跑 python / pytest / 采集命令，不用再手打路径。
REM
REM  主要命令速查（进入后直接复制粘贴）：
REM    python -c "import lerobot; print(lerobot.__file__)"
REM    python -X utf8 -m pytest -q -p no:cacheprovider src\lerobot\nero_tools\test_episode_capture.py
REM    python -X utf8 tools\realsense_bandwidth_check.py --list
REM ============================================================================

setlocal
chcp 65001 >nul
title NERO - lerobot environment

REM 定位 Conda 与项目路径
call "%~dp0_nero_env.bat"
if errorlevel 1 (
  echo.
  pause
  exit /b 1
)
if not defined NERO_PYTHON (
  echo [ERROR] 环境定位脚本没有返回 NERO_PYTHON。
  pause
  exit /b 1
)

cd /d "%NERO_PROJECT%"

echo ============================================================
echo  NERO 环境已就绪
echo ------------------------------------------------------------
echo  Conda 根目录 : %CONDA_ROOT%
echo  Python       : %NERO_PYTHON%
echo  项目目录     : %NERO_PROJECT%
echo  PYTHONPATH   : %PYTHONPATH%
echo ============================================================
echo.

REM 验证导入能否正常工作（editable 安装指向错误目录时这里会立刻暴露）
"%NERO_PYTHON%" -c "import lerobot, sys; print('lerobot ->', lerobot.__file__); print('python  ->', sys.version.split()[0])" 2>nul
if errorlevel 1 (
  echo.
  echo [WARNING] import lerobot 失败。editable 安装可能指向了错误目录。
  echo 修复方法：
  echo     "%NERO_PYTHON%" -m pip install -e . --no-deps
  echo.
)

REM 让 PATH 优先使用本环境的可执行文件，这样 python / pip 指向正确版本
set "PATH=%CONDA_ROOT%\envs\lerobot;%CONDA_ROOT%\envs\lerobot\Scripts;%CONDA_ROOT%\envs\lerobot\Library\bin;%PATH%"

echo 现在可以运行命令了。输入 exit 关闭窗口。
echo.
cmd /k
