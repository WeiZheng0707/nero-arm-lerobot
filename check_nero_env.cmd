@echo off
REM ============================================================================
REM  NERO 环境体检 - 直接双击运行
REM
REM  一次跑完：Python 版本 -> 修复 editable 安装 -> 导入自检 -> 离线测试 -> 相机枚举
REM  全程不连接、不使能、不移动机械臂。
REM
REM  输出同时打印到屏幕并保存到 verify_output.txt，方便粘贴反馈。
REM ============================================================================

setlocal
chcp 65001 >nul
title NERO - environment check

call "%~dp0_nero_env.bat"
if errorlevel 1 (
  echo.
  pause
  exit /b 1
)

cd /d "%NERO_PROJECT%"
set "LOG=%NERO_PROJECT%\verify_output.txt"
set "PY=%NERO_PYTHON%"
set "PATH=%CONDA_ROOT%\envs\lerobot;%CONDA_ROOT%\envs\lerobot\Scripts;%CONDA_ROOT%\envs\lerobot\Library\bin;%PATH%"

> "%LOG%" echo NERO environment check
>> "%LOG%" echo conda=%CONDA_ROOT%
>> "%LOG%" echo python=%PY%
>> "%LOG%" echo project=%NERO_PROJECT%
>> "%LOG%" echo ------------------------------------------------------------

echo ============================================================
echo  [1/5] Python 版本与解释器路径
echo ============================================================
>> "%LOG%" echo [1/5] python version
"%PY%" -V 2>&1
"%PY%" -V >> "%LOG%" 2>&1
"%PY%" -c "import sys; print(sys.executable)" 2>&1
"%PY%" -c "import sys; print(sys.executable)" >> "%LOG%" 2>&1
echo.

echo ============================================================
echo  [2/5] 修复 lerobot editable 安装
echo ============================================================
echo 说明：当前 .pth 指向一个空的内层目录，不修则 import lerobot 失败。
>> "%LOG%" echo [2/5] pip install -e .
%PY% -m pip install -e . --no-deps 2>&1
%PY% -m pip install -e . --no-deps >> "%LOG%" 2>&1
echo.

echo ============================================================
echo  [3/5] 导入自检
echo ============================================================
>> "%LOG%" echo [3/5] imports
%PY% -c "import lerobot; print('lerobot ->', lerobot.__file__)" 2>&1
%PY% -c "import lerobot; print('lerobot ->', lerobot.__file__)" >> "%LOG%" 2>&1
%PY% -c "import pyAgxArm, can, draccus, torch; print('deps ok; torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1
%PY% -c "import pyAgxArm, can, draccus, torch; print('deps ok; torch', torch.__version__, 'cuda', torch.cuda.is_available())" >> "%LOG%" 2>&1
echo.

echo ============================================================
echo  [4/5] 离线测试（不连硬件）
echo ============================================================
>> "%LOG%" echo [4/5] pytest
%PY% -X utf8 -m pytest -q -p no:cacheprovider src\lerobot\nero_tools\test_episode_capture.py src\lerobot\nero_tools\test_color_detector.py src\lerobot\nero_tools\test_convert_episodes.py src\lerobot\motors\nero\test_teleoperation.py src\lerobot\robots\nero_follower\test_nero_follower.py src\lerobot\teleoperators\nero_leader\test_nero_leader.py 2>&1
%PY% -X utf8 -m pytest -q -p no:cacheprovider src\lerobot\nero_tools\test_episode_capture.py src\lerobot\nero_tools\test_color_detector.py src\lerobot\nero_tools\test_convert_episodes.py src\lerobot\motors\nero\test_teleoperation.py src\lerobot\robots\nero_follower\test_nero_follower.py src\lerobot\teleoperators\nero_leader\test_nero_leader.py >> "%LOG%" 2>&1
echo.

echo ============================================================
echo  [5/5] RealSense 相机枚举（只读，不连机械臂）
echo ============================================================
>> "%LOG%" echo [5/5] realsense devices
%PY% -X utf8 tools\realsense_bandwidth_check.py --list 2>&1
%PY% -X utf8 tools\realsense_bandwidth_check.py --list >> "%LOG%" 2>&1
echo.

echo ============================================================
echo  完成。完整输出已保存到：
echo    %LOG%
echo ============================================================
pause
exit /b 0
