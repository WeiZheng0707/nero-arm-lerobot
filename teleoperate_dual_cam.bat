@echo off
chcp 65001 > nul
REM ============================================================================
REM  双机位遥操作（腕部 + base），带恢复点归位
REM
REM  通道映射是 2026-10-27 实测确认的，不要凭记忆改：
REM      channel 1 = 主臂 leader  （固件 1.20）
REM      channel 0 = 从臂 follower（固件 1.21）
REM  相机 index 1 和 3（不是 0/1），640x480 YUY2@30
REM
REM  恢复点用的是 examples\nero\saved_start_pose.json —— 10-16 那版，
REM  leader / follower 存的是同一组坐标，j7 = 45deg。
REM
REM  ⚠️ 运行时注意：
REM    - 使能后**两臂都会自动走到恢复点**（先从臂、再主臂），已放行 60deg 上限
REM    - 主臂走的时候会自己动，手不要放在主臂上，也别站在它的运动范围里
REM    - 会先打印两行「归位预览: 单轴最大 XX deg」，数字离谱就按 Q 或拍急停
REM    - 归位完主臂切回零力拖拽，然后才开始跟随
REM    - 退出：点一下预览窗口让它获得焦点，按 Q 或 Esc；
REM            在控制台窗口按 Q 也可以
REM    - 没有时间限制，想跑多久跑多久
REM    - 想追加参数直接在本窗口传，会透传进去
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
REM --- 切到项目根目录，让后面的相对路径（examples\...）始终指向本项目
cd /d "%NERO_PROJECT%"

echo.
echo ============================================================
echo  解释器: %NERO_PYTHON%
echo  主臂: channel 1 (v120)    从臂: channel 0 (v121)
echo  相机: index 1 (wrist) + index 3 (base)，双摄并排
echo  速度: 1.2 rad/s, speed-percent 60, fps 120
echo  退出: 预览窗口按 Q，或控制台按 Q。无时间限制。
echo ============================================================
echo.
echo  确认两臂已支撑、工作区清空、急停在手，然后按键开始。
echo.
pause

"%NERO_PYTHON%" -X utf8 examples\nero\teleoperate.py ^
  --interface agx_cando ^
  --leader-channel 1 --follower-channel 0 ^
  --leader-firmware v120 --follower-firmware v121 ^
  --leader-gripper-width 0.10 --follower-gripper-width 0.07 --gripper-force 1.0 ^
  --speed-percent 60 --fps 120 --max-joint-speed 1.2 --max-gripper-speed 2.5 ^
  --camera-index 1 3 --camera-labels wrist base --preview-fps 30 ^
  --return-to-start ^
  --allow-large-return ^
  --pose-file examples\nero\saved_start_pose.json --return-duration 8 ^
  %*

echo.
echo [退出] 遥操作进程已结束，两臂应已失能。
pause
