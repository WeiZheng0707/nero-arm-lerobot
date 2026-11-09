@echo off
setlocal
cd /d "%~dp0"

rem Pure NERO leader-follower teleoperation. No object recognition and no data recording.
rem Verified hardware identity: channel 0=v120 leader, channel 1=v121 follower.
set LEADER_CHANNEL=0
set FOLLOWER_CHANNEL=1
set RETURN_DURATION=5
set SPEED_PERCENT=15
set FPS=50
set POSE_FILE=examples/nero/saved_start_pose.json

call D:\Miniconda\Scripts\activate.bat lerobot
if errorlevel 1 exit /b 1
set "PATH=D:\Miniconda\envs\lerobot;D:\Miniconda\envs\lerobot\Scripts;%PATH%"
set PYTHONPATH=%CD%\src

echo.
echo NERO pure teleoperation
echo Startup sequence: connect -> move both arms to saved start pose -> enter leader drag mode.
echo Press Ctrl+C only when necessary; the program cleanup requests joint and gripper disable.
echo.

D:\Miniconda\envs\lerobot\python.exe -X utf8 -m lerobot.nero_teleoperate ^
  --interface agx_cando ^
  --leader-channel %LEADER_CHANNEL% ^
  --follower-channel %FOLLOWER_CHANNEL% ^
  --leader-firmware v120 ^
  --follower-firmware v121 ^
  --leader-angle-unit auto ^
  --angle-source auto ^
  --mapping relative ^
  --fps %FPS% ^
  --max-joint-speed 0.30 ^
  --max-gripper-speed 1.0 ^
  --max-control-dt 0.10 ^
  --speed-percent %SPEED_PERCENT% ^
  --gripper-mode width ^
  --leader-gripper-source auto ^
  --leader-gripper-width 0.10 ^
  --follower-gripper-width 0.07 ^
  --gripper-force 1.0 ^
  --return-to-start ^
  --pose-file %POSE_FILE% ^
  --return-duration %RETURN_DURATION%

set EXIT_CODE=%ERRORLEVEL%
echo.
echo Teleoperation exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
