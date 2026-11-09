@echo off
setlocal
chcp 65001 >nul
title NERO Control Center
cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Conda was not found.
  echo Activate the "lerobot" environment and run:
  echo python -X utf8 examples\nero\control_center.py
  pause
  exit /b 1
)

echo Starting the local control center from conda environment "lerobot"...
conda run --no-capture-output -n lerobot python -X utf8 examples\nero\control_center.py
set "RESULT=%ERRORLEVEL%"

if not "%RESULT%"=="0" (
  echo.
  echo [ERROR] Control center exited with code %RESULT%.
  echo If teleoperation ran, keep both arms supported and verify updated feedback
  echo showing all seven joints and the AgxGripper disabled.
)

pause
exit /b %RESULT%
