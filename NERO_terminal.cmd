@echo off
REM ============================================================================
REM  NERO project terminal  (ASCII filename, ASCII content)
REM
REM  Double-click to open a cmd window already set up for this project:
REM    - using D:\miniconda\envs\lerobot
REM    - cd'ed into the project root
REM    - PYTHONPATH = <project>\src;<project>\tools
REM
REM  Same as NERO_terminal.cmd but with a pure-English filename, in case the
REM  Chinese filename is what is blocking the double-click.
REM ============================================================================

setlocal

set "PROBE=%~dp0"
if "%PROBE:~-1%"=="\" set "PROBE=%PROBE:~0,-1%"
set "NERO_PROJECT="
for /l %%I in (1,1,4) do (
  if not defined NERO_PROJECT (
    if exist "!PROBE!\src\lerobot\__init__.py" (
      set "NERO_PROJECT=!PROBE!"
    ) else (
      for %%P in ("!PROBE!\..") do set "PROBE=%%~fP"
    )
  )
)
endlocal & set "NERO_PROJECT=%NERO_PROJECT%" & set "PROBE=%PROBE%"

if not defined NERO_PROJECT (
  echo [ERROR] project root not found: no src\lerobot\__init__.py within 4 levels up.
  echo         Script location: %~dp0
  echo         Last folder tried: %PROBE%
  echo.
  pause
  exit /b 1
)

set "CONDA_ROOT=D:\miniconda"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
  echo [ERROR] not found: %CONDA_ROOT%\Scripts\activate.bat
  echo         Edit the CONDA_ROOT line in this file if conda lives elsewhere.
  echo.
  pause
  exit /b 1
)
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%\envs\lerobot"

cd /d "%NERO_PROJECT%"
set "PYTHONPATH=%NERO_PROJECT%\src;%NERO_PROJECT%\tools"
set "PYTHONIOENCODING=utf-8"
set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin;%PATH%"

echo ============================================================
echo  NERO project terminal
echo ------------------------------------------------------------
echo  Conda prefix : %CONDA_PREFIX%
echo  Project root : %NERO_PROJECT%
echo  PYTHONPATH   : %PYTHONPATH%
echo ============================================================
echo  Checking python (expected D:\miniconda\envs\lerobot\python.exe):
where python
python -c "import sys; print('  python  ->', sys.executable); print('  version ->', sys.version.split()[0])"
echo ============================================================
echo  Useful commands:
echo    python -X utf8 -m pytest -q -p no:cacheprovider src\lerobot\nero_tools\test_episode_capture.py
echo    python -X utf8 tools\realsense_bandwidth_check.py --list
echo    python -X utf8 -m lerobot.nero_tools.episode_capture --help
echo ============================================================
echo.

cmd /k
