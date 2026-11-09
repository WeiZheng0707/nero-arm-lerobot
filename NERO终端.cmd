@echo off
REM ============================================================================
REM  NERO project terminal
REM
REM  Double-click to open a cmd window that is already:
REM    - using D:\miniconda\envs\lerobot
REM    - cd'ed into the project root
REM    - with PYTHONPATH set to <project>\src;<project>\tools
REM
REM  IMPORTANT: this file is deliberately pure ASCII. CMD reads a .bat/.cmd file
REM  using the ANSI code page before any chcp runs, so non-ASCII bytes here can
REM  corrupt parsing and crash the script. Keep it ASCII-only.
REM ============================================================================

setlocal

REM ---------- 1) locate project root (the folder that has src\lerobot) ----------
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

REM ---------- 2) activate the lerobot environment ----------
set "CONDA_ROOT=D:\miniconda"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
  echo [ERROR] not found: %CONDA_ROOT%\Scripts\activate.bat
  echo         Edit the CONDA_ROOT line in this file if conda lives elsewhere.
  echo.
  pause
  exit /b 1
)
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%\envs\lerobot"

REM ---------- 3) enter project, set paths ----------
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

REM ---------- 4) hand over to an interactive prompt ----------
cmd /k
