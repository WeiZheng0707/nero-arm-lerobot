@echo off
REM Pure ASCII diagnostic. No Chinese characters, no chcp. Safe to double-click.
setlocal

echo ============================================================
echo  NERO ENV DIAGNOSTIC
echo ============================================================
echo  Script dir : %~dp0
echo.

echo --- [1] conda root check
if exist "D:\miniconda\Scripts\activate.bat" (
  echo  OK   D:\miniconda\Scripts\activate.bat
) else (
  echo  FAIL D:\miniconda\Scripts\activate.bat NOT FOUND
)
if exist "D:\miniconda\envs\lerobot\python.exe" (
  echo  OK   D:\miniconda\envs\lerobot\python.exe
) else (
  echo  FAIL D:\miniconda\envs\lerobot\python.exe NOT FOUND
)
echo.

echo --- [2] find project root (dir containing src\lerobot\__init__.py)
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
if defined NERO_PROJECT (
  echo  OK   project root = %NERO_PROJECT%
) else (
  echo  FAIL no project root found. last probe = %PROBE%
)
echo.

echo --- [3] python version (direct path, no activate)
"D:\miniconda\envs\lerobot\python.exe" -V
echo.

echo --- [4] import lerobot
"D:\miniconda\envs\lerobot\python.exe" -c "import lerobot; print('lerobot ->', lerobot.__file__)"
echo.

echo --- [5] activate.bat test
call "D:\miniconda\Scripts\activate.bat" "D:\miniconda\envs\lerobot"
echo  CONDA_PREFIX = %CONDA_PREFIX%
where python
echo.

echo ============================================================
echo  DONE. Copy everything above and send it back.
echo ============================================================
pause
