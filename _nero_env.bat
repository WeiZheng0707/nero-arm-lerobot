@echo off
REM ============================================================================
REM  NERO 环境定位器 - 被其它 .bat 调用，不要直接双击
REM
REM  作用：找到 Miniconda 根目录和 lerobot 环境的 python.exe，
REM        并把它们交给调用者（设置 CONDA_ROOT / NERO_PYTHON / NERO_PROJECT）。
REM
REM  为什么需要它：conda 没有初始化到 PATH，每次开新终端都要手打完整路径。
REM  本脚本按顺序尝试，找到即用，并把结果缓存到 conda_path.txt，
REM  以后换机器只要把项目搬到新位置重跑一次即可。
REM ============================================================================

setlocal EnableDelayedExpansion

REM --- 项目根目录：向上查找"含 src\lerobot\__init__.py 的那一层"。
REM     备份解压时多套了一层同名目录，内层是空的（没有 src），
REM     所以不能直接用 %~dp0，否则 PYTHONPATH 会指向不存在的 src。
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
if not defined NERO_PROJECT (
  echo [ERROR] 没找到项目根目录（需要存在 src\lerobot\__init__.py）。
  echo         从 %~dp0 向上找了 4 层都没有。
  exit /b 1
)
set "NERO_CACHE=%NERO_PROJECT%\.nero_conda_path.txt"

REM --- 1) 优先用上次记住的路径
if exist "%NERO_CACHE%" (
  for /f "usebackq delims=" %%P in ("%NERO_CACHE%") do set "CACHED=%%P"
  if defined CACHED if exist "!CACHED!\envs\lerobot\python.exe" (
    set "CONDA_ROOT=!CACHED!"
    goto :found
  )
)

REM --- 2) 依次尝试常见安装位置（Windows 路径不区分大小写，Miniconda == miniconda）
for %%R in (
  "D:\Anaconda3"
  "D:\anaconda3"
  "D:\miniconda"
  "D:\Miniconda3"
  "D:\miniconda3"
  "C:\Anaconda3"
  "C:\anaconda3"
  "C:\miniconda"
  "C:\miniconda3"
  "C:\ProgramData\miniconda3"
  "C:\ProgramData\Miniconda3"
  "C:\ProgramData\Anaconda3"
  "%USERPROFILE%\miniconda3"
  "%USERPROFILE%\Miniconda3"
  "%USERPROFILE%\anaconda3"
  "%USERPROFILE%\Anaconda3"
  "%LOCALAPPDATA%\miniconda3"
  "%LOCALAPPDATA%\Continuum\miniconda3"
) do (
  if not defined CONDA_ROOT if exist "%%~R\envs\lerobot\python.exe" set "CONDA_ROOT=%%~R"
)

REM --- 3) 用 where 从 PATH 里找（conda.exe 通常在 <根>\Scripts 或 <根>\condabin）
if not defined CONDA_ROOT (
  for /f "delims=" %%P in ('where conda 2^>nul') do (
    if not defined CONDA_ROOT (
      call :probe_root "%%~dpP"
      if defined NERO_PROBE (
        set "CONDA_ROOT=!NERO_PROBE!"
        set "NERO_PROBE="
      )
    )
  )
)

REM --- 4) 全部失败则报错退出
if not defined CONDA_ROOT (
  echo [ERROR] 找不到带 lerobot 环境的 Conda 安装。
  echo.
  echo   需要存在： ^<Conda根目录^>\envs\lerobot\python.exe
  echo   已尝试： D:\Anaconda3、D:\Miniconda3、C:\Anaconda3、用户目录等常见位置。
  echo.
  echo   两个办法：
  echo     1) 在本脚本第 47 行附近的列表里加入你的 Conda 根目录
  echo     2) 直接创建文件 .nero_conda_path.txt，写一行 Conda 根目录（如 D:\Anaconda3）
  exit /b 1
)

:found
REM --- 规范化：去掉可能残留的尾部反斜杠
if "%CONDA_ROOT:~-1%"=="\" set "CONDA_ROOT=%CONDA_ROOT:~0,-1%"
set "NERO_PYTHON=%CONDA_ROOT%\envs\lerobot\python.exe"

if not exist "%NERO_PYTHON%" (
  echo [ERROR] 找到 Conda 根目录 "%CONDA_ROOT%"，但里面没有 envs\lerobot。
  exit /b 1
)

REM --- 记住结果，下次秒开（注意 echo 与 > 之间不能有空格，否则会写进文件）
>"%NERO_CACHE%" echo %CONDA_ROOT%

REM --- 把结果交给调用者。必须先 ENDLOCAL 再 set 到外层作用域，
REM     且不能包在括号块里（块内的 % 会提前展开，取到的是旧值）。
ENDLOCAL
set "CONDA_ROOT=%CONDA_ROOT%"
set "NERO_PYTHON=%NERO_PYTHON%"
set "NERO_PROJECT=%NERO_PROJECT%"
set "PYTHONPATH=%NERO_PROJECT%\src;%NERO_PROJECT%\tools"
set "PYTHONIOENCODING=utf-8"
exit /b 0

:probe_root
REM %1 = conda.exe 所在目录（尾部带反斜杠）。
REM 子程序里 set 的值会随 endlocal 丢失，所以这里直接写到调用者的外层。
REM Scripts\conda.exe -> 根目录是 "%~1.."；condabin\conda.exe -> 根目录也是 "%~1.."。
for %%D in ("%~1..") do (
  if exist "%%~fD\envs\lerobot\python.exe" set "NERO_PROBE=%%~fD"
)
exit /b 0
