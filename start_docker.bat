@echo off
REM Double-click entry point for the containerised demo.
REM
REM It does nothing itself: it hands over to scripts\docker_up.ps1, which checks
REM the engine, checks the ports, builds, starts and waits for both healthchecks.
REM -ExecutionPolicy Bypass is scoped to this one process, so it changes no
REM machine policy; without it a default Windows install refuses to run the .ps1.

setlocal
set SCRIPT_DIR=%~dp0

echo Starting ED Intelligence in Docker...
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\docker_up.ps1" -Open
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% NEQ 0 (
  echo.
  echo Startup failed with exit code %EXITCODE%.
  echo The messages above say what to fix.
)

echo.
echo This window can be closed; the containers keep running.
echo Stop them with scripts\docker_down.ps1
echo.
pause
endlocal
exit /b %EXITCODE%
