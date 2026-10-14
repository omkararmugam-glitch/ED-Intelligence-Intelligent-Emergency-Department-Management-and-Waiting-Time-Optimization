@echo off
REM Double-click entry point for stopping the local demo.
REM
REM Hands over to scripts\stop_demo.ps1, which stops only the processes recorded
REM in runs\demo.pids and nothing else. It never touches Docker.
REM
REM %~dp0 is this file's own folder, so a double-click works from anywhere.

setlocal
set "SCRIPT_DIR=%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\stop_demo.ps1" %*
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% NEQ 0 (
  echo.
  echo Stop failed with exit code %EXITCODE%.
  echo The messages above say what happened.
  echo.
  pause
)

endlocal
exit /b %EXITCODE%
