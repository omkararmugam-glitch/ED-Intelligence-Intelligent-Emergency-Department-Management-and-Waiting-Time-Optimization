@echo off
REM Double-click entry point for the local demo.
REM
REM This file does nothing itself. It hands over to scripts\start_demo.ps1, which
REM checks the environment, starts the API and the dashboard, waits for both and
REM opens the browser.
REM
REM %~dp0 is this file's own folder, with a trailing backslash, so a double-click
REM works regardless of what the current directory happens to be -- which is not
REM the project folder when Explorer or a shortcut launches it.
REM
REM -ExecutionPolicy Bypass is scoped to this one process. It changes no machine
REM policy; without it a default Windows install refuses to run the .ps1 at all.
REM -NoProfile keeps someone's PowerShell profile from changing how this behaves.

setlocal
set "SCRIPT_DIR=%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\start_demo.ps1" %*
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% NEQ 0 (
  echo.
  echo Startup failed with exit code %EXITCODE%.
  echo The messages above say what to fix.
  echo.
  REM Only on failure. A successful start should not leave a window demanding a
  REM keypress before the person can go and look at the dashboard.
  pause
) else (
  echo This window can be closed; the demo keeps running.
  echo Stop it with Stop-Demo.cmd
)

endlocal
exit /b %EXITCODE%
