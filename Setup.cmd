@echo off
REM Double-click entry point for a fresh machine. Needed once, not every time.
REM
REM It hands over to scripts\setup.ps1, which finds Python 3.12, creates .venv and
REM installs the pinned dependencies from prebuilt wheels only.
REM
REM %~dp0 is this file's own folder, so a double-click works from anywhere.
REM -ExecutionPolicy Bypass is scoped to this one process and changes no machine
REM policy; without it a default Windows install refuses to run the .ps1 at all.

setlocal
set "SCRIPT_DIR=%~dp0"

echo Setting up ED Intelligence. This installs into .venv and takes a few minutes.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\setup.ps1" %*
set EXITCODE=%ERRORLEVEL%

echo.
if %EXITCODE% NEQ 0 (
  echo Setup failed with exit code %EXITCODE%.
  echo The messages above say what to fix. The usual cause is that Python 3.12
  echo is not installed, or was installed without "Add python.exe to PATH".
) else (
  echo Next: double-click Start-Demo.cmd
)

echo.
REM Always pause: this is run once, by someone who needs to read the outcome before
REM the window disappears.
pause

endlocal
exit /b %EXITCODE%
