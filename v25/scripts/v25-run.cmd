@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0v25-run.ps1" %*
exit /b %errorlevel%
