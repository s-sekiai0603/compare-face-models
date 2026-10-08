@echo off
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
.venv\Scripts\python.exe -m verify_reid
if errorlevel 1 pause
