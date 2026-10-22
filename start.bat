@echo off
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
.venv\Scripts\python.exe -m compare_face %*
if errorlevel 1 pause
