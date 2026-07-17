@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" run_all.py --trials 10000 --skip-raw
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" validate_outputs.py

