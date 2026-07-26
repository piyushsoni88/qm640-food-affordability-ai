@echo off
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python scripts\07_collect_agmarknet_current.py
pause
