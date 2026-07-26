@echo off
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python scripts\08_collect_state_mandi_accessible.py
pause
