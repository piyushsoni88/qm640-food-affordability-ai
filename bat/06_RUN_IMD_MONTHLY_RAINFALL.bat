@echo off
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python scripts\06_collect_imd_monthly_rainfall.py
pause
