@echo off
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python scripts\04_collect_crop_production_upag_des.py --max-files 100
pause
