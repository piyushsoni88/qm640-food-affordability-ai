@echo off
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python scripts\01_extract_mospi_cpi_cfpi.py --start 2021-01-01 --end 2025-12-31
pause
