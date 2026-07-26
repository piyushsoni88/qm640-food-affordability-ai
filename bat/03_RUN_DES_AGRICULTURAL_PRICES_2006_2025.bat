@echo off
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python scripts\03_download_parse_des_agricultural_prices.py --start-year 2006 --end-year 2025
pause
