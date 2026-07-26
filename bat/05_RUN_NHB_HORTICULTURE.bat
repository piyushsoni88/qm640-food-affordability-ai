@echo off
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python scripts\05_download_parse_nhb_horticulture.py --max-files 80
pause
