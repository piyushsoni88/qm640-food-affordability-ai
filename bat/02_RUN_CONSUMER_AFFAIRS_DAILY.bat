@echo off
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
REM Start with a short range. Expand only after confirming page_reported_date matches requested_date.
python scripts\02_extract_consumer_affairs_daily.py --start 2025-01-01 --end 2025-01-07
pause
