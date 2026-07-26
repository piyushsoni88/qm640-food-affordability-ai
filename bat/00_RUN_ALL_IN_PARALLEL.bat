@echo off
setlocal
cd /d "%~dp0"
start "01 MoSPI" cmd /k 01_RUN_MOSPI_CPI_CFPI.bat
start "02 Consumer Affairs" cmd /k 02_RUN_CONSUMER_AFFAIRS_DAILY.bat
start "03 DES Prices" cmd /k 03_RUN_DES_AGRICULTURAL_PRICES_2006_2025.bat
start "04 Crop Production" cmd /k 04_RUN_CROP_PRODUCTION_UPAG_DES.bat
start "05 NHB" cmd /k 05_RUN_NHB_HORTICULTURE.bat
start "06 IMD" cmd /k 06_RUN_IMD_MONTHLY_RAINFALL.bat
start "07 AGMARKNET" cmd /k 07_RUN_AGMARKNET_CURRENT.bat
start "08 State Mandi" cmd /k 08_RUN_STATE_MANDI_ACCESSIBLE.bat
echo All collectors started in separate windows.
pause
