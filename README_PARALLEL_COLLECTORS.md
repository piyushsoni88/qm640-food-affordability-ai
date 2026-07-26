# QM640 Parallel Data Collectors

Copy the contents of this package into the repository root.

## Install

```powershell
cd <repository-root>
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_collectors.txt
```

Keep the data.gov.in key in `.env`:

```text
DATA_GOV_IN_API_KEY=your_key
```

## Run

Use `bat\00_RUN_ALL_IN_PARALLEL.bat`, or run each numbered BAT independently.
Each job writes a separate log under `data/logs/` and stores data in its intended source folder.

## Output map

| Job | Main destination |
|---|---|
| 01 MoSPI CPI/CFPI | `data/raw/mospi/` |
| 02 Consumer Affairs | `data/raw/consumer_affairs/`, `data/processed/consumer_affairs/` |
| 03 DES Agricultural Prices | `data/raw/des/agricultural_prices/`, `data/interim/des/agricultural_prices_tables/` |
| 04 Crop production | `data/raw/des/crop_production/`, `data/interim/des/crop_production/` |
| 05 NHB horticulture | `data/raw/nhb/`, `data/interim/nhb/tables/` |
| 06 IMD rainfall | `data/raw/imd/monthly/`, `data/metadata/imd/` |
| 07 AGMARKNET current | `data/raw/agmarknet/current/` |
| 08 State mandi | `data/raw/state_mandi/<state>/` |

## Important limitations

Government sites change HTML, links, certificates, and anti-bot controls. Each collector:
- retains raw source files;
- records a source manifest and SHA-256 checksum;
- logs failures separately;
- does not fabricate missing historical data.

The Consumer Affairs site may ignore a requested historical date. Validate `page_reported_date` before launching a long date range.

Public IMD pages primarily provide current products. Complete historical monthly data may require the IMD Data Service Portal or an official request.

PDF extraction is best-effort and must be manually validated before analysis.
