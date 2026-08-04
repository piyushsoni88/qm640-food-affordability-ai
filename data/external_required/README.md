# Required official datasets: acquisition status

Extraction attempt date: 2026-07-30 (Asia/Calcutta)

This directory is an auditable record of an incomplete acquisition attempt. None
of the five required derived datasets is present, and none is claimed complete.
No synthetic data, proxy substitutions, or empty files using the requested
dataset filenames were created.

## Confirmed blockers

1. The execution environment cannot establish direct HTTPS connections to the
   official portals. Both sandboxed and approved out-of-sandbox attempts to
   `enam.gov.in` timed out.
2. The in-app browser runtime failed to start because Windows denied access to
   the application profile (`EPERM` while resolving `C:\Users\piyus\AppData`).
3. e-NAM historical trade data and DES APY are interactive query/export
   dashboards, so their complete exports could not be generated through the
   search-only web fallback.
4. The Labour Bureau rural-wages page is public, but the full response exceeds
   the web-fetch size limit. Direct download was unavailable.
5. MoSPI requires registered-user login for microdata downloads. The repository
   has local credentials in `.env`, but they were neither displayed nor copied.
   Without a working authenticated browser or direct HTTPS access, the HCES
   archive could not be downloaded.

## Official sources and units

| Dataset | Official source | Units and coverage verified from official metadata |
|---|---|---|
| e-NAM historical arrivals/trades | https://enam.gov.in/web/dashboard/Historical | Arrival quantity, traded quantity, portal-reported unit, and min/modal/max price; state/APMC/commodity/date filters |
| DES state-crop APY | https://data.desagri.gov.in/website/crops-apy-report-web | State, district, crop, season and year reports; area, production and yield in report-selected units |
| Labour Bureau rural wages | https://labourbureau.gov.in/en/rural-wages | State/occupation/item monthly wages for men and women; INR per normalized 8-hour working day |
| HCES 2022-23 | https://microdata.gov.in/NADA/index.php/catalog/224 | Public-use unit-level survey files; expenditure in INR and survey multipliers/weights per MoSPI survey design |
| MoSPI CPI/CFPI | https://www.cpi.mospi.gov.in/ | Monthly CPI/CFPI index points; current series base 2012=100; rural, urban and combined sectors |

## Existing local evidence not used as a substitute

The repository already preserves an official data.gov.in AGMARKNET price
archive under `data/raw/agmarknet/historical_8_commodities/`. Its validated
snapshot contains 18,836,462 source rows in 192 gzip partitions, spans
2001-01-10 through 2026-07-28, contains 34 state/label values and eight
commodities, and has the columns:

`Arrival_Date, Commodity, Commodity_Code, District, Grade, Market, Max_Price,
Min_Price, Modal_Price, State, Variety`

Despite the field name `Arrival_Date`, these files do not include arrival
quantity, traded quantity, or quantity unit. They cannot satisfy the requested
e-NAM/AGMARKNET arrivals dataset and were not relabelled.

The repository also contains official DES agricultural-price PDFs. These are
not state-crop area/production/yield data and were not used as APY substitutes.
No FAOSTAT, World Bank, NASA, aggregate HCES, or other proxy data were used to
populate the missing official outputs.

## Validation rule

A dataset may be marked complete only when its required gzip CSV exists and
passes all of the following:

- exact required columns are present;
- rows are non-empty and derived from the stated official source;
- duplicates and key duplicates are reported;
- missingness is calculated by column;
- dates fall within the released portion of 2006-2025 plus 2026 YTD (HCES is
  survey-year specific);
- state/UT and commodity/group coverage is enumerated;
- units are explicit and internally consistent;
- file size and SHA-256 checksum are recorded.

`download_manifest.csv` and `data_quality_summary.csv` deliberately retain
blocked statuses and blank checksums for absent outputs.
