# Curated interim datasets

These compressed CSV files are the evaluator-accessible analytical layer. They
are ordinary gzip-compressed CSVs and can be read directly with pandas:

```python
import pandas as pd

panel = pd.read_csv(
    "data/curated/india_food_affordability_panel_15x8_2005_2025.csv.gz"
)
```

| File | Rows | Description |
|---|---:|---|
| `faostat_india_consumer_price_indices.csv.gz` | 924 | All India rows extracted from the FAOSTAT CPI bulk archive |
| `faostat_india_producer_prices.csv.gz` | 5,665 | All India rows extracted from the FAOSTAT producer-price archive |
| `faostat_india_crop_production.csv.gz` | 26,432 | All India rows extracted from the FAOSTAT crop/livestock archive |
| `nasa_power_india_15_regions_monthly_2005_2025.csv.gz` | 3,780 | Five monthly climate variables for 15 selected Indian regional points |
| `world_bank_pink_sheet_food_energy_monthly.csv.gz` | 792 | Selected monthly international food and energy benchmarks |
| `india_commodity_annual_production_prices.csv.gz` | 512 | Commodity-mapped annual India production and producer-price features |
| `india_food_affordability_national_monthly.csv.gz` | 252 | National monthly modeling table with lags and exogenous predictors |
| `india_food_affordability_panel_15x8_2005_2025.csv.gz` | 30,240 | Date-region-commodity integration panel |
| `nasa_power_india_all_states_uts_daily_2000_2026_ytd.csv.gz` | 349,272 | Daily climate observations for 28 states and eight union territories through 24 July 2026 |
| `nasa_power_india_all_states_uts_monthly_2000_2026_ytd.csv.gz` | 11,484 | Monthly climate aggregates with partial-period flags |
| `india_food_affordability_national_monthly_2000_2026_ytd.csv.gz` | 319 | Expanded national monthly feature table |
| `india_food_affordability_panel_36x8_2000_2026_ytd.csv.gz` | 91,872 | All-state/UT, eight-commodity integration panel |
| `agmarknet_official_daily_snapshot_2026-07-28.csv.gz` | 3,824 | Official data.gov.in market-price snapshot covering 27 states and 26 returned commodity labels |

The 30,240-row panel combines observed region-specific NASA climate with
national FAOSTAT and World Bank series. National variables are explicitly
proxies and must not be described as regional mandi prices. The planned
AGMARKNET backfill requires a user-authorized data.gov.in API key.

The expanded climate files use one reproducible administrative-capital point
per state/UT. They provide a consistent regional indicator but are not
area-weighted state averages. July 2026 is partial and is explicitly flagged.
The AGMARKNET file is a current-day snapshot, not a historical backfill.

Provenance, checksums, sizes, access times, and quality results are in
`data/metadata/`.
