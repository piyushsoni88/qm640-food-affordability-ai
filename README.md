# Explainable AI for Food Price Shocks and Household Affordability in India

## Project title
**Forecasting Essential Food Price Shocks and Household Affordability Stress in India: An Explainable AI Decision-Intelligence Framework Using Public Market, Agricultural, Climatic, and Economic Data**

## Purpose
This QM640 Data Analytics Capstone develops an explainable forecasting and early-warning framework for essential-food price shocks in India and translates forecasted commodity-price changes into a Household Food Affordability Stress Index (HFASI).

## Research questions
1. Which lagged-price, market-arrival, climatic, seasonal, agricultural-production, and macroeconomic variables significantly influence one- to three-month essential-food price changes in India?
2. Do machine-learning and ensemble forecasting models predict essential-food prices and price shocks more accurately than conventional statistical forecasting models?
3. How can forecasted commodity-price changes be translated into a statistically reliable, interpretable, and stable HFASI across rural, urban, and expenditure segments?
4. How effectively can the proposed early-warning framework support household budgeting, retail inventory, enterprise procurement, and policy-monitoring decisions under alternative supply and price scenarios?

## Structure
- `config/`: project and source configuration
- `data/`: raw, interim, processed, dictionaries, and manifests
- `notebooks/`: reproducible analysis notebooks
- `src/food_affordability_ai/`: reusable Python package
- `reports/`: figures, tables, and model cards
- `docs/`: methodology, lineage, ethics, and reproducibility
- `tests/`: automated tests

## Quick start
```bash
python -m venv .venv
pip install -r requirements.txt
pip install -e .
pytest
python scripts/run_phase1.py
```

## Interim assignment release

The interim release contains a reproducible, India-focused research dataset and
time-aware preliminary analysis:

- 51.5 MB of authoritative raw source files retained locally;
- 37,593 source-specific India, climate, and benchmark records in the curated layer;
- a 30,240-row `date x region x commodity` integration panel;
- 15 regional NASA POWER series, eight commodity groups, and 2005-2025 coverage;
- source URLs, access timestamps, sizes, SHA-256 checksums, and row counts;
- rolling-origin validation, a fixed 2022-2025 holdout, statistical inference,
  and report-ready figures/tables.

Install the collector and modeling dependencies, then run:

```powershell
python -m pip install -r requirements_collectors.txt
python -m pip install -r requirements.txt
python scripts/09_build_interim_research_dataset.py --start-year 2005 --end-year 2025
python scripts/10_run_interim_analysis.py
pytest
```

The final DOCX report is under
`reports/Piyush_Soni_QM640_Interim_Report.docx`.

## All-India 2000-2026 expansion

The extended collector covers all 28 states and eight union territories using
one documented administrative-capital point per region. It downloads NASA
POWER daily climate observations from 2000 through the latest safely available
2026 date, retains the raw API payloads locally, creates monthly aggregates,
and builds an eight-commodity integration panel:

```powershell
python scripts/12_collect_all_india_state_daily_climate.py
```

The 2026 records are explicitly flagged as year-to-date/provisional. Regional
climate fields are observed at representative points; FAOSTAT and World Bank
price/production fields remain India-level proxies and must not be interpreted
as state mandi prices. Full MoSPI state CPI requires a registered API token,
and historical AGMARKNET data requires a personal data.gov.in API key.

## Data sources and storage policy

| Source | Interim use | Access |
|---|---|---|
| FAOSTAT | India food CPI, producer prices, production | Public bulk downloads |
| NASA POWER | Monthly climate at 15 Indian regional points | Public API |
| World Bank Pink Sheet | Monthly global food and energy benchmarks | Public workbook |
| DES, Government of India | Agricultural price publication/validation | Public PDF |
| AGMARKNET / data.gov.in | Planned mandi prices and arrivals | Personal API key required |

Evaluator-ready India extracts and the 30,240-row derived panel are committed
under `data/curated/`. Large immutable archives and NASA response payloads are
downloaded into `data/raw/`, verified by checksums, and excluded from Git
history. This avoids GitHub's large-file limits while preserving full
reproducibility. Put `DATA_GOV_IN_API_KEY` in an untracked `.env` file before
running the AGMARKNET collector; never commit the credential.

## Preliminary findings

On the 47-month 2022-2025 holdout, ridge regression achieved MAE 1.362, RMSE
1.933, sMAPE 0.910%, and R-squared 0.957. Persistence remained a strong control
(RMSE 2.045), and the paired improvement was not statistically significant
(one-sided Wilcoxon p = 0.202). A level-based random forest failed to
extrapolate (RMSE 21.948), motivating differenced targets and trend-capable
models. The rare-event classifier ranked cases reasonably (ROC-AUC 0.818) but
detected none of the three holdout shocks at the default threshold; threshold
tuning and precision-recall evaluation remain final-report work.
