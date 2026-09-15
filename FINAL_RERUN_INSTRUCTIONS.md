# Final rerun instructions

Use the notebooks in:

`notebooks/FinalNotebooks/`

The genuine AGMARKNET arrivals input has been standardized here:

`data/external_required/agmarknet_enam_arrivals.csv.gz`

Coverage audit:

`data/external_required/agmarknet_enam_arrivals_coverage.csv`

## Execution order

Run all notebooks in this order because each notebook consumes artifacts from earlier notebooks:

1. `01_data_acquisition_synopsis_aligned.ipynb`
2. `02_data_quality_and_cleaning_synopsis_aligned.ipynb`
3. `03_exploratory_analysis_synopsis_aligned.ipynb`
4. `04_statistical_analysis_synopsis_aligned.ipynb`
5. `05_forecasting_models_synopsis_aligned.ipynb`
6. `06_shock_classification_synopsis_aligned.ipynb`
7. `07_explainability_synopsis_aligned.ipynb`
8. `08_affordability_index_synopsis_aligned.ipynb`
9. `09_decision_scenario_analysis_synopsis_aligned.ipynb`
10. `10_final_results_synopsis_aligned.ipynb`

## What changed in this final notebook set

- Notebook 02 will now integrate genuine AGMARKNET market-arrival quantity if the standardized file is present.
- Notebook 04 includes `log_market_arrival_quantity` as a genuine arrivals predictor when available.
- Notebook 05 uses 24 monthly rolling origins instead of 8 quarterly origins.
- Notebook 05 uses target-date-safe training windows for 1-, 2-, and 3-month targets.
- Notebook 06 uses target-date-safe training windows for shock classification.
- Notebooks 05, 06, and 07 include `log_market_arrival_quantity` as an available model/explainability feature.

## Models compared

Forecasting, Notebook 05:

- `hist_gradient_boosting`
- `ridge`
- `zero_change`
- `seasonal_change`
- National long-horizon path: `damped_ets`

Shock classification, Notebook 06:

- `hist_gradient_boosting`
- `logistic_balanced`
- `persistence`

Statistical analysis, Notebook 04:

- State-clustered fixed-effects OLS by forecast horizon.
- Horizons: 1 month, 2 months, 3 months.
- Multiple-testing control: Benjamini-Hochberg FDR.

Affordability, Notebook 08:

- HFASI scenario index.
- HCES food-expenditure shares by state, sector, and decile.
- Purchasing-power scenarios.

Decision analysis, Notebook 09:

- Stakeholders: household budgeting, retail inventory, enterprise procurement, policy monitoring.
- Scenario stress testing: favorable supply, baseline, moderate stress, severe stress.

## Arrival data caveat for report

The AGMARKNET arrivals file is a 2021-2026 confirmatory panel for ten selected states, not a full all-India all-market historical panel. Soyabean has six-state coverage. Treat arrival results as a strengthened robustness test and a partial adjudication of the original market-arrival hypothesis.

Suggested report wording:

`A genuine AGMARKNET arrival-quantity panel was added for eight study commodities across ten analytically important states for January 2021 to July 2026. Because the arrivals panel is narrower than the full price panel, the market-arrival component of RQ1 is treated as partially adjudicated rather than as a full national causal conclusion.`

## Rubric coverage notes

- Executive summary: state which RQs are fully addressed and which remain partially constrained.
- Data and EDA: report the new arrivals coverage table.
- Architecture: show arrivals entering data validation, statistical modelling, forecasting, and explainability.
- Model building: emphasize 24 monthly origins and target-date-safe training windows.
- Results: avoid claiming statistical superiority unless the rerun supports it.
- Implementation: position the dashboard as decision support, not automatic procurement or policy action.
- Limitations: clearly report partial arrivals, limited Soyabean state coverage, and non-causal interpretation.
