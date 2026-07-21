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
