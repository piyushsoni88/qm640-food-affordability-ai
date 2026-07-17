# QM640 Food Affordability AI

**Capstone Project:** Explainable AI for Food Price Shocks and Household Affordability in India

This repository contains the data-acquisition pipeline, analytical notebooks, statistical models, machine-learning experiments, explainability outputs, and reproducibility documentation for the QM640 Data Analytics Capstone.

## Project Objective

The project develops an explainable decision-intelligence framework that:

- forecasts one- to three-month price movements for selected essential food commodities;
- identifies abnormal price-shock risk;
- explains the main drivers behind each prediction;
- estimates household affordability pressure using food-expenditure weights and purchasing-power indicators;
- evaluates whether model-based alerts improve household, procurement, inventory, and policy decisions.

## Study Scope

**Country:** India  
**Planned study period:** January 2011 to December 2025  
**Primary analytical unit:** Commodity-region-month  
**Forecast horizons:** 1, 2, and 3 months

### Initial commodities

- Onion
- Tomato
- Potato
- Arhar/Tur Dal
- Wheat

### Main analytical outputs

- Price forecasts
- Prediction intervals
- Price-shock probabilities
- SHAP-based explanations
- Household Food Affordability Stress Index
- Decision-support recommendations

## Research Questions

1. Which market, climatic, seasonal, agricultural, and macroeconomic variables are associated with future essential-food price changes?
2. Do machine-learning and ensemble models outperform conventional statistical forecasting models?
3. How can forecasted food-price changes be translated into a reliable and interpretable household affordability-stress measure?
4. Do model-driven alerts reduce retrospective household, procurement, inventory, or policy decision loss?

## Data Sources

The project uses official public or openly licensed data sources.

| Source | Main variables | Role |
|---|---|---|
| AGMARKNET / data.gov.in | Mandi prices, markets, commodities, varieties, arrivals where available | Primary forecasting panel |
| MOSPI CPI/CFPI | Consumer-price and food-price indices | Inflation benchmark |
| India Meteorological Department | Rainfall and rainfall anomalies | Climatic predictors |
| Directorate of Economics and Statistics | Crop area, production, and yield | Supply-side predictors |
| National Horticulture Board | Horticultural area and production, including tomato | Horticulture data |
| Household Consumption Expenditure Survey | Rural, urban, and expenditure-group food shares | Affordability weights |
| Labour Bureau | Rural wages and CPI-AL/RL | Purchasing-power adjustment |
| RBI / Office of Economic Adviser | Fuel, inflation, and macroeconomic indicators | Additional predictors |

The repository does not assume that every official source contains the entire planned study period. Actual coverage, missingness, licensing conditions, and extraction dates are recorded in the source manifest.

## Repository Structure

```text
qm640-food-affordability-ai/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── config/
│   ├── project.yaml
│   └── sources.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── source_registry.csv
│   ├── source_manifest.csv
│   └── data_dictionary.csv
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_quality_and_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_forecasting_models.ipynb
│   ├── 06_shock_classification.ipynb
│   ├── 07_explainability.ipynb
│   └── 08_affordability_index.ipynb
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── dashboard/
├── reports/
│   ├── figures/
│   ├── tables/
│   └── model_cards/
├── tests/
└── docs/
    ├── data_lineage.md
    ├── reproducibility.md
    └── ethical_and_limitations_statement.md
```

## Current Project Status

### Completed

- Capstone topic and research design
- Research questions and hypotheses
- Public-data source identification
- APA 7 synopsis
- Initial calculations using actual MOSPI CPI/CFPI and HCES data
- Phase 1 repository structure
- AGMARKNET ingestion script
- Source registry and data dictionary
- Raw-data audit framework
- Initial automated tests

### In progress

- Full official data acquisition
- Historical coverage verification
- Missingness and data-quality audit
- Commodity-region-month panel construction
- Exploratory data analysis

### Planned

- Statistical baseline models
- Machine-learning models
- Shock classification
- SHAP explainability
- HFASI construction and validation
- Decision-value backtesting
- Final dashboard and capstone report

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/piyushsoni88/qm640-food-affordability-ai.git
cd qm640-food-affordability-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the data.gov.in API key

Create a local `.env` file:

```text
DATA_GOV_API_KEY=your_personal_api_key
```

Never commit the `.env` file.

### 5. Run the local Phase 1 workflow

```bash
python scripts/run_phase1.py --mode all-local
pytest
```

## Data Governance

- Raw files are preserved unchanged.
- Derived data are stored separately in `data/interim` and `data/processed`.
- API keys, cookies, and credentials are never committed.
- Each source is recorded with its official URL, extraction date, checksum, coverage, and licence or usage terms.
- Large NetCDF files should be managed with Git LFS or external archival storage.
- Restricted unit-level microdata should not be redistributed unless the source terms explicitly permit it.
- All reported statistics must be reproducible from code and documented source files.

## Reproducibility

The final project will include:

- exact package versions;
- source manifests and file hashes;
- data dictionaries;
- transformation logs;
- temporal train-validation-test splits;
- model cards;
- evaluation metrics;
- random seeds where applicable;
- limitations and ethical-use documentation.

## Academic Integrity

This repository supports an academic capstone project. All external datasets, papers, and official publications must be cited appropriately. Results should distinguish clearly among observed evidence, statistical association, predictive inference, simulated decision outcomes, and causal claims.

## Author

**Piyush Soni**  
QM640: Data Analytics Capstone  
Walsh College

## Licence

The source code in this repository is released under the MIT License. Datasets remain subject to the licences and terms of their original publishers.
