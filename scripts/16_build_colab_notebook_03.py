"""Build the Google Colab version of Notebook 03."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "03_exploratory_analysis.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    markdown(
        """# QM640 Food Price Affordability AI
## Notebook 03 — Exploratory Data Analysis

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Prerequisite:** Notebooks 01 and 02 completed successfully

### Purpose

This notebook explores the verified cleaned data before formal modeling. It asks:

1. How have food CPI and mandi prices changed over time?
2. Which commodities show the strongest growth and volatility?
3. Are price changes seasonal?
4. How does recent food-price pressure vary across reporting states/UTs?
5. How has official reporting coverage changed?
6. What descriptive relationships exist between price changes and climate?
7. When did high national food-inflation periods occur?

The notebook saves every plotted dataset as CSV, so the interim report can cite
exact values rather than relying only on figures.
"""
    ),
    markdown(
        """## 1. Colab and Google Drive setup

The cleaned datasets are read from the same persistent Drive folder used by the
first two notebooks. Colab includes matplotlib and seaborn. Figures use a
consistent accessible palette and are saved at 180 DPI for report use.
"""
    ),
    code(
        """from pathlib import Path
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 170)

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

SAVE_TO_DRIVE = True
if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/QM640_Food_Affordability")
else:
    default_root = Path.cwd().resolve()
    if default_root.name == "notebooks":
        default_root = default_root.parent
    OUTPUT_ROOT = Path(os.environ.get("QM640_OUTPUT_ROOT", str(default_root)))

PROCESSED = OUTPUT_ROOT / "data" / "processed"
REPORT_OUTPUT = OUTPUT_ROOT / "reports" / "notebook_outputs"
FIGURES = REPORT_OUTPUT / "figures"
REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Plot skipping is only for automated validation in environments without
# matplotlib. In Google Colab it remains False and all figures are produced.
SKIP_PLOTS = os.environ.get("QM640_SKIP_PLOTS", "0") == "1"
if not SKIP_PLOTS:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 110,
        "savefig.dpi": 180,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

print(f"Running in Google Colab: {IN_COLAB}")
print(f"Output root: {OUTPUT_ROOT}")
print(f"Create figures: {not SKIP_PLOTS}")
"""
    ),
    markdown(
        """## 2. Load and verify cleaned inputs

Only cleaned outputs from Notebook 02 are used. Expected row counts are read from
its JSON summary, so accidental truncation or selecting the wrong Drive folder
stops the notebook immediately.
"""
    ),
    code(
        """STATE_FILE = PROCESSED / "cleaned_state_monthly.csv.gz"
NATIONAL_FILE = PROCESSED / "cleaned_national_monthly.csv.gz"
NB02_SUMMARY_FILE = REPORT_OUTPUT / "02_execution_summary.json"

required = [STATE_FILE, NATIONAL_FILE, NB02_SUMMARY_FILE]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Notebook 02 outputs are missing. Run Notebook 02 first.\\n"
        + "\\n".join(missing)
    )

# Category dtypes reduce repeated-string memory in the 41,792-row state panel.
state = pd.read_csv(
    STATE_FILE,
    parse_dates=["date"],
    dtype={"region": "category", "Commodity": "category"},
    low_memory=False,
)
national = pd.read_csv(NATIONAL_FILE, parse_dates=["date"], low_memory=False)
nb02 = json.loads(NB02_SUMMARY_FILE.read_text(encoding="utf-8"))

if len(state) != int(nb02["clean_state_rows"]):
    raise AssertionError("State row count does not match Notebook 02 summary.")
if len(national) != int(nb02["clean_national_rows"]):
    raise AssertionError("National row count does not match Notebook 02 summary.")

print(f"Verified state rows: {len(state):,}")
print(f"Verified national rows: {len(national):,}")
print(f"Regions with mandi reporting: {state['region'].nunique()}")
print(f"Commodities: {state['Commodity'].nunique()}")
"""
    ),
    markdown(
        """## 3. Overall descriptive statistics

Raw price levels differ structurally across commodities, so pooled price averages
are not interpreted as a national basket. The table reports each commodity
separately and uses within-series percentage changes for comparison.
"""
    ),
    code(
        """commodity_summary = (
    state.groupby("Commodity", observed=True, as_index=False)
    .agg(
        observations=("date", "size"),
        states_uts=("region", "nunique"),
        official_source_rows=("source_rows", "sum"),
        median_price_rs_per_quintal=("price_cleaned_rs_per_quintal", "median"),
        mean_price_rs_per_quintal=("price_cleaned_rs_per_quintal", "mean"),
        price_standard_deviation=("price_cleaned_rs_per_quintal", "std"),
        median_mom_pct=("price_mom_pct", "median"),
        mean_yoy_pct=("price_yoy_pct", "mean"),
        yoy_volatility=("price_yoy_pct", "std"),
        extreme_prices=("price_extreme_flag", "sum"),
    )
)
commodity_summary["coefficient_of_variation"] = (
    commodity_summary["price_standard_deviation"]
    / commodity_summary["mean_price_rs_per_quintal"]
)
commodity_summary = commodity_summary.sort_values(
    "yoy_volatility", ascending=False
)
commodity_summary.to_csv(REPORT_OUTPUT / "03_commodity_summary.csv", index=False)
print(commodity_summary.round(3).to_string(index=False))
"""
    ),
    markdown(
        """## 4. National food CPI and mandi-price trends

Food CPI and the mandi index are conceptually different. Both are displayed on
their existing 2015=100 scales, and only dates with at least one value are used.
The underlying table is saved for exact reporting.
"""
    ),
    code(
        """national_trend = national[
    [
        "date",
        "food_cpi_2015_100",
        "mandi_price_index_2015_100",
        "food_cpi_yoy_pct_exact",
        "mandi_index_yoy_pct",
        "mandi_source_rows",
        "commodities_observed",
    ]
].copy()
national_trend.to_csv(REPORT_OUTPUT / "03_national_price_trends.csv", index=False)

if not SKIP_PLOTS:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        national_trend["date"],
        national_trend["food_cpi_2015_100"],
        label="Food CPI (2015=100)",
        color="#1f4e79",
        linewidth=2.2,
    )
    ax.plot(
        national_trend["date"],
        national_trend["mandi_price_index_2015_100"],
        label="AGMARKNET mandi index (2015=100)",
        color="#d97706",
        linewidth=1.8,
        alpha=0.9,
    )
    ax.axvline(pd.Timestamp("2026-01-01"), color="#9ca3af", linestyle="--", linewidth=1)
    ax.text(
        pd.Timestamp("2026-01-01"),
        ax.get_ylim()[0],
        " 2026 partial",
        color="#6b7280",
        va="bottom",
    )
    ax.set(
        title="National food CPI and official mandi-price index",
        xlabel="Month",
        ylabel="Index (2015=100)",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_national_cpi_mandi_trend.png", bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        """## 5. Commodity-specific national price indices

State prices are aggregated using official observation counts. Each commodity is
rebased to its own 2015 national weighted mean. This vectorized aggregation is
efficient and prevents high-priced commodities from dominating simply because
of their rupee level.
"""
    ),
    code(
        """weighted = state[
    ["date", "Commodity", "price_cleaned_rs_per_quintal", "source_rows"]
].copy()
weighted["weighted_price"] = (
    weighted["price_cleaned_rs_per_quintal"] * weighted["source_rows"]
)
commodity_monthly = (
    weighted.groupby(
        ["date", "Commodity"], observed=True, as_index=False, sort=False
    )
    .agg(
        weighted_price_sum=("weighted_price", "sum"),
        official_observations=("source_rows", "sum"),
    )
)
commodity_monthly["national_weighted_price"] = (
    commodity_monthly["weighted_price_sum"]
    / commodity_monthly["official_observations"]
)

base_prices = (
    commodity_monthly.loc[commodity_monthly["date"].dt.year.eq(2015)]
    .groupby("Commodity", observed=True)["national_weighted_price"]
    .mean()
)
commodity_monthly["base_price_2015"] = (
    commodity_monthly["Commodity"].map(base_prices).astype("float64")
)
commodity_monthly["price_index_2015_100"] = (
    100
    * commodity_monthly["national_weighted_price"]
    / commodity_monthly["base_price_2015"]
)
commodity_monthly = commodity_monthly.sort_values(["Commodity", "date"])
commodity_monthly["index_yoy_pct"] = (
    commodity_monthly.groupby("Commodity", observed=True)[
        "price_index_2015_100"
    ].pct_change(12)
    * 100
)
commodity_monthly.to_csv(
    REPORT_OUTPUT / "03_national_commodity_indices.csv", index=False
)

if not SKIP_PLOTS:
    fig, ax = plt.subplots(figsize=(14, 7))
    for commodity, group in commodity_monthly.groupby("Commodity", observed=True):
        ax.plot(
            group["date"],
            group["price_index_2015_100"],
            label=str(commodity),
            linewidth=1.6,
        )
    ax.set(
        title="National weighted mandi-price indices by commodity",
        xlabel="Month",
        ylabel="Commodity index (2015=100)",
    )
    ax.legend(ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_commodity_price_indices.png", bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        """## 6. Seasonal price-change patterns

Median month-on-month change is used because price changes are skewed. The
heatmap is descriptive: it reveals recurring calendar patterns but does not prove
that season alone causes them.
"""
    ),
    code(
        """seasonality_source = state.loc[state["price_mom_pct"].notna()].copy()
seasonality_source["calendar_month"] = seasonality_source["date"].dt.month
seasonality = (
    seasonality_source.groupby(
        ["Commodity", "calendar_month"], observed=True
    )["price_mom_pct"]
    .median()
    .unstack()
    .reindex(columns=range(1, 13))
)
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
seasonality.columns = month_labels
seasonality.to_csv(REPORT_OUTPUT / "03_price_seasonality.csv")
print("Median month-on-month price change (%)")
print(seasonality.round(2).to_string())

if not SKIP_PLOTS:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        seasonality,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        cbar_kws={"label": "Median monthly price change (%)"},
        ax=ax,
    )
    ax.set(
        title="Seasonality of state-level mandi price changes",
        xlabel="Calendar month",
        ylabel="Commodity",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "03_price_seasonality_heatmap.png", bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        """## 7. Regional price pressure over 2, 5, and 10 years

Regional results are calculated separately for trailing 24-, 60-, and 120-month
windows. Comparisons use within-series year-on-year changes, not raw price levels.
For a window of *N* months, a state/UT must have at least *N* valid observations;
this conservative rule reduces unstable rankings from sparse reporting.

The median is the ranking statistic because large percentage changes can make the
arithmetic mean misleading. Mean, volatility, commodity coverage, and official
observation counts remain in the saved audit table.
"""
    ),
    code(
        """latest_month = state["date"].max()
WINDOWS = {
    "2-year": 24,
    "5-year": 60,
    "10-year": 120,
}


def regional_window_summary(
    frame: pd.DataFrame, label: str, months: int
) -> pd.DataFrame:
    \"\"\"Summarize robust regional price pressure for one trailing window.\"\"\"
    window_start = latest_month - pd.DateOffset(months=months - 1)
    selected = frame.loc[
        frame["date"].between(window_start, latest_month)
        & frame["price_yoy_pct"].notna()
    ]
    result = (
        selected.groupby("region", observed=True, as_index=False)
        .agg(
            mean_yoy_price_change=("price_yoy_pct", "mean"),
            median_yoy_price_change=("price_yoy_pct", "median"),
            yoy_volatility=("price_yoy_pct", "std"),
            valid_observations=("price_yoy_pct", "size"),
            commodities=("Commodity", "nunique"),
            active_months=("date", "nunique"),
            official_source_rows=("source_rows", "sum"),
        )
    )
    result = result.loc[result["valid_observations"] >= months].copy()
    result["window"] = label
    result["window_months"] = months
    result["window_start"] = window_start
    result["window_end"] = latest_month
    return result.sort_values(
        "median_yoy_price_change", ascending=False
    ).reset_index(drop=True)


regional_tables = [
    regional_window_summary(state, label, months)
    for label, months in WINDOWS.items()
]
regional_comparison = pd.concat(regional_tables, ignore_index=True)
regional_comparison.to_csv(
    REPORT_OUTPUT / "03_regional_pressure_2_5_10_years.csv", index=False
)

for label in WINDOWS:
    print(f"\\nTop regional median price pressure — {label}")
    display_columns = [
        "region",
        "median_yoy_price_change",
        "mean_yoy_price_change",
        "yoy_volatility",
        "valid_observations",
        "active_months",
        "commodities",
    ]
    print(
        regional_comparison.loc[
            regional_comparison["window"].eq(label), display_columns
        ]
        .head(10)
        .round(2)
        .to_string(index=False)
    )

if not SKIP_PLOTS:
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), sharex=False)
    for ax, (label, months) in zip(axes, WINDOWS.items()):
        plot_data = (
            regional_comparison.loc[regional_comparison["window"].eq(label)]
            .head(12)
            .sort_values("median_yoy_price_change", ascending=True)
        )
        colors = np.where(
            plot_data["median_yoy_price_change"] >= 0, "#c2410c", "#2563eb"
        )
        ax.barh(
            plot_data["region"].astype(str),
            plot_data["median_yoy_price_change"],
            color=colors,
        )
        ax.axvline(0, color="#374151", linewidth=0.8)
        start = latest_month - pd.DateOffset(months=months - 1)
        ax.set(
            title=f"{label}: {start:%b %Y}–{latest_month:%b %Y}",
            xlabel="Median YoY price change (%)",
            ylabel="Reporting state/UT",
        )
    fig.suptitle(
        "Regional mandi-price pressure across 2-, 5-, and 10-year windows",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "03_regional_pressure_2_5_10_years.png",
        bbox_inches="tight",
    )
    plt.show()
"""
    ),
    markdown(
        """## 8. Official reporting coverage over time

Coverage affects apparent trends. The panel is first reduced to unique keys, then
states, commodities, rows, and underlying official observations are counted by
month. This distinguishes broader reporting from actual price movement.
"""
    ),
    code(
        """monthly_coverage = (
    state.groupby("date", as_index=False, observed=True)
    .agg(
        reporting_states_uts=("region", "nunique"),
        commodities_observed=("Commodity", "nunique"),
        state_commodity_rows=("Commodity", "size"),
        official_source_rows=("source_rows", "sum"),
    )
    .sort_values("date")
)
monthly_coverage.to_csv(REPORT_OUTPUT / "03_monthly_reporting_coverage.csv", index=False)

if not SKIP_PLOTS:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(
        monthly_coverage["date"],
        monthly_coverage["reporting_states_uts"],
        color="#1f4e79",
        label="Reporting states/UTs",
    )
    axes[0].plot(
        monthly_coverage["date"],
        monthly_coverage["commodities_observed"],
        color="#059669",
        label="Commodities observed",
    )
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)
    axes[0].set_title("AGMARKNET reporting coverage over time")
    axes[1].plot(
        monthly_coverage["date"],
        monthly_coverage["official_source_rows"],
        color="#d97706",
    )
    axes[1].set(
        xlabel="Month",
        ylabel="Official observations",
        title="Underlying observations represented in monthly aggregates",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "03_reporting_coverage.png", bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        """## 9. Descriptive climate–price relationships

Correlations use all valid rows but are interpreted cautiously. They do not
control for seasonality, commodity, state, market arrivals, policy, or common
trends. Formal multivariate analysis belongs in Notebook 04.

For the scatter figure, at most 15,000 rows are sampled reproducibly to keep
rendering responsive; the correlation table itself still uses the complete data.
"""
    ),
    code(
        """correlation_variables = [
    "price_mom_pct",
    "price_yoy_pct",
    "rainfall_anomaly_pct",
    "temperature_anomaly_c",
    "relative_humidity_pct",
    "source_rows",
]
correlation_matrix = state[correlation_variables].corr(method="pearson")
correlation_matrix.to_csv(REPORT_OUTPUT / "03_climate_price_correlations.csv")
print(correlation_matrix.round(3).to_string())

climate_price = state[
    [
        "Commodity",
        "price_yoy_pct",
        "rainfall_anomaly_pct",
        "temperature_anomaly_c",
    ]
].dropna()
plot_sample = climate_price.sample(
    n=min(15_000, len(climate_price)),
    random_state=640,
)

if not SKIP_PLOTS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hexbin(
        plot_sample["rainfall_anomaly_pct"].clip(-150, 250),
        plot_sample["price_yoy_pct"].clip(-100, 200),
        gridsize=45,
        mincnt=1,
        cmap="Blues",
    )
    axes[0].axhline(0, color="#6b7280", linewidth=0.7)
    axes[0].set(
        title="Rainfall anomaly and mandi price change",
        xlabel="Rainfall anomaly (%) — clipped for display",
        ylabel="Year-on-year price change (%) — clipped for display",
    )
    axes[1].hexbin(
        plot_sample["temperature_anomaly_c"].clip(-4, 4),
        plot_sample["price_yoy_pct"].clip(-100, 200),
        gridsize=45,
        mincnt=1,
        cmap="Oranges",
    )
    axes[1].axhline(0, color="#6b7280", linewidth=0.7)
    axes[1].set(
        title="Temperature anomaly and mandi price change",
        xlabel="Temperature anomaly (°C) — clipped for display",
        ylabel="Year-on-year price change (%) — clipped for display",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "03_climate_price_hexbin.png", bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        """## 10. Exploratory national food-inflation shock periods

For description only, a high-inflation month is one at or above the 75th
percentile of observed national year-on-year food CPI inflation. Notebook 06 will
re-estimate its classification threshold from training data only to avoid leakage.
"""
    ),
    code(
        """inflation = national[
    ["date", "food_cpi_yoy_pct_exact", "food_cpi_2015_100"]
].dropna(subset=["food_cpi_yoy_pct_exact"]).copy()
exploratory_threshold = inflation["food_cpi_yoy_pct_exact"].quantile(0.75)
inflation["exploratory_high_inflation"] = (
    inflation["food_cpi_yoy_pct_exact"] >= exploratory_threshold
)
shock_periods = inflation.loc[inflation["exploratory_high_inflation"]].sort_values(
    "food_cpi_yoy_pct_exact", ascending=False
)
shock_periods.to_csv(REPORT_OUTPUT / "03_exploratory_inflation_shocks.csv", index=False)

print(f"Exploratory 75th-percentile threshold: {exploratory_threshold:.2f}%")
print("Highest food-inflation months")
print(shock_periods.head(15).round(2).to_string(index=False))

if not SKIP_PLOTS:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        inflation["date"],
        inflation["food_cpi_yoy_pct_exact"],
        color="#1f4e79",
        linewidth=1.8,
        label="Food CPI YoY",
    )
    ax.axhline(
        exploratory_threshold,
        color="#b91c1c",
        linestyle="--",
        label=f"75th percentile ({exploratory_threshold:.1f}%)",
    )
    highlighted = inflation.loc[inflation["exploratory_high_inflation"]]
    ax.scatter(
        highlighted["date"],
        highlighted["food_cpi_yoy_pct_exact"],
        color="#dc2626",
        s=18,
        alpha=0.75,
        label="Exploratory high-inflation month",
    )
    ax.axhline(0, color="#6b7280", linewidth=0.7)
    ax.set(
        title="National food-inflation history and exploratory shock threshold",
        xlabel="Month",
        ylabel="Food CPI year-on-year change (%)",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_food_inflation_shocks.png", bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        """## 11. Consolidated EDA findings

The code calculates findings directly from saved tables. No conclusion is
hard-coded, which keeps the narrative synchronized with the actual Colab run.
"""
    ),
    code(
        """highest_volatility = commodity_summary.iloc[0]
regional_window_rankings = {}
for label, months in WINDOWS.items():
    ranked = regional_comparison.loc[regional_comparison["window"].eq(label)]
    top = ranked.iloc[0] if len(ranked) else None
    regional_window_rankings[label] = {
        "months": months,
        "start": str((latest_month - pd.DateOffset(months=months - 1)).date()),
        "end": str(latest_month.date()),
        "eligible_regions": int(len(ranked)),
        "highest_pressure_region": str(top["region"]) if top is not None else None,
        "highest_median_yoy_pct": (
            float(top["median_yoy_price_change"]) if top is not None else None
        ),
    }
latest_national = national_trend.dropna(
    subset=["food_cpi_2015_100"]
).iloc[-1]

findings = {
    "notebook": "03_exploratory_analysis",
    "status": "completed",
    "state_rows_analyzed": int(len(state)),
    "national_months_analyzed": int(len(national)),
    "reporting_states_uts": int(state["region"].nunique()),
    "commodities": int(state["Commodity"].nunique()),
    "most_volatile_commodity_by_yoy_sd": str(highest_volatility["Commodity"]),
    "most_volatile_commodity_yoy_sd": float(
        highest_volatility["yoy_volatility"]
    ),
    "regional_window_rankings": regional_window_rankings,
    "exploratory_food_inflation_shock_threshold_pct": float(
        exploratory_threshold
    ),
    "exploratory_high_inflation_months": int(
        inflation["exploratory_high_inflation"].sum()
    ),
    "latest_cpi_date": str(latest_national["date"].date()),
    "latest_food_cpi_2015_100": float(latest_national["food_cpi_2015_100"]),
    "rainfall_yoy_price_correlation": float(
        correlation_matrix.loc["rainfall_anomaly_pct", "price_yoy_pct"]
    ),
    "temperature_yoy_price_correlation": float(
        correlation_matrix.loc["temperature_anomaly_c", "price_yoy_pct"]
    ),
    "figures_created": 0 if SKIP_PLOTS else 7,
    "interpretation_warning": (
        "EDA associations are descriptive and must not be interpreted as causal."
    ),
    "output_root": str(OUTPUT_ROOT),
}

SUMMARY_FILE = REPORT_OUTPUT / "03_execution_summary.json"
SUMMARY_FILE.write_text(json.dumps(findings, indent=2), encoding="utf-8")

print("=" * 72)
print("NOTEBOOK 03 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT")
print("=" * 72)
print(json.dumps(findings, indent=2))
"""
    ),
    markdown(
        """## 12. Generated artifact inventory

This final check confirms that every table exists. In Colab, it also verifies all
seven figures before declaring the notebook complete.
"""
    ),
    code(
        """expected_tables = [
    REPORT_OUTPUT / "03_commodity_summary.csv",
    REPORT_OUTPUT / "03_national_price_trends.csv",
    REPORT_OUTPUT / "03_national_commodity_indices.csv",
    REPORT_OUTPUT / "03_price_seasonality.csv",
    REPORT_OUTPUT / "03_regional_pressure_2_5_10_years.csv",
    REPORT_OUTPUT / "03_monthly_reporting_coverage.csv",
    REPORT_OUTPUT / "03_climate_price_correlations.csv",
    REPORT_OUTPUT / "03_exploratory_inflation_shocks.csv",
    SUMMARY_FILE,
]
expected_figures = [
    FIGURES / "03_national_cpi_mandi_trend.png",
    FIGURES / "03_commodity_price_indices.png",
    FIGURES / "03_price_seasonality_heatmap.png",
    FIGURES / "03_regional_pressure_2_5_10_years.png",
    FIGURES / "03_reporting_coverage.png",
    FIGURES / "03_climate_price_hexbin.png",
    FIGURES / "03_food_inflation_shocks.png",
]

files_to_check = expected_tables + ([] if SKIP_PLOTS else expected_figures)
missing_artifacts = [str(path) for path in files_to_check if not path.exists()]
if missing_artifacts:
    raise FileNotFoundError(
        "Expected EDA artifacts were not created:\\n" + "\\n".join(missing_artifacts)
    )

print(f"Verified generated tables: {len(expected_tables)}")
print(f"Verified generated figures: {0 if SKIP_PLOTS else len(expected_figures)}")
print("\\nArtifacts:")
for path in files_to_check:
    print(f"- {path} ({path.stat().st_size / 1_000_000:.3f} MB)")
"""
    ),
    markdown(
        """## Notebook 03 conclusion

This EDA establishes the temporal, commodity, seasonal, regional, reporting, and
climate context needed for formal hypothesis testing. Figures are suitable for
the interim report, while their source tables preserve exact reproducibility.

### What to send back

After choosing **Runtime → Run all**, send:

1. the `NOTEBOOK 03 EXECUTION SUMMARY`;
2. any warnings or red error output; and
3. optionally the figures you find most important for the report.

Notebook 04 will use these findings to specify multivariate statistical models
with seasonality, trend, commodity effects, regional effects, and robust
time-series interpretation.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {
            "name": "03_exploratory_analysis.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    for position, cell in enumerate(cells, start=1):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{OUTPUT.name}:cell-{position}", "exec")

    print(f"Created and syntax-validated: {OUTPUT}")
    print(f"Notebook cells: {len(cells)}")


if __name__ == "__main__":
    main()
