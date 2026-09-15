"""Build the synopsis-aligned exploratory-analysis notebook (Notebook 03)."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "03_exploratory_analysis.ipynb"
TARGET = ROOT / "notebooks" / "03_exploratory_analysis_synopsis_aligned.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


nb = json.loads(SOURCE.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

nb["cells"][0]["source"] = [
    "# QM640 Food Price Affordability AI\n",
    "## Notebook 03 — Synopsis-Aligned Exploratory Analysis\n",
    "\n",
    "**Purpose.** Explore the original synopsis outcome—future essential-food price "
    "change—at exact 1-, 2-, and 3-month horizons, while preserving the distinction "
    "between available predictors and missing confirmatory sources.\n",
]

alignment_md = md(
    """
## 11. Synopsis-aligned horizon and variable analysis

The original synopsis is controlling. The dependent variables are exact future
commodity-price changes at one, two, and three months. Available lagged-price,
climate, seasonal, macroeconomic, and coverage-control variables are explored
descriptively. `source_rows` measures reporting coverage and is **not** market
arrival quantity. Market arrivals, crop production/yield, rural wages, and HCES
microdata remain confirmatory gaps unless genuine files were integrated by
Notebook 02.
"""
)

alignment_code = code(
    r"""
HORIZON_TARGETS = {
    "1_month": "future_price_change_h1m_pct",
    "2_month": "future_price_change_h2m_pct",
    "3_month": "future_price_change_h3m_pct",
}
missing_targets = [column for column in HORIZON_TARGETS.values() if column not in state.columns]
if missing_targets:
    raise KeyError(
        "Notebook 02 must create exact future targets before Notebook 03. Missing: "
        + ", ".join(missing_targets)
    )

# Long form avoids repeating groupby logic and makes the three horizons directly comparable.
horizon_long = state[
    ["date", "region", "Commodity", *HORIZON_TARGETS.values()]
].melt(
    id_vars=["date", "region", "Commodity"],
    value_vars=list(HORIZON_TARGETS.values()),
    var_name="target_column",
    value_name="future_price_change_pct",
)
horizon_lookup = {value: key for key, value in HORIZON_TARGETS.items()}
horizon_long["horizon"] = horizon_long["target_column"].map(horizon_lookup)
horizon_long["horizon_months"] = horizon_long["horizon"].str.extract(r"(\d+)").astype(int)

horizon_summary = (
    horizon_long.groupby(["horizon", "horizon_months"], as_index=False)
    .agg(
        observations=("future_price_change_pct", "count"),
        mean_change_pct=("future_price_change_pct", "mean"),
        median_change_pct=("future_price_change_pct", "median"),
        standard_deviation_pct=("future_price_change_pct", "std"),
        p05_pct=("future_price_change_pct", lambda x: x.quantile(0.05)),
        p95_pct=("future_price_change_pct", lambda x: x.quantile(0.95)),
        positive_change_share=("future_price_change_pct", lambda x: x.gt(0).mean()),
    )
    .sort_values("horizon_months")
)
horizon_summary.to_csv(REPORT_OUTPUT / "03_future_horizon_summary.csv", index=False)

commodity_horizon_summary = (
    horizon_long.dropna(subset=["future_price_change_pct"])
    .groupby(["Commodity", "horizon", "horizon_months"], observed=True, as_index=False)
    .agg(
        observations=("future_price_change_pct", "size"),
        median_change_pct=("future_price_change_pct", "median"),
        volatility_pct=("future_price_change_pct", "std"),
    )
    .sort_values(["horizon_months", "volatility_pct"], ascending=[True, False])
)
commodity_horizon_summary.to_csv(
    REPORT_OUTPUT / "03_future_horizon_commodity_summary.csv", index=False
)

region_horizon_summary = (
    horizon_long.dropna(subset=["future_price_change_pct"])
    .groupby(["region", "horizon", "horizon_months"], observed=True, as_index=False)
    .agg(
        observations=("future_price_change_pct", "size"),
        median_change_pct=("future_price_change_pct", "median"),
        volatility_pct=("future_price_change_pct", "std"),
    )
)
region_horizon_summary.to_csv(
    REPORT_OUTPUT / "03_future_horizon_region_summary.csv", index=False
)
print("Exact future-price target summary")
print(horizon_summary.round(3).to_string(index=False))

# Correlations are screening statistics only; Notebook 04 performs controlled inference.
candidate_predictors = [
    "price_lag_1m",
    "price_lag_12m",
    "price_mom_pct",
    "price_yoy_pct",
    "rainfall_anomaly_pct",
    "temperature_anomaly_c",
    "relative_humidity_pct",
    "production_tonne",
    "yield_kg_per_hectare",
    "rural_wage_rs_per_day",
    "rural_wage_yoy_pct",
    "source_rows",  # reporting-coverage control, never interpreted as arrivals
]
candidate_predictors = [column for column in candidate_predictors if column in state.columns]
correlation_rows = []
for horizon, target in HORIZON_TARGETS.items():
    for predictor in candidate_predictors:
        pair = state[[predictor, target]].dropna()
        correlation_rows.append(
            {
                "horizon": horizon,
                "horizon_months": int(horizon.split("_")[0]),
                "predictor": predictor,
                "n_complete": int(len(pair)),
                "pearson_correlation": (
                    float(pair[predictor].corr(pair[target])) if len(pair) >= 3 else np.nan
                ),
                "interpretation": (
                    "reporting coverage control; not market arrivals"
                    if predictor == "source_rows"
                    else "descriptive association; not causal"
                ),
            }
        )
predictor_correlations = pd.DataFrame(correlation_rows)
predictor_correlations.to_csv(
    REPORT_OUTPUT / "03_available_predictor_correlations.csv", index=False
)

confirmatory_status = nb02.get("confirmatory_source_status", {})
readiness_rows = [
    {
        "hypothesis_variable_group": "Future essential-food price change",
        "role": "Dependent variable",
        "variables": ", ".join(HORIZON_TARGETS.values()),
        "status": "available",
    },
    {
        "hypothesis_variable_group": "Lagged price and seasonality",
        "role": "Core predictor/control",
        "variables": "price_lag_1m, price_lag_12m, price_mom_pct, price_yoy_pct, calendar month",
        "status": "available",
    },
    {
        "hypothesis_variable_group": "Climate",
        "role": "Core predictor",
        "variables": "rainfall_anomaly_pct, temperature_anomaly_c, relative_humidity_pct",
        "status": "available; representative administrative-capital points",
    },
    {
        "hypothesis_variable_group": "Market supply",
        "role": "Confirmatory predictor",
        "variables": "genuine market arrival quantity",
        "status": confirmatory_status.get("market_arrivals", "missing"),
    },
    {
        "hypothesis_variable_group": "Agricultural production",
        "role": "Confirmatory predictor",
        "variables": "state crop area, production, yield",
        "status": confirmatory_status.get("state_crop_apy", "missing"),
    },
    {
        "hypothesis_variable_group": "Purchasing power",
        "role": "Confirmatory predictor",
        "variables": "rural wage or income growth",
        "status": confirmatory_status.get("rural_wages", "missing"),
    },
    {
        "hypothesis_variable_group": "Household validation",
        "role": "RQ3 validation",
        "variables": "HCES household/segment expenditure microdata",
        "status": confirmatory_status.get("hces_microdata", "missing"),
    },
]
synopsis_variable_readiness = pd.DataFrame(readiness_rows)
synopsis_variable_readiness.to_csv(
    REPORT_OUTPUT / "03_synopsis_variable_readiness.csv", index=False
)
print("\nSynopsis variable readiness")
print(synopsis_variable_readiness.to_string(index=False))

if not SKIP_PLOTS:
    plot_data = horizon_long.dropna(subset=["future_price_change_pct"]).copy()
    plot_data["display_change"] = plot_data["future_price_change_pct"].clip(-100, 200)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.boxplot(
        data=plot_data.sample(min(30000, len(plot_data)), random_state=640),
        x="horizon",
        y="display_change",
        order=list(HORIZON_TARGETS),
        showfliers=False,
        ax=axes[0],
    )
    axes[0].axhline(0, color="#6b7280", linewidth=0.8)
    axes[0].set(
        title="Exact future price-change distributions",
        xlabel="Forecast horizon",
        ylabel="Future price change (%) — clipped for display",
    )
    volatility_pivot = commodity_horizon_summary.pivot(
        index="Commodity", columns="horizon", values="volatility_pct"
    ).reindex(columns=list(HORIZON_TARGETS))
    sns.heatmap(volatility_pivot, cmap="YlOrRd", annot=True, fmt=".1f", ax=axes[1])
    axes[1].set(
        title="Future price-change volatility by commodity",
        xlabel="Forecast horizon",
        ylabel="Commodity",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "03_future_price_change_horizons.png", bbox_inches="tight")
    plt.show()
"""
)

# Place the new analysis before the consolidated findings.
nb["cells"][21:21] = [alignment_md, alignment_code]

# Summary moved from index 22 to 24 after insertion.
summary_index = next(
    i
    for i, cell in enumerate(nb["cells"])
    if cell["cell_type"] == "code"
    and "NOTEBOOK 03 EXECUTION SUMMARY" in "".join(cell.get("source", []))
)
summary_source = "".join(nb["cells"][summary_index]["source"])
summary_source = summary_source.replace(
    '"notebook": "03_exploratory_analysis",',
    '"notebook": "03_exploratory_analysis_synopsis_aligned",\n'
    '    "original_research_questions_preserved": True,\n'
    '    "dependent_variable_horizons_months": [1, 2, 3],\n'
    '    "future_target_rows": {\n'
    '        row["horizon"]: int(row["observations"])\n'
    '        for _, row in horizon_summary.iterrows()\n'
    '    },\n'
    '    "confirmatory_source_status": confirmatory_status,\n'
    '    "reporting_counts_not_used_as_arrivals": True,',
)
summary_source = summary_source.replace(
    '"figures_created": 0 if SKIP_PLOTS else 7,',
    '"figures_created": 0 if SKIP_PLOTS else 8,',
)
summary_source = summary_source.replace(
    'NOTEBOOK 03 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT',
    'NOTEBOOK 03 ALIGNED EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT',
)
nb["cells"][summary_index]["source"] = [
    line + "\n" for line in summary_source.rstrip().splitlines()
]

inventory_index = next(
    i
    for i, cell in enumerate(nb["cells"])
    if cell["cell_type"] == "code"
    and "expected_tables = [" in "".join(cell.get("source", []))
)
inventory_source = "".join(nb["cells"][inventory_index]["source"])
inventory_source = inventory_source.replace(
    '    REPORT_OUTPUT / "03_exploratory_inflation_shocks.csv",',
    '    REPORT_OUTPUT / "03_exploratory_inflation_shocks.csv",\n'
    '    REPORT_OUTPUT / "03_future_horizon_summary.csv",\n'
    '    REPORT_OUTPUT / "03_future_horizon_commodity_summary.csv",\n'
    '    REPORT_OUTPUT / "03_future_horizon_region_summary.csv",\n'
    '    REPORT_OUTPUT / "03_available_predictor_correlations.csv",\n'
    '    REPORT_OUTPUT / "03_synopsis_variable_readiness.csv",',
)
inventory_source = inventory_source.replace(
    '    FIGURES / "03_food_inflation_shocks.png",',
    '    FIGURES / "03_food_inflation_shocks.png",\n'
    '    FIGURES / "03_future_price_change_horizons.png",',
)
nb["cells"][inventory_index]["source"] = [
    line + "\n" for line in inventory_source.rstrip().splitlines()
]

nb["cells"][-1]["source"] = [
    "## Notebook 03 conclusion\n",
    "\n",
    "Notebook 03 preserves the original synopsis by describing exact one-, two-, "
    "and three-month future price changes and the available predictor groups. "
    "All relationships remain exploratory and non-causal. Missing genuine arrivals, "
    "crop APY, rural wages, and HCES microdata are explicitly carried forward rather "
    "than replaced by invalid proxies.\n",
]

TARGET.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {TARGET}")
