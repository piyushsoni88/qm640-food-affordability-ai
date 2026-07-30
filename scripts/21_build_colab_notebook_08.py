"""Build the Google Colab version of Notebook 08."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "08_affordability_index.ipynb"


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
## Notebook 08 — Household Food Affordability Stress Index

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Prerequisite:** Notebooks 01–07 completed successfully

### Objectives

This notebook:

1. converts the Notebook 05 food-CPI path into 2026 year-on-year food-cost growth;
2. applies the synopsis affordability formula separately to rural and urban
   household segments;
3. evaluates downside, baseline, and upside purchasing-power scenarios;
4. connects affordability stress with Notebook 06 shock probabilities;
5. tests sensitivity to household food shares and purchasing-power growth; and
6. saves report-ready tables and figures.

HCES microdata have not been supplied. Therefore, this notebook uses published
aggregate food-expenditure shares as transparent segment benchmarks:
**46.3% rural** and **39.1% urban**. Results describe representative segments,
not individual households or distributional causal effects. Source:
[MoSPI, NSS Report No. 591, HCES 2022–23](https://mospi.gov.in/sites/default/files/publication_reports/Report_591_HCES_2022-23New.pdf).
"""
    ),
    markdown("## 1. Runtime and Google Drive setup"),
    code(
        """from pathlib import Path
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/QM640_Food_Affordability")
else:
    root = Path.cwd().resolve()
    if root.name == "notebooks":
        root = root.parent
    OUTPUT_ROOT = Path(os.environ.get("QM640_OUTPUT_ROOT", str(root)))

PROCESSED = OUTPUT_ROOT / "data" / "processed"
REPORT_OUTPUT = OUTPUT_ROOT / "reports" / "notebook_outputs"
FIGURES = REPORT_OUTPUT / "figures"
for folder in [REPORT_OUTPUT, FIGURES]:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "savefig.dpi": 180,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
print(f"Output root: {OUTPUT_ROOT}")
"""
    ),
    markdown(
        """## 2. Load verified upstream artifacts

The notebook reads model outputs rather than manually copying forecast values.
"""
    ),
    code(
        """NATIONAL_FILE = PROCESSED / "cleaned_national_monthly.csv.gz"
FORECAST_FILE = REPORT_OUTPUT / "05_twenty_four_month_conditional_forecast.csv"
FORECAST_SUMMARY_FILE = REPORT_OUTPUT / "05_execution_summary.json"
RISK_FILE = REPORT_OUTPUT / "06_2026_monthly_shock_risk.csv"
RISK_SUMMARY_FILE = REPORT_OUTPUT / "06_execution_summary.json"
EXPLAIN_SUMMARY_FILE = REPORT_OUTPUT / "07_execution_summary.json"

required = [
    NATIONAL_FILE, FORECAST_FILE, FORECAST_SUMMARY_FILE,
    RISK_FILE, RISK_SUMMARY_FILE, EXPLAIN_SUMMARY_FILE,
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Run Notebooks 01–07 first.\\n" + "\\n".join(missing)
    )

national = pd.read_csv(NATIONAL_FILE, parse_dates=["date"], low_memory=False)
forecast = pd.read_csv(FORECAST_FILE, parse_dates=["date"])
risk = pd.read_csv(RISK_FILE, parse_dates=["feature_month", "target_date"])
forecast_summary = json.loads(
    FORECAST_SUMMARY_FILE.read_text(encoding="utf-8")
)
risk_summary = json.loads(RISK_SUMMARY_FILE.read_text(encoding="utf-8"))
explain_summary = json.loads(
    EXPLAIN_SUMMARY_FILE.read_text(encoding="utf-8")
)

if len(risk) != 12:
    raise AssertionError("Notebook 06 must provide twelve 2026 risk months.")
print(f"Observed national rows: {len(national):,}")
print(f"Forecast model: {forecast_summary['selected_long_horizon_model']}")
print(f"Shock classifier: {risk_summary['selected_classifier']}")
"""
    ),
    markdown(
        """## 3. Construct the 2026 food-cost growth path

Year-on-year growth compares each forecast month with the corresponding observed
month of 2025. This preserves seasonality and avoids comparing index levels with
different calendar months.
"""
    ),
    code(
        """TARGET = "food_cpi_2015_100"
observed = (
    national.loc[national[TARGET].notna(), ["date", TARGET]]
    .sort_values("date")
    .drop_duplicates("date", keep="last")
)
forecast_2026 = (
    forecast.loc[
        forecast["date"].between("2026-01-01", "2026-12-01"),
        ["date", "forecast", "lower_95_empirical", "upper_95_empirical"],
    ]
    .sort_values("date")
    .reset_index(drop=True)
)
if len(forecast_2026) != 12:
    raise AssertionError("Expected all twelve forecast months in 2026.")

previous_year = observed.rename(
    columns={"date": "previous_date", TARGET: "food_cpi_previous_year"}
)
forecast_2026["previous_date"] = forecast_2026["date"] - pd.DateOffset(years=1)
cost_path = forecast_2026.merge(
    previous_year,
    on="previous_date",
    how="left",
    validate="one_to_one",
)
for level_column, growth_column in [
    ("forecast", "food_cost_growth_yoy_pct"),
    ("lower_95_empirical", "food_cost_growth_lower_yoy_pct"),
    ("upper_95_empirical", "food_cost_growth_upper_yoy_pct"),
]:
    cost_path[growth_column] = (
        100
        * (
            cost_path[level_column]
            / cost_path["food_cpi_previous_year"]
            - 1
        )
    )
if cost_path["food_cost_growth_yoy_pct"].isna().any():
    raise AssertionError("Observed 2025 CPI denominators are incomplete.")

cost_path = cost_path.merge(
    risk[["target_date", "selected_model_probability", "input_status"]],
    left_on="date",
    right_on="target_date",
    how="left",
    validate="one_to_one",
).drop(columns="target_date")
cost_path.to_csv(REPORT_OUTPUT / "08_2026_food_cost_path.csv", index=False)
print(cost_path[[
    "date", "forecast", "food_cost_growth_yoy_pct",
    "food_cost_growth_lower_yoy_pct", "food_cost_growth_upper_yoy_pct",
    "selected_model_probability",
]].round(3).to_string(index=False))
"""
    ),
    markdown(
        """## 4. Define the HFASI transparently

For segment \\(s\\) and month \\(t\\):

\\[
HFASI_{s,t}=100+w_s(g^{food}_t-g^{power}_s)
\\]

where \\(w_s\\) is the household food-expenditure share expressed as a proportion,
and both growth rates are percentages. `100` is neutral: values above 100 imply
food costs grow faster than assumed purchasing power; values below 100 imply
improving affordability.

Purchasing-power paths are scenarios, not observed household-income forecasts.
"""
    ),
    code(
        """# Published HCES 2022–23 aggregate expenditure-share benchmarks.
SEGMENTS = {
    "Rural": 0.463,
    "Urban": 0.391,
}

# User-adjustable annual nominal purchasing-power assumptions.
POWER_SCENARIOS = {
    "Downside (0%)": 0.0,
    "Baseline (4%)": 4.0,
    "Upside (8%)": 8.0,
}


def hfasi(food_share, food_growth_pct, power_growth_pct):
    \"\"\"Baseline-100 affordability stress index from the approved synopsis.\"\"\"
    return 100 + food_share * (food_growth_pct - power_growth_pct)


rows = []
for segment, food_share in SEGMENTS.items():
    for scenario, power_growth in POWER_SCENARIOS.items():
        for record in cost_path.to_dict("records"):
            rows.append({
                "date": record["date"],
                "segment": segment,
                "food_expenditure_share": food_share,
                "purchasing_power_scenario": scenario,
                "purchasing_power_growth_pct": power_growth,
                "food_cost_growth_yoy_pct": record[
                    "food_cost_growth_yoy_pct"
                ],
                "hfasi": hfasi(
                    food_share,
                    record["food_cost_growth_yoy_pct"],
                    power_growth,
                ),
                "shock_probability": record["selected_model_probability"],
                "input_status": record["input_status"],
            })
hfasi_monthly = pd.DataFrame(rows)
hfasi_monthly["affordability_state"] = np.select(
    [
        hfasi_monthly["hfasi"] >= 102,
        hfasi_monthly["hfasi"] > 100,
    ],
    ["Elevated stress", "Mild stress"],
    default="Stable or improving",
)
hfasi_monthly.to_csv(REPORT_OUTPUT / "08_hfasi_monthly.csv", index=False)
print(hfasi_monthly.head(12).round(3).to_string(index=False))
"""
    ),
    markdown("## 5. Monthly rural and urban affordability under the baseline scenario"),
    code(
        """baseline = hfasi_monthly.loc[
    hfasi_monthly["purchasing_power_scenario"].eq("Baseline (4%)")
].copy()

fig, ax = plt.subplots(figsize=(12, 6))
for segment, group in baseline.groupby("segment", sort=False):
    ax.plot(
        group["date"],
        group["hfasi"],
        marker="o",
        linewidth=2,
        label=segment,
    )
ax.axhline(100, color="#374151", linestyle="--", linewidth=1, label="Neutral")
ax.set(
    title="2026 Household Food Affordability Stress Index — baseline scenario",
    xlabel="Month",
    ylabel="HFASI (neutral = 100)",
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "08_hfasi_baseline_2026.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 6. Scenario comparison

The scenario range is a decision aid. It shows how conclusions change when
household purchasing power grows more slowly or quickly than the baseline.
"""
    ),
    code(
        """annual_summary = (
    hfasi_monthly.groupby(
        ["segment", "purchasing_power_scenario"],
        as_index=False,
    )
    .agg(
        mean_hfasi=("hfasi", "mean"),
        maximum_hfasi=("hfasi", "max"),
        peak_month=("date", lambda x: x.loc[
            hfasi_monthly.loc[x.index, "hfasi"].idxmax()
        ]),
        months_above_100=("hfasi", lambda x: int((x > 100).sum())),
        months_at_or_above_102=("hfasi", lambda x: int((x >= 102).sum())),
    )
)
annual_summary.to_csv(
    REPORT_OUTPUT / "08_hfasi_annual_summary.csv", index=False
)
print(annual_summary.round(3).to_string(index=False))

scenario_plot = annual_summary.copy()
scenario_plot["mean_hfasi_deviation"] = scenario_plot["mean_hfasi"] - 100
fig, ax = plt.subplots(figsize=(11, 6))
sns.barplot(
    data=scenario_plot,
    x="purchasing_power_scenario",
    y="mean_hfasi_deviation",
    hue="segment",
    order=list(POWER_SCENARIOS),
    ax=ax,
)
ax.axhline(0, color="#374151", linestyle="--", linewidth=1)
ax.set(
    title="Average 2026 HFASI deviation under purchasing-power scenarios",
    xlabel="Scenario",
    ylabel="Mean HFASI minus 100",
)
ax.legend(title="Segment", frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "08_hfasi_scenario_comparison.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 7. Sensitivity analysis

Food shares from 30% to 60% and purchasing-power growth from −2% to 10% test
whether the interpretation depends on a single benchmark. The annual mean
forecast food-cost growth is used for a compact two-dimensional diagnostic.
"""
    ),
    code(
        """mean_food_growth = float(cost_path["food_cost_growth_yoy_pct"].mean())
share_grid = np.round(np.arange(0.30, 0.601, 0.025), 3)
power_grid = np.arange(-2.0, 10.1, 1.0)
sensitivity = pd.DataFrame([
    {
        "food_expenditure_share": share,
        "purchasing_power_growth_pct": power,
        "mean_forecast_food_cost_growth_pct": mean_food_growth,
        "hfasi": hfasi(share, mean_food_growth, power),
    }
    for share in share_grid
    for power in power_grid
])
sensitivity.to_csv(
    REPORT_OUTPUT / "08_hfasi_sensitivity.csv", index=False
)

heatmap_data = sensitivity.pivot(
    index="food_expenditure_share",
    columns="purchasing_power_growth_pct",
    values="hfasi",
).sort_index(ascending=False)
fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(
    heatmap_data,
    cmap="RdYlGn_r",
    center=100,
    annot=False,
    cbar_kws={"label": "HFASI"},
    ax=ax,
)
ax.set(
    title="HFASI sensitivity to food share and purchasing-power growth",
    xlabel="Annual purchasing-power growth (%)",
    ylabel="Food-expenditure share",
)
fig.tight_layout()
fig.savefig(FIGURES / "08_hfasi_sensitivity.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 8. Forecast uncertainty translated into affordability uncertainty

The empirical CPI forecast interval is mapped through the same formula. This is
forecast uncertainty conditional on each scenario; it is not a household-level
confidence interval.
"""
    ),
    code(
        """uncertainty_rows = []
for segment, share in SEGMENTS.items():
    for record in cost_path.to_dict("records"):
        uncertainty_rows.append({
            "date": record["date"],
            "segment": segment,
            "hfasi_point": hfasi(
                share, record["food_cost_growth_yoy_pct"], 4.0
            ),
            "hfasi_lower": hfasi(
                share, record["food_cost_growth_lower_yoy_pct"], 4.0
            ),
            "hfasi_upper": hfasi(
                share, record["food_cost_growth_upper_yoy_pct"], 4.0
            ),
        })
hfasi_uncertainty = pd.DataFrame(uncertainty_rows)
hfasi_uncertainty[["hfasi_lower", "hfasi_upper"]] = np.sort(
    hfasi_uncertainty[["hfasi_lower", "hfasi_upper"]].to_numpy(),
    axis=1,
)
hfasi_uncertainty.to_csv(
    REPORT_OUTPUT / "08_hfasi_forecast_uncertainty.csv", index=False
)

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
for ax, segment in zip(axes, SEGMENTS):
    group = hfasi_uncertainty.loc[
        hfasi_uncertainty["segment"].eq(segment)
    ]
    x = group["date"].to_numpy()
    ax.plot(x, group["hfasi_point"], color="#1f4e79", marker="o")
    ax.fill_between(
        x,
        group["hfasi_lower"].to_numpy(),
        group["hfasi_upper"].to_numpy(),
        color="#93c5fd",
        alpha=0.35,
    )
    ax.axhline(100, color="#374151", linestyle="--", linewidth=1)
    ax.set_title(f"{segment}: baseline purchasing-power scenario")
    ax.set_ylabel("HFASI")
axes[-1].set_xlabel("Month")
fig.suptitle("Affordability range implied by CPI forecast uncertainty", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "08_hfasi_forecast_uncertainty.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown("## 9. Execution summary and artifact validation"),
    code(
        """baseline_peak = baseline.loc[baseline["hfasi"].idxmax()]
summary = {
    "notebook": "08_affordability_index",
    "status": "completed",
    "index_name": "Household Food Affordability Stress Index (HFASI)",
    "formula": (
        "100 + food_expenditure_share * "
        "(food_cost_growth_pct - purchasing_power_growth_pct)"
    ),
    "segments": {
        segment: share for segment, share in SEGMENTS.items()
    },
    "segment_weight_source": (
        "Published HCES 2022-23 aggregate food-expenditure shares; "
        "microdata not used."
    ),
    "purchasing_power_scenarios_pct": POWER_SCENARIOS,
    "forecast_months": int(cost_path["date"].nunique()),
    "mean_2026_food_cost_growth_pct": mean_food_growth,
    "baseline_highest_hfasi_segment": str(baseline_peak["segment"]),
    "baseline_highest_hfasi_month": str(baseline_peak["date"].date()),
    "baseline_highest_hfasi": float(baseline_peak["hfasi"]),
    "highest_2026_shock_probability": float(
        cost_path["selected_model_probability"].max()
    ),
    "interpretation": (
        "Scenario-based representative-segment stress; not a household-level "
        "estimate, causal effect, or welfare threshold."
    ),
    "output_root": str(OUTPUT_ROOT),
}
SUMMARY_FILE = REPORT_OUTPUT / "08_execution_summary.json"
_ = SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

expected_files = [
    REPORT_OUTPUT / "08_2026_food_cost_path.csv",
    REPORT_OUTPUT / "08_hfasi_monthly.csv",
    REPORT_OUTPUT / "08_hfasi_annual_summary.csv",
    REPORT_OUTPUT / "08_hfasi_sensitivity.csv",
    REPORT_OUTPUT / "08_hfasi_forecast_uncertainty.csv",
    FIGURES / "08_hfasi_baseline_2026.png",
    FIGURES / "08_hfasi_scenario_comparison.png",
    FIGURES / "08_hfasi_sensitivity.png",
    FIGURES / "08_hfasi_forecast_uncertainty.png",
    SUMMARY_FILE,
]
missing_artifacts = [str(path) for path in expected_files if not path.exists()]
if missing_artifacts:
    raise FileNotFoundError(
        "Notebook 08 artifacts missing:\\n" + "\\n".join(missing_artifacts)
    )

print("=" * 72)
print("NOTEBOOK 08 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT")
print("=" * 72)
print(json.dumps(summary, indent=2))
print(f"\\nVerified artifacts: {len(expected_files)}")
"""
    ),
    markdown(
        """## Notebook 08 conclusion

HFASI translates forecast food-cost growth into representative rural and urban
affordability pressure. The rural benchmark responds more strongly because food
has a larger expenditure share. Scenario and sensitivity results must accompany
every point estimate because household purchasing-power growth is assumed.

After **Runtime → Run all**, send the Notebook 08 execution summary, annual
scenario table, food-cost growth table, and any red error output.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {
            "name": "08_affordability_index.ipynb",
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
            compile("".join(cell["source"]), f"cell_{position}", "exec")
    print(f"Created and syntax-validated: {OUTPUT}")
    print(f"Notebook cells: {len(cells)}")


if __name__ == "__main__":
    main()
