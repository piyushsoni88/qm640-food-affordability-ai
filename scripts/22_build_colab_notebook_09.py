"""Build the Google Colab version of Notebook 09."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "09_scenario_analysis.ipynb"


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
## Notebook 09 — Scenario and Decision Analysis

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Prerequisite:** Notebooks 01–08 completed successfully

### Objectives

This notebook stress-tests the 2026 baseline by:

1. applying explicit favorable, moderate-stress, and severe-stress assumptions;
2. rescoring the fitted shock classifier under each feature scenario;
3. translating assumed food-cost overlays into rural and urban HFASI;
4. comparing household-budget and procurement-warning signals; and
5. documenting every assumption in report-ready outputs.

Scenarios are **what-if tests**, not forecasts. Their purpose is to examine model
behavior and decision robustness under controlled changes.
"""
    ),
    markdown("## 1. Runtime and Google Drive setup"),
    code(
        """from pathlib import Path
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 190)

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

REPORT_OUTPUT = OUTPUT_ROOT / "reports" / "notebook_outputs"
FIGURES = REPORT_OUTPUT / "figures"
MODELS = OUTPUT_ROOT / "models"
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
    markdown("## 2. Load and verify upstream outputs"),
    code(
        """FEATURE_FILE = REPORT_OUTPUT / "06_2026_conditional_features.csv"
RISK_FILE = REPORT_OUTPUT / "06_2026_monthly_shock_risk.csv"
RISK_SUMMARY_FILE = REPORT_OUTPUT / "06_execution_summary.json"
COST_FILE = REPORT_OUTPUT / "08_2026_food_cost_path.csv"
HFASI_FILE = REPORT_OUTPUT / "08_hfasi_monthly.csv"
HFASI_SUMMARY_FILE = REPORT_OUTPUT / "08_execution_summary.json"

required = [
    FEATURE_FILE, RISK_FILE, RISK_SUMMARY_FILE,
    COST_FILE, HFASI_FILE, HFASI_SUMMARY_FILE,
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Run Notebooks 01–08 first.\\n" + "\\n".join(missing)
    )

features_2026 = pd.read_csv(
    FEATURE_FILE, parse_dates=["feature_month", "target_date"]
)
baseline_risk = pd.read_csv(
    RISK_FILE, parse_dates=["feature_month", "target_date"]
)
cost_path = pd.read_csv(COST_FILE, parse_dates=["date", "previous_date"])
hfasi_monthly = pd.read_csv(HFASI_FILE, parse_dates=["date"])
risk_summary = json.loads(RISK_SUMMARY_FILE.read_text(encoding="utf-8"))
hfasi_summary = json.loads(HFASI_SUMMARY_FILE.read_text(encoding="utf-8"))

SELECTED_MODEL = risk_summary["selected_classifier"]
MODEL_FILE = MODELS / f"06_{SELECTED_MODEL}.joblib"
if not MODEL_FILE.exists():
    raise FileNotFoundError(str(MODEL_FILE))
classifier = joblib.load(MODEL_FILE)

if not (len(features_2026) == len(baseline_risk) == len(cost_path) == 12):
    raise AssertionError("All upstream 2026 paths must contain twelve months.")
print(f"Shock model: {SELECTED_MODEL}")
print(f"HFASI formula: {hfasi_summary['formula']}")
"""
    ),
    markdown(
        """## 3. Define transparent scenarios

Each number is an additive change relative to the Notebook 06/08 baseline.
Climate changes use the feature's original units: rainfall anomaly percentage
points and temperature anomaly degrees Celsius. External-price changes use
year-on-year percentage points. The food-cost overlay directly modifies the
forecast YoY food-cost-growth path for affordability stress.
"""
    ),
    code(
        """SCENARIOS = {
    "Baseline": {
        "food_cost_overlay_pp": 0.0,
        "mandi_yoy_delta_pp": 0.0,
        "world_wheat_yoy_delta_pp": 0.0,
        "world_crude_yoy_delta_pp": 0.0,
        "rainfall_anomaly_delta_pp": 0.0,
        "temperature_anomaly_delta_c": 0.0,
    },
    "Favorable supply": {
        "food_cost_overlay_pp": -2.0,
        "mandi_yoy_delta_pp": -5.0,
        "world_wheat_yoy_delta_pp": -5.0,
        "world_crude_yoy_delta_pp": -10.0,
        "rainfall_anomaly_delta_pp": 10.0,
        "temperature_anomaly_delta_c": -0.5,
    },
    "Moderate stress": {
        "food_cost_overlay_pp": 3.0,
        "mandi_yoy_delta_pp": 10.0,
        "world_wheat_yoy_delta_pp": 10.0,
        "world_crude_yoy_delta_pp": 15.0,
        "rainfall_anomaly_delta_pp": -15.0,
        "temperature_anomaly_delta_c": 1.5,
    },
    "Severe stress": {
        "food_cost_overlay_pp": 6.0,
        "mandi_yoy_delta_pp": 20.0,
        "world_wheat_yoy_delta_pp": 20.0,
        "world_crude_yoy_delta_pp": 30.0,
        "rainfall_anomaly_delta_pp": -30.0,
        "temperature_anomaly_delta_c": 3.0,
    },
}
scenario_assumptions = (
    pd.DataFrame(SCENARIOS)
    .T.rename_axis("scenario")
    .reset_index()
)
scenario_assumptions.to_csv(
    REPORT_OUTPUT / "09_scenario_assumptions.csv", index=False
)
print(scenario_assumptions.to_string(index=False))
"""
    ),
    markdown(
        """## 4. Rescore the shock classifier

The target CPI growth features are also adjusted by the food-cost overlay. This
keeps the scenario internally aligned: a higher food-cost path affects both the
affordability calculation and the classifier's current/lagged inflation signals.
"""
    ),
    code(
        """SHOCK_FEATURES = [
    "food_cpi_yoy_current", "food_cpi_yoy_lag1", "food_cpi_mom_current",
    "mandi_yoy_current", "world_wheat_yoy_current",
    "world_crude_yoy_current", "rainfall_anomaly_current",
    "temperature_anomaly_current", "target_month_sin",
    "target_month_cos", "trend_months",
]
base_matrix = features_2026[SHOCK_FEATURES].copy()
scenario_risk_rows = []

for scenario, changes in SCENARIOS.items():
    matrix = base_matrix.copy()
    food_overlay = changes["food_cost_overlay_pp"]
    matrix["food_cpi_yoy_current"] += food_overlay
    matrix["food_cpi_yoy_lag1"] += food_overlay
    # Approximate a smooth overlay as one-twelfth of the annual percentage-point
    # shift in the month-on-month feature.
    matrix["food_cpi_mom_current"] += food_overlay / 12
    matrix["mandi_yoy_current"] += changes["mandi_yoy_delta_pp"]
    matrix["world_wheat_yoy_current"] += changes[
        "world_wheat_yoy_delta_pp"
    ]
    matrix["world_crude_yoy_current"] += changes[
        "world_crude_yoy_delta_pp"
    ]
    matrix["rainfall_anomaly_current"] += changes[
        "rainfall_anomaly_delta_pp"
    ]
    matrix["temperature_anomaly_current"] += changes[
        "temperature_anomaly_delta_c"
    ]
    probabilities = classifier.predict_proba(matrix)[:, 1]
    for row_number, probability in enumerate(probabilities):
        scenario_risk_rows.append({
            "scenario": scenario,
            "target_date": features_2026.loc[row_number, "target_date"],
            "horizon_months": int(row_number + 1),
            "shock_probability": float(probability),
        })

scenario_risk = pd.DataFrame(scenario_risk_rows)
baseline_reproduced = scenario_risk.loc[
    scenario_risk["scenario"].eq("Baseline"), "shock_probability"
].to_numpy()
if not np.allclose(
    baseline_reproduced,
    baseline_risk["selected_model_probability"].to_numpy(),
    atol=1e-10,
):
    raise AssertionError("Baseline scenario does not reproduce Notebook 06.")
scenario_risk.to_csv(
    REPORT_OUTPUT / "09_scenario_shock_probabilities.csv", index=False
)
"""
    ),
    markdown("## 5. Visualize scenario shock probabilities"),
    code(
        """fig, ax = plt.subplots(figsize=(13, 6))
palette = {
    "Baseline": "#1f4e79",
    "Favorable supply": "#15803d",
    "Moderate stress": "#d97706",
    "Severe stress": "#b91c1c",
}
for scenario, group in scenario_risk.groupby("scenario", sort=False):
    ax.plot(
        group["target_date"],
        group["shock_probability"],
        marker="o",
        linewidth=2,
        label=scenario,
        color=palette[scenario],
    )
ax.axhline(0.50, color="#374151", linestyle="--", linewidth=1, label="0.50 cutoff")
ax.set(
    title="Conditional 2026 shock probability under alternative scenarios",
    xlabel="Target month",
    ylabel="Predicted shock probability",
    ylim=(0, 1),
)
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(FIGURES / "09_scenario_shock_probabilities.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 6. Translate scenarios into HFASI

The comparison holds nominal purchasing-power growth at 4% so that differences
come only from the scenario food-cost path. Rural and urban shares match
Notebook 08.
"""
    ),
    code(
        """SEGMENTS = hfasi_summary["segments"]
POWER_GROWTH = 4.0


def hfasi(food_share, food_growth_pct, power_growth_pct):
    return 100 + food_share * (food_growth_pct - power_growth_pct)


hfasi_rows = []
for scenario, changes in SCENARIOS.items():
    adjusted_growth = (
        cost_path["food_cost_growth_yoy_pct"]
        + changes["food_cost_overlay_pp"]
    )
    for segment, share in SEGMENTS.items():
        for row_number, date in enumerate(cost_path["date"]):
            hfasi_rows.append({
                "scenario": scenario,
                "date": date,
                "segment": segment,
                "food_expenditure_share": float(share),
                "purchasing_power_growth_pct": POWER_GROWTH,
                "scenario_food_cost_growth_yoy_pct": float(
                    adjusted_growth.iloc[row_number]
                ),
                "hfasi": float(hfasi(
                    share,
                    adjusted_growth.iloc[row_number],
                    POWER_GROWTH,
                )),
            })
scenario_hfasi = pd.DataFrame(hfasi_rows)
scenario_hfasi.to_csv(
    REPORT_OUTPUT / "09_scenario_hfasi.csv", index=False
)

baseline_hfasi = (
    scenario_hfasi.loc[scenario_hfasi["scenario"].eq("Baseline")]
    .sort_values(["segment", "date"])["hfasi"]
    .to_numpy()
)
notebook08_baseline = (
    hfasi_monthly.loc[
        hfasi_monthly["purchasing_power_scenario"].eq("Baseline (4%)")
    ]
    .sort_values(["segment", "date"])["hfasi"]
    .to_numpy()
)
if not np.allclose(baseline_hfasi, notebook08_baseline, atol=1e-10):
    raise AssertionError("Baseline HFASI does not reproduce Notebook 08.")
"""
    ),
    markdown("## 7. Scenario summary and decision signals"),
    code(
        """risk_summary_table = (
    scenario_risk.groupby("scenario", as_index=False)
    .agg(
        mean_shock_probability=("shock_probability", "mean"),
        maximum_shock_probability=("shock_probability", "max"),
        peak_risk_month=("target_date", lambda x: x.loc[
            scenario_risk.loc[x.index, "shock_probability"].idxmax()
        ]),
        months_probability_at_or_above_10pct=(
            "shock_probability", lambda x: int((x >= 0.10).sum())
        ),
        months_probability_at_or_above_50pct=(
            "shock_probability", lambda x: int((x >= 0.50).sum())
        ),
    )
)
hfasi_summary_table = (
    scenario_hfasi.groupby(["scenario", "segment"], as_index=False)
    .agg(
        mean_hfasi=("hfasi", "mean"),
        maximum_hfasi=("hfasi", "max"),
        months_above_neutral=("hfasi", lambda x: int((x > 100).sum())),
        months_at_or_above_102=("hfasi", lambda x: int((x >= 102).sum())),
    )
)
scenario_summary = hfasi_summary_table.merge(
    risk_summary_table, on="scenario", how="left", validate="many_to_one"
)
scenario_summary["household_budget_signal"] = np.select(
    [
        scenario_summary["maximum_hfasi"] >= 102,
        scenario_summary["maximum_hfasi"] > 100,
    ],
    ["Elevated", "Watch"],
    default="Normal",
)
scenario_summary["procurement_signal"] = np.select(
    [
        scenario_summary["maximum_shock_probability"] >= 0.50,
        scenario_summary["maximum_shock_probability"] >= 0.10,
    ],
    ["Alert", "Watch"],
    default="Normal",
)
scenario_summary.to_csv(
    REPORT_OUTPUT / "09_scenario_decision_summary.csv", index=False
)
print(scenario_summary.round(4).to_string(index=False))
"""
    ),
    markdown("## 8. Compare affordability deviations from neutral"),
    code(
        """plot_summary = hfasi_summary_table.copy()
plot_summary["mean_hfasi_deviation"] = plot_summary["mean_hfasi"] - 100
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    data=plot_summary,
    x="scenario",
    y="mean_hfasi_deviation",
    hue="segment",
    order=list(SCENARIOS),
    ax=ax,
)
ax.axhline(0, color="#374151", linestyle="--", linewidth=1)
ax.set(
    title="Mean 2026 HFASI deviation from neutral under stress scenarios",
    xlabel="Scenario",
    ylabel="Mean HFASI minus 100",
)
ax.tick_params(axis="x", rotation=15)
ax.legend(title="Segment", frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "09_scenario_hfasi_comparison.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 9. Joint household and procurement decision map

The 10% watch and 50% alert thresholds are illustrative operating rules, not
statistically optimized policy thresholds.
"""
    ),
    code(
        """decision_plot = scenario_summary.copy()
fig, ax = plt.subplots(figsize=(10, 7))
for scenario, group in decision_plot.groupby("scenario"):
    ax.scatter(
        group["maximum_hfasi"],
        group["maximum_shock_probability"],
        s=130,
        label=scenario,
        color=palette[scenario],
    )
    for _, row in group.iterrows():
        ax.annotate(
            row["segment"],
            (row["maximum_hfasi"], row["maximum_shock_probability"]),
            xytext=(5, 4),
            textcoords="offset points",
        )
ax.axvline(100, color="#6b7280", linestyle="--", linewidth=1)
ax.axhline(0.10, color="#d97706", linestyle="--", linewidth=1)
ax.axhline(0.50, color="#b91c1c", linestyle="--", linewidth=1)
ax.set(
    title="Joint affordability and shock-risk scenario map",
    xlabel="Maximum monthly HFASI",
    ylabel="Maximum shock probability",
    ylim=(0, 1),
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "09_joint_decision_map.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown("## 10. Execution summary and artifact validation"),
    code(
        """worst_row = scenario_summary.loc[
    scenario_summary["maximum_hfasi"].idxmax()
]
highest_risk_row = scenario_summary.loc[
    scenario_summary["maximum_shock_probability"].idxmax()
]
summary = {
    "notebook": "09_scenario_analysis",
    "status": "completed",
    "scenarios": list(SCENARIOS),
    "scenario_months": int(scenario_risk["target_date"].nunique()),
    "baseline_reproduces_notebook_06": True,
    "baseline_reproduces_notebook_08": True,
    "purchasing_power_growth_held_pct": POWER_GROWTH,
    "highest_hfasi_scenario": str(worst_row["scenario"]),
    "highest_hfasi_segment": str(worst_row["segment"]),
    "highest_monthly_hfasi": float(worst_row["maximum_hfasi"]),
    "highest_shock_probability_scenario": str(
        highest_risk_row["scenario"]
    ),
    "highest_scenario_shock_probability": float(
        highest_risk_row["maximum_shock_probability"]
    ),
    "interpretation": (
        "Controlled what-if stress test; scenario assumptions are not forecasts "
        "or causal effect estimates."
    ),
    "output_root": str(OUTPUT_ROOT),
}
SUMMARY_FILE = REPORT_OUTPUT / "09_execution_summary.json"
_ = SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

expected_files = [
    REPORT_OUTPUT / "09_scenario_assumptions.csv",
    REPORT_OUTPUT / "09_scenario_shock_probabilities.csv",
    REPORT_OUTPUT / "09_scenario_hfasi.csv",
    REPORT_OUTPUT / "09_scenario_decision_summary.csv",
    FIGURES / "09_scenario_shock_probabilities.png",
    FIGURES / "09_scenario_hfasi_comparison.png",
    FIGURES / "09_joint_decision_map.png",
    SUMMARY_FILE,
]
missing_artifacts = [str(path) for path in expected_files if not path.exists()]
if missing_artifacts:
    raise FileNotFoundError(
        "Notebook 09 artifacts missing:\\n" + "\\n".join(missing_artifacts)
    )

print("=" * 72)
print("NOTEBOOK 09 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT")
print("=" * 72)
print(json.dumps(summary, indent=2))
print(f"\\nVerified artifacts: {len(expected_files)}")
"""
    ),
    markdown(
        """## Notebook 09 conclusion

The scenario framework separates the baseline forecast from controlled stress
assumptions and checks two decision dimensions: household affordability and
procurement-oriented shock risk. Results must always be accompanied by the
assumption table and should not be described as event probabilities for the
scenario itself.

After **Runtime → Run all**, send the Notebook 09 execution summary, scenario
decision table, and any red error output.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {
            "name": "09_scenario_analysis.ipynb",
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
