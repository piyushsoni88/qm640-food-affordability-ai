"""Build the Google Colab version of Notebook 10."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "10_final_results.ipynb"


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
## Notebook 10 — Final Results Consolidation

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Prerequisite:** Notebooks 01–09 completed successfully

### Objectives

This notebook:

1. verifies that every preceding notebook completed;
2. consolidates the final numerical findings without manual transcription;
3. maps evidence to the four research questions;
4. creates model, decision, and limitation scorecards;
5. produces a final executive-results dashboard;
6. inventories report artifacts for reproducibility; and
7. writes a concise evidence brief for the interim report.

No new model is selected here. This notebook reports the results already produced
under their original chronological validation and scenario assumptions.
"""
    ),
    markdown("## 1. Runtime and Google Drive setup"),
    code(
        """from pathlib import Path
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 200)

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
    markdown("## 2. Verify execution summaries from Notebooks 01–09"),
    code(
        """summaries = {}
missing_summaries = []
for number in range(1, 10):
    path = REPORT_OUTPUT / f"{number:02d}_execution_summary.json"
    if not path.exists():
        missing_summaries.append(str(path))
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries[number] = payload
if missing_summaries:
    raise FileNotFoundError(
        "Missing notebook summaries:\\n" + "\\n".join(missing_summaries)
    )
not_completed = {
    number: payload.get("status")
    for number, payload in summaries.items()
    if payload.get("status") != "completed"
}
if not_completed:
    raise AssertionError(f"Incomplete upstream notebooks: {not_completed}")

execution_status = pd.DataFrame([
    {
        "notebook_number": number,
        "notebook": payload.get("notebook"),
        "status": payload.get("status"),
    }
    for number, payload in summaries.items()
])
execution_status.to_csv(
    REPORT_OUTPUT / "10_upstream_execution_status.csv", index=False
)
print(execution_status.to_string(index=False))
"""
    ),
    markdown("## 3. Load final analytical outputs"),
    code(
        """NATIONAL_FILE = PROCESSED / "cleaned_national_monthly.csv.gz"
FORECAST_FILE = REPORT_OUTPUT / "05_twenty_four_month_conditional_forecast.csv"
RISK_FILE = REPORT_OUTPUT / "06_2026_monthly_shock_risk.csv"
HFASI_FILE = REPORT_OUTPUT / "08_hfasi_monthly.csv"
SCENARIO_FILE = REPORT_OUTPUT / "09_scenario_decision_summary.csv"

required = [
    NATIONAL_FILE, FORECAST_FILE, RISK_FILE, HFASI_FILE, SCENARIO_FILE
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("\\n".join(missing))

national = pd.read_csv(NATIONAL_FILE, parse_dates=["date"], low_memory=False)
forecast = pd.read_csv(FORECAST_FILE, parse_dates=["date"])
risk = pd.read_csv(RISK_FILE, parse_dates=["feature_month", "target_date"])
hfasi = pd.read_csv(HFASI_FILE, parse_dates=["date"])
scenario = pd.read_csv(SCENARIO_FILE, parse_dates=["peak_risk_month"])

forecast_2026 = forecast.loc[
    forecast["date"].between("2026-01-01", "2026-12-01")
].copy()
hfasi_baseline = hfasi.loc[
    hfasi["purchasing_power_scenario"].eq("Baseline (4%)")
].copy()
"""
    ),
    markdown(
        """## 4. Final key findings

The table stores exact report numbers and separates baseline forecasts from
conditional scenarios.
"""
    ),
    code(
        """s03, s04, s05, s06, s07, s08, s09 = [
    summaries[number] for number in [3, 4, 5, 6, 7, 8, 9]
]
key_findings = pd.DataFrame([
    {
        "domain": "Data",
        "finding": "Clean state-month-commodity observations",
        "value": summaries[2]["clean_state_rows"],
        "scope": "2001–2026 YTD; unbalanced official reporting panel",
    },
    {
        "domain": "EDA",
        "finding": "Most volatile commodity by YoY standard deviation",
        "value": s03["most_volatile_commodity_by_yoy_sd"],
        "scope": "Descriptive association",
    },
    {
        "domain": "Statistical analysis",
        "finding": "National full-model adjusted R-squared",
        "value": s04["national_full_adjusted_r_squared"],
        "scope": "Conditional association; no significant national transmission terms",
    },
    {
        "domain": "Forecasting",
        "finding": "One-step selected model",
        "value": s05["selected_one_step_model"],
        "scope": f"Rolling RMSE {s05['selected_one_step_rmse']:.3f}",
    },
    {
        "domain": "Forecasting",
        "finding": "Twenty-four-step selected model",
        "value": s05["selected_long_horizon_model"],
        "scope": f"All-step RMSE {s05['selected_long_horizon_rmse_all_steps']:.3f}",
    },
    {
        "domain": "Shock classification",
        "finding": "Selected classifier balanced accuracy",
        "value": s06["selected_balanced_accuracy"],
        "scope": "60 chronological targets; only five observed shocks",
    },
    {
        "domain": "2026 baseline",
        "finding": "Highest conditional shock probability",
        "value": s06["highest_2026_selected_model_probability"],
        "scope": s06["highest_2026_risk_month"],
    },
    {
        "domain": "Affordability",
        "finding": "Mean forecast food-cost growth",
        "value": s08["mean_2026_food_cost_growth_pct"],
        "scope": "2026 YoY percentage",
    },
    {
        "domain": "Affordability",
        "finding": "Highest baseline HFASI",
        "value": s08["baseline_highest_hfasi"],
        "scope": (
            f"{s08['baseline_highest_hfasi_segment']}; "
            f"{s08['baseline_highest_hfasi_month']}"
        ),
    },
    {
        "domain": "Stress scenario",
        "finding": "Highest severe-scenario shock probability",
        "value": s09["highest_scenario_shock_probability"],
        "scope": "Controlled what-if scenario; not a forecast",
    },
    {
        "domain": "Stress scenario",
        "finding": "Highest severe-scenario HFASI",
        "value": s09["highest_monthly_hfasi"],
        "scope": f"{s09['highest_hfasi_segment']} representative segment",
    },
])
key_findings.to_csv(REPORT_OUTPUT / "10_final_key_findings.csv", index=False)
print(key_findings.to_string(index=False))
"""
    ),
    markdown("## 5. Research-question evidence map"),
    code(
        """rq_evidence = pd.DataFrame([
    {
        "research_question": "RQ1 — What drives food-price change?",
        "evidence": (
            "National HAC model found no 5% significant transmission terms; "
            "temperature anomaly was Holm-significant in 2-year and 5-year "
            "state models but not the 10-year model. Current YoY CPI dominated "
            "shock-model permutation importance."
        ),
        "answer_strength": "Mixed/conditional",
        "causal_claim": "No",
        "primary_notebooks": "03, 04, 07",
    },
    {
        "research_question": "RQ2 — How accurately can food prices be forecast?",
        "evidence": (
            "Last-value persistence won 60 one-step origins (RMSE 2.063). "
            "Damped ETS won 24-step validation (all-step RMSE 6.044) and "
            "generated the 2026–2027 conditional CPI path."
        ),
        "answer_strength": "Chronologically validated",
        "causal_claim": "No",
        "primary_notebooks": "05, 07",
    },
    {
        "research_question": "RQ3 — Can abnormal shocks and affordability stress be identified?",
        "evidence": (
            "HistGradientBoosting achieved balanced accuracy 0.773 with "
            "recall 0.600. Baseline 2026 shock risk stayed below 4.2%. "
            "HFASI distinguished rural/urban outcomes under explicit "
            "purchasing-power assumptions."
        ),
        "answer_strength": "Predictive with rare-event limitation",
        "causal_claim": "No",
        "primary_notebooks": "06, 08",
    },
    {
        "research_question": "RQ4 — How can the framework support decisions?",
        "evidence": (
            "Baseline and moderate scenarios did not trigger procurement "
            "alerts. The severe scenario triggered both elevated household "
            "stress and a procurement alert, demonstrating an auditable "
            "two-dimensional warning framework."
        ),
        "answer_strength": "Scenario demonstration",
        "causal_claim": "No",
        "primary_notebooks": "09",
    },
])
rq_evidence.to_csv(
    REPORT_OUTPUT / "10_research_question_evidence.csv", index=False
)
print(rq_evidence.to_string(index=False))
"""
    ),
    markdown("## 6. Final model and decision scorecard"),
    code(
        """scorecard = pd.DataFrame([
    {
        "component": "One-step food CPI forecast",
        "selected_method": s05["selected_one_step_model"],
        "primary_metric": "RMSE",
        "metric_value": s05["selected_one_step_rmse"],
        "validation": "60 expanding chronological origins",
        "decision_use": "Near-term monitoring",
    },
    {
        "component": "24-month food CPI forecast",
        "selected_method": s05["selected_long_horizon_model"],
        "primary_metric": "All-step RMSE",
        "metric_value": s05["selected_long_horizon_rmse_all_steps"],
        "validation": "24 origins × 24 horizons",
        "decision_use": "Conditional planning path",
    },
    {
        "component": "Next-month shock classification",
        "selected_method": s06["selected_classifier"],
        "primary_metric": "Balanced accuracy",
        "metric_value": s06["selected_balanced_accuracy"],
        "validation": "60 expanding chronological origins",
        "decision_use": "Risk watch; not automatic policy trigger",
    },
    {
        "component": "Household affordability",
        "selected_method": "HFASI scenario index",
        "primary_metric": "Neutral reference",
        "metric_value": 100.0,
        "validation": "Formula and sensitivity consistency",
        "decision_use": "Representative rural/urban budget stress",
    },
])
scorecard.to_csv(REPORT_OUTPUT / "10_model_scorecard.csv", index=False)
print(scorecard.to_string(index=False))
"""
    ),
    markdown("## 7. Limitations register"),
    code(
        """limitations = pd.DataFrame([
    {
        "area": "Household representation",
        "limitation": (
            "HFASI uses published aggregate rural/urban food shares because "
            "HCES microdata were not supplied."
        ),
        "impact": "No household-level or expenditure-quantile inference",
        "mitigation": "Label representative segments and report sensitivity grid",
    },
    {
        "area": "Rare shocks",
        "limitation": "Only five shocks occurred in the 60-month evaluation window.",
        "impact": "Classifier recall and precision have high sampling uncertainty",
        "mitigation": "Report confusion matrix, PR metrics, and avoid policy automation",
    },
    {
        "area": "Scenario non-monotonicity",
        "limitation": (
            "The favorable scenario has slightly higher modeled risk than "
            "baseline due to nonlinear interactions."
        ),
        "impact": "Scenario probabilities are model-sensitive",
        "mitigation": "Report assumptions, HFASI, and classifier outputs together",
    },
    {
        "area": "Causal interpretation",
        "limitation": "Regression and explainability outputs are associational.",
        "impact": "No driver is established as a causal intervention target",
        "mitigation": "Use causal language prohibitions throughout the report",
    },
    {
        "area": "Long-horizon uncertainty",
        "limitation": "Only 24 historical origins calibrate 24-step intervals.",
        "impact": "Tail uncertainty is approximate",
        "mitigation": "Report horizon-specific intervals and conditional assumptions",
    },
    {
        "area": "Market coverage",
        "limitation": "AGMARKNET reporting is an unbalanced state-market panel.",
        "impact": "Coverage changes can resemble price changes",
        "mitigation": "Retain coverage measures and avoid interpolating missing prices",
    },
])
limitations.to_csv(REPORT_OUTPUT / "10_limitations_register.csv", index=False)
print(limitations.to_string(index=False))
"""
    ),
    markdown("## 8. Executive results dashboard"),
    code(
        """fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# A. Observed history and long-horizon forecast
history = national.loc[
    national["food_cpi_2015_100"].notna(),
    ["date", "food_cpi_2015_100"],
].tail(60)
axes[0, 0].plot(
    history["date"], history["food_cpi_2015_100"],
    color="#1f4e79", linewidth=2, label="Observed"
)
axes[0, 0].plot(
    forecast["date"], forecast["forecast"],
    color="#d97706", linewidth=2, label="Damped ETS"
)
axes[0, 0].fill_between(
    forecast["date"].to_numpy(),
    forecast["lower_95_empirical"].to_numpy(),
    forecast["upper_95_empirical"].to_numpy(),
    color="#fbbf24", alpha=0.25,
)
axes[0, 0].set_title("A. National food CPI forecast")
axes[0, 0].set_ylabel("Food CPI (2015=100)")
axes[0, 0].legend(frameon=False)

# B. Baseline monthly shock risk
axes[0, 1].plot(
    risk["target_date"], risk["selected_model_probability"],
    color="#b91c1c", marker="o", linewidth=2,
)
axes[0, 1].axhline(0.50, color="#6b7280", linestyle="--", linewidth=1)
axes[0, 1].set_ylim(0, 1)
axes[0, 1].set_title("B. Conditional 2026 shock risk")
axes[0, 1].set_ylabel("Probability")

# C. Baseline HFASI
for segment, group in hfasi_baseline.groupby("segment", sort=False):
    axes[1, 0].plot(
        group["date"], group["hfasi"],
        marker="o", linewidth=2, label=segment,
    )
axes[1, 0].axhline(100, color="#374151", linestyle="--", linewidth=1)
axes[1, 0].set_title("C. Baseline 2026 HFASI")
axes[1, 0].set_ylabel("HFASI (neutral=100)")
axes[1, 0].legend(frameon=False)

# D. Joint scenario decision space
palette = {
    "Baseline": "#1f4e79",
    "Favorable supply": "#15803d",
    "Moderate stress": "#d97706",
    "Severe stress": "#b91c1c",
}
for scenario_name, group in scenario.groupby("scenario"):
    axes[1, 1].scatter(
        group["maximum_hfasi"],
        group["maximum_shock_probability"],
        s=100,
        label=scenario_name,
        color=palette[scenario_name],
    )
axes[1, 1].axvline(100, color="#6b7280", linestyle="--", linewidth=1)
axes[1, 1].axhline(0.10, color="#d97706", linestyle="--", linewidth=1)
axes[1, 1].axhline(0.50, color="#b91c1c", linestyle="--", linewidth=1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_title("D. Joint scenario decision space")
axes[1, 1].set_xlabel("Maximum HFASI")
axes[1, 1].set_ylabel("Maximum shock probability")
axes[1, 1].legend(frameon=False, fontsize=9)

for ax in axes.flat:
    ax.tick_params(axis="x", rotation=20)
fig.suptitle(
    "Food Price Affordability AI — consolidated interim results",
    fontsize=17,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(FIGURES / "10_executive_results_dashboard.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 9. Reproducibility inventory

Hashes cover the report tables, summaries, and figures present before the final
manifest is written. They can detect accidental changes between analysis and
report preparation.
"""
    ),
    code(
        """inventory_rows = []
inventory_candidates = sorted(
    [
        path for path in REPORT_OUTPUT.glob("*")
        if path.is_file() and path.name != "10_artifact_inventory.csv"
    ]
    + [path for path in FIGURES.glob("*.png") if path.is_file()]
)
for path in inventory_candidates:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    inventory_rows.append({
        "artifact": str(path.relative_to(OUTPUT_ROOT)),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    })
artifact_inventory = pd.DataFrame(inventory_rows)
artifact_inventory.to_csv(
    REPORT_OUTPUT / "10_artifact_inventory.csv", index=False
)
print(f"Inventoried report artifacts: {len(artifact_inventory):,}")
"""
    ),
    markdown("## 10. Write interim-report evidence brief"),
    code(
        """brief_lines = [
    "# QM640 Interim Report Evidence Brief",
    "",
    "## Core validated results",
    "",
    f"- The cleaned state panel contains {summaries[2]['clean_state_rows']:,} rows.",
    f"- The one-step winner is `{s05['selected_one_step_model']}` "
    f"(RMSE {s05['selected_one_step_rmse']:.3f}).",
    f"- The 24-step winner is `{s05['selected_long_horizon_model']}` "
    f"(all-step RMSE {s05['selected_long_horizon_rmse_all_steps']:.3f}).",
    f"- The shock classifier is `{s06['selected_classifier']}` with balanced "
    f"accuracy {s06['selected_balanced_accuracy']:.3f} and recall "
    f"{s06['selected_recall']:.3f}.",
    f"- Baseline 2026 maximum shock probability is "
    f"{100*s06['highest_2026_selected_model_probability']:.2f}%.",
    f"- Mean forecast 2026 food-cost growth is "
    f"{s08['mean_2026_food_cost_growth_pct']:.2f}%.",
    f"- Severe-scenario maximum shock probability is "
    f"{100*s09['highest_scenario_shock_probability']:.2f}% and maximum "
    f"HFASI is {s09['highest_monthly_hfasi']:.2f}.",
    "",
    "## Required interpretation",
    "",
    "- Forecast and classification results are predictive, not causal.",
    "- January 2026 uses observed inputs; later 2026 risks are conditional.",
    "- HFASI uses representative aggregate rural/urban food shares.",
    "- Scenario assumptions are controlled stress tests, not forecasts.",
    "- The rare-event evaluation contains only five observed shocks.",
]
BRIEF_FILE = REPORT_OUTPUT / "10_interim_report_evidence_brief.md"
_ = BRIEF_FILE.write_text("\\n".join(brief_lines), encoding="utf-8")
print("\\n".join(brief_lines))
"""
    ),
    markdown("## 11. Execution summary and final validation"),
    code(
        """summary = {
    "notebook": "10_final_results",
    "status": "completed",
    "upstream_notebooks_verified": int(len(summaries)),
    "research_questions_mapped": int(len(rq_evidence)),
    "key_findings_consolidated": int(len(key_findings)),
    "limitations_registered": int(len(limitations)),
    "artifacts_in_inventory": int(len(artifact_inventory)),
    "one_step_model": str(s05["selected_one_step_model"]),
    "long_horizon_model": str(s05["selected_long_horizon_model"]),
    "shock_classifier": str(s06["selected_classifier"]),
    "baseline_2026_max_shock_probability": float(
        s06["highest_2026_selected_model_probability"]
    ),
    "mean_2026_food_cost_growth_pct": float(
        s08["mean_2026_food_cost_growth_pct"]
    ),
    "severe_scenario_max_shock_probability": float(
        s09["highest_scenario_shock_probability"]
    ),
    "severe_scenario_max_hfasi": float(s09["highest_monthly_hfasi"]),
    "report_readiness": (
        "Analysis artifacts consolidated; final narrative must retain "
        "documented assumptions and limitations."
    ),
    "output_root": str(OUTPUT_ROOT),
}
SUMMARY_FILE = REPORT_OUTPUT / "10_execution_summary.json"
_ = SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

expected_files = [
    REPORT_OUTPUT / "10_upstream_execution_status.csv",
    REPORT_OUTPUT / "10_final_key_findings.csv",
    REPORT_OUTPUT / "10_research_question_evidence.csv",
    REPORT_OUTPUT / "10_model_scorecard.csv",
    REPORT_OUTPUT / "10_limitations_register.csv",
    REPORT_OUTPUT / "10_artifact_inventory.csv",
    REPORT_OUTPUT / "10_interim_report_evidence_brief.md",
    FIGURES / "10_executive_results_dashboard.png",
    SUMMARY_FILE,
]
missing_artifacts = [str(path) for path in expected_files if not path.exists()]
if missing_artifacts:
    raise FileNotFoundError(
        "Notebook 10 artifacts missing:\\n" + "\\n".join(missing_artifacts)
    )

print("=" * 72)
print("NOTEBOOK 10 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT")
print("=" * 72)
print(json.dumps(summary, indent=2))
print(f"\\nVerified final artifacts: {len(expected_files)}")
"""
    ),
    markdown(
        """## Notebook 10 conclusion

All analytical stages are now connected from acquisition through decision
scenarios. The next step is to update the interim-report template using the
verified evidence brief, tables, dashboard, and limitations register.

After **Runtime → Run all**, send the Notebook 10 execution summary, research
question evidence table, executive dashboard, and any red error output.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {
            "name": "10_final_results.ipynb",
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
