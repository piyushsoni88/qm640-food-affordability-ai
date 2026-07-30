"""Build the Google Colab version of Notebook 07."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "07_explainability.ipynb"


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
## Notebook 07 — Forecast and Shock-Model Explainability

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Prerequisite:** Notebooks 01–06 completed successfully

### Explainability objectives

This notebook explains:

1. why a one-step naïve forecast beat complex alternatives;
2. which variables drove the closest complex one-step model (Ridge ARX);
3. how damped ETS combines level, trend, and seasonality for 2026–2027;
4. which variables contributed to shock-classifier discrimination; and
5. which inputs increased or decreased predicted shock risk across all of 2026.

Permutation importance and local replacement sensitivity are predictive
explanations. They do not establish causal effects.
"""
    ),
    markdown("## 1. Runtime and Google Drive setup"),
    code(
        """from pathlib import Path
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import joblib

try:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance, PartialDependenceDisplay
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "scikit-learn>=1.5", "statsmodels>=0.14"],
        check=True,
    )
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance, PartialDependenceDisplay
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
print(f"scikit-learn: {sklearn.__version__}")
print(f"Output root: {OUTPUT_ROOT}")
"""
    ),
    markdown(
        """## 2. Load verified model results

The selected model names and performance are read from the saved summaries rather
than typed manually.
"""
    ),
    code(
        """NATIONAL_FILE = PROCESSED / "cleaned_national_monthly.csv.gz"
NB05_FILE = REPORT_OUTPUT / "05_execution_summary.json"
NB06_FILE = REPORT_OUTPUT / "06_execution_summary.json"
FORECAST_FILE = REPORT_OUTPUT / "05_twenty_four_month_conditional_forecast.csv"
ANNUAL_RISK_FILE = REPORT_OUTPUT / "06_2026_monthly_shock_risk.csv"
ANNUAL_FEATURE_FILE = REPORT_OUTPUT / "06_2026_conditional_features.csv"

required = [
    NATIONAL_FILE,
    NB05_FILE,
    NB06_FILE,
    FORECAST_FILE,
    ANNUAL_RISK_FILE,
    ANNUAL_FEATURE_FILE,
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Run Notebooks 01–06 first.\\n" + "\\n".join(missing)
    )

national = pd.read_csv(NATIONAL_FILE, parse_dates=["date"], low_memory=False)
nb05 = json.loads(NB05_FILE.read_text(encoding="utf-8"))
nb06 = json.loads(NB06_FILE.read_text(encoding="utf-8"))
FINAL_SHOCK_MODEL_FILE = (
    OUTPUT_ROOT / "models" / f"06_{nb06['selected_classifier']}.joblib"
)
if not FINAL_SHOCK_MODEL_FILE.exists():
    raise FileNotFoundError(str(FINAL_SHOCK_MODEL_FILE))
forecast_24 = pd.read_csv(FORECAST_FILE, parse_dates=["date"])
annual_risk = pd.read_csv(
    ANNUAL_RISK_FILE, parse_dates=["feature_month", "target_date"]
)
annual_features = pd.read_csv(
    ANNUAL_FEATURE_FILE, parse_dates=["feature_month", "target_date"]
)
final_shock_classifier = joblib.load(FINAL_SHOCK_MODEL_FILE)

print(f"One-step selected model: {nb05['selected_one_step_model']}")
print(f"Long-horizon selected model: {nb05['selected_long_horizon_model']}")
print(f"Shock selected model: {nb06['selected_classifier']}")
"""
    ),
    markdown(
        """## 3. Reconstruct one-step forecasting features

The specification exactly matches Notebook 05. The final 60 months are retained
as a fixed chronological holdout for explanation.
"""
    ),
    code(
        """data = national.sort_values("date").copy()
TARGET = "food_cpi_2015_100"
for lag in [1, 2, 3, 6, 12]:
    data[f"cpi_lag_{lag}"] = data[TARGET].shift(lag)
external_levels = [
    "mandi_price_index_2015_100",
    "world_wheat_usd_per_mt",
    "world_crude_oil_usd_per_bbl",
    "state_avg_rainfall_mm",
    "state_avg_temperature_c",
]
for variable in external_levels:
    data[variable + "_lag1"] = data[variable].shift(1)
data["cpi_rolling_mean_3_lag1"] = data[TARGET].shift(1).rolling(3).mean()
data["cpi_rolling_mean_12_lag1"] = data[TARGET].shift(1).rolling(12).mean()
data["cpi_rolling_std_12_lag1"] = data[TARGET].shift(1).rolling(12).std()
data["month_sin"] = np.sin(2 * np.pi * data["date"].dt.month / 12)
data["month_cos"] = np.cos(2 * np.pi * data["date"].dt.month / 12)
data["trend_months"] = np.arange(len(data), dtype=float)
FORECAST_FEATURES = [
    "cpi_lag_1", "cpi_lag_2", "cpi_lag_3", "cpi_lag_6", "cpi_lag_12",
    "cpi_rolling_mean_3_lag1", "cpi_rolling_mean_12_lag1",
    "cpi_rolling_std_12_lag1",
    *[variable + "_lag1" for variable in external_levels],
    "month_sin", "month_cos", "trend_months",
]
forecast_model_data = (
    data[["date", TARGET, *FORECAST_FEATURES]]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)
forecast_train = forecast_model_data.iloc[:-60]
forecast_test = forecast_model_data.iloc[-60:]
"""
    ),
    markdown(
        """## 4. Explain the closest complex one-step model

The naïve winner has no fitted features: its complete explanation is “use last
month's CPI.” Ridge ARX was the strongest complex candidate, so standardized
coefficients and holdout permutation importance show what that alternative used.
"""
    ),
    code(
        """ridge = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=10.0)),
])
ridge.fit(forecast_train[FORECAST_FEATURES], forecast_train[TARGET])
standardized_coefficients = pd.DataFrame({
    "feature": FORECAST_FEATURES,
    "standardized_coefficient": ridge.named_steps["model"].coef_,
})
standardized_coefficients["absolute_coefficient"] = (
    standardized_coefficients["standardized_coefficient"].abs()
)
standardized_coefficients = standardized_coefficients.sort_values(
    "absolute_coefficient", ascending=False
)

ridge_permutation_raw = permutation_importance(
    ridge,
    forecast_test[FORECAST_FEATURES],
    forecast_test[TARGET],
    scoring="neg_root_mean_squared_error",
    n_repeats=100,
    random_state=640,
    n_jobs=-1,
)
ridge_permutation = pd.DataFrame({
    "feature": FORECAST_FEATURES,
    "rmse_importance_mean": ridge_permutation_raw.importances_mean,
    "rmse_importance_sd": ridge_permutation_raw.importances_std,
}).sort_values("rmse_importance_mean", ascending=False)

standardized_coefficients.to_csv(
    REPORT_OUTPUT / "07_ridge_standardized_coefficients.csv", index=False
)
ridge_permutation.to_csv(
    REPORT_OUTPUT / "07_ridge_permutation_importance.csv", index=False
)
print("Ridge standardized coefficients")
print(standardized_coefficients.head(12).round(4).to_string(index=False))
print("\\nRidge holdout permutation importance")
print(ridge_permutation.head(12).round(4).to_string(index=False))
"""
    ),
    code(
        """plot_data = standardized_coefficients.head(12).sort_values(
    "standardized_coefficient"
)
fig, ax = plt.subplots(figsize=(10, 7))
colors = np.where(
    plot_data["standardized_coefficient"] >= 0, "#2563eb", "#dc2626"
)
ax.barh(
    plot_data["feature"],
    plot_data["standardized_coefficient"],
    color=colors,
)
ax.axvline(0, color="#374151", linewidth=0.8)
ax.set(
    title="Ridge ARX standardized feature coefficients",
    xlabel="Food CPI association per one-standard-deviation feature change",
    ylabel="Feature",
)
fig.tight_layout()
fig.savefig(FIGURES / "07_ridge_coefficients.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 5. Explain damped ETS

Damped ETS is univariate. It extrapolates a gradually diminishing trend while
repeating an estimated 12-month seasonal pattern. The monthly average fitted
seasonal component is reported directly.
"""
    ),
    code(
        """cpi_history = data[["date", TARGET]].dropna().sort_values("date")
ets = ExponentialSmoothing(
    cpi_history[TARGET].to_numpy(),
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated",
).fit(optimized=True, remove_bias=True)

ets_parameters = pd.DataFrame([
    {"parameter": key, "value": value}
    for key, value in ets.params.items()
    if np.isscalar(value)
])
seasonal_series = pd.DataFrame({
    "date": cpi_history["date"].to_numpy(),
    "seasonal_component": np.asarray(ets.season),
})
seasonal_by_month = (
    seasonal_series.assign(month=seasonal_series["date"].dt.month)
    .groupby("month", as_index=False)["seasonal_component"]
    .mean()
)
seasonal_by_month["month_name"] = pd.to_datetime(
    seasonal_by_month["month"], format="%m"
).dt.month_name().str[:3]

ets_parameters.to_csv(REPORT_OUTPUT / "07_ets_parameters.csv", index=False)
seasonal_by_month.to_csv(
    REPORT_OUTPUT / "07_ets_seasonal_component.csv", index=False
)
print("Damped ETS scalar parameters")
print(ets_parameters.round(5).to_string(index=False))
print("\\nAverage fitted monthly seasonal component")
print(seasonal_by_month.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(
    seasonal_by_month["month_name"],
    seasonal_by_month["seasonal_component"],
    color="#d97706",
)
ax.axhline(0, color="#374151", linewidth=0.8)
ax.set(
    title="Damped ETS estimated monthly seasonal contribution",
    xlabel="Calendar month",
    ylabel="Food CPI index-point contribution",
)
fig.tight_layout()
fig.savefig(FIGURES / "07_ets_seasonality.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 6. Reconstruct the shock-classification sample

The feature definitions and fixed threshold exactly match Notebook 06. A single
model trained before the final 60 months is used for out-of-sample explanations.
"""
    ),
    code(
        """shock_data = national.sort_values("date").copy()
shock_data["food_cpi_yoy_current"] = shock_data["food_cpi_yoy_pct_exact"]
shock_data["food_cpi_yoy_lag1"] = shock_data["food_cpi_yoy_current"].shift(1)
shock_data["food_cpi_mom_current"] = shock_data["food_cpi_mom_pct_exact"]
shock_data["mandi_yoy_current"] = shock_data["mandi_index_yoy_pct"]
shock_data["world_wheat_yoy_current"] = (
    shock_data["world_wheat_usd_per_mt"].pct_change(12) * 100
)
shock_data["world_crude_yoy_current"] = (
    shock_data["world_crude_oil_usd_per_bbl"].pct_change(12) * 100
)
reference = shock_data.loc[shock_data["date"].dt.year <= 2025]
rain_normal = reference.groupby(reference["date"].dt.month)[
    "state_avg_rainfall_mm"
].mean()
temp_normal = reference.groupby(reference["date"].dt.month)[
    "state_avg_temperature_c"
].mean()
shock_data["rainfall_anomaly_current"] = (
    100 * (
        shock_data["state_avg_rainfall_mm"]
        - shock_data["date"].dt.month.map(rain_normal)
    ) / shock_data["date"].dt.month.map(rain_normal).replace(0, np.nan)
)
shock_data["temperature_anomaly_current"] = (
    shock_data["state_avg_temperature_c"]
    - shock_data["date"].dt.month.map(temp_normal)
)
shock_data["target_date"] = shock_data["date"].shift(-1)
shock_data["target_food_cpi_yoy"] = (
    shock_data["food_cpi_yoy_current"].shift(-1)
)
shock_data["target_month_sin"] = np.sin(
    2 * np.pi * shock_data["target_date"].dt.month / 12
)
shock_data["target_month_cos"] = np.cos(
    2 * np.pi * shock_data["target_date"].dt.month / 12
)
shock_data["trend_months"] = np.arange(len(shock_data), dtype=float)
SHOCK_FEATURES = [
    "food_cpi_yoy_current", "food_cpi_yoy_lag1", "food_cpi_mom_current",
    "mandi_yoy_current", "world_wheat_yoy_current",
    "world_crude_yoy_current", "rainfall_anomaly_current",
    "temperature_anomaly_current", "target_month_sin",
    "target_month_cos", "trend_months",
]
shock_work = (
    shock_data[["date", "target_date", "target_food_cpi_yoy", *SHOCK_FEATURES]]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)
SHOCK_THRESHOLD = float(nb06["training_only_shock_threshold_yoy_pct"])
shock_work["shock_next"] = (
    shock_work["target_food_cpi_yoy"] >= SHOCK_THRESHOLD
).astype(int)
shock_train, shock_test = shock_work.iloc[:-60], shock_work.iloc[-60:]


def balanced_weights(y):
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    mapping = {
        cls: len(y) / (2 * count) if count else 1.0
        for cls, count in enumerate(counts)
    }
    return np.array([mapping[value] for value in y])


shock_classifier = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=180,
    max_leaf_nodes=15,
    min_samples_leaf=12,
    l2_regularization=2.0,
    random_state=640,
)
_ = shock_classifier.fit(
    shock_train[SHOCK_FEATURES],
    shock_train["shock_next"],
    sample_weight=balanced_weights(shock_train["shock_next"]),
)
"""
    ),
    markdown(
        """## 7. Out-of-sample shock permutation importance

Each feature is shuffled 100 times in the chronological holdout. The reported
decrease in ROC-AUC measures how much discrimination depended on that feature.
Correlated features can share or substitute importance.
"""
    ),
    code(
        """shock_permutation_raw = permutation_importance(
    shock_classifier,
    shock_test[SHOCK_FEATURES],
    shock_test["shock_next"],
    scoring="roc_auc",
    n_repeats=100,
    random_state=640,
    n_jobs=-1,
)
shock_importance = pd.DataFrame({
    "feature": SHOCK_FEATURES,
    "roc_auc_decrease_mean": shock_permutation_raw.importances_mean,
    "roc_auc_decrease_sd": shock_permutation_raw.importances_std,
}).sort_values("roc_auc_decrease_mean", ascending=False)
shock_importance.to_csv(
    REPORT_OUTPUT / "07_shock_permutation_importance.csv", index=False
)
print(shock_importance.round(4).to_string(index=False))

plot_data = shock_importance.sort_values("roc_auc_decrease_mean")
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    plot_data["feature"],
    plot_data["roc_auc_decrease_mean"],
    xerr=plot_data["roc_auc_decrease_sd"],
    color="#1f4e79",
    alpha=0.9,
)
ax.axvline(0, color="#374151", linewidth=0.8)
ax.set(
    title="Out-of-sample shock-classifier permutation importance",
    xlabel="Mean decrease in ROC-AUC after shuffling",
    ylabel="Feature",
)
fig.tight_layout()
fig.savefig(FIGURES / "07_shock_permutation_importance.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 8. January–December 2026 local sensitivity explanations

For every target month, each feature is replaced individually with its full-model
training median and the row is rescored. `probability_contribution` is the
original probability minus the replacement probability. Positive values indicate
that the monthly input raised modeled risk relative to a typical training value.

January uses observed December 2025 features. Explanations for later months
inherit the conditional CPI, external-price, and climate assumptions from
Notebook 06.
"""
    ),
    code(
        """annual_feature_matrix = (
    annual_features.set_index("target_date")
    .loc[annual_risk["target_date"], SHOCK_FEATURES]
    .replace([np.inf, -np.inf], np.nan)
)
if annual_feature_matrix.shape != (12, len(SHOCK_FEATURES)):
    raise AssertionError("Expected a complete 12-month explanation matrix.")
if annual_feature_matrix.isna().any(axis=None):
    raise AssertionError("The 2026 explanation feature path is incomplete.")

base_probabilities = final_shock_classifier.predict_proba(
    annual_feature_matrix
)[:, 1]
saved_probabilities = annual_risk[
    "selected_model_probability"
].to_numpy()
if not np.allclose(base_probabilities, saved_probabilities, atol=1e-10):
    raise AssertionError(
        "Notebook 07 probabilities do not reproduce Notebook 06."
    )

training_medians = shock_work[SHOCK_FEATURES].median()
local_rows = []
for row_number, target_date in enumerate(annual_risk["target_date"]):
    month_features = annual_feature_matrix.iloc[[row_number]].copy()
    base_probability = float(base_probabilities[row_number])
    for feature in SHOCK_FEATURES:
        counterfactual = month_features.copy()
        counterfactual[feature] = training_medians[feature]
        replacement_probability = float(
            final_shock_classifier.predict_proba(counterfactual)[0, 1]
        )
        local_rows.append({
            "target_date": target_date,
            "horizon_months": int(row_number + 1),
            "input_status": annual_risk.loc[row_number, "input_status"],
            "feature": feature,
            "feature_value": float(month_features[feature].iloc[0]),
            "training_median": float(training_medians[feature]),
            "base_probability": base_probability,
            "median_replacement_probability": replacement_probability,
            "probability_contribution": (
                base_probability - replacement_probability
            ),
        })
annual_local_explanation = pd.DataFrame(local_rows)
annual_local_explanation["absolute_contribution"] = (
    annual_local_explanation["probability_contribution"].abs()
)
annual_local_explanation.to_csv(
    REPORT_OUTPUT / "07_2026_monthly_local_explanations.csv", index=False
)

monthly_leaders = (
    annual_local_explanation.sort_values(
        ["target_date", "absolute_contribution"],
        ascending=[True, False],
    )
    .groupby("target_date", as_index=False)
    .first()
)
monthly_leaders.to_csv(
    REPORT_OUTPUT / "07_2026_monthly_leading_drivers.csv", index=False
)
print("Leading local sensitivity feature by 2026 target month")
print(monthly_leaders[[
    "target_date", "horizon_months", "input_status", "base_probability",
    "feature", "probability_contribution",
]].round(4).to_string(index=False))

plot_table = annual_local_explanation.pivot(
    index="feature",
    columns="target_date",
    values="probability_contribution",
)
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(
    plot_table,
    cmap="RdBu_r",
    center=0,
    linewidths=0.3,
    cbar_kws={"label": "Probability contribution vs median replacement"},
    ax=ax,
)
ax.set(
    title="Monthly local shock-risk sensitivity across 2026",
    xlabel="Target month",
    ylabel="Feature",
)
ax.set_xticklabels(
    [date.strftime("%b") for date in plot_table.columns],
    rotation=0,
)
fig.tight_layout()
fig.savefig(
    FIGURES / "07_2026_monthly_local_explanations.png",
    bbox_inches="tight",
)
plt.show()
"""
    ),
    markdown(
        """## 9. Partial dependence for leading shock features

Partial dependence varies one feature over its observed range while averaging
over other holdout features. It describes the fitted model, not a causal response.
"""
    ),
    code(
        """top_features = shock_importance.head(2)["feature"].tolist()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
PartialDependenceDisplay.from_estimator(
    shock_classifier,
    shock_test[SHOCK_FEATURES],
    features=top_features,
    ax=axes,
    grid_resolution=30,
    # Sample-weighted HistGradientBoosting does not support the faster
    # recursion algorithm. Brute force is exact for this small holdout.
    method="brute",
)
fig.suptitle("Partial dependence of leading shock predictors", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "07_shock_partial_dependence.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown("## 10. Execution summary and artifact validation"),
    code(
        """top_ridge = ridge_permutation.iloc[0]
top_shock = shock_importance.iloc[0]
highest_risk_row = annual_risk.loc[
    annual_risk["selected_model_probability"].idxmax()
]
largest_annual_driver = annual_local_explanation.loc[
    annual_local_explanation["absolute_contribution"].idxmax()
]
summary = {
    "notebook": "07_explainability",
    "status": "completed",
    "one_step_winner_explanation": (
        "Naive last-month CPI; no fitted feature coefficients."
    ),
    "explained_complex_forecast_model": "ridge_arx",
    "top_ridge_permutation_feature": str(top_ridge["feature"]),
    "top_ridge_rmse_importance": float(top_ridge["rmse_importance_mean"]),
    "long_horizon_model": str(nb05["selected_long_horizon_model"]),
    "long_horizon_explanation": (
        "Damped trend plus estimated 12-month additive seasonality."
    ),
    "shock_model": str(nb06["selected_classifier"]),
    "top_shock_permutation_feature": str(top_shock["feature"]),
    "top_shock_roc_auc_decrease": float(top_shock["roc_auc_decrease_mean"]),
    "annual_explanation_months": int(annual_risk["target_date"].nunique()),
    "january_2026_probability": float(
        annual_risk.loc[
            annual_risk["target_date"].eq(pd.Timestamp("2026-01-01")),
            "selected_model_probability",
        ].iloc[0]
    ),
    "highest_2026_risk_month": str(highest_risk_row["target_date"].date()),
    "highest_2026_risk_probability": float(
        highest_risk_row["selected_model_probability"]
    ),
    "largest_annual_local_feature": str(largest_annual_driver["feature"]),
    "largest_annual_local_feature_month": str(
        largest_annual_driver["target_date"].date()
    ),
    "annual_explanation_scope": (
        "January uses observed December 2025 inputs; February–December "
        "explain conditional scenario inputs from Notebook 06."
    ),
    "interpretation": (
        "Predictive explanations and sensitivity; not causal attribution."
    ),
    "output_root": str(OUTPUT_ROOT),
}
SUMMARY_FILE = REPORT_OUTPUT / "07_execution_summary.json"
_ = SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

expected_files = [
    REPORT_OUTPUT / "07_ridge_standardized_coefficients.csv",
    REPORT_OUTPUT / "07_ridge_permutation_importance.csv",
    REPORT_OUTPUT / "07_ets_parameters.csv",
    REPORT_OUTPUT / "07_ets_seasonal_component.csv",
    REPORT_OUTPUT / "07_shock_permutation_importance.csv",
    REPORT_OUTPUT / "07_2026_monthly_local_explanations.csv",
    REPORT_OUTPUT / "07_2026_monthly_leading_drivers.csv",
    FIGURES / "07_ridge_coefficients.png",
    FIGURES / "07_ets_seasonality.png",
    FIGURES / "07_shock_permutation_importance.png",
    FIGURES / "07_2026_monthly_local_explanations.png",
    FIGURES / "07_shock_partial_dependence.png",
    SUMMARY_FILE,
]
missing_artifacts = [str(path) for path in expected_files if not path.exists()]
if missing_artifacts:
    raise FileNotFoundError(
        "Notebook 07 artifacts missing:\\n" + "\\n".join(missing_artifacts)
    )

print("=" * 70)
print("NOTEBOOK 07 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT")
print("=" * 70)
print(json.dumps(summary, indent=2))
print(f"\\nVerified artifacts: {len(expected_files)}")
"""
    ),
    markdown(
        """## Notebook 07 conclusion

The one-step result favors a transparent persistence rule. Ridge explanations
show what the strongest complex alternative used, ETS decomposition explains the
two-year path, and shock-model discrimination is explained outside the training
period. The 2026 local explanations distinguish January's observed inputs from
the increasingly assumption-dependent February–December scenario. None should
be interpreted causally.

After **Runtime → Run all**, send the Notebook 07 summary, the shock permutation
table, 2026 monthly leading-driver table, and any red error output.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {
            "name": "07_explainability.ipynb",
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
