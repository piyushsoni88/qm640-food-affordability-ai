"""Build the Google Colab version of Notebook 06."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "06_shock_classification.ipynb"


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
## Notebook 06 — Food-Inflation Shock Classification

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Prerequisite:** Notebooks 01–05 completed successfully

### Classification objective

Predict whether **the following month** will be a high national food-inflation
month. A shock is defined using the 75th percentile of food CPI year-on-year
inflation in the initial training period only.

The notebook compares:

1. persistence baseline: next month repeats the current shock state;
2. class-weighted logistic regression;
3. class-weighted Random Forest; and
4. histogram gradient boosting.

Evaluation is chronological. Recall and balanced accuracy receive special
attention because missing a shock can matter more than correctly identifying the
larger number of normal months.
"""
    ),
    markdown(
        """## 1. Runtime and Google Drive setup

Colab normally includes all required packages. Conditional installation is used
only when scikit-learn is unavailable.
"""
    ),
    code(
        """from pathlib import Path
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

try:
    import sklearn
    import joblib
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "scikit-learn>=1.5", "joblib"],
        check=True,
    )
    import sklearn
    import joblib
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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
MODELS = OUTPUT_ROOT / "models"
for folder in [REPORT_OUTPUT, FIGURES, MODELS]:
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
        """## 2. Load and verify forecasting outputs

Notebook 05 is checked before classification begins. The cleaned national panel
remains the source of features and outcomes.
"""
    ),
    code(
        """NATIONAL_FILE = PROCESSED / "cleaned_national_monthly.csv.gz"
NB05_SUMMARY = REPORT_OUTPUT / "05_execution_summary.json"
NB05_FORECAST = REPORT_OUTPUT / "05_twenty_four_month_conditional_forecast.csv"
required = [NATIONAL_FILE, NB05_SUMMARY, NB05_FORECAST]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Run Notebooks 01–05 first.\\n" + "\\n".join(missing)
    )

national = pd.read_csv(NATIONAL_FILE, parse_dates=["date"], low_memory=False)
nb05 = json.loads(NB05_SUMMARY.read_text(encoding="utf-8"))
forecast_24 = pd.read_csv(NB05_FORECAST, parse_dates=["date"])
if nb05.get("status") != "completed":
    raise AssertionError("Notebook 05 summary is not marked completed.")
print(f"National rows: {len(national):,}")
print(f"One-step winner: {nb05['selected_one_step_model']}")
print(f"Long-horizon winner: {nb05['selected_long_horizon_model']}")
"""
    ),
    markdown(
        """## 3. Leakage-safe predictors and next-month outcome

Every feature is observed by the end of month `t`; the target is food inflation
in month `t+1`. World-price changes are computed from historical levels. Climate
anomalies are relative to month-specific normals through 2025.
"""
    ),
    code(
        """data = national.sort_values("date").copy()
expected = pd.date_range(data["date"].min(), data["date"].max(), freq="MS")
if not data["date"].reset_index(drop=True).equals(pd.Series(expected)):
    raise AssertionError("National calendar contains missing months.")

data["food_cpi_yoy_current"] = data["food_cpi_yoy_pct_exact"]
data["food_cpi_yoy_lag1"] = data["food_cpi_yoy_current"].shift(1)
data["food_cpi_mom_current"] = data["food_cpi_mom_pct_exact"]
data["mandi_yoy_current"] = data["mandi_index_yoy_pct"]
data["world_wheat_yoy_current"] = (
    data["world_wheat_usd_per_mt"].pct_change(12) * 100
)
data["world_crude_yoy_current"] = (
    data["world_crude_oil_usd_per_bbl"].pct_change(12) * 100
)

reference = data.loc[data["date"].dt.year <= 2025]
rain_normal = reference.groupby(reference["date"].dt.month)[
    "state_avg_rainfall_mm"
].mean()
temp_normal = reference.groupby(reference["date"].dt.month)[
    "state_avg_temperature_c"
].mean()
data["rainfall_anomaly_current"] = (
    100
    * (
        data["state_avg_rainfall_mm"]
        - data["date"].dt.month.map(rain_normal)
    )
    / data["date"].dt.month.map(rain_normal).replace(0, np.nan)
)
data["temperature_anomaly_current"] = (
    data["state_avg_temperature_c"]
    - data["date"].dt.month.map(temp_normal)
)
data["target_date"] = data["date"].shift(-1)
data["target_food_cpi_yoy"] = data["food_cpi_yoy_current"].shift(-1)
data["target_month_sin"] = np.sin(
    2 * np.pi * data["target_date"].dt.month / 12
)
data["target_month_cos"] = np.cos(
    2 * np.pi * data["target_date"].dt.month / 12
)
data["trend_months"] = np.arange(len(data), dtype=float)

FEATURES = [
    "food_cpi_yoy_current",
    "food_cpi_yoy_lag1",
    "food_cpi_mom_current",
    "mandi_yoy_current",
    "world_wheat_yoy_current",
    "world_crude_yoy_current",
    "rainfall_anomaly_current",
    "temperature_anomaly_current",
    "target_month_sin",
    "target_month_cos",
    "trend_months",
]
work = (
    data[["date", "target_date", "target_food_cpi_yoy", *FEATURES]]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)
print(f"Model-ready prediction rows: {len(work):,}")
print(f"Target coverage: {work.target_date.min().date()} to {work.target_date.max().date()}")
"""
    ),
    markdown(
        """## 4. Training-only shock definition

The final 60 target months form the chronological evaluation period. The 75th
percentile threshold is estimated from earlier targets only, then held fixed.
This avoids redefining a shock using information from the test period.
"""
    ),
    code(
        """N_EVALUATION_MONTHS = 60
first_test_index = len(work) - N_EVALUATION_MONTHS
initial_training_targets = work.loc[
    : first_test_index - 1, "target_food_cpi_yoy"
]
SHOCK_THRESHOLD = float(initial_training_targets.quantile(0.75))
work["shock_next"] = (
    work["target_food_cpi_yoy"] >= SHOCK_THRESHOLD
).astype(int)
work["shock_current"] = (
    work["food_cpi_yoy_current"] >= SHOCK_THRESHOLD
).astype(int)

training_prevalence = work.loc[
    : first_test_index - 1, "shock_next"
].mean()
evaluation_prevalence = work.loc[
    first_test_index:, "shock_next"
].mean()

print(f"Training-only shock threshold: {SHOCK_THRESHOLD:.3f}% YoY")
print(f"Training shock prevalence: {training_prevalence:.2%}")
print(f"Evaluation shock prevalence: {evaluation_prevalence:.2%}")
"""
    ),
    markdown(
        """## 5. Classifier definitions

Balanced class weights prevent the majority normal class from dominating
logistic and forest training. Histogram boosting receives equivalent per-row
sample weights during fitting.
"""
    ),
    code(
        """def make_classifiers():
    return {
        "logistic_balanced": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=2000,
                random_state=640,
            )),
        ]),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=4,
            max_features=0.75,
            class_weight="balanced",
            random_state=640,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=2.0,
            random_state=640,
        ),
    }


def balanced_sample_weights(y):
    \"\"\"Inverse-frequency weights with mean approximately one.\"\"\"
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    weights = {
        cls: len(y) / (2 * count) if count else 1.0
        for cls, count in enumerate(counts)
    }
    return np.array([weights[value] for value in y])
"""
    ),
    markdown(
        """## 6. Expanding-window rolling-origin classification

Each evaluation month is predicted after refitting on all earlier observations.
The fixed shock definition is unchanged. Probabilities are retained for ROC-AUC,
average precision, calibration, and future risk scoring.
"""
    ),
    code(
        """prediction_rows = []
started = time.perf_counter()
for test_index in range(first_test_index, len(work)):
    train = work.iloc[:test_index]
    test = work.iloc[[test_index]]
    if train["target_date"].max() >= test["target_date"].iloc[0]:
        raise AssertionError("Chronological leakage detected.")

    result = {
        "feature_date": test["date"].iloc[0],
        "target_date": test["target_date"].iloc[0],
        "actual_shock": int(test["shock_next"].iloc[0]),
        "target_food_cpi_yoy": float(test["target_food_cpi_yoy"].iloc[0]),
        "persistence_probability": float(test["shock_current"].iloc[0]),
    }
    X_train, y_train = train[FEATURES], train["shock_next"]
    X_test = test[FEATURES]
    for name, classifier in make_classifiers().items():
        if name == "hist_gradient_boosting":
            classifier.fit(
                X_train,
                y_train,
                sample_weight=balanced_sample_weights(y_train),
            )
        else:
            classifier.fit(X_train, y_train)
        result[name + "_probability"] = float(
            classifier.predict_proba(X_test)[0, 1]
        )
    prediction_rows.append(result)
    if (test_index - first_test_index + 1) % 12 == 0:
        print(f"Completed {test_index - first_test_index + 1}/{N_EVALUATION_MONTHS} origins")

predictions = pd.DataFrame(prediction_rows)
predictions.to_csv(
    REPORT_OUTPUT / "06_rolling_shock_predictions.csv", index=False
)
print(f"Elapsed: {time.perf_counter() - started:.1f} seconds")
"""
    ),
    markdown(
        """## 7. Class-sensitive evaluation

The decision cutoff is 0.50. Balanced accuracy averages shock recall and normal
recall. ROC-AUC and average precision evaluate probabilities without fixing one
cutoff. Brier score measures probability calibration; lower is better.
"""
    ),
    code(
        """PROBABILITY_COLUMNS = {
    "persistence": "persistence_probability",
    "logistic_balanced": "logistic_balanced_probability",
    "random_forest_balanced": "random_forest_balanced_probability",
    "hist_gradient_boosting": "hist_gradient_boosting_probability",
}
actual = predictions["actual_shock"].to_numpy()
metric_rows = []
confusion_rows = []
for model_name, probability_column in PROBABILITY_COLUMNS.items():
    probability = predictions[probability_column].to_numpy()
    predicted = (probability >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    metric_rows.append({
        "model": model_name,
        "n": len(actual),
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(actual, predicted, zero_division=0),
        "roc_auc": roc_auc_score(actual, probability),
        "average_precision": average_precision_score(actual, probability),
        "brier_score": brier_score_loss(actual, probability),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    })
    confusion_rows.append({
        "model": model_name,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    })

metrics = (
    pd.DataFrame(metric_rows)
    .sort_values(["balanced_accuracy", "f1"], ascending=False)
    .reset_index(drop=True)
)
confusions = pd.DataFrame(confusion_rows)
metrics.to_csv(REPORT_OUTPUT / "06_shock_classifier_metrics.csv", index=False)
confusions.to_csv(REPORT_OUTPUT / "06_confusion_matrices.csv", index=False)
BEST_CLASSIFIER = str(metrics.iloc[0]["model"])
print(metrics.round(4).to_string(index=False))
print(f"\\nSelected classifier: {BEST_CLASSIFIER}")
"""
    ),
    markdown(
        """## 8. Classification figures

The probability chart shows when models anticipated test-period shocks. Metric
and confusion-matrix figures expose trade-offs between missed shocks and false
alarms.
"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(14, 7))
for model_name, probability_column in PROBABILITY_COLUMNS.items():
    ax.plot(
        predictions["target_date"],
        predictions[probability_column],
        label=model_name.replace("_", " ").title(),
        linewidth=1.4,
    )
shock_dates = predictions.loc[predictions["actual_shock"].eq(1), "target_date"]
ax.scatter(
    shock_dates,
    np.ones(len(shock_dates)) * 1.03,
    marker="v",
    color="#b91c1c",
    label="Observed shock",
    clip_on=False,
)
ax.axhline(0.50, color="#6b7280", linestyle="--", linewidth=1)
ax.set(
    title="Rolling next-month food-inflation shock probabilities",
    xlabel="Target month",
    ylabel="Predicted shock probability",
    ylim=(-0.03, 1.08),
)
ax.legend(ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "06_shock_probabilities.png", bbox_inches="tight")
plt.show()

plot_metrics = metrics.sort_values("balanced_accuracy")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(plot_metrics))
ax.barh(x - 0.2, plot_metrics["balanced_accuracy"], 0.4,
        label="Balanced accuracy", color="#1f4e79")
ax.barh(x + 0.2, plot_metrics["recall"], 0.4,
        label="Shock recall", color="#d97706")
ax.set_yticks(x, plot_metrics["model"].str.replace("_", " ").str.title())
ax.set(xlabel="Score", ylabel="Model",
       title="Shock-classification performance")
ax.set_xlim(0, 1)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "06_classifier_metrics.png", bbox_inches="tight")
plt.show()

best_confusion = confusions.loc[
    confusions["model"].eq(BEST_CLASSIFIER)
].iloc[0]
matrix = np.array([
    [best_confusion["true_negative"], best_confusion["false_positive"]],
    [best_confusion["false_negative"], best_confusion["true_positive"]],
], dtype=int)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Predicted normal", "Predicted shock"],
    yticklabels=["Actual normal", "Actual shock"],
    ax=ax,
)
ax.set_title(f"Confusion matrix: {BEST_CLASSIFIER.replace('_', ' ').title()}")
fig.tight_layout()
fig.savefig(FIGURES / "06_best_confusion_matrix.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 9. Final models and January–December 2026 risk

Classifiers are refitted on all labeled observations. December 2025 features
produce a genuine one-month-ahead estimate for January 2026. February–December
use the Notebook 05 CPI forecast recursively, hold external price levels at their
December 2025 values, and set future climate anomalies to seasonal normal
(`0`). These later probabilities are conditional scenario estimates whose
uncertainty increases with the horizon.
"""
    ),
    code(
        """final_classifiers = make_classifiers()
for name, classifier in final_classifiers.items():
    if name == "hist_gradient_boosting":
        classifier.fit(
            work[FEATURES],
            work["shock_next"],
            sample_weight=balanced_sample_weights(work["shock_next"]),
        )
    else:
        classifier.fit(work[FEATURES], work["shock_next"])
    joblib.dump(classifier, MODELS / f"06_{name}.joblib")

# Build a monthly scenario table. A row dated month t contains the information
# available at the end of t and predicts the shock state for month t+1.
scenario_levels = data[[
    "date",
    "food_cpi_2015_100",
    "mandi_price_index_2015_100",
    "world_wheat_usd_per_mt",
    "world_crude_oil_usd_per_bbl",
]].loc[
    lambda frame: frame["date"].le(pd.Timestamp("2025-12-01"))
].copy()
future_levels = forecast_24.loc[
    forecast_24["date"].between("2026-01-01", "2026-12-01"),
    ["date", "forecast"],
].rename(columns={"forecast": "food_cpi_2015_100"})
if len(future_levels) != 12:
    raise AssertionError("Notebook 05 must provide all 12 months of 2026.")

# Holding future external levels constant is explicit and reproducible. It does
# not claim that mandi, wheat, or crude prices will actually remain unchanged.
for variable in [
    "mandi_price_index_2015_100",
    "world_wheat_usd_per_mt",
    "world_crude_oil_usd_per_bbl",
]:
    future_levels[variable] = float(
        scenario_levels.loc[
            scenario_levels["date"].eq(pd.Timestamp("2025-12-01")), variable
        ].iloc[0]
    )
scenario_levels = pd.concat(
    [scenario_levels, future_levels], ignore_index=True
).sort_values("date").reset_index(drop=True)

scenario_levels["food_cpi_yoy_current"] = (
    scenario_levels["food_cpi_2015_100"].pct_change(12) * 100
)
scenario_levels["food_cpi_yoy_lag1"] = (
    scenario_levels["food_cpi_yoy_current"].shift(1)
)
scenario_levels["food_cpi_mom_current"] = (
    scenario_levels["food_cpi_2015_100"].pct_change() * 100
)
scenario_levels["mandi_yoy_current"] = (
    scenario_levels["mandi_price_index_2015_100"].pct_change(12) * 100
)
scenario_levels["world_wheat_yoy_current"] = (
    scenario_levels["world_wheat_usd_per_mt"].pct_change(12) * 100
)
scenario_levels["world_crude_yoy_current"] = (
    scenario_levels["world_crude_oil_usd_per_bbl"].pct_change(12) * 100
)
scenario_levels["rainfall_anomaly_current"] = 0.0
scenario_levels["temperature_anomaly_current"] = 0.0

# Preserve the observed December 2025 climate anomalies for the genuine
# one-step January forecast; only future feature months use normal climate.
observed_december = data.loc[data["date"].eq(pd.Timestamp("2025-12-01"))].iloc[0]
for variable in ["rainfall_anomaly_current", "temperature_anomaly_current"]:
    scenario_levels.loc[
        scenario_levels["date"].eq(pd.Timestamp("2025-12-01")), variable
    ] = float(observed_december[variable])

scenario_levels["target_date"] = scenario_levels["date"] + pd.offsets.MonthBegin(1)
scenario_levels["target_month_sin"] = np.sin(
    2 * np.pi * scenario_levels["target_date"].dt.month / 12
)
scenario_levels["target_month_cos"] = np.cos(
    2 * np.pi * scenario_levels["target_date"].dt.month / 12
)
historical_trend_lookup = data.set_index("date")["trend_months"]
scenario_levels["trend_months"] = (
    (scenario_levels["date"].dt.year - data["date"].dt.year.min()) * 12
    + scenario_levels["date"].dt.month
    - data["date"].dt.month.min()
).astype(float)
# Use the original index exactly for all historical months.
historical_mask = scenario_levels["date"].isin(historical_trend_lookup.index)
scenario_levels.loc[historical_mask, "trend_months"] = (
    scenario_levels.loc[historical_mask, "date"]
    .map(historical_trend_lookup)
    .astype(float)
)

score_rows = scenario_levels.loc[
    scenario_levels["date"].between("2025-12-01", "2026-11-01")
].copy()
if len(score_rows) != 12:
    raise AssertionError("Expected 12 feature months for the 2026 outlook.")
if score_rows[FEATURES].replace([np.inf, -np.inf], np.nan).isna().any(axis=None):
    raise AssertionError("The 2026 conditional feature path is incomplete.")

future_risk = score_rows[["date", "target_date"]].rename(
    columns={"date": "feature_month"}
).reset_index(drop=True)
future_risk["horizon_months"] = np.arange(1, 13)
future_risk["input_status"] = np.where(
    future_risk["horizon_months"].eq(1),
    "observed inputs",
    "conditional forecast inputs",
)
future_risk["shock_threshold_yoy_pct"] = SHOCK_THRESHOLD
future_risk["persistence_probability"] = (
    score_rows["food_cpi_yoy_current"].to_numpy() >= SHOCK_THRESHOLD
).astype(float)
for name, classifier in final_classifiers.items():
    future_risk[name + "_probability"] = classifier.predict_proba(
        score_rows[FEATURES]
    )[:, 1]
future_risk["selected_model"] = BEST_CLASSIFIER
future_risk["selected_model_probability"] = future_risk[
    BEST_CLASSIFIER + "_probability"
]
future_risk["scenario_assumption"] = (
    "Notebook 05 CPI path; external price levels held at Dec-2025; "
    "future climate anomalies set to seasonal normal"
)

ANNUAL_RISK_FILE = REPORT_OUTPUT / "06_2026_monthly_shock_risk.csv"
future_risk.to_csv(ANNUAL_RISK_FILE, index=False)
annual_features = score_rows[["date", "target_date", *FEATURES]].rename(
    columns={"date": "feature_month"}
)
annual_features.to_csv(
    REPORT_OUTPUT / "06_2026_conditional_features.csv", index=False
)
# Retain the original one-row artifact for backward compatibility.
future_risk.iloc[[0]].to_csv(
    REPORT_OUTPUT / "06_january_2026_shock_risk.csv", index=False
)
print(future_risk.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    future_risk["target_date"],
    future_risk["selected_model_probability"],
    color="#b91c1c",
    marker="o",
    linewidth=2,
    label=f"{BEST_CLASSIFIER} conditional probability",
)
ax.axhline(0.5, color="#6b7280", linestyle="--", linewidth=1, label="0.50 cutoff")
ax.fill_between(
    future_risk["target_date"],
    0,
    future_risk["selected_model_probability"],
    color="#fca5a5",
    alpha=0.25,
)
ax.set(
    title="Conditional monthly food-inflation shock risk: 2026",
    xlabel="Target month",
    ylabel="Predicted shock probability",
    ylim=(0, 1),
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "06_2026_monthly_shock_risk.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 10. Execution summary and validation

The summary distinguishes predictive skill from causal interpretation and records
both discrimination and missed-shock performance.
"""
    ),
    code(
        """best = metrics.iloc[0]
summary = {
    "notebook": "06_shock_classification",
    "status": "completed",
    "shock_definition": (
        "Next-month food CPI YoY at or above initial-training 75th percentile"
    ),
    "training_only_shock_threshold_yoy_pct": SHOCK_THRESHOLD,
    "training_shock_prevalence": float(training_prevalence),
    "evaluation_shock_prevalence": float(evaluation_prevalence),
    "rolling_origins": int(len(predictions)),
    "evaluation_start": str(predictions["target_date"].min().date()),
    "evaluation_end": str(predictions["target_date"].max().date()),
    "selected_classifier": BEST_CLASSIFIER,
    "selected_balanced_accuracy": float(best["balanced_accuracy"]),
    "selected_precision": float(best["precision"]),
    "selected_recall": float(best["recall"]),
    "selected_specificity": float(best["specificity"]),
    "selected_f1": float(best["f1"]),
    "selected_roc_auc": float(best["roc_auc"]),
    "selected_average_precision": float(best["average_precision"]),
    "selected_brier_score": float(best["brier_score"]),
    "january_2026_selected_model_probability": float(
        future_risk.loc[
            future_risk["target_date"].eq(pd.Timestamp("2026-01-01")),
            "selected_model_probability",
        ].iloc[0]
    ),
    "conditional_outlook_start": str(future_risk["target_date"].min().date()),
    "conditional_outlook_end": str(future_risk["target_date"].max().date()),
    "conditional_outlook_months": int(len(future_risk)),
    "fully_observed_input_forecasts": 1,
    "conditional_input_forecasts": 11,
    "highest_2026_risk_month": str(
        future_risk.loc[
            future_risk["selected_model_probability"].idxmax(), "target_date"
        ].date()
    ),
    "highest_2026_selected_model_probability": float(
        future_risk["selected_model_probability"].max()
    ),
    "annual_outlook_method": (
        "January uses observed December 2025 inputs; February–December use "
        "the Notebook 05 CPI path, constant December 2025 external price "
        "levels, and seasonal-normal climate anomalies."
    ),
    "interpretation": (
        "Predictive risk classification; not a causal or policy trigger."
    ),
    "output_root": str(OUTPUT_ROOT),
}
SUMMARY_FILE = REPORT_OUTPUT / "06_execution_summary.json"
_ = SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

expected_files = [
    REPORT_OUTPUT / "06_rolling_shock_predictions.csv",
    REPORT_OUTPUT / "06_shock_classifier_metrics.csv",
    REPORT_OUTPUT / "06_confusion_matrices.csv",
    REPORT_OUTPUT / "06_january_2026_shock_risk.csv",
    REPORT_OUTPUT / "06_2026_monthly_shock_risk.csv",
    REPORT_OUTPUT / "06_2026_conditional_features.csv",
    FIGURES / "06_shock_probabilities.png",
    FIGURES / "06_classifier_metrics.png",
    FIGURES / "06_best_confusion_matrix.png",
    FIGURES / "06_2026_monthly_shock_risk.png",
    MODELS / "06_logistic_balanced.joblib",
    MODELS / "06_random_forest_balanced.joblib",
    MODELS / "06_hist_gradient_boosting.joblib",
    SUMMARY_FILE,
]
missing_artifacts = [str(path) for path in expected_files if not path.exists()]
if missing_artifacts:
    raise FileNotFoundError(
        "Notebook 06 artifacts missing:\\n" + "\\n".join(missing_artifacts)
    )

print("=" * 74)
print("NOTEBOOK 06 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT")
print("=" * 74)
print(json.dumps(summary, indent=2))
print(f"\\nVerified artifacts: {len(expected_files)}")
"""
    ),
    markdown(
        """## Notebook 06 conclusion

Shock classification is evaluated with fixed, training-only definitions and
chronological predictions. The selected model balances detection of shocks
against false alarms; it does not establish why a shock occurs.

### What to send back

After **Runtime → Run all**, send the Notebook 06 execution summary, metric table,
complete 2026 monthly risk table, and any red error output. Notebook 07 will
explain the forecasting and classification models globally and across 2026.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {
            "name": "06_shock_classification.ipynb",
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
