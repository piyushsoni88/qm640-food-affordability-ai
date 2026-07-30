"""Build the Google Colab version of Notebook 05."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "05_forecasting_models.ipynb"


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
## Notebook 05 — Forecasting Models and Rolling-Origin Validation

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Prerequisite:** Notebooks 01–04 completed successfully

### Forecasting objective

Forecast the national food CPI level using only information available before each
prediction month. The notebook compares:

1. last-observation naïve;
2. seasonal naïve (12-month lag);
3. autoregressive linear regression with Ridge regularization;
4. Random Forest; and
5. histogram gradient boosting.

Models are judged by expanding-window, one-step rolling-origin validation. This
is stricter and more realistic than a random train/test split. A complex model is
accepted only if it improves on simple persistence baselines.
"""
    ),
    markdown(
        """## 1. Runtime and persistent storage

Colab normally includes scikit-learn, matplotlib, and joblib. The conditional
installation executes only if those packages are unavailable.
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
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "scikit-learn>=1.5", "joblib", "statsmodels>=0.14"],
        check=True,
    )
    import sklearn
    import joblib
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
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
        """## 2. Load and verify inputs

The cleaned national table is verified against Notebook 04. Notebook 04 results
are used for interpretation, not to select a winner using future forecast errors.
"""
    ),
    code(
        """NATIONAL_FILE = PROCESSED / "cleaned_national_monthly.csv.gz"
NB04_SUMMARY = REPORT_OUTPUT / "04_execution_summary.json"
required = [NATIONAL_FILE, NB04_SUMMARY]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Run Notebooks 01–04 before Notebook 05.\\n" + "\\n".join(missing)
    )

national = pd.read_csv(NATIONAL_FILE, parse_dates=["date"], low_memory=False)
nb04 = json.loads(NB04_SUMMARY.read_text(encoding="utf-8"))
if len(national) != 319:
    raise AssertionError("Expected 319 national months from Notebook 02.")
print(f"National rows: {len(national):,}")
print(f"Notebook 04 full-model R²: {nb04['national_full_r_squared']:.4f}")
print(
    "Significant national transmission terms:",
    nb04["significant_national_transmission_terms_5pct"],
)
"""
    ),
    markdown(
        """## 3. Leakage-safe forecasting features

Food CPI lags use 1, 2, 3, 6, and 12 months. External variables enter with at
least a one-month lag. Calendar terms are known in advance. No centered rolling
window, backward fill, or future observation is used.
"""
    ),
    code(
        """data = national.sort_values("date").copy()
TARGET = "food_cpi_2015_100"

# A complete monthly calendar is required for row shifts to equal calendar lags.
expected = pd.date_range(data["date"].min(), data["date"].max(), freq="MS")
if not data["date"].reset_index(drop=True).equals(pd.Series(expected)):
    raise AssertionError("National calendar is not consecutive monthly data.")

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

# Past-only rolling features summarize recent momentum and volatility.
data["cpi_rolling_mean_3_lag1"] = data[TARGET].shift(1).rolling(3).mean()
data["cpi_rolling_mean_12_lag1"] = data[TARGET].shift(1).rolling(12).mean()
data["cpi_rolling_std_12_lag1"] = data[TARGET].shift(1).rolling(12).std()
data["month_sin"] = np.sin(2 * np.pi * data["date"].dt.month / 12)
data["month_cos"] = np.cos(2 * np.pi * data["date"].dt.month / 12)
data["trend_months"] = np.arange(len(data), dtype=float)

FEATURES = [
    "cpi_lag_1", "cpi_lag_2", "cpi_lag_3", "cpi_lag_6", "cpi_lag_12",
    "cpi_rolling_mean_3_lag1", "cpi_rolling_mean_12_lag1",
    "cpi_rolling_std_12_lag1",
    *[variable + "_lag1" for variable in external_levels],
    "month_sin", "month_cos", "trend_months",
]
model_data = (
    data[["date", TARGET, *FEATURES]]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)
print(f"Model-ready months: {len(model_data):,}")
print(f"Coverage: {model_data.date.min().date()} to {model_data.date.max().date()}")
"""
    ),
    markdown(
        """## 4. Model definitions

Ridge handles correlated CPI lags. Random Forest and histogram boosting allow
nonlinearities and interactions. Their depth/leaf constraints are deliberately
conservative because the national sample is small.
"""
    ),
    code(
        """def make_models():
    \"\"\"Return fresh estimators so no fitted state leaks between origins.\"\"\"
    return {
        "ridge_arx": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=4,
            max_features=0.75,
            random_state=640,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=2.0,
            random_state=640,
        ),
    }
"""
    ),
    markdown(
        """## 5. Expanding-window rolling-origin validation

The final 60 model-ready months are forecast one at a time. For prediction month
`t`, every fitted model uses only rows strictly earlier than `t`. Models are
refitted at each origin; this is computationally heavier but prevents future
information from entering earlier forecasts.
"""
    ),
    code(
        """N_EVALUATION_MONTHS = 60
MIN_TRAIN_MONTHS = 120
first_test_index = len(model_data) - N_EVALUATION_MONTHS
if first_test_index < MIN_TRAIN_MONTHS:
    raise AssertionError("Insufficient history for the requested rolling evaluation.")

prediction_rows = []
started = time.perf_counter()

for test_index in range(first_test_index, len(model_data)):
    train = model_data.iloc[:test_index]
    test = model_data.iloc[[test_index]]
    if train["date"].max() >= test["date"].iloc[0]:
        raise AssertionError("Temporal leakage detected.")

    actual = float(test[TARGET].iloc[0])
    row = {
        "date": test["date"].iloc[0],
        "actual": actual,
        "train_end": train["date"].max(),
        "train_n": len(train),
        "naive_last": float(test["cpi_lag_1"].iloc[0]),
        "seasonal_naive": float(test["cpi_lag_12"].iloc[0]),
    }
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test = test[FEATURES]
    for name, estimator in make_models().items():
        estimator.fit(X_train, y_train)
        row[name] = float(estimator.predict(X_test)[0])
    prediction_rows.append(row)

    if (test_index - first_test_index + 1) % 12 == 0:
        print(f"Completed {test_index - first_test_index + 1}/{N_EVALUATION_MONTHS} origins")

rolling_predictions = pd.DataFrame(prediction_rows)
rolling_predictions.to_csv(
    REPORT_OUTPUT / "05_rolling_origin_predictions.csv", index=False
)
print(f"Rolling validation elapsed: {time.perf_counter() - started:.1f} seconds")
"""
    ),
    markdown(
        """## 6. Forecast metrics and model selection

RMSE is the primary metric because large CPI errors matter operationally. MAE,
MAPE, symmetric MAPE, MASE, bias, and directional accuracy provide complementary
evidence. MASE below 1 means improvement over the in-sample one-month naïve scale.
"""
    ),
    code(
        """MODEL_NAMES = [
    "naive_last",
    "seasonal_naive",
    "ridge_arx",
    "random_forest",
    "hist_gradient_boosting",
]

# The MASE denominator uses only pre-evaluation training data.
initial_train = model_data.iloc[:first_test_index][TARGET]
mase_scale = initial_train.diff().abs().dropna().mean()

metric_rows = []
actual = rolling_predictions["actual"].to_numpy()
previous_actual = rolling_predictions["naive_last"].to_numpy()
actual_direction = np.sign(actual - previous_actual)
for model_name in MODEL_NAMES:
    predicted = rolling_predictions[model_name].to_numpy()
    error = actual - predicted
    # Direction is evaluated against the CPI value known at the forecast origin,
    # not against the previous prediction from a different fitted model.
    predicted_direction = np.sign(predicted - previous_actual)
    metric_rows.append({
        "model": model_name,
        "n_forecasts": len(actual),
        "mae": np.mean(np.abs(error)),
        "rmse": np.sqrt(np.mean(error ** 2)),
        "mape_pct": np.mean(np.abs(error / actual)) * 100,
        "smape_pct": np.mean(
            200 * np.abs(error) / (np.abs(actual) + np.abs(predicted))
        ),
        "mase": np.mean(np.abs(error)) / mase_scale,
        "mean_error_bias": np.mean(error),
        "directional_accuracy": np.mean(
            predicted_direction == actual_direction
        ),
    })
metrics = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
metrics["rmse_improvement_vs_naive_pct"] = (
    100
    * (
        metrics.loc[metrics["model"].eq("naive_last"), "rmse"].iloc[0]
        - metrics["rmse"]
    )
    / metrics.loc[metrics["model"].eq("naive_last"), "rmse"].iloc[0]
)
metrics.to_csv(REPORT_OUTPUT / "05_forecast_model_metrics.csv", index=False)
BEST_MODEL = str(metrics.iloc[0]["model"])
print(metrics.round(4).to_string(index=False))
print(f"\\nSelected model by rolling RMSE: {BEST_MODEL}")
"""
    ),
    markdown(
        """## 7. Forecast comparison figures

The first figure preserves time order; the second compares RMSE and MAE directly.
"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(
    rolling_predictions["date"],
    rolling_predictions["actual"],
    color="#111827",
    linewidth=2.3,
    label="Observed",
)
palette = {
    "naive_last": "#64748b",
    "seasonal_naive": "#a855f7",
    "ridge_arx": "#2563eb",
    "random_forest": "#059669",
    "hist_gradient_boosting": "#d97706",
}
for name in MODEL_NAMES:
    ax.plot(
        rolling_predictions["date"],
        rolling_predictions[name],
        label=name.replace("_", " ").title(),
        color=palette[name],
        linewidth=1.2,
        alpha=0.85,
    )
ax.set(
    title="Rolling-origin national food CPI forecasts",
    xlabel="Forecast month",
    ylabel="Food CPI (2015=100)",
)
ax.legend(ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "05_rolling_forecasts.png", bbox_inches="tight")
plt.show()

plot_metrics = metrics.sort_values("rmse", ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(plot_metrics))
ax.barh(x - 0.18, plot_metrics["rmse"], height=0.35, label="RMSE", color="#1f4e79")
ax.barh(x + 0.18, plot_metrics["mae"], height=0.35, label="MAE", color="#d97706")
ax.set_yticks(x, plot_metrics["model"].str.replace("_", " ").str.title())
ax.set(
    title="Chronological forecast-error comparison",
    xlabel="Food CPI index points",
    ylabel="Model",
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "05_forecast_metrics.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 8. Fit final candidate models and preserve them

All model-ready observations are used only after validation is complete. Fitted
objects and feature names are saved so Notebook 07 can explain the models without
silently retraining a different specification.
"""
    ),
    code(
        """final_models = make_models()
for name, estimator in final_models.items():
    estimator.fit(model_data[FEATURES], model_data[TARGET])
    joblib.dump(estimator, MODELS / f"05_{name}.joblib")

metadata = {
    "target": TARGET,
    "features": FEATURES,
    "training_start": str(model_data["date"].min().date()),
    "training_end": str(model_data["date"].max().date()),
    "training_rows": len(model_data),
    "selected_model": BEST_MODEL,
    "selection_metric": "rolling-origin RMSE",
}
_ = (MODELS / "05_model_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
"""
    ),
    markdown(
        """## 9. Genuine 24-step backtesting and final two-year forecast

The best one-month model is not automatically the best two-year model. A second
backtest therefore evaluates complete 24-month paths from 24 historical forecast
origins. Four long-horizon methods are compared:

- flat naïve;
- repeating seasonal naïve;
- five-year local drift; and
- damped-trend seasonal exponential smoothing.

The winning long-horizon method is selected by RMSE across all 24 steps. Final
uncertainty limits use the observed error distribution at each corresponding
horizon, rather than extrapolating one-step errors.
"""
    ),
    code(
        """FORECAST_HORIZON_MONTHS = 24
LONG_MODELS = ["flat_naive", "seasonal_naive", "local_drift", "damped_ets"]
history = data[["date", TARGET]].dropna().sort_values("date").reset_index(drop=True)


def long_horizon_forecasts(values, horizon=24):
    \"\"\"Forecast a complete path using methods suitable for extrapolation.\"\"\"
    values = np.asarray(values, dtype=float)
    if len(values) < 72:
        raise ValueError("At least six years are required for long-horizon models.")

    flat = np.repeat(values[-1], horizon)
    seasonal = np.resize(values[-12:], horizon)

    # Local drift uses the latest five years, limiting dependence on very old
    # structural regimes while still smoothing short-run noise.
    drift_per_month = (values[-1] - values[-61]) / 60
    drift = values[-1] + drift_per_month * np.arange(1, horizon + 1)

    ets_fit = ExponentialSmoothing(
        values,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True, remove_bias=True)
    ets = np.asarray(ets_fit.forecast(horizon), dtype=float)
    return {
        "flat_naive": flat,
        "seasonal_naive": seasonal,
        "local_drift": drift,
        "damped_ets": ets,
    }


# The last 24 eligible origins each have a complete 24-month future path.
first_origin = len(history) - 2 * FORECAST_HORIZON_MONTHS
last_origin_exclusive = len(history) - FORECAST_HORIZON_MONTHS
multistep_rows = []
for origin_index in range(first_origin, last_origin_exclusive):
    train_values = history.loc[:origin_index, TARGET].to_numpy()
    origin_date = history.loc[origin_index, "date"]
    actual_future = history.loc[
        origin_index + 1 : origin_index + FORECAST_HORIZON_MONTHS, TARGET
    ].to_numpy()
    forecast_paths = long_horizon_forecasts(
        train_values, FORECAST_HORIZON_MONTHS
    )
    for model_name, path in forecast_paths.items():
        for horizon, (actual_value, predicted_value) in enumerate(
            zip(actual_future, path), start=1
        ):
            multistep_rows.append({
                "origin_date": origin_date,
                "forecast_date": origin_date + pd.DateOffset(months=horizon),
                "horizon_months": horizon,
                "model": model_name,
                "actual": actual_value,
                "forecast": predicted_value,
                "error": actual_value - predicted_value,
            })

multistep_predictions = pd.DataFrame(multistep_rows)
multistep_predictions.to_csv(
    REPORT_OUTPUT / "05_multihorizon_backtest_predictions.csv", index=False
)

multihorizon_metrics = (
    multistep_predictions.groupby("model", as_index=False)
    .apply(
        lambda group: pd.Series({
            "backtest_origins": group["origin_date"].nunique(),
            "forecast_errors": len(group),
            "mae_all_horizons": group["error"].abs().mean(),
            "rmse_all_horizons": np.sqrt(np.mean(group["error"] ** 2)),
            "bias_all_horizons": group["error"].mean(),
            "rmse_h1": np.sqrt(np.mean(
                group.loc[group["horizon_months"].eq(1), "error"] ** 2
            )),
            "rmse_h6": np.sqrt(np.mean(
                group.loc[group["horizon_months"].eq(6), "error"] ** 2
            )),
            "rmse_h12": np.sqrt(np.mean(
                group.loc[group["horizon_months"].eq(12), "error"] ** 2
            )),
            "rmse_h24": np.sqrt(np.mean(
                group.loc[group["horizon_months"].eq(24), "error"] ** 2
            )),
        }),
        include_groups=False,
    )
    .reset_index(drop=True)
    .sort_values("rmse_all_horizons")
)
multihorizon_metrics.to_csv(
    REPORT_OUTPUT / "05_multihorizon_model_metrics.csv", index=False
)
BEST_LONG_MODEL = str(multihorizon_metrics.iloc[0]["model"])
print("Twenty-four-step backtest metrics")
print(multihorizon_metrics.round(3).to_string(index=False))
print(f"\\nSelected long-horizon model: {BEST_LONG_MODEL}")

# Refit the selected long-horizon method to all CPI observations.
last_date = history["date"].max()
requested_start = pd.Timestamp("2026-01-01")
if last_date + pd.DateOffset(months=1) != requested_start:
    raise AssertionError("Available CPI history no longer implies a January 2026 start.")
final_paths = long_horizon_forecasts(
    history[TARGET].to_numpy(), FORECAST_HORIZON_MONTHS
)
point_path = final_paths[BEST_LONG_MODEL]
forward_forecast = pd.DataFrame({
    "date": pd.date_range(requested_start, periods=24, freq="MS"),
    "horizon_months": np.arange(1, 25),
    "forecast": point_path,
})

# Each horizon uses errors observed at that same horizon across 24 backtest origins.
selected_errors = multistep_predictions.loc[
    multistep_predictions["model"].eq(BEST_LONG_MODEL)
]
intervals = (
    selected_errors.groupby("horizon_months")["error"]
    .quantile([0.025, 0.975])
    .unstack()
    .rename(columns={0.025: "error_q025", 0.975: "error_q975"})
    .reset_index()
)
forward_forecast = forward_forecast.merge(
    intervals, on="horizon_months", how="left", validate="one_to_one"
)
forward_forecast["lower_95_empirical"] = (
    forward_forecast["forecast"] + forward_forecast["error_q025"]
)
forward_forecast["upper_95_empirical"] = (
    forward_forecast["forecast"] + forward_forecast["error_q975"]
)
forward_forecast["model"] = BEST_LONG_MODEL
forward_forecast["assumption"] = (
    "Univariate CPI path; interval calibrated from matching historical horizons"
)
forward_forecast.to_csv(
    REPORT_OUTPUT / "05_twenty_four_month_conditional_forecast.csv", index=False
)
print("\\nFinal January 2026–December 2027 forecast")
print(forward_forecast.round(3).to_string(index=False))
"""
    ),
    markdown(
        """## 10. Forward-forecast visualization

The displayed history is limited to five years for readability; model fitting
still uses the complete model-ready history.
"""
    ),
    code(
        """recent_history = history.tail(60)
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(
    recent_history["date"],
    recent_history[TARGET],
    color="#1f4e79",
    linewidth=2,
    label="Observed history",
)
ax.plot(
    forward_forecast["date"],
    forward_forecast["forecast"],
    color="#d97706",
    linewidth=2,
    marker="o",
    label=f"{BEST_LONG_MODEL} 24-step forecast",
)
ax.fill_between(
    forward_forecast["date"],
    forward_forecast["lower_95_empirical"],
    forward_forecast["upper_95_empirical"],
    color="#fbbf24",
    alpha=0.25,
    label="Empirical 95% error interval",
)
ax.axvline(last_date, color="#6b7280", linestyle="--", linewidth=1)
ax.set(
    title="Twenty-four-month conditional national food CPI forecast",
    xlabel="Month",
    ylabel="Food CPI (2015=100)",
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "05_twenty_four_month_forecast.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 11. Execution summary and artifact validation

The summary records whether a complex model actually beat the naïve baseline.
This prevents the final report from selecting a model merely because it sounds
more advanced.
"""
    ),
    code(
        """best_metrics = metrics.iloc[0]
naive_metrics = metrics.loc[metrics["model"].eq("naive_last")].iloc[0]
best_long_metrics = multihorizon_metrics.iloc[0]
summary = {
    "notebook": "05_forecasting_models",
    "status": "completed",
    "model_ready_months": int(len(model_data)),
    "rolling_origins": int(len(rolling_predictions)),
    "evaluation_start": str(rolling_predictions["date"].min().date()),
    "evaluation_end": str(rolling_predictions["date"].max().date()),
    "models_compared": MODEL_NAMES,
    "selected_one_step_model": BEST_MODEL,
    "selected_one_step_rmse": float(best_metrics["rmse"]),
    "selected_one_step_mae": float(best_metrics["mae"]),
    "selected_one_step_mape_pct": float(best_metrics["mape_pct"]),
    "selected_one_step_mase": float(best_metrics["mase"]),
    "naive_rmse": float(naive_metrics["rmse"]),
    "rmse_improvement_vs_naive_pct": float(
        best_metrics["rmse_improvement_vs_naive_pct"]
    ),
    "complex_model_beats_naive": bool(
        BEST_MODEL not in ["naive_last", "seasonal_naive"]
    ),
    "forecast_horizon_months": int(FORECAST_HORIZON_MONTHS),
    "multihorizon_backtest_origins": int(
        best_long_metrics["backtest_origins"]
    ),
    "selected_long_horizon_model": BEST_LONG_MODEL,
    "selected_long_horizon_rmse_all_steps": float(
        best_long_metrics["rmse_all_horizons"]
    ),
    "selected_long_horizon_rmse_h24": float(
        best_long_metrics["rmse_h24"]
    ),
    "forward_forecast_start": str(forward_forecast["date"].min().date()),
    "forward_forecast_end": str(forward_forecast["date"].max().date()),
    "forward_forecast_assumption": (
        "Univariate CPI forecast selected by 24-step backtesting; "
        "uncertainty calibrated separately at each historical horizon."
    ),
    "output_root": str(OUTPUT_ROOT),
}
SUMMARY_FILE = REPORT_OUTPUT / "05_execution_summary.json"
SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

expected_files = [
    REPORT_OUTPUT / "05_rolling_origin_predictions.csv",
    REPORT_OUTPUT / "05_forecast_model_metrics.csv",
    REPORT_OUTPUT / "05_multihorizon_backtest_predictions.csv",
    REPORT_OUTPUT / "05_multihorizon_model_metrics.csv",
    REPORT_OUTPUT / "05_twenty_four_month_conditional_forecast.csv",
    FIGURES / "05_rolling_forecasts.png",
    FIGURES / "05_forecast_metrics.png",
    FIGURES / "05_twenty_four_month_forecast.png",
    MODELS / "05_ridge_arx.joblib",
    MODELS / "05_random_forest.joblib",
    MODELS / "05_hist_gradient_boosting.joblib",
    MODELS / "05_model_metadata.json",
    SUMMARY_FILE,
]
missing_artifacts = [str(path) for path in expected_files if not path.exists()]
if missing_artifacts:
    raise FileNotFoundError(
        "Notebook 05 artifacts missing:\\n" + "\\n".join(missing_artifacts)
    )

print("=" * 73)
print("NOTEBOOK 05 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT")
print("=" * 73)
print(json.dumps(summary, indent=2))
print(f"\\nVerified artifacts: {len(expected_files)}")
"""
    ),
    markdown(
        """## Notebook 05 conclusion

One-step and 24-step models are selected by separate chronological out-of-sample
tests. The January 2026–December 2027 path uses a long-horizon method and
horizon-matched empirical uncertainty intervals.

### What to send back

After **Runtime → Run all**, send the `NOTEBOOK 05 EXECUTION SUMMARY`, the printed
metric table, and any red error output. Notebook 06 will define food-price shocks
using training-only thresholds and compare classification models.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {
            "name": "05_forecasting_models.ipynb",
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
