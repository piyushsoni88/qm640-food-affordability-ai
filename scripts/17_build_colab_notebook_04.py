"""Build the Google Colab version of Notebook 04."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "04_statistical_analysis.ipynb"


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
## Notebook 04 — Statistical Analysis and Hypothesis Testing

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Prerequisite:** Notebooks 01–03 completed successfully

### Statistical objectives

This notebook tests conditional—not causal—relationships suggested by EDA:

1. **National persistence and transmission:** Is food CPI inflation associated
   with its own lag, lagged mandi inflation, lagged world wheat and energy-price
   changes, and lagged climate conditions?
2. **State climate association:** After controlling for reporting intensity,
   seasonality, trend, commodity effects, and state/UT effects, are lagged
   rainfall or temperature anomalies associated with mandi-price changes?
3. **Horizon stability:** Do the state-level climate associations differ across
   trailing 2-, 5-, and 10-year windows?

National models use Newey–West/HAC standard errors. State models use standard
errors clustered by state/UT. These choices address serial dependence and
within-region correlation more appropriately than ordinary standard errors.
"""
    ),
    markdown(
        """## 1. Runtime, packages, and persistent storage

Google Colab normally includes `statsmodels`. A conditional installation runs
only if the import is missing. This avoids unnecessary package downloads during
normal execution.
"""
    ),
    code(
        """from pathlib import Path
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

try:
    import statsmodels as statsmodels_package
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    # This fallback is rarely needed in Colab but makes the notebook portable.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "statsmodels>=0.14"],
        check=True,
    )
    import statsmodels as statsmodels_package
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tsa.stattools import adfuller

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
    default_root = Path.cwd().resolve()
    if default_root.name == "notebooks":
        default_root = default_root.parent
    OUTPUT_ROOT = Path(os.environ.get("QM640_OUTPUT_ROOT", str(default_root)))

PROCESSED = OUTPUT_ROOT / "data" / "processed"
REPORT_OUTPUT = OUTPUT_ROOT / "reports" / "notebook_outputs"
FIGURES = REPORT_OUTPUT / "figures"
REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "savefig.dpi": 180,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print(f"Running in Colab: {IN_COLAB}")
print(f"statsmodels: {statsmodels_package.__version__}")
print(f"Output root: {OUTPUT_ROOT}")
"""
    ),
    markdown(
        """## 2. Load verified inputs

Notebook 03's JSON is checked before modeling. The underlying state and national
files remain the cleaned outputs from Notebook 02; Notebook 03 did not modify
them.
"""
    ),
    code(
        """STATE_FILE = PROCESSED / "cleaned_state_monthly.csv.gz"
NATIONAL_FILE = PROCESSED / "cleaned_national_monthly.csv.gz"
NB03_SUMMARY_FILE = REPORT_OUTPUT / "03_execution_summary.json"

required = [STATE_FILE, NATIONAL_FILE, NB03_SUMMARY_FILE]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Required outputs are missing. Run Notebooks 01–03 first.\\n"
        + "\\n".join(missing)
    )

state = pd.read_csv(
    STATE_FILE,
    parse_dates=["date"],
    dtype={"region": "category", "Commodity": "category"},
    low_memory=False,
)
national = pd.read_csv(NATIONAL_FILE, parse_dates=["date"], low_memory=False)
nb03 = json.loads(NB03_SUMMARY_FILE.read_text(encoding="utf-8"))

if len(state) != int(nb03["state_rows_analyzed"]):
    raise AssertionError("State rows differ from Notebook 03.")
if len(national) != int(nb03["national_months_analyzed"]):
    raise AssertionError("National rows differ from Notebook 03.")

print(f"Verified state rows: {len(state):,}")
print(f"Verified national months: {len(national):,}")
"""
    ),
    markdown(
        """## 3. National feature engineering

All transmission variables are lagged one month so the model never uses a future
driver to explain an earlier outcome. International prices are converted to
year-on-year percentage changes. Rainfall and temperature are converted to
month-specific anomalies relative to completed years through 2025.

The national monthly calendar is checked for gaps before row-based lagging.
"""
    ),
    code(
        """national = national.sort_values("date").copy()
expected_dates = pd.date_range(
    national["date"].min(), national["date"].max(), freq="MS"
)
if not national["date"].reset_index(drop=True).equals(pd.Series(expected_dates)):
    raise AssertionError("National monthly calendar has gaps; exact lagging is required.")

# Calculate year-on-year international-price changes from levels.
national["world_wheat_yoy_pct"] = (
    national["world_wheat_usd_per_mt"].pct_change(12) * 100
)
national["world_crude_yoy_pct"] = (
    national["world_crude_oil_usd_per_bbl"].pct_change(12) * 100
)

# Month-specific normals remove expected monsoon and temperature seasonality.
reference = national.loc[national["date"].dt.year <= 2025].copy()
rain_norm = reference.groupby(reference["date"].dt.month)[
    "state_avg_rainfall_mm"
].mean()
temp_norm = reference.groupby(reference["date"].dt.month)[
    "state_avg_temperature_c"
].mean()
national["rainfall_anomaly_pct"] = (
    100
    * (
        national["state_avg_rainfall_mm"]
        - national["date"].dt.month.map(rain_norm)
    )
    / national["date"].dt.month.map(rain_norm).replace(0, np.nan)
)
national["temperature_anomaly_c"] = (
    national["state_avg_temperature_c"]
    - national["date"].dt.month.map(temp_norm)
)

# The dependent variable is exact-calendar food CPI inflation from Notebook 02.
national["food_cpi_yoy"] = national["food_cpi_yoy_pct_exact"]
lag_sources = {
    "food_cpi_yoy": "food_cpi_yoy_lag1",
    "mandi_index_yoy_pct": "mandi_yoy_lag1",
    "world_wheat_yoy_pct": "world_wheat_yoy_lag1",
    "world_crude_yoy_pct": "world_crude_yoy_lag1",
    "rainfall_anomaly_pct": "rainfall_anomaly_lag1",
    "temperature_anomaly_c": "temperature_anomaly_lag1",
}
for source, destination in lag_sources.items():
    national[destination] = national[source].shift(1)

national["trend_years"] = (
    (national["date"] - national["date"].min()).dt.days / 365.25
)
national["month_sin"] = np.sin(2 * np.pi * national["date"].dt.month / 12)
national["month_cos"] = np.cos(2 * np.pi * national["date"].dt.month / 12)
"""
    ),
    markdown(
        """## 4. Stationarity diagnostics

The Augmented Dickey–Fuller test is applied to levels and year-on-year changes.
A small p-value rejects the unit-root null. Regression emphasis is placed on
inflation/change variables rather than non-stationary index levels.
"""
    ),
    code(
        """series_for_adf = {
    "food_cpi_level": national["food_cpi_2015_100"],
    "food_cpi_yoy": national["food_cpi_yoy"],
    "mandi_index_level": national["mandi_price_index_2015_100"],
    "mandi_index_yoy": national["mandi_index_yoy_pct"],
}
adf_rows = []
for name, series in series_for_adf.items():
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    statistic, p_value, used_lags, n_obs, critical, _ = adfuller(
        values, autolag="AIC"
    )
    adf_rows.append({
        "series": name,
        "n": len(values),
        "adf_statistic": statistic,
        "p_value": p_value,
        "used_lags": used_lags,
        "critical_value_5pct": critical["5%"],
        "reject_unit_root_at_5pct": p_value < 0.05,
    })
adf_results = pd.DataFrame(adf_rows)
adf_results.to_csv(REPORT_OUTPUT / "04_adf_stationarity_tests.csv", index=False)
print(adf_results.round(4).to_string(index=False))
"""
    ),
    markdown(
        """## 5. National HAC regressions

Two nested specifications are compared:

- **Baseline:** lagged food inflation, trend, and seasonal terms.
- **Full:** baseline plus lagged mandi, world wheat, crude oil, rainfall anomaly,
  and temperature anomaly.

Predictors are standardized using the estimation sample. Coefficients therefore
represent the change in food-inflation percentage points associated with a
one-standard-deviation predictor change. HAC standard errors use 12 monthly lags.
"""
    ),
    code(
        """baseline_features = [
    "food_cpi_yoy_lag1",
    "trend_years",
    "month_sin",
    "month_cos",
]
transmission_features = [
    "mandi_yoy_lag1",
    "world_wheat_yoy_lag1",
    "world_crude_yoy_lag1",
    "rainfall_anomaly_lag1",
    "temperature_anomaly_lag1",
]
full_features = baseline_features + transmission_features

national_model = national[
    ["date", "food_cpi_yoy", *full_features]
].replace([np.inf, -np.inf], np.nan).dropna().copy()

# One common sample ensures the nested model comparison is fair.
feature_means = national_model[full_features].mean()
feature_scales = national_model[full_features].std().replace(0, 1)
standardized_names = []
for feature in full_features:
    name = feature + "_z"
    national_model[name] = (
        national_model[feature] - feature_means[feature]
    ) / feature_scales[feature]
    standardized_names.append(name)

baseline_z = [name + "_z" for name in baseline_features]
full_z = [name + "_z" for name in full_features]

baseline_fit = sm.OLS(
    national_model["food_cpi_yoy"],
    sm.add_constant(national_model[baseline_z]),
).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
full_fit = sm.OLS(
    national_model["food_cpi_yoy"],
    sm.add_constant(national_model[full_z]),
).fit(cov_type="HAC", cov_kwds={"maxlags": 12})


def coefficient_table(result, model_name: str) -> pd.DataFrame:
    \"\"\"Convert a statsmodels result into a tidy, report-ready table.\"\"\"
    confidence = result.conf_int()
    return pd.DataFrame({
        "model": model_name,
        "term": result.params.index,
        "coefficient": result.params.values,
        "std_error": result.bse.values,
        "t_stat": result.tvalues.values,
        "p_value": result.pvalues.values,
        "ci_95_lower": confidence.iloc[:, 0].values,
        "ci_95_upper": confidence.iloc[:, 1].values,
    })


national_coefficients = pd.concat(
    [
        coefficient_table(baseline_fit, "baseline_hac"),
        coefficient_table(full_fit, "full_hac"),
    ],
    ignore_index=True,
)
national_coefficients.to_csv(
    REPORT_OUTPUT / "04_national_hac_coefficients.csv", index=False
)

national_comparison = pd.DataFrame([
    {
        "model": "baseline_hac",
        "n": int(baseline_fit.nobs),
        "r_squared": baseline_fit.rsquared,
        "adjusted_r_squared": baseline_fit.rsquared_adj,
        "aic": baseline_fit.aic,
        "bic": baseline_fit.bic,
        "residual_rmse": np.sqrt(np.mean(baseline_fit.resid ** 2)),
    },
    {
        "model": "full_hac",
        "n": int(full_fit.nobs),
        "r_squared": full_fit.rsquared,
        "adjusted_r_squared": full_fit.rsquared_adj,
        "aic": full_fit.aic,
        "bic": full_fit.bic,
        "residual_rmse": np.sqrt(np.mean(full_fit.resid ** 2)),
    },
])
national_comparison.to_csv(
    REPORT_OUTPUT / "04_national_model_comparison.csv", index=False
)
print(national_comparison.round(3).to_string(index=False))
print("\\nFull HAC model coefficients")
print(
    national_coefficients.loc[
        national_coefficients["model"].eq("full_hac")
    ].round(4).to_string(index=False)
)
"""
    ),
    markdown(
        """## 6. Multicollinearity check

Variance inflation factors (VIFs) are diagnostic rather than hypothesis tests.
Large values indicate that coefficient separation may be unstable even when
overall prediction is useful.
"""
    ),
    code(
        """vif_design = sm.add_constant(national_model[full_z])
vif = pd.DataFrame({
    "term": vif_design.columns,
    "vif": [
        variance_inflation_factor(vif_design.to_numpy(), index)
        for index in range(vif_design.shape[1])
    ],
})
vif.to_csv(REPORT_OUTPUT / "04_national_vif.csv", index=False)
print(vif.round(3).to_string(index=False))
"""
    ),
    markdown(
        """## 7. National diagnostics and coefficient figures

The fitted-value chart retains chronological order. The coefficient chart shows
95% HAC confidence intervals for standardized full-model predictors.
"""
    ),
    code(
        """national_diagnostics = national_model[["date", "food_cpi_yoy"]].copy()
national_diagnostics["baseline_fitted"] = baseline_fit.fittedvalues
national_diagnostics["full_fitted"] = full_fit.fittedvalues
national_diagnostics["full_residual"] = full_fit.resid
national_diagnostics.to_csv(
    REPORT_OUTPUT / "04_national_fitted_residuals.csv", index=False
)

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(
    national_diagnostics["date"],
    national_diagnostics["food_cpi_yoy"],
    label="Observed food CPI YoY",
    color="#1f4e79",
    linewidth=2,
)
ax.plot(
    national_diagnostics["date"],
    national_diagnostics["full_fitted"],
    label="Full HAC fitted",
    color="#d97706",
    linewidth=1.6,
)
ax.axhline(0, color="#6b7280", linewidth=0.7)
ax.set(
    title="Observed and fitted national food inflation",
    xlabel="Month",
    ylabel="Year-on-year food CPI change (%)",
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "04_national_observed_fitted.png", bbox_inches="tight")
plt.show()

full_plot = national_coefficients.loc[
    national_coefficients["model"].eq("full_hac")
    & national_coefficients["term"].ne("const")
].sort_values("coefficient")
fig, ax = plt.subplots(figsize=(10, 7))
ax.errorbar(
    full_plot["coefficient"],
    full_plot["term"],
    xerr=[
        full_plot["coefficient"] - full_plot["ci_95_lower"],
        full_plot["ci_95_upper"] - full_plot["coefficient"],
    ],
    fmt="o",
    color="#1f4e79",
    ecolor="#94a3b8",
    capsize=3,
)
ax.axvline(0, color="#b91c1c", linestyle="--", linewidth=1)
ax.set(
    title="National standardized coefficients with 95% HAC intervals",
    xlabel="Food-inflation percentage-point association",
    ylabel="Predictor",
)
fig.tight_layout()
fig.savefig(FIGURES / "04_national_hac_coefficients.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 8. State-panel preparation

State models use exact one-month lagged climate anomalies. The outcome is bounded
within commodity at its 1st and 99th percentiles **only in the modeling copy** to
prevent tiny lag denominators from dominating least squares. Notebook 02's raw
and cleaned price values remain unchanged.
"""
    ),
    code(
        """state_model = state.sort_values(["region", "Commodity", "date"]).copy()

# Create one climate row per state-month before lagging; otherwise commodities
# with more reporting would duplicate the same climate observation.
climate_lookup = (
    state_model[
        ["region", "date", "rainfall_anomaly_pct", "temperature_anomaly_c"]
    ]
    .drop_duplicates(["region", "date"])
    .copy()
)
climate_lookup["date"] = climate_lookup["date"] + pd.DateOffset(months=1)
climate_lookup = climate_lookup.rename(columns={
    "rainfall_anomaly_pct": "rainfall_anomaly_lag1",
    "temperature_anomaly_c": "temperature_anomaly_lag1",
})
state_model = state_model.merge(
    climate_lookup,
    on=["region", "date"],
    how="left",
    validate="many_to_one",
)

outcome_bounds = (
    state_model.groupby("Commodity", observed=True)["price_yoy_pct"]
    .quantile([0.01, 0.99])
    .unstack()
    .rename(columns={0.01: "yoy_lower", 0.99: "yoy_upper"})
)
state_model = state_model.join(outcome_bounds, on="Commodity")
state_model["price_yoy_model"] = state_model["price_yoy_pct"].clip(
    lower=state_model["yoy_lower"],
    upper=state_model["yoy_upper"],
)
state_model["log_source_rows"] = np.log1p(state_model["source_rows"])
state_model["calendar_month"] = state_model["date"].dt.month.astype("category")
state_model["trend_years"] = (
    (state_model["date"] - state_model["date"].min()).dt.days / 365.25
)
"""
    ),
    markdown(
        """## 9. Cluster-robust fixed-effects models: 2, 5, and 10 years

Each model includes:

- lagged rainfall and temperature anomalies;
- humidity and log reporting intensity;
- linear trend;
- calendar-month fixed effects;
- commodity fixed effects; and
- state/UT fixed effects.

Continuous predictors are standardized within each horizon. Standard errors are
clustered by state/UT to allow arbitrary within-region dependence.
"""
    ),
    code(
        """WINDOWS = {"2-year": 24, "5-year": 60, "10-year": 120}
continuous = [
    "rainfall_anomaly_lag1",
    "temperature_anomaly_lag1",
    "relative_humidity_pct",
    "log_source_rows",
    "trend_years",
]
state_coefficient_tables = []
state_metric_rows = []
state_fits = {}
latest_month = state_model["date"].max()

for label, months in WINDOWS.items():
    start = latest_month - pd.DateOffset(months=months - 1)
    window = state_model.loc[
        state_model["date"].between(start, latest_month),
        [
            "price_yoy_model",
            "region",
            "Commodity",
            "calendar_month",
            *continuous,
        ],
    ].replace([np.inf, -np.inf], np.nan).dropna().copy()

    # Standardization makes continuous coefficients comparable within the model.
    for variable in continuous:
        scale = window[variable].std()
        window[variable + "_z"] = (
            window[variable] - window[variable].mean()
        ) / (scale if scale and np.isfinite(scale) else 1.0)

    formula = (
        "price_yoy_model ~ "
        + " + ".join(variable + "_z" for variable in continuous)
        + " + C(calendar_month) + C(Commodity) + C(region)"
    )
    fit = smf.ols(formula, data=window).fit(
        cov_type="cluster",
        cov_kwds={"groups": window["region"].astype(str)},
    )
    state_fits[label] = fit
    table = coefficient_table(fit, f"state_{label}")
    table["window_start"] = start
    table["window_end"] = latest_month
    state_coefficient_tables.append(table)
    state_metric_rows.append({
        "window": label,
        "months": months,
        "start": start,
        "end": latest_month,
        "n": int(fit.nobs),
        "regions": int(window["region"].nunique()),
        "commodities": int(window["Commodity"].nunique()),
        "r_squared": fit.rsquared,
        "adjusted_r_squared": fit.rsquared_adj,
        "residual_rmse": np.sqrt(np.mean(fit.resid ** 2)),
    })
    print(f"Completed {label}: n={int(fit.nobs):,}, regions={window['region'].nunique()}")

state_coefficients = pd.concat(state_coefficient_tables, ignore_index=True)
state_metrics = pd.DataFrame(state_metric_rows)
state_coefficients.to_csv(
    REPORT_OUTPUT / "04_state_clustered_coefficients.csv", index=False
)
state_metrics.to_csv(
    REPORT_OUTPUT / "04_state_model_metrics_2_5_10_years.csv", index=False
)
print("\\nState model performance")
print(state_metrics.round(3).to_string(index=False))
"""
    ),
    markdown(
        """## 10. Multiple-testing adjustment and horizon comparison

Six focal climate tests are performed: rainfall and temperature across three
horizons. Holm adjustment controls the family-wise error rate. Both raw and
adjusted p-values are reported.
"""
    ),
    code(
        """focal_terms = [
    "rainfall_anomaly_lag1_z",
    "temperature_anomaly_lag1_z",
]
climate_tests = state_coefficients.loc[
    state_coefficients["term"].isin(focal_terms)
].copy()


def holm_adjust(p_values: pd.Series) -> np.ndarray:
    \"\"\"Return Holm-adjusted p-values in the original row order.\"\"\"
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty(len(p))
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = (len(p) - rank) * p[original_index]
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(running_max, 1.0)
    adjusted = np.empty(len(p))
    adjusted[order] = adjusted_sorted
    return adjusted


climate_tests["p_value_holm"] = holm_adjust(climate_tests["p_value"])
climate_tests["significant_holm_5pct"] = climate_tests["p_value_holm"] < 0.05
climate_tests.to_csv(
    REPORT_OUTPUT / "04_climate_horizon_tests.csv", index=False
)
print(climate_tests[
    [
        "model", "term", "coefficient", "std_error",
        "p_value", "p_value_holm", "significant_holm_5pct",
    ]
].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 6))
plot_order = ["state_2-year", "state_5-year", "state_10-year"]
offsets = {
    "rainfall_anomaly_lag1_z": -0.10,
    "temperature_anomaly_lag1_z": 0.10,
}
colors = {
    "rainfall_anomaly_lag1_z": "#2563eb",
    "temperature_anomaly_lag1_z": "#dc2626",
}
labels = {
    "rainfall_anomaly_lag1_z": "Lagged rainfall anomaly",
    "temperature_anomaly_lag1_z": "Lagged temperature anomaly",
}
for term in focal_terms:
    subset = climate_tests.loc[
        climate_tests["term"].eq(term)
    ].set_index("model").reindex(plot_order)
    x = np.arange(len(plot_order)) + offsets[term]
    ax.errorbar(
        x,
        subset["coefficient"],
        yerr=1.96 * subset["std_error"],
        fmt="o",
        capsize=4,
        color=colors[term],
        label=labels[term],
    )
ax.axhline(0, color="#374151", linestyle="--", linewidth=1)
ax.set_xticks(np.arange(len(plot_order)), ["2-year", "5-year", "10-year"])
ax.set(
    title="Climate-price associations across statistical horizons",
    xlabel="Trailing estimation window",
    ylabel="Price-change association per 1 SD predictor change",
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "04_climate_horizon_coefficients.png", bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """## 11. Hypothesis evidence table

Decisions use a 5% significance level but avoid claiming causality. The national
table reports lagged transmission terms; state climate decisions use Holm-adjusted
p-values.
"""
    ),
    code(
        """full_terms = national_coefficients.loc[
    national_coefficients["model"].eq("full_hac")
].set_index("term")
hypotheses = []
for term, description in {
    "mandi_yoy_lag1_z": "Lagged mandi inflation",
    "world_wheat_yoy_lag1_z": "Lagged world wheat-price change",
    "world_crude_yoy_lag1_z": "Lagged crude-oil price change",
    "rainfall_anomaly_lag1_z": "Lagged national rainfall anomaly",
    "temperature_anomaly_lag1_z": "Lagged national temperature anomaly",
}.items():
    row = full_terms.loc[term]
    hypotheses.append({
        "scope": "national",
        "horizon": "full available estimation sample",
        "driver": description,
        "coefficient": row["coefficient"],
        "p_value": row["p_value"],
        "adjusted_p_value": np.nan,
        "adjustment_method": "not applied; prespecified national transmission terms",
        "evidence_at_5pct": bool(row["p_value"] < 0.05),
        "interpretation": "conditional association; not causal",
    })

for _, row in climate_tests.iterrows():
    hypotheses.append({
        "scope": "state fixed effects",
        "horizon": row["model"].replace("state_", ""),
        "driver": row["term"].replace("_z", ""),
        "coefficient": row["coefficient"],
        "p_value": row["p_value"],
        "adjusted_p_value": row["p_value_holm"],
        "adjustment_method": "Holm across six state climate horizon tests",
        "evidence_at_5pct": bool(row["significant_holm_5pct"]),
        "interpretation": "cluster-robust conditional association; not causal",
    })

hypothesis_evidence = pd.DataFrame(hypotheses)
hypothesis_evidence.to_csv(
    REPORT_OUTPUT / "04_hypothesis_evidence.csv", index=False
)
print(hypothesis_evidence.round(4).to_string(index=False))
"""
    ),
    markdown(
        """## 12. Execution summary and artifact validation

The summary records model sizes, fit, significant conditional associations, and
window-specific estimates. These values will guide forecasting feature selection
in Notebook 05 without using future test outcomes.
"""
    ),
    code(
        """significant_national = hypothesis_evidence.loc[
    hypothesis_evidence["scope"].eq("national")
    & hypothesis_evidence["evidence_at_5pct"],
    "driver",
].tolist()
significant_state = climate_tests.loc[
    climate_tests["significant_holm_5pct"],
    ["model", "term", "coefficient", "p_value_holm"],
].to_dict(orient="records")

window_summaries = {}
for _, row in state_metrics.iterrows():
    window_summaries[row["window"]] = {
        "n": int(row["n"]),
        "regions": int(row["regions"]),
        "commodities": int(row["commodities"]),
        "r_squared": float(row["r_squared"]),
        "adjusted_r_squared": float(row["adjusted_r_squared"]),
        "residual_rmse": float(row["residual_rmse"]),
    }

summary = {
    "notebook": "04_statistical_analysis",
    "status": "completed",
    "national_model_n": int(full_fit.nobs),
    "national_baseline_r_squared": float(baseline_fit.rsquared),
    "national_full_r_squared": float(full_fit.rsquared),
    "national_full_adjusted_r_squared": float(full_fit.rsquared_adj),
    "national_full_residual_rmse": float(
        np.sqrt(np.mean(full_fit.resid ** 2))
    ),
    "national_hac_maxlags": 12,
    "significant_national_transmission_terms_5pct": significant_national,
    "state_window_models": window_summaries,
    "holm_significant_state_climate_terms": significant_state,
    "statistical_interpretation": (
        "Results are conditional associations, not causal effects."
    ),
    "output_root": str(OUTPUT_ROOT),
}

SUMMARY_FILE = REPORT_OUTPUT / "04_execution_summary.json"
SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

expected_files = [
    REPORT_OUTPUT / "04_adf_stationarity_tests.csv",
    REPORT_OUTPUT / "04_national_hac_coefficients.csv",
    REPORT_OUTPUT / "04_national_model_comparison.csv",
    REPORT_OUTPUT / "04_national_vif.csv",
    REPORT_OUTPUT / "04_national_fitted_residuals.csv",
    REPORT_OUTPUT / "04_state_clustered_coefficients.csv",
    REPORT_OUTPUT / "04_state_model_metrics_2_5_10_years.csv",
    REPORT_OUTPUT / "04_climate_horizon_tests.csv",
    REPORT_OUTPUT / "04_hypothesis_evidence.csv",
    FIGURES / "04_national_observed_fitted.png",
    FIGURES / "04_national_hac_coefficients.png",
    FIGURES / "04_climate_horizon_coefficients.png",
    SUMMARY_FILE,
]
missing_artifacts = [str(path) for path in expected_files if not path.exists()]
if missing_artifacts:
    raise FileNotFoundError(
        "Expected Notebook 04 artifacts are missing:\\n"
        + "\\n".join(missing_artifacts)
    )

print("=" * 75)
print("NOTEBOOK 04 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT")
print("=" * 75)
print(json.dumps(summary, indent=2))
print(f"\\nVerified artifacts: {len(expected_files)}")
"""
    ),
    markdown(
        """## Notebook 04 conclusion

The statistical models quantify conditional relationships while respecting time
order, seasonality, persistence, fixed effects, serial correlation, clustered
regional dependence, and multiple climate tests. They do not identify policy or
climate causality.

### What to send back

After selecting **Runtime → Run all**, send:

1. the `NOTEBOOK 04 EXECUTION SUMMARY`;
2. any red error output; and
3. the printed climate horizon table if possible.

Notebook 05 will compare forecasting models with chronological validation and
will use only information available before each forecast date.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {
            "name": "04_statistical_analysis.ipynb",
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
