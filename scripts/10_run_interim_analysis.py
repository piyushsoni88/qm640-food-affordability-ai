from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "curated"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
METADATA = ROOT / "data" / "metadata"

FEATURES = [
    "food_cpi_2015_100",
    "food_cpi_lag_1",
    "food_cpi_lag_3",
    "food_cpi_lag_12",
    "world_rice_usd_per_mt",
    "world_wheat_usd_per_mt",
    "world_palm_oil_usd_per_mt",
    "world_soybean_oil_usd_per_mt",
    "world_sugar_usd_per_kg",
    "world_crude_oil_usd_per_bbl",
    "rainfall_mm",
    "temperature_c",
    "relative_humidity_pct",
    "month_sin",
    "month_cos",
]


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    values = np.where(
        denominator == 0,
        0,
        200 * np.abs(predicted - actual) / denominator,
    )
    return float(np.mean(values))


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "sMAPE_pct": smape(actual.to_numpy(), np.asarray(predicted)),
        "R2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)

    national = pd.read_csv(
        DATA / "india_food_affordability_national_monthly.csv.gz",
        parse_dates=["date"],
    ).sort_values("date")
    climate = pd.read_csv(
        DATA / "nasa_power_india_15_regions_monthly_2005_2025.csv.gz",
        parse_dates=["date"],
    )
    national["world_wheat_usd_per_mt"] = national[
        "world_wheat_usd_per_mt"
    ].interpolate(limit_direction="both")
    national["target_next_cpi"] = national["food_cpi_2015_100"].shift(-1)
    national["target_next_mom_pct"] = national["food_cpi_mom_pct"].shift(-1)
    rolling_mean = national["food_cpi_mom_pct"].rolling(12, min_periods=6).mean()
    rolling_std = national["food_cpi_mom_pct"].rolling(12, min_periods=6).std()
    national["shock_z"] = (
        national["target_next_mom_pct"] - rolling_mean
    ) / rolling_std.replace(0, np.nan)
    national["target_next_shock"] = (
        national["shock_z"].abs().ge(2)
        | national["target_next_mom_pct"].abs().ge(10)
    ).astype(int)

    model_data = national.dropna(subset=FEATURES + ["target_next_cpi"]).copy()
    split_date = pd.Timestamp("2022-01-01")
    train = model_data.loc[model_data["date"] < split_date].copy()
    test = model_data.loc[model_data["date"] >= split_date].copy()
    x_train, y_train = train[FEATURES], train["target_next_cpi"]
    x_test, y_test = test[FEATURES], test["target_next_cpi"]

    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    ridge.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    predictions = {
        "Persistence": test["food_cpi_2015_100"].to_numpy(),
        "Ridge regression": ridge.predict(x_test),
        "Random forest": forest.predict(x_test),
    }
    rows = []
    for model, predicted in predictions.items():
        rows.append({"Model": model, **regression_metrics(y_test, predicted)})
    regression_results = pd.DataFrame(rows).sort_values("RMSE")
    regression_results.to_csv(TABLES / "preliminary_regression_performance.csv", index=False)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_rows = []
    for model_name, estimator in [("Ridge regression", ridge), ("Random forest", forest)]:
        scores = -cross_val_score(
            estimator,
            x_train,
            y_train,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        cv_rows.append(
            {
                "Model": model_name,
                "CV_MAE_mean": float(scores.mean()),
                "CV_MAE_std": float(scores.std(ddof=1)),
                "folds": len(scores),
            }
        )
    cv_results = pd.DataFrame(cv_rows)
    cv_results.to_csv(TABLES / "rolling_origin_cross_validation.csv", index=False)

    best_model_name = regression_results.iloc[0]["Model"]
    best_predictions = predictions[best_model_name]
    baseline_errors = np.abs(y_test.to_numpy() - predictions["Persistence"])
    best_errors = np.abs(y_test.to_numpy() - best_predictions)
    if np.allclose(baseline_errors, best_errors):
        wilcoxon_stat, wilcoxon_p = np.nan, 1.0
    else:
        wilcoxon_stat, wilcoxon_p = wilcoxon(
            baseline_errors,
            best_errors,
            alternative="greater",
            zero_method="wilcox",
        )

    classification_data = model_data.dropna(subset=["target_next_shock"]).copy()
    train_c = classification_data.loc[classification_data["date"] < split_date]
    test_c = classification_data.loc[classification_data["date"] >= split_date]
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    classifier.fit(train_c[FEATURES], train_c["target_next_shock"])
    shock_prediction = classifier.predict(test_c[FEATURES])
    shock_probability = classifier.predict_proba(test_c[FEATURES])[:, 1]
    classification_results = pd.DataFrame(
        [
            {
                "Model": "Random forest classifier",
                "test_rows": len(test_c),
                "positive_events": int(test_c["target_next_shock"].sum()),
                "Accuracy": accuracy_score(test_c["target_next_shock"], shock_prediction),
                "Precision": precision_score(
                    test_c["target_next_shock"], shock_prediction, zero_division=0
                ),
                "Recall": recall_score(
                    test_c["target_next_shock"], shock_prediction, zero_division=0
                ),
                "F1": f1_score(
                    test_c["target_next_shock"], shock_prediction, zero_division=0
                ),
                "ROC_AUC": (
                    roc_auc_score(test_c["target_next_shock"], shock_probability)
                    if test_c["target_next_shock"].nunique() == 2
                    else np.nan
                ),
            }
        ]
    )
    classification_results.to_csv(
        TABLES / "preliminary_shock_classification_performance.csv", index=False
    )

    forest_estimator = forest.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": forest_estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(TABLES / "random_forest_feature_importance.csv", index=False)

    inference_features = [
        "food_cpi_lag_1",
        "world_rice_usd_per_mt",
        "world_wheat_usd_per_mt",
        "world_palm_oil_usd_per_mt",
        "rainfall_mm",
        "temperature_c",
        "month_sin",
        "month_cos",
    ]
    inference = national.dropna(
        subset=inference_features + ["target_next_mom_pct"]
    ).copy()
    standardized = inference[inference_features].apply(
        lambda column: (column - column.mean()) / column.std(ddof=0)
    )
    ols = sm.OLS(
        inference["target_next_mom_pct"],
        sm.add_constant(standardized),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    coefficients = pd.DataFrame(
        {
            "term": ols.params.index,
            "coefficient": ols.params.values,
            "std_error_HAC": ols.bse.values,
            "p_value": ols.pvalues.values,
            "ci_95_low": ols.conf_int()[0].values,
            "ci_95_high": ols.conf_int()[1].values,
        }
    )
    coefficients.to_csv(TABLES / "preliminary_ols_hac_coefficients.csv", index=False)

    profile = pd.DataFrame(
        {
            "dataset": [
                "FAOSTAT India consumer price indices",
                "FAOSTAT India producer prices",
                "FAOSTAT India crop/livestock production",
                "NASA POWER 15-region monthly climate",
                "World Bank Pink Sheet monthly benchmarks",
                "Derived 15-region x 8-commodity panel",
            ],
            "rows": [
                924,
                5665,
                26432,
                len(climate),
                len(
                    pd.read_csv(
                        DATA / "world_bank_pink_sheet_food_energy_monthly.csv.gz"
                    )
                ),
                len(
                    pd.read_csv(
                        DATA
                        / "india_food_affordability_panel_15x8_2005_2025.csv.gz"
                    )
                ),
            ],
        }
    )
    profile.to_csv(TABLES / "dataset_profile.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    accent = "#1f4e79"
    orange = "#e67e22"

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(national["date"], national["food_cpi_2015_100"], color=accent, linewidth=2)
    ax.set(
        title="India Food Consumer Price Index, 2005-2025",
        xlabel="Year",
        ylabel="Index (2015 = 100)",
    )
    ax.text(
        0.01,
        0.98,
        "Observed monthly FAOSTAT series",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "food_cpi_trend.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ordered = regression_results.sort_values("RMSE", ascending=True)
    ax.barh(ordered["Model"], ordered["RMSE"], color=[accent, "#5b9bd5", orange])
    ax.set(
        title="One-Month-Ahead Food CPI Forecast Error",
        xlabel="Test RMSE (index points; lower is better)",
        ylabel="",
    )
    for index, value in enumerate(ordered["RMSE"]):
        ax.text(value, index, f" {value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_performance_rmse.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(test["date"], y_test, label="Actual", color="#111111", linewidth=2.2)
    ax.plot(
        test["date"],
        best_predictions,
        label=best_model_name,
        color=orange,
        linewidth=1.8,
    )
    ax.set(
        title=f"Actual vs. Predicted Food CPI ({best_model_name})",
        xlabel="Test period",
        ylabel="Index (2015 = 100)",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "actual_vs_predicted_food_cpi.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    top = importance.head(10).sort_values("importance")
    ax.barh(top["feature"], top["importance"], color=accent)
    ax.set(
        title="Random Forest Feature Importance",
        xlabel="Mean decrease in impurity",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "random_forest_feature_importance.png", dpi=220)
    plt.close(fig)

    annual_rainfall = (
        climate.assign(year=climate["date"].dt.year)
        .groupby(["region", "year"], as_index=False)["prectotcorr"]
        .mean()
        .pivot(index="region", columns="year", values="prectotcorr")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(annual_rainfall, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(len(annual_rainfall.index)))
    ax.set_yticklabels(annual_rainfall.index, fontsize=8)
    selected_ticks = list(range(0, len(annual_rainfall.columns), 3))
    ax.set_xticks(selected_ticks)
    ax.set_xticklabels(
        [annual_rainfall.columns[index] for index in selected_ticks],
        rotation=45,
        ha="right",
    )
    ax.set(title="Regional Monthly-Mean Rainfall by Year", xlabel="Year", ylabel="Region")
    fig.colorbar(image, ax=ax, label="NASA POWER precipitation (mm/day)")
    fig.tight_layout()
    fig.savefig(FIGURES / "regional_rainfall_heatmap.png", dpi=220)
    plt.close(fig)

    summary = {
        "analysis_generated_at": pd.Timestamp.utcnow().isoformat(),
        "train_period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
        "train_rows": len(train),
        "test_rows": len(test),
        "best_regression_model": best_model_name,
        "regression_results": regression_results.to_dict("records"),
        "cross_validation": cv_results.to_dict("records"),
        "shock_classification": classification_results.to_dict("records"),
        "paired_wilcoxon_best_vs_persistence": {
            "statistic": None if np.isnan(wilcoxon_stat) else float(wilcoxon_stat),
            "p_value_one_sided": float(wilcoxon_p),
        },
        "ols_adjusted_r_squared": float(ols.rsquared_adj),
        "ols_n": int(ols.nobs),
        "top_features": importance.head(8).to_dict("records"),
        "interpretation_warning": (
            "These are interim predictive associations using national price proxies. "
            "They do not establish causality or replace the planned AGMARKNET panel."
        ),
    }
    (METADATA / "interim_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
