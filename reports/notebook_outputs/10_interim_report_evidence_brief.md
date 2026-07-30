# QM640 Interim Report Evidence Brief

## Core validated results

- The cleaned state panel contains 41,792 rows.
- The one-step winner is `naive_last` (RMSE 2.063).
- The 24-step winner is `damped_ets` (all-step RMSE 6.044).
- The shock classifier is `hist_gradient_boosting` with balanced accuracy 0.773 and recall 0.600.
- Baseline 2026 maximum shock probability is 4.12%.
- Mean forecast 2026 food-cost growth is 2.58%.
- Severe-scenario maximum shock probability is 74.88% and maximum HFASI is 102.54.

## Required interpretation

- Forecast and classification results are predictive, not causal.
- January 2026 uses observed inputs; later 2026 risks are conditional.
- HFASI uses representative aggregate rural/urban food shares.
- Scenario assumptions are controlled stress tests, not forecasts.
- The rare-event evaluation contains only five observed shocks.