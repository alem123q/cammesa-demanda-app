# Pre-registration — evaluation on 2022–2025

Written and committed **before** any model was evaluated on the test period.
The git history is the timestamp: this file's commit comes before the commit with the test results.

## Question

In the course (TP2) no model beat the seasonal rule "same month last year" on Sep 2019 – Feb 2020,
and the exploratory analysis suggested that the missing information was weather, activity and
tariffs. Does that hold on four years nobody has looked at?

## Data

- Demand: `data/processed/serie_regional_extendida.csv` (9 CAMMESA regions, monthly, 2017-01 to 2025-12).
- Weather: Open-Meteo, monthly heating/cooling degree-days per region (base 18 °C).
- Activity: INDEC EMAE (original series).
- Tariff proxy: regional CPI "housing, water, electricity, gas and other fuels" divided by the general CPI.

## Partition

| Set | Months | Rows |
|---|---|---|
| Train (and validation inside it) | 2018-06 to 2021-12 | 43 × 9 = 387 |
| Test (opened once) | 2022-01 to 2025-12 | 48 × 9 = 432 |

Training starts in June 2018 so every model uses the same rows (the lagged tariff variable needs 18 months of CPI).
Validation: rolling origin inside the training period, 4 blocks of 6 months (2020-01 to 2021-12), each block
predicted by a model trained on everything before it.

The test period has never been used: the only things seen about it are annual national totals while checking the
data sources.

## Task

Forecast each region's demand for month *t*, one month ahead.

## Candidates

| Name | Target | Predictors | Uses information unknown at forecast time? |
|---|---|---|---|
| **Seasonal rule** | — | demand of month *t − 12* | no |
| **TP2 Ridge** | log demand | region, log lag 1, log lag 12, month, quarter (as in the course) | no |
| **Forecast model** | log(demand / lag 12) | last month's year-over-year change; weather normal of month *t* minus actual weather of *t − 12* (heating and cooling degree-days, per region); EMAE year-over-year change at *t − 2* (last published); real tariff year-over-year change at *t − 6* | no |
| **Explanatory model** | log(demand / lag 12) | last month's year-over-year change; actual weather of *t* minus *t − 12* (per region); EMAE year-over-year change at *t*; real tariff year-over-year change at *t − 6* | **yes** (actual weather and same-month EMAE) |

All regression models are Ridge with standardised predictors. The only hyperparameter, alpha, is chosen per model
on the validation blocks from {0.01, 0.1, 1, 3, 10, 30, 100}. Weather normals are computed from 2016–2021 only.

## Metric and goals

- Metric: RMSE in GWh over the 432 test region-months. Also reported: MAE, MAPE and RMSE per year.
- **Goal 1 (forecasting):** the forecast model has at least **10 % less RMSE** than the seasonal rule
  (same threshold as TP2).
- **Goal 2 (explanation):** the explanatory model has at least 10 % less RMSE than the seasonal rule.
  Meeting it would confirm that weather, activity and tariffs explain most of what the rule misses, even if
  they cannot be known in advance.
- The TP2 Ridge is reported as a reference; no goal is attached to it.

Whatever the result, it is reported as it comes out. Any change made after seeing the test is reported as post-hoc.
