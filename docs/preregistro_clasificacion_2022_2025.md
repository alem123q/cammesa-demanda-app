# Pre-registration — classification on 2022–2025

Written and committed **before** any classification metric was computed on the test period.

## Disclosure

The regression test on 2022–2025 had already been opened (see `preregistro_2022_2025.md`): its RMSE by model and
year is known. No classification metric (F1, kappa, confusion matrix, share of "up" months) has been computed on
2022–2025 before this commit.

## Question

In the course (TP3) no classifier beat the persistence rule ("same direction as last month") on whether demand is
above the same month last year. Does the same information that helped the regression (weather, activity, tariffs)
change that on 2022–2025?

## Data and split

Same as the regression pre-registration: train (and validation) Jun 2018 – Dec 2021, 387 region-months;
test Jan 2022 – Dec 2025, 432 region-months. Validation: 4 rolling-origin blocks of 6 months inside the training period.

Target: **sube** = 1 if the month's demand is above the same month last year.

## Candidates

| Name | How it decides | Role |
|---|---|---|
| **Persistence rule** | *sube* if last month was above its same month a year earlier | baseline |
| Always *sube* | always 1 | reference only |
| **Forecast classifier** | *sube* if the pre-registered *forecast model* (regression, α = 100) predicts a positive year-over-year change | **goal 1** |
| **Explanatory classifier** | *sube* if the *explanatory model* (regression, α = 30) predicts a positive change | **goal 2** |
| Logistic, forecast predictors | logistic regression (balanced classes) on the forecast model's predictors; C chosen on validation F1 from {0.01, 0.1, 1, 10} | reported, no goal |
| Logistic, explanatory predictors | same with the explanatory predictors | reported, no goal |
| TP3 SVM | default SVC on the TP3 predictors, as pre-registered in the course | reported, no goal |

The two goal classifiers reuse the regression models exactly as pre-registered (no new fitting or tuning), which
keeps the number of tested options small.

## Metrics and goals

- Primary metric: **F1 of the class *sube*** (as in TP3). Also reported: Cohen's kappa, accuracy, balanced accuracy, confusion matrix.
- A classifier **meets its goal** if it beats the persistence rule on **both** F1 *sube* and Cohen's kappa on the test.
  Kappa is required too so that a classifier cannot win just by saying *sube* more often.
- **Goal 1:** the forecast classifier meets it. **Goal 2:** the explanatory classifier meets it.

Results are reported as they come out; anything tried afterwards is post-hoc.
