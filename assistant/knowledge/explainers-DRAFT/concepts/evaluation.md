---
id: concept-evaluation
title: How trials are evaluated
kind: concept
about: [field:evaluation_scheme, field:evaluation_value, field:time_column, field:time_gap, field:horizon, field:series_column, field:season, field:seed]
related: [concept-metrics, concept-tasks-and-models, page-setup-step-1, figure-forecast]
keywords: [validation method, cross-validation, folds, dataset split, holdout, held out, backtest, rolling origin, time order, order by, gap, horizon, series column, season, forecast, reduction, lags, seasonal naive, ETS, stratified]
status: draft
sources: [ui/forms.py:25-52, ui/forms.py:118-170, ui/forms.py:201-231, ui/forms.py:307-320, ui/forms.py:322-365, core/splits.py:1-260, core/io.py:296-316, core/io.py:356-386, core/io.py:859-893, core/forecasting.py:1-71, core/forecasting.py:142-149, core/forecasters.py:73-157, core/forecasters.py:158-200, core/models/seasonal_naive.py, core/models/ets.py]
---

## In short

A trial fits the model on some rows and scores it on rows it has not seen. The validation method decides how the rows are divided: k-fold cross-validation, one held-out share, or backtests for a forecast. It is chosen in step 1 and fixed for the experiment's life, because trials evaluated in different ways cannot be compared.

## In detail

### The three methods

- **Cross-validation** (`kfold`, "Folds", 2–20, default 5). Each of k folds is validated once by a model fitted on the others, and the trial's score is the average. One model fit per fold. Classifications are stratified by class when possible. Without an order column, folds are shuffled using the experiment's seed.
- **Dataset split** (`holdout`, "Share held out for validation", 0.05–0.5, default 0.2). One division, one fit per trial. Cheapest, but noisy on a small table: the search can end up chasing the split.
- **Backtests** (`backtest`, 1–20, default 3). These make the experiment a forecast (below).

A number outside the range is clamped to it.

### Time order

"Order by" (`time_column`) names a date or number column. Rows are then sorted oldest first, and every trial trains on earlier rows and validates on later ones. Nothing is shuffled or stratified, and the seed plays no part. Rows with no value in the column are left out.

- With a dataset split, the latest share of rows is validated.
- With cross-validation, rolling origin [Tas00] is used: successive blocks of the latest rows, each validated after training on everything before it. The earliest block is never validated.

"Gap (rows)" (`time_gap`) leaves rows out between training and validation, for labels that arrive late.

### Backtests and forecasting

Forecasting is a way of evaluating, not a task: the next labels or the next numbers. Backtests need the order column and:

- "Horizon (steps)" (`horizon`, default 12): how many time steps each forecast predicts.
- "Series column" (`series_column`): which series a row belongs to, when the file holds several. Each series is forecast from its own history.
- "Season length (steps)" (`season`): empty means inferred from the spacing of the times: 24 hourly, 7 daily, 52 weekly, 12 monthly, 4 quarterly, otherwise 1.

The latest origin is one horizon before the end of the data, and each earlier one a horizon before that. A backtest trains on everything before its origin, less the gap (counted in time steps), and forecasts the next horizon steps. This needs at least backtests × horizon + gap + 1 distinct times [Tas00, HA21].

**By reduction.** Every built-in model can forecast. Each training row becomes one example per step ahead, with these features:

- the step;
- the series' last `forecast:lags` values;
- a summary of the last season;
- the columns known ahead of time (the time's parts and the series).

Other columns are never used, because their future values are unknown. `forecast:lags` joins the search space. An uploaded model must declare `forecaster = True`.

**Baselines.** "Seasonal Naive" repeats last season, repeats the last value, or draws a straight line (`strategy`). "ETS" is Holt–Winters exponential smoothing per series (`trend`, `damped`, `seasonal`). MASE scores forecasts against the seasonal naive forecast [HK06].

## In Codesigner

- Setup step 1: "Validation method" (`evaluation_scheme`) and its number (`evaluation_value`), whose label follows the method. Then "Order by", "Gap (rows)", "Seed", and under backtests "Horizon (steps)", "Series column" and "Season length (steps)".
- The form refuses an order the data cannot support, and lists the columns it can be ordered by.
- A forecast's dashboard has the "Forecast" figure, which shows what the selected trial forecast at every backtest.

## Questions people ask

**Which method should I use?** 5-fold cross-validation for small tables, a held-out share for large ones, backtests for a time series to forecast.

**Why can't I change it later?** Trials scored differently are not comparable.

**Why does a 5-fold trial take five times as long?** One fit per fold.

**Why are some rows never validated in time order?** Nothing comes before the earliest rows to learn them from.
