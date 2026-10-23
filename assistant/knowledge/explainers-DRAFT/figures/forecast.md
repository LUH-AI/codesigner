---
id: figure-forecast
title: Forecast
kind: figure
about: [figure:forecast, forecast-series-select, forecast-compute-btn]
related: [figure-selected-configuration, concept-metrics, page-dashboard-sidebar]
keywords: [forecast, backtest, time series, predicted vs actual, series, horizon]
status: draft
sources:
  - ui/figures/catalog.py:605-620 (Forecast, forecast_only, deferred)
  - ui/figures/base.py:100-120 (forecast_only, applies)
  - ui/templates/ui/figures/forecast.html
  - ui/figures/plots.py:797-841 (forecast_plot)
  - ui/views.py:803-852 (forecast_figure), 3374-3389 (_evaluation_label)
  - ui/templates/ui/experiment_detail.html:1553-1613 (refreshForecast), 1198-1205 (noTrialSelected)
  - ui/services/settings.py (autocompute_forecast off by default)
  - config/figure_layout.toml (overview, misc)
---

## What it shows

The time series, and what the selected trial's configuration forecast at every backtest. It shows how close that configuration's forecasts came to what actually happened, on the same backtests the trial was scored on.

## How to read it

- **Blue line**: the actual series, in time order.
- **Green dotted lines with points** ("Backtest 1", "Backtest 2", …): one per backtest, each a horizon's worth of steps after its cut-off. The points are what was scored.
- **Y axis**: the target column.
- For a classification forecast the y axis lists the labels, the actual series steps between them, and each forecast is drawn as points only, since a line between two labels would claim a value in between.

Drawing it runs the model again, once per backtest, exactly as the trial did.

## Controls

- **Series** (`forecast-series-select`): appears only when the data holds several series; one is drawn at a time.
- **Compute** (`forecast-compute-btn`): shown when the forecast waits to be asked.
- The figure follows the selected trial: click another trial in any figure or the Trials table to forecast with its configuration.

## When it appears

- Only on an experiment evaluated by backtests (a forecast). On any other experiment it is left off entirely.
- **Misc** tab, also pinned to **Overview**.
- Deferred: "Forecast" under "Analytics computed as soon as a figure needs them" is off by default, so a **Compute** button appears and nothing runs until pressed. Each new selection asks again, unless already computed.
- While a run is adding trials it does not compute on its own.
- Warnings replace the plot when the model is an uploaded one ("drawn for built-in models only"), when the configuration fails to forecast, or when a proposal rather than a trial is selected.
- Switched off with "Forecast" in the experiment's settings.

## Questions people ask

**Why do I have to press Compute?** Drawing it re-runs the model on every backtest, so it waits to be asked unless its autocompute setting is on.

**Which configuration is this?** The one in Selected Configuration — the best trial until you click another.

**Why is there no forecast for my uploaded model?** The page does not start an uploaded model's own environment, so it draws forecasts for built-in models only.

**Why don't I see this figure?** The experiment is not a forecast, or the figure is switched off in settings.
