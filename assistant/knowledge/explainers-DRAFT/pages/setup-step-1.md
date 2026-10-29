---
id: page-setup-step-1
title: Setup (step 1)
kind: page
about: [page:ui:setup, field:name, field:group, field:task, field:use_demo_model, field:model_name, field:model_file, field:mounted_model, field:use_demo_dataset, field:demo_dataset, field:dataset_file, field:target_column, field:evaluation_scheme, field:evaluation_value, field:time_column, field:time_gap, field:horizon, field:series_column, field:season, field:seed, data-assist:save]
related: [page-new-experiment, page-setup-step-2, page-setup-step-3, concept-evaluation, concept-metrics, concept-permissions]
keywords: [setup, experiment name, task, classification, regression, model, upload model, custom model, dataset, csv, demo dataset, target, validation method, cross-validation, folds, dataset split, held out, backtests, forecast, order by, gap, horizon, series column, season length, seed]
status: draft
sources:
  - ui/templates/ui/setup_step_setup.html
  - ui/forms.py:22-52
  - ui/forms.py:90-367
  - ui/forms.py:77-87
  - ui/views/setup.py:41-57
  - ui/views/setup.py:128-144
  - ui/drafts.py:254-327
  - ui/setup_steps.py
  - core/splits.py:21
  - core/datasets.py:17
  - core/registry.py:17-30
  - access/policy.py:61-65
---

## What it is for

The first setup step says what the experiment is: its data, what it predicts, which model is tuned, and how each trial is scored. Nothing else can be opened until it is saved.

## What is on it

- **Experiment name** (`name`): optional; empty shows as "Untitled Experiment".
- **Group** (`group`): only for someone in more than one group; required then.
- **Task** (`task`): Classification or Regression. Only models that can do the task are offered, so choose it first.
- **Model** (`model_name`): a built-in model. If you may upload models, a **Use demo models** box (`use_demo_model`) switches between the built-in list and **…or upload a model .py** (`model_file`), plus **…or a mounted model .py** (`mounted_model`) where the instance has some. An uploaded model already on the draft shows "Currently *file*; kept unless another is chosen."
- **Dataset**: a CSV upload (`dataset_file`, "Upload", × removes it), or, with **Use demo dataset** ticked (`use_demo_dataset`), a demo (`demo_dataset`).
- **Target** (`target_column`): the column to predict; empty means the last column. Column names are suggested from the dataset.
- **Validation method** (`evaluation_scheme`) and the number beside it (`evaluation_value`), whose label follows the method:
  - **Cross-validation** (default): **Folds**, 2–20, default 5. One model fit per fold.
  - **Dataset split**: **Share held out for validation**, 0.05–0.5, default 0.2. Cheapest; noisier on small tables.
  - **Backtests**: **Backtests**, 1–20, default 3. Makes the experiment a forecast.
- **Order by** (`time_column`): a date or number column; rows are divided in time order, training on earlier rows and validating on later ones. Empty divides at random. Rows with no value in it are left out.
- **Gap (rows)** (`time_gap`): with an order only; rows left out between training and validation.
- Backtests only: **Horizon (steps)** (`horizon`, default 12), **Series column** (`series_column`, for a file holding several series), **Season length (steps)** (`season`, empty infers it).
- **Seed** (`seed`, default 0): drives every random part of the experiment; a negative seed picks one at random.
- **Save and continue** (`data-assist="save"`).

The ⓘ beside a label explains the field.

## How to use it

1. Name it, choose the task, then the model.
2. Upload a CSV or tick **Use demo dataset** and pick one. Set **Target** if the column to predict is not the last.
3. Choose the validation method and its number. For a forecast choose **Backtests** and fill in **Order by** and **Horizon**.
4. Press **Save and continue**. Errors appear under the fields.

## Rules and limits

- Task and validation method are fixed once the experiment is created: trials scored different ways cannot be compared.
- A model that only forecasts (Seasonal Naive, ETS) needs **Backtests**. An uploaded model can be backtested only if it declares `forecaster = True`.
- Regression needs a numeric target. Backtests need **Order by** and a horizon of at least one step.
- A number outside its range is clamped to the nearest bound.
- Saving again with a different dataset, target, task or model asks for step 3 (Data) to be saved again. A different model also withdraws the priors stated in step 4.
- Uploading a custom model is only offered to people allowed to (the instance setting, and with accounts the "Upload and run custom models" permission). It runs code, so only a person may upload one; the assistant never does. Its environment starts being prepared in the background once the step is saved.
- If the group has no room for the files, the draft is deleted and a message says why.

## Questions people ask

**Which validation method should I pick?** Cross-validation is the default. A dataset split is cheaper but noisier on a small table. Backtests are for forecasting a series.

**Why is a model greyed out?** It cannot do the chosen task, or it only forecasts and the method is not Backtests.

**Can I change the task later?** Not after creating the experiment. While it is a draft, save step 1 again.
