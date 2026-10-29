---
id: concept-tasks-and-models
title: Tasks and models
kind: concept
about: [field:task, field:target_column, field:model_name, field:model_file, field:mounted_model]
related: [concept-evaluation, concept-metrics, concept-optimizers, concept-permissions, page-setup-step-1, page-data-handling]
keywords: [task, classification, regression, forecasting, target, target column, last column, model, built-in models, random forest, SVM, extra trees, gradient boosting, LightGBM, XGBoost, CatBoost, k-nearest neighbors, logistic regression, elastic net, seasonal naive, ETS, custom model, upload, model file]
status: draft
sources:
  - core/tasks.py
  - ui/forms.py:109-124
  - ui/forms.py:194-206
  - ui/forms.py:259-313
  - core/datasets.py:42-53
  - core/registry.py:13-30
  - core/models/
  - core/forecasting.py
  - core/model_source.py
  - ui/setup_steps.py
  - access/policy.py:61-65
  - ui/forms.py:195-198
  - ui/templates/ui/setup_step_setup.html:33-110
  - ui/templates/ui/_experiment_header.html:33-39
---

## In short

The task says what the experiment predicts: one of a set of labels (classification) or a number (regression). It is chosen in step 1, decides which models and metrics are offered, and is fixed once the experiment exists. Forecasting is not a third task. It is a way of evaluating either task, chosen by picking backtests as the validation method.

## In detail

### How the task is decided

The task is asked, never guessed: whole numbers can be classes to one person and a quantity to another. Saving step 1 (`_resolve_task`) checks that:

- a regression target must be all numbers ("…is not numbers, so it cannot be regressed on");
- the model must support the task;
- a model that only forecasts needs backtests as the validation method;
- an uploaded model can be backtested only if it declares `forecaster = True`.

### The target column

"Target" (`target_column`) names the column to predict. Left empty, the dataset's last column is the target. Rows whose target is missing are left out: there is nothing to train on and nothing to score against. A different target counts as a different dataset to everything that reads it.

### Built-in models

All of them run on the CPU, inside Codesigner itself. Each comes with its own search space.

| Model | Tasks | What it is |
|---|---|---|
| Random Forest | both | scikit-learn's random forest |
| SVM | both | `SVC`/`SVR`; tunes C, kernel, gamma and more |
| Extra Trees | both | randomized-threshold forest; adds class weighting for classification |
| HistGradientBoosting | both | scikit-learn's histogram boosting, early stopping off |
| LightGBM | both | leaf-wise boosted trees |
| XGBoost | both | boosted trees, histogram method |
| CatBoost | both | ordered boosting on symmetric trees |
| k-Nearest Neighbors | both | majority or mean of the closest rows; tunes k, weighting, distance |
| Logistic Regression | classification | L2-penalized linear model; tunes C and class weighting |
| Elastic Net | regression | linear model with L1 + L2 penalty; tunes alpha and the blend |
| Seasonal Naive | regression, forecast only | last season, last value or drift |
| ETS | regression, forecast only | Holt–Winters exponential smoothing |

Under backtests, every model except the two forecasters forecasts by reduction and gains a `forecast:lags` hyperparameter (see concept-evaluation). The models that measure distances or refuse missing values (SVM, k-NN, Logistic Regression, Elastic Net) get their columns filled in and scaled first. How columns are processed is set on the Data step.

### Custom models

A model can also be a `.py` file defining a `BaseModel` subclass (`from codesigner_model import BaseModel`), with its dependencies in a PEP 723 header. The file is read without being run. It then runs in its own environment, which is prepared in the background and can take minutes. The class states its `name`, its `tasks` (classification when it says nothing), and whether it is a `forecaster`.

Uploading is arbitrary code execution, so it is offered only where the instance allows custom models (`ALLOW_CUSTOM_MODELS`). With accounts, it also needs the "use custom models" permission. Only the person may upload. The assistant never can, even for someone who may.

### Changing the choice

Saving step 1 again with a different dataset, task or model asks for the Data step to be saved again. A different model also withdraws the stated priors, because they described the old model's search space.

## In Codesigner

- Setup step 1 ("Setup"): "Task" (`task`), then the model: "Model" (`model_name`), or, where uploads are allowed, "…or upload a model .py" (`model_file`) and "…or a mounted model .py" (`mounted_model`). A "Use demo models" checkbox switches to the built-ins. Then the dataset, and "Target" (`target_column`, placeholder "Last column").
- Models that cannot do the chosen task, evaluated the chosen way, are greyed out in the list.
- The dashboard header line reads "Model: … · Optimizer: … · Metric: … · Evaluation: … · Seed: …". The task is not shown there.

## Questions people ask

**My target is 0/1. Classification or regression?** Classification if each value is a class. Regression only if the number is a quantity whose size matters.

**Why is a model greyed out?** It does not do the task you chose, or it only forecasts and the validation method is not backtests.

**Which column is predicted if I leave Target empty?** The last column of the file.

**Can I forecast with a random forest?** Yes. Choose backtests, and it forecasts by reduction.

**Why can't the assistant upload my model?** Uploading code is reserved for the person.
