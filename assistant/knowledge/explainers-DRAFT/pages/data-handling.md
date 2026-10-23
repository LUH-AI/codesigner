---
id: page-data-handling
title: Data Handling
kind: page
about: [page:datahandling:data_overview, page:datahandling:data_section, field:missing, field:scale, field:labels, data-assist:save]
related: [page-setup-step-3, page-dashboard, page-timeline, concept-evaluation, concept-permissions]
keywords: [data handling, processing, missing values, scaling, label columns, overview, columns, quality, cleaning, features, augmentation, splits, sources, versions, planned, not built yet]
status: draft
sources: [datahandling/urls.py:1-18, datahandling/views.py:16-120, datahandling/sections.py:1-185, datahandling/panels.py:1-150, datahandling/templates/datahandling/section.html:1-10, datahandling/templates/datahandling/_tabs.html:1-57, datahandling/templates/datahandling/panels/processing_step.html:1-16, datahandling/templates/datahandling/panels/planned.html:1-9, config/figure_layout.toml:57-140, core/processing.py:1-62, templates/base.html:163-175, ui/timeline.py:291-299]
---

## What it is for

The experiment's view of its dataset and of how its columns are processed before the model sees them. It is opened from **Data Handling** under the experiment's row in the sidebar, beside Dashboard and Timeline, and carries the same header as the dashboard. Most of it is a plan of what Data Handling will do; three processing choices are built and can be set.

## What is on it

Nine tabs, each its own address:

| Tab | Built | Planned |
|---|---|---|
| **Overview** | — | Profile, Column kinds, Where values are missing, Target distribution, Sample |
| **Columns** | — | Target, Kinds, Rename and drop, Roles, Known in the future |
| **Quality** | — | Missing-value patterns, Duplicates, Constant columns, Outliers, Class imbalance, Label noise, Target leakage, Train and test overlap |
| **Cleaning** | **Missing values** (`missing`) | Imputation, Drop rows or columns, De-duplication, Outlier handling, Type coercion, Value mapping |
| **Features** | **Label columns** (`labels`), **Scaling** (`scale`) | Encodings, Other scaling, Transforms, Calendar features, Interactions, Lags and windows, Feature selection |
| **Augmentation** | — | Resampling, SMOTE, Noise injection, Mixup, Synthetic data, Time-series augmentation |
| **Splits** | — | Scheme, Final test set, Fold preview |
| **Sources** | — | Replace, Append, Join, Subsample |
| **Versions** | — | History, Recipe, Runs per version, Differences, Export |

- A **built** card is a select with a line explaining it:
  - **Missing values**: Fill in / Leave for the model.
  - **Scaling**: Standardize / Leave as they are.
  - **Label columns**: One column per label / As categories.
  
  The model's own default is labelled, for example "Standardize (SVM default)". An option the model cannot take is greyed out.
- A **planned** card has a **Planned** badge and one line saying what it will do. It has no controls.
- On Cleaning and Features, a **Save** button (`data-assist="save"`) under the cards, or a line saying why they cannot be changed.
- With an uploaded model: "An uploaded model is given the columns as they are; these choices apply to the built-in models."

## How to use it

1. Click **Data Handling** under the open experiment in the sidebar.
2. Open **Cleaning** or **Features**, change a select, press **Save**. "Saved." confirms it.
3. Read the planned cards to see what the page is meant to offer.

## Rules and limits

- The processing choices can be changed only before the experiment's first trial and not during a run. After that the selects are disabled with the reason: "This experiment has trials, and they were scored on its columns processed as they are now; changing that would make the next ones incomparable."
- Saving needs permission to change the experiment (owner, contributor or group lead). Anyone else sees "Only people who may change this experiment can set these."
- Each change is recorded on the timeline, for example "Scaling: leave as they are → standardize."
- The cards cannot be moved, pinned or resized here. Their places are fixed for everyone.
- During setup, the same page is step 3, with an Overview that gathers the three built choices.

## Questions people ask

**Why can I not change anything?** The experiment already has trials, a run is under way, or you may not change this experiment.

**Where do I choose the target column?** In setup step 1 (Target). The Columns tab's Target card is planned.

**Are Quality, Augmentation and the other tabs working?** Not yet. Their cards are marked Planned.
