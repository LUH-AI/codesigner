---
id: page-setup-step-3
title: Data (step 3)
kind: page
about: [page:ui:setup_data, field:missing, field:scale, field:labels, data-assist:save]
related: [page-setup-step-1, page-setup-step-4, page-data-handling, page-timeline]
keywords: [data step, data handling, processing, missing values, fill in, impute, scaling, standardize, label columns, one-hot, categories, model default, planned]
status: draft
sources: [ui/setup_views.py:72-93, ui/templates/ui/setup_step_data.html:1-12, datahandling/templates/datahandling/_tabs.html:1-57, datahandling/templates/datahandling/panels/processing_step.html:1-16, datahandling/templates/datahandling/panels/planned.html:1-9, datahandling/panels.py:27-89, datahandling/sections.py:91-134, datahandling/views.py:33-55, core/processing.py:1-62, config/figure_layout.toml:142-149, ui/services/drafts.py:143-168, ui/services/drafts.py:353-354, core/models/svm.py:24-25, core/models/knn.py:19-20, core/models/logistic_regression.py:15-16, core/models/elastic_net.py:17-18]
---

## What it is for

The third setup step decides how the dataset's columns are processed before the model sees them. It shows the Data Handling page as the experiment will have it, with an **Overview** of its own that gathers the three choices that can be made now.

## What is on it

- The tab bar of Data Handling: **Overview**, **Columns**, **Quality**, **Cleaning**, **Features**, **Augmentation**, **Splits**, **Sources**, **Versions**.
- On **Overview**, three cards, each a select:
  - **Missing values** (`missing`): **Fill in** (a number with its column's median, a label with its commonest label) or **Leave for the model**.
  - **Scaling** (`scale`): **Standardize** (zero mean, unit variance) or **Leave as they are**.
  - **Label columns** (`labels`): **One column per label**, or **As categories** (codes for a booster, words for CatBoost).
  
  The option that is the model's own default is named so, for example "Leave as they are (Random Forest default)", and is selected until you choose another. A choice the model cannot take is offered greyed out; for example, a model that cannot take gaps always has them filled.
- The other tabs hold cards marked **Planned**: capabilities that are not built yet, each with a line saying what it will do. The three choices above also appear on their own tabs (Missing values under Cleaning, Scaling and Label columns under Features).
- **Save and continue** (`data-assist="save"`) under the controls.
- With an uploaded model, the note "An uploaded model is given the columns as they are; these choices apply to the built-in models." Its selects offer "As the model needs" instead of a named default.

## How to use it

1. Leave each choice on the model's default unless you have a reason to change it.
2. To change one, pick another option in its card.
3. Press **Save and continue** to go to step 4, Priors & constraints. Saving this step is required even if nothing was changed.

## Rules and limits

- Step 1 must be saved first.
- Saving step 1 again with a different dataset, target, task or model clears this step's "Saved" mark; save it again before creating.
- Processing is fitted on each fold's training rows only, so validation rows never inform how they are scaled or filled.
- Once the created experiment has trials, these choices are fixed; before that they can be changed on its Data Handling page. The choices made here are listed on the timeline as the "Data handling" entry under the experiment's set-up.
- Planned cards have no controls; nothing on them can be set yet.

## Questions people ask

**Should I standardize?** SVM, k-Nearest Neighbors, Logistic Regression and Elastic Net are standardized by default; the tree models and boosters are not. The "(… default)" label tells you what the chosen model gets.

**Why is "Leave for the model" greyed out?** The chosen model cannot take missing values (SVM, k-Nearest Neighbors, Logistic Regression, Elastic Net), so its gaps are always filled.

**What are the Planned cards?** Capabilities that are not built yet. They show what Data Handling is meant to offer but cannot be used.
