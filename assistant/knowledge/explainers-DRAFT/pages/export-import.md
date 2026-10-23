---
id: page-export-import
title: Export, load and the bin
kind: page
about: [page:ui:experiment_export, page:ui:experiment_export_model, page:ui:experiment_export_trial_model, page:ui:import_experiment, page:ui:experiment_bin, page:ui:experiment_delete, field:timestamps, field:tracebacks, field:file, field:dataset, field:model, field:smac_dir, field:group, element:export-model, element:export-trial-parameters]
related: [page-dashboard, page-dashboard-sidebar, page-experiment-settings, page-account-and-groups, page-timeline, concept-permissions]
keywords: [export, ihpo, download, timestamps, real times, relative times, no times, stacktrace, traceback, export model, python script, trial parameters, fitted model, zip, load, import, SMAC run, runhistory, read-only, bin, restore, delete, remove from my bin, delete for good]
status: draft
sources: [ui/templates/ui/_experiment_header.html:43-75, ui/templates/ui/export_confirm.html:1-52, ui/views.py:2364-2389, ui/views.py:2406-2482, ui/views.py:2495-2559, ui/services/timestamps.py:1-111, ui/services/model_export.py:1-58, ui/templates/ui/import.html:1-55, ui/templates/ui/_group_picker.html:1-12, ui/views.py:2827-2948, core/provenance.py:228-251, ui/templates/ui/bin.html:1-45, ui/templates/ui/bin_dismiss_confirm.html:1-24, ui/templates/ui/delete_confirm.html:1-19, ui/views.py:2735-2825, ui/services/bin.py:1-80, config/settings.py:47, ui/templates/ui/experiment_detail.html:720-757]
---

## What it is for

Taking an experiment, its tuned model or a trial's fitted model off the instance; bringing an experiment in from a `.ihpo` file or a SMAC output folder; and getting a deleted experiment back from the bin.

## What is on it

**Export** (header buttons on an experiment's page):
- **💾 Export** opens "Export *name*": "Some of what the file records is about you rather than about the search." Choose:
  - **Time** (`timestamps`): **Real Times** (UTC times), **Relative Times** (times relative to the start of the first run) or **No Times** (events in order only). Preselected from the experiment's setting, else your preference (Settings → Export). Never more precise than the experiment's own times.
  - A checkbox for the stacktraces of failed trials (`tracebacks`, off), shown only when there are any. Without them the failed trials stay in the file with their one-line reason.
  - **Export** downloads `name.ihpo` (or `ID.ihpo` when unnamed). Paths on the server are always removed.
- **📦 Export model**: a runnable `.py` script with the model set to the selected trial's hyperparameters and the experiment's data processing, named like `random-forest-trial-15.py`.
- **🧮 Export trial parameters**: a `.zip` of the selected trial's fitted model in its library's own format. It is partially trained, fitted on one fold's training rows.

**Load** (📂 Load in the sidebar, "Load an experiment"):
- **Experiment file (.ihpo)** (`file`), **Dataset CSV (optional)** (`dataset`), **Model .py (optional, custom models only)** (`model`, only if you may upload models), **Group** when you are in more than one, then **Load**.
- **…or a SMAC run**: **SMAC output directory** (`smac_dir`, the folder holding `runhistory.json`), **Group**, **Load run**.

**Bin** (🗑 in Experiment Selection): a table of Experiment, Owner, Deleted (and by whom), **Leaves your bin** (with accounts), and per row **Restore**, **Download .ihpo**, and **Remove from my bin** (**Delete for good** without accounts).

## How to use it

1. To share results as a file: **💾 Export**, choose Time and stacktraces, **Export**.
2. To use the tuned model elsewhere: select a trial in any figure, then **📦 Export model**.
3. To load: **📂 Load**, choose the `.ihpo`, and attach its dataset if you want to run it further.
4. To undo a delete: open 🗑, press **Restore**. It comes back exactly as it was.

## Rules and limits

- Anyone who can open an experiment may export it.
- **Export model** is unavailable when the experiment was imported without its model, its model file is missing, or a forecast's dataset is not on this instance.
- **Export trial parameters** works only for built-in models and only for trials whose model a run kept (Settings: Trial models kept for export). Otherwise the button is greyed out and its tooltip says why. A proposal has no model to export.
- A loaded `.ihpo` without its dataset is read-only. An attached dataset must be the one the trials were run on, or loading is refused. A SMAC run always loads read-only; no dataset or model can be attached, and only one run folder at a time.
- **🗑 Delete** (owner and contributors) stops any run and puts the experiment in the bins of its owner and contributors. With accounts it stays there for the retention period (30 days by default), then moves to the site administrators' bin.
- **Remove from my bin** takes it out of your bin only and ends your access. If nobody else has it, it goes to the site administrators' bin. Without accounts, **Delete for good** destroys it.
- Deleting, restoring, removing and uploading files are done by the person. The assistant never chooses files or presses these buttons.

## Questions people ask

**Can I get a deleted experiment back?** Yes, with **Restore** in your bin until it leaves it.

**Why is my loaded experiment read-only?** No dataset was attached, or it came from a SMAC folder.

**What is the difference between Export model and Export trial parameters?** The first is a script that trains the model with that trial's settings; the second is that trial's already-fitted model.
