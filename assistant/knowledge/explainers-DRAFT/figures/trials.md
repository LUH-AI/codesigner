---
id: figure-trials
title: Trials
kind: figure
about: [figure:trials, element:trials-page, element:trials-page-size]
related: [figure-selected-configuration, figure-best-configuration, figure-performance-over-time, figure-trial-duration, page-dashboard-sidebar]
keywords: [trials table, all trials, sort, page, rows per page, failed trials, scores, durations, configurations]
status: draft
sources:
  - ui/figures/catalog.py:372-395 (Trials, page_size)
  - ui/templates/ui/figures/trials.html, _trial_rows.html
  - ui/views/dashboard.py:148-169 (_trial_rows)
  - ui/views/runs.py:69-107 (_live_payloads rows_html)
  - ui/templates/ui/experiment_detail.html:77-131 (sorting), 612-700 (pager, table selection), 501-515 (row clicks), 3266-3294 (appendTrialRows)
  - ui/figure_layout.toml (overview, misc)
---

## What it shows

Every trial as one row of a table: its outcome first — a score for every metric and its duration — then the hyperparameter values that produced it. The heading counts the trials, "Trials (N)".

## How to read it

- **#**: the trial number. A **✕** after it, and a red-tinted row, mark a trial that failed; hover the row for the reason.
- **One column per metric**: the trial's score. A failed trial shows its stored placeholder (0) rather than a blank, so the columns still sort; the tint is what says it was never measured.
- **Duration**: seconds the trial took.
- **Hyperparameter columns**: the configuration.
- The highlighted row is the selected trial — the same one every figure and the Selected Configuration panel show.

Unlike the figures, the table shows every metric side by side, whatever the metric selector says.

## Controls

- **Sorting**: click a column header among #, the metric columns and Duration. Scores and durations sort largest first on the first click, # in run order; click again to reverse. Hyperparameter columns do not sort.
- **Selecting**: click a row, or focus it and press Enter or Space, to select that trial on the whole page.
- **Pager**: shown only when there are more rows than fit on one page. Previous/next buttons, a **Page** select (`trials-page`) and **Rows per page** (`trials-page-size`, 50 by default). Selecting a trial elsewhere turns to the page it is on.

## When it appears

- On **Overview** and **Misc** by default.
- Needs no computing. During a run, new trials are appended as they arrive, keeping your sort, page and selection.
- Only shown once the experiment has trials.
- Switched off with "Trials" in the experiment's settings.

## Questions people ask

**How do I find the best trial?** Click the metric's column header; it sorts largest first, then click again for smallest first if lower is better.

**Why does a row show 0 for every metric?** The trial failed; the row is tinted and marked ✕. Hover it for the reason, or select it to see the traceback link.

**Why can't I see all trials?** The table is paged; use the pager or raise Rows per page.

**Can I sort by a hyperparameter?** No, only the outcome columns sort.
