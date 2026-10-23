---
id: page-timeline
title: Timeline
kind: page
about: [page:ui:experiment_timeline]
related: [page-dashboard, page-data-handling, page-setup-step-4, page-export-import, concept-priors, concept-stopping, concept-metrics]
keywords: [timeline, history, what changed, runs, set up, data handling, prior stated, prior applied, prior refused, optimizer changed, metric changed, scheduled, now, details, stopped because]
status: draft
sources: [ui/templates/ui/experiment_timeline.html:1-60, ui/templates/ui/_timeline_details.html:1-10, ui/views.py:2392-2403, ui/timeline.py:1-11, ui/timeline.py:118-131, ui/timeline.py:216-370, ui/timeline.py:371-467, ui/timeline.py:470-586, ui/services/history.py:1-45, ui/services/timestamps.py:1-27, templates/base.html:163-175]
---

## What it is for

The record of everything that changed how the experiment searches, newest first, down to how it was set up. It answers "what was done to this experiment, when, and by whom". Opened with **Timeline** under the experiment's row in the sidebar.

## What is on it

- The experiment's header, then the heading **Timeline** and a vertical list of entries.
- Each entry has, on the left, when it happened and "after N trials". On the right are a one-line summary, the person who did it ("— *name*"), and a **Details** fold.
- Kinds of entry:
  - **Set up**: "Set up: *model* on *dataset*, tuned by *optimizer*." Details: Task, Model, Dataset, Evaluation, Ordered by, Optimizer, Optimizer settings, Seed. Always at the bottom.
  - **Data handling**: just above it, the three processing choices it was set up with, for example "Data handling: as Random Forest does by default."
  - **Imported**: from a `.ihpo` file or a SMAC run, and whether its dataset was attached.
  - Changes between runs: a Data Handling choice ("Scaling: … → …"), a prior stated, changed, withdrawn or reset on a hyperparameter, a prior drawn in a view (HP Space, PCA or PLS) stated, changed or withdrawn, the optimizer or its settings changed, the model's environment rebuilt.
  - **Runs**: "Run N ran X trials (a–b), optimizing *metric*. Stopped because …", or "waiting to start", "under way", "Then it failed: …", "Cancelled by *name*". A run of one proposal says so. Details: Status, Queued, Started, Finished, Trial range, Optimized metric, Stopping (the criteria it was given), Stopped by, Time in trials, Priors, Priors drawn in a view, Proposal, Trial timeout. Inside a run, its own events: the optimized metric changing, a prior applied after trial N, a prior not applied or refused, and trial models not kept for lack of room.
- **Scheduled** entries sit above a **Now** stop on a dashed line, with no time: a prior that takes effect "after the initial design" and has not yet.

## How to use it

1. Read from the top for the latest change, or from the bottom for how the experiment began.
2. Open **Details** on an entry for its full values, such as a run's stopping criteria or a prior's distribution and decay.
3. Use a run's "Stopped because …" to see which criterion ended it.

## Rules and limits

- Only what changes the search is recorded. Changing which figures are shown, the layout or the page settings is not.
- A prior counts once a run applied it, or while it is the one standing. A prior that was replaced or withdrawn before any run used it is left out, because it never affected the search.
- Times are shown as the experiment has them: by the clock, as time since its first run, or not at all. An experiment imported from a file without exact times cannot show more than the file had.
- Anyone who can open the experiment can read its timeline. Nothing can be edited here.
- The history travels in the exported `.ihpo`, so an experiment opened elsewhere keeps its timeline.
- Drafts have no timeline; it begins when the experiment is created.

## Questions people ask

**Why is a prior I set not listed?** It was replaced or withdrawn before any run applied it.

**What does "Scheduled — after the initial design" mean?** A prior set to wait until the initial design ends. It has not taken effect yet.

**Who changed the optimizer?** The name after the summary is the person who did it.

**Why are there no dates?** The experiment was imported with relative times or none.
