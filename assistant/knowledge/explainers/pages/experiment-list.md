---
id: page-experiment-list
title: All experiments
kind: page
about: [page:ui:experiment_list]
related: [page-home, page-dashboard, page-new-experiment, concept-permissions]
keywords: [all experiments, see all, experiment list, every experiment, older experiments, pages, previous, next]
status: draft
sources:
  - ui/templates/ui/experiment_list.html
  - ui/views/pages.py:19-34
  - ui/context_processors.py:5-8
  - access/policy.py:130-134
  - access/policy.py:49-53
  - ui/permissions.py:64-80
---

## What it is for

The full list of experiments you can see. The sidebar's Experiment Selection shows only the 20 most recent; this page, reached with **See all…** at the bottom of the sidebar, lists all of them, 50 to a page.

## What is on it

- Heading **All experiments**.
- A table with four columns:
  - **Name**: the experiment's name ("Untitled Experiment" when it has none), linked to its page, with its ID in brackets. A spinner (tooltip "Running") marks one with a run under way.
  - **Model**: the model it tunes.
  - **Optimizer**: the optimizer it searches with.
  - **Created**: when it was created.
- When there is more than one page: **← Previous**, "Page N of M", **Next →**.
- "No experiments yet." when there are none.

## How to use it

1. Click **See all…** in the sidebar.
2. Find the experiment by name, ID, model or optimizer, using the page links if there are several pages.
3. Click its name to open it.

## Rules and limits

- The list is the same selection the sidebar uses, so it never shows an experiment that would then refuse to open. With accounts, that is your own experiments and the ones shared with you. A group lead's colleagues' experiments are not listed here; they are on the Group page (👥).
- Deleted experiments are not listed; they are in the bin.
- A draft listed here opens its setup rather than a dashboard. With accounts, only your own drafts can appear.
- The list cannot be sorted or filtered on this page.

## Questions people ask

**How do I find an experiment that is not in the sidebar?** Open **See all…**; the sidebar only shows the 20 most recent.

**Why can I not see a colleague's experiment?** It has not been shared with you. Its owner shares it from their Group page. If you lead the group, it is under "Everybody else's" on the Group page.

**What is the code in brackets?** The experiment's ID, which tells experiments with the same name apart.
