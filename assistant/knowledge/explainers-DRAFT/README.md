# DRAFT explainers: awaiting review

**Not reviewed. Not used by the assistant yet.** The library builder does not
read this folder. Once a file is reviewed, it moves to
`assistant/knowledge/explainers/`, where the assistant reads it with
`read_explainer` and finds it through `search_knowledge`.

Each file explains one figure, page or concept, for the assistant to answer
from and to point at. The files were drafted from Codesigner's own texts: figure
descriptions, ⓘ help texts, form help, template comments, and the user docs
deleted in commit `e933531`. Each file's `sources` lists where its facts came
from, so they can be checked.

## Folders

| Folder | One file per |
|---|---|
| `figures/` | figure on the experiment dashboard (`ui/figures/catalog.py`) |
| `pages/` | page or panel: setup steps, the dashboard's sidebar, Data Handling, settings, timeline, … |
| `concepts/` | idea a person may ask about: metrics, evaluation, optimizers, priors, explanation games, … |

## The format

```markdown
---
id: figure-performance-over-time        # unique; figure-/page-/concept- prefix
title: Trial performance                # as the page names it
kind: figure                            # figure | page | concept
about: [figure:performance_over_time]   # the page targets it explains (figure keys, data-assist names, field names)
related: [concept-metrics]              # other explainers' ids
keywords: [score over time, running best, incumbent]
status: draft
sources: [ui/figures/catalog.py, ui/templates/ui/figures/performance_over_time.html]
---
```

The body has short sections with headings. Which sections a file has depends on
its kind; typical ones are "What it shows", "How to read it", "Controls", "When
it appears", and "Questions people ask".

A note that needs a decision or a check is an HTML comment:
`<!-- REVIEW: … -->`.

## Checklist for the reviewer

- Is it right? Every statement should match what Codesigner does today.
- Is it what a person would want to know, in words they would use?
- Are the `<!-- REVIEW -->` notes resolved or deleted?
- Then set `status: reviewed` and move the file to `explainers/`.
