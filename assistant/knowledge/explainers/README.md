# Explainers

Each file explains one figure, page or concept, for the assistant to answer
from and to point at, or weighs one choice: when moving a setting from its
default, the metric or the validation method is worth it, and what it costs.
The files were drafted from Codesigner's own texts: figure descriptions, ⓘ help
texts, form help, template comments, and the user docs deleted in commit
`e933531`; the trade-offs also from the papers in the library. Each file's
`sources` lists where its facts came from, so they can be checked.

`assistant/explainers.py` reads `tradeoffs/` and `pages/`:
`list_optimizers`, `list_metrics` and `read_experiment settings` hand the
trade-off notes to the model (`assistant/tradeoffs.py`), and `about_page`
hands it a page's explainer. Nothing reads `figures/` or `concepts/` yet; the
library builder does not index this folder.

## Folders

| Folder | One file per |
|---|---|
| `figures/` | figure on the experiment dashboard (`ui/figures/catalog.py`) |
| `pages/` | page or panel: setup steps, the dashboard's sidebar, Data Handling, settings, timeline, … |
| `concepts/` | idea a person may ask about: metrics, evaluation, optimizers, priors, explanation games, … |
| `tradeoffs/` | choice with a default worth weighing: a SMAC setting or group of them, the optimized metric, the validation method |

## The format

Each kind has a template in `_templates/`: `figure.md`, `page.md`,
`concept.md` and `tradeoff.md`. A file has its template's frontmatter fields in
the same order, an `id` made of its kind and file name
(`figures/trial_duration.md` is `figure-trial-duration`), and its template's `##` sections in the same order,
with no others. The comments in each template say what goes in each section.

- `about` lists what the file explains, each as `kind:name`: `figure:` a key of
  `ui/figures/catalog.py`, `page:` a URL name, `field:` a form field,
  `element:` an element's id, `data-assist:` a page target, `action:` a policy
  action.
- `related` lists other explainers' ids.
- `sources` lists, one per line, the files the facts came from, with line
  ranges where they help, and papers by their library key.
- "Questions people ask" holds only bold questions, each answered on the same
  line.

`tests/assistant/test_explainers.py` checks every file against its template,
and that every target, related id and cited file exists.

A note that needs a decision or a check is an HTML comment:
`<!-- REVIEW: … -->`.

## Checklist for the reviewer

- Is it right? Every statement should match what Codesigner does today.
- Is it what a person would want to know, in words they would use?
- Are the `<!-- REVIEW -->` notes resolved or deleted?
- Then set `status: reviewed`.
