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

Each kind has a template in `_templates/`: `figure.md`, `page.md` and
`concept.md`. A file has its template's frontmatter fields in the same order,
an `id` made of its kind and file name (`figures/trial_duration.md` is
`figure-trial-duration`), and its template's `##` sections in the same order,
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
- Then set `status: reviewed` and move the file to `explainers/`.
