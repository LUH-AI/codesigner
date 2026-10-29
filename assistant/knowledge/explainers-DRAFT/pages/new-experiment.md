---
id: page-new-experiment
title: New experiment
kind: page
about: [page:ui:new_experiment, page:ui:setup_resume, data-assist:discard, data-assist:steps]
related: [page-home, page-setup-step-1, page-setup-step-2, page-setup-step-3, page-setup-step-4, concept-permissions]
keywords: [new experiment, draft, start, resume, continue setup, discard draft, throw away, setup steps, Draft badge]
status: draft
sources:
  - ui/templates/ui/base.html:139-145
  - ui/templates/ui/base.html:158-162
  - ui/templates/ui/new_experiment.html
  - ui/views/pages.py:66-72
  - ui/drafts.py:50-54
  - ui/setup_steps.py:47-52
  - ui/views/setup.py:35-38
  - ui/views/setup.py:116-126
  - ui/templates/ui/_setup_heading.html
  - ui/templates/ui/_setup_steps.html
  - access/policy.py:34-38
  - ui/drafts.py:57-82 (settle_storage)
  - access/policy.py:79-81
---

## What it is for

How a new experiment begins. Pressing **＋ New** in the sidebar makes a **draft** straight away and opens its setup. The draft appears in Experiment Selection with a **Draft** badge from that moment, so you can leave it and come back to it. It becomes an experiment only when you press **Create experiment** on the last setup step.

## What is on it

- **＋ New** (sidebar): creates the draft and opens step 1, Setup.
- A page titled **New experiment** with one button, **Start a new experiment**. You only see this page if you reach New through a link or bookmark; it does the same as ＋ New. It says: "A new experiment starts as a draft that only you can see, and is set up in three steps before it is created."
- At the top of every setup step:
  - The heading "New experiment: *name*" and the line "A draft until it is created: only you can see it, and nothing about it runs yet."
  - **🗑 Discard draft** (`data-assist="discard"`, tooltip "Throw this draft away").
  - The step bar (`data-assist="steps"`): **1 Setup**, **2 Optimizer**, **3 Data**, **4 Priors & constraints**. Each step says **Saved** (with a ✓), **Save to continue** (required and not saved yet) or **Optional** (step 4). Steps 2 to 4 can be opened only once step 1 is saved; any reachable step can be gone back to.

## How to use it

1. **Start:** press **＋ New**. Fill in step 1 and press **Save and continue**; each step leads to the next.
2. **Resume:** click the draft (Draft badge) in the sidebar. It opens at the first required step not yet saved, or at step 4 when steps 1 to 3 are all saved.
3. **Discard:** on any step, press **🗑 Discard draft**. The draft and its uploaded files are deleted, and you are taken back to the home page with the message "Discarded the draft “*name*”."

## Rules and limits

- Steps 1 to 3 must each be saved before **Create experiment** works; step 4 is optional.
- With accounts, a draft is its owner's alone: not visible to their group lead, colleagues or site administrators. A draft has no dashboard, runs, export or timeline; its history begins when it is created.
- Discarding is permanent. A draft never goes to the bin, because nothing was run.
- Discarding, creating and saving steps are done by the person. The assistant may fill in a step's fields but never presses these buttons.
- If the group is out of storage when the dataset or model is saved, the draft is deleted again, and a message says why.

## Questions people ask

**I pressed New by accident. What now?** Press **🗑 Discard draft**, or leave the draft. It stays in the sidebar until you discard it.

**Can a colleague see my draft?** With accounts, no. Only after **Create experiment** can it be shared.

**Where did my half-finished setup go?** It is the sidebar row with the Draft badge. Clicking it picks up at the first step not yet saved.

**Is a discarded draft in the bin?** No. Discarding deletes it for good.

<!-- REVIEW: new_experiment.html and the _setup_steps.html comment say "three steps"; the setup has four (three required). Consider updating the page's caption. -->
