---
id: page-home
title: Home
kind: page
about: [page:ui:home, sidebar:experiment-selection, field:text, field:assistant-demo, field:assistant-use-demo]
related: [page-new-experiment, page-experiment-list, page-export-import, page-dashboard, concept-permissions]
keywords: [start page, front page, sidebar, Experiment Selection, New, Load, Bin, recent experiments, See all, draft badge, assistant chat, plan, approve]
status: draft
sources: [ui/templates/ui/home.html:1-15, templates/base.html:12-80, templates/base.html:128-184, templates/base.html:211-221, ui/context_processors.py:1-60, ui/views.py:286-287, assistant/templates/assistant/_chat.html:1-124, assistant/context_processors.py:1-20, assistant/views.py:207-247]
---

## What it is for

The page Codesigner opens on (the "Codesigner" heading). When the assistant is switched on, the home page is the assistant's chat, full size: you describe what the experiment is for, choose a dataset, and the assistant plans the experiment with you. When the assistant is off, the page only says "Use the sidebar to create a new experiment." Either way, the sidebar's **Experiment Selection** is where experiments are started, loaded and opened.

## What is on it

- **Icon rail** (far left): 🧪 Experiments, 👥 Group, 🏛 Site (administrators), ⚙ Settings, ⏻ sign out, and the 🌐 language select (`language-select`).
- **Experiment Selection** (sidebar):
  - **＋ New**: makes a new draft at once and opens its setup.
  - **📂 Load**: opens the page for loading a `.ihpo` file or a SMAC output folder.
  - **🗑** (tooltip "Bin"): your bin of deleted experiments.
  - **Recent list**: the 20 most recent experiments you can see, newest first. A draft has a **Draft** badge and opens its setup. A created experiment shows its ID in brackets and opens its dashboard. A spinner (tooltip "Running") means a run is under way, and ⏳ ("Preparing its environment") means an uploaded model's environment is still being built. Under the open experiment, three links: **Dashboard**, **Data Handling**, **Timeline**.
  - **See all…**: the full list of experiments.
- **The chat** (assistant on): the message box (`text`, "Send"), and above it the dataset to plan with: an upload box ("Upload Dataset"), a demo select (`assistant-demo`, "Choose a demo dataset") with a **Use demo dataset** checkbox (`assistant-use-demo`), and a **Group** select when you are in more than one group. **Reset** deletes the chat and its dataset, after asking. On other pages the chat is a tab docked at the bottom right.

## How to use it

1. To start yourself: press **＋ New** in the sidebar and fill in the setup steps.
2. To start with the assistant: choose a dataset (upload a CSV, or tick **Use demo dataset** and pick one), type what you want to find out, and press **Send**. When the plan is ready, press **Approve and set up**. Approving makes the draft and puts the dataset on it. Only you can approve.
3. To open an earlier experiment, click its row in the sidebar. If it is not among the recent ones, use **See all…**.

## Rules and limits

- The sidebar lists only experiments you may see: your own, ones shared with you and, for an instance without accounts, all of them. A group lead's list shows their own and shared work; their colleagues' work is on the Group page.
- With accounts, another person's drafts are never listed: a draft is its owner's alone until it is created. Without accounts there is no owner, so every draft is listed.
- Approving a plan, saving setup steps, creating, running and deleting are always done by you. The assistant can fill in fields and point at things, but it never presses those buttons.

## Questions people ask

**Why is my experiment not in the sidebar?** The sidebar shows the 20 most recent. Use **See all…** for the rest.

**What does the Draft badge mean?** The experiment is still being set up. Click it to continue where you stopped.

**Where did a deleted experiment go?** Into the bin (🗑 in Experiment Selection), where it can be restored.

<!-- REVIEW: the brief described the home page as "the experiment list otherwise"; in the code (home.html:12-14, also on main) it is only the sentence "Use the sidebar to create a new experiment." Confirm which is intended. -->
