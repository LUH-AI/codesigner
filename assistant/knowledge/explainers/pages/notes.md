---
id: page-notes
title: Notes
kind: page
about: [page:ui:experiment_notes, field:text]
related: [page-dashboard, page-timeline, page-account-and-groups]
keywords: [notes, notepad, scratchpad, markdown, directives, goals, decisions, what was tried, shared, colleagues, assistant notes, preview, save, changed meanwhile]
status: draft
sources:
  - ui/templates/ui/experiment_notes.html
  - ui/views/notes.py
  - ui/static/ui/notes.js
  - ui/static/ui/markdown.js
  - assistant/notes.py
  - assistant/tools/notes.py
  - ui/templates/ui/base.html:163-190
---

## What it is for

One Markdown document per experiment for what everyone working on it should know: the person's goals and rules, decisions and why they were made, what was tried and dropped. It is shared with everyone who can open the experiment and with their assistants, which read it at every turn and write to it. Opened with **Notes** under the experiment's row in the sidebar, for a draft as well as a created experiment.

## What is on it

- The experiment's header, then the heading **Notes** and who changed the notes last, and when. A change by an assistant says "*name*'s assistant".
- For a person who may edit the experiment: the editor. **Write** shows the text box; **Preview** shows the notes as they will be drawn. A line under the box lists the Markdown it understands: headings, bold, italics, lists, numbered lists, quotes, code, links and tables. Then **Save**.
- For a person who may only view the experiment: the notes as drawn, or "Nothing written down yet."

## How to use it

1. Write in **Write**, check it in **Preview**, then press **Save**.
2. To have the assistant write something down, ask it in the experiment's chat, for example "remember that we report balanced accuracy". It adds a paragraph or changes the one on that subject.
3. Leaving the page with changes unsaved asks first.

## Rules and limits

- Viewers read the notes; the owner and contributors, and their assistants, write them.
- The notes hold at most 20000 characters.
- If someone, or an assistant, saved the notes after you opened them, your save is refused. The page shows their version above the editor, which still holds yours: merge the two and save again.
- An assistant changes only the passage it names, or adds one at the end, so it never writes over what someone saved meanwhile.
- Links open only to http and https addresses; nothing else in the notes can add to the page.
- The notes stay on this instance: they are not in the experiment's exported `.ihpo`.

## Questions people ask

**Who can see the notes?** Everyone who can open the experiment, and their assistants.

**Why was my save refused?** The notes changed after you opened them. Their version is shown above yours; merge and save again.

**Does the assistant read what I write here?** Yes. Every chat about the experiment is given the notes at each turn, as information rather than instructions.
