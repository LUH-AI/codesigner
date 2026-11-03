---
id: concept-permissions
title: Who may do what
kind: concept
about: [action:view, action:edit, action:run, action:compute, action:export, action:rename, action:delete, action:share]
related: [concept-deferred-computation, concept-tasks-and-models, page-dashboard, page-experiment-settings]
keywords: [permission, owner, group, group lead, primary lead, site admin, member, share, viewer, contributor, transfer, hand over, draft, private, delete, bin, rename, export, run, cancel, assistant limits, human only, custom models, defaults]
status: draft
sources:
  - ui/permissions.py:14-82
  - access/policy.py:212-370
  - access/policy.py:378-386
  - access/models.py
  - experiments/models.py:286-344
  - ui/views/sharing.py:16-20
  - experiments/services/bin.py
  - ui/setup_steps.py
  - assistant/gateway.py
  - assistant/instructions/core.md
  - ui/permissions.py:52-82 (experiment_view actions per route)
  - ui/templates/ui/panels/group.html
  - ui/templates/ui/base.html:81-103
---

## In short

Without accounts, everyone may do everything. With accounts, an experiment belongs to its owner and sits in a group; others reach it only if the owner shares it or they lead the group. A draft is its owner's alone. The assistant acts as the person, never as more, and some actions are reserved for the person.

## In detail

### Actions

Every page and request declares one of these actions:

- **view**: see the experiment and its figures. Analyses fetched for a figure count as viewing.
- **edit**: change settings, the optimizer (before the first trial), priors, and setup steps.
- **run**: start or cancel a run, force-stop a stuck one, run a proposal, prepare the model's environment.
- **compute**: spend the machine on "Evaluate Prior" or "Compute explanations".
- **export**: download the `.ihpo` file, the model script, or a trial's fitted model.
- **rename**, **delete** (including discarding a draft), and **share** (set who else has access, or hand the experiment over).

### Without accounts

With no login, there are no owners and nothing is refused (except the assistant's limits below).

### With accounts

**What you can reach:**

- your own experiments;
- experiments shared with you, in a group you are still in;
- every experiment in a group you lead.

Someone else's draft is never reachable, not even by their group lead or a site admin. An experiment you cannot reach looks as if it does not exist.

**What you may do:**

| | view, export | edit, run, compute | delete | rename, share |
|---|---|---|---|---|
| Owner | yes | yes | yes | yes |
| Contributor | yes | yes | yes | no |
| Viewer | yes | no | no | no |
| Group lead (its group) | yes | yes | no | no |

**Sharing.** The owner gives one colleague from the experiment's group at a time a level: none, Viewer or Contributor. The owner can also hand the experiment over to a contributor. The new owner takes over, and the previous owner stays on as a contributor.

**Roles.**

- A *member* creates and runs experiments.
- A *group lead* also manages the group's people, and sees colleagues' work in a separate panel.
- The *primary group lead* appoints and removes leads.
- A *site admin* manages groups (size, storage, active or not) and sees no experiments unless in a group.

A person can be in several groups, with a role in each.

**Instance-wide permissions:**

- "use custom models" is needed to upload and run model code, and only where custom models are allowed at all.
- "change defaults" is needed to edit the settings every inheriting experiment follows.

**Deleting.** A deleted experiment goes into the bins of its owner and contributors, and any of them can restore it. Once nobody's bin holds it, a site admin restores it to a group or destroys it.

### What the assistant may never do

The assistant acts only as the person, through the same checks as every page, and is always refused these, whatever the person may do:

- start, cancel or force-stop a run, or run a proposal (run);
- delete an experiment or discard a draft (delete);
- share an experiment or hand it over (share);
- upload or run custom model code;
- change the instance's default settings;
- manage groups, see the site bin, or use any permission that reaches past the person's own work;
- change the person's password or account.

It cannot press buttons either: saving a step, creating the experiment and running are the person's. It can fill fields, explain, navigate and highlight. Asked for any of these, it explains how the person does it.

## In Codesigner

- The sharing and hand-over controls are on the group page.
- Some refusals are said where they apply, for example "You can view this experiment but not run it." on a proposal's "Run as Trial".
- The Bin is the 🗑 button in the sidebar's "Experiment Selection" section. A site admin settles the site bin from the site panel.

## Questions people ask

**Why can my colleague see but not run my experiment?** You shared it as a Viewer. Make them a Contributor.

**Can my group lead delete my experiment?** No. A lead can view, run, compute and edit it, but not delete, rename or share it.

**Who can see my draft?** Only you, until it is created.

**Can the assistant start the run for me?** No. Starting a run is always the person's own action.
