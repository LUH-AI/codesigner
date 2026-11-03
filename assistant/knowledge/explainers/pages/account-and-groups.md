---
id: page-account-and-groups
title: Settings, account and groups
kind: page
about: [page:ui:appearance, page:ui:export_preferences, page:ui:account, page:ui:group_index, page:ui:group_detail, field:theme, field:export_timestamps, field:level, field:role, field:email, field:password]
related: [concept-permissions, page-dashboard, page-export-import, page-experiment-settings]
keywords: [settings, appearance, theme, dark mode, export preference, account, permissions, password, group, members, lead, primary lead, roles, seats, storage, share, sharing, viewer, contributor, make owner, transfer, add somebody, remove]
status: draft
sources:
  - ui/templates/ui/base.html:12-127
  - ui/templates/ui/appearance.html
  - ui/templates/ui/export_preferences.html
  - ui/templates/ui/account.html
  - ui/views/preferences.py:86-101
  - ui/context_processors.py:106-120
  - core/metrics.py:23-58
  - ui/templates/ui/panels/group.html
  - ui/templates/ui/panels/group_none.html
  - ui/views/groups.py
  - experiments/models.py:297-313
  - access/models.py:91-104
  - access/policy.py:55-57
  - access/signals.py:12-18
  - ui/urls.py
  - ui/views/sharing.py:41-67
  - ui/templates/ui/_experiment_header.html:78-101
  - access/policy.py:79-119
  - assistant/tools/client.py
---

## What it is for

The pages about you rather than about one experiment: how Codesigner looks, how your exports carry times, what your account may do, and your groups, including who each of your experiments is shared with. The assistant never changes any of them.

## What is on it

**Settings** (⚙ on the rail):
- **Appearance**: **Theme** (`theme`): Light, Dark or Same as my system. It is saved for this browser, not the account.
- **Export**: **Time** (`export_timestamps`): Real Times, Relative Times (the default) or No Times. This is what the export page preselects unless the experiment says otherwise.
- **Account** (with accounts only): "Signed in as *name*", and **What you may do here**, ticking: See every experiment on this instance; Run, edit and delete other people's experiments; Change the settings every experiment inherits; Upload and run custom models. **Change password** asks for the current one.
- **Default experiment settings**: only for people allowed to change them.

**Group** (👥 on the rail, with accounts): the sidebar lists your groups; with none, "You are not in a group yet. A group's lead can add you by your email address." A group's page has:
- **Storage**: used of the limit, and how much is kept trial models. Leads also see the seats left.
- **Your experiments**: each with **Shared with**. Open it to set each colleague to **Not shared**, **Viewer** or **Contributor** (`level`). A contributor also gets **Make owner**.
- **Shared with you**: experiments, their owner, and whether you are a viewer or a contributor.
- **Everybody else's** (leads only): the rest of the group's experiments.
- **People**: name, **Role**, number of experiments. The primary lead can set others' roles (Member, Group lead) and pass on **Make primary lead** to another lead. Leads can **Remove** people.
- **Add somebody** (leads only): **Email address**, **First password (optional)**, **Role**, **Add**.

**Site** (🏛, site administrators only): Groups, Usage, Jobs and the site Bin. It shows no experiments.

## How to use it

1. **Share an experiment**: Group → your group → Your experiments → open **Shared with** → set the colleague's level. It applies at once.
2. **Hand it over**: make them a Contributor first, then **Make owner**. You stay on as a contributor.
3. **Change your password**: Settings → Account → **Change password**.
4. **Switch to dark mode**: Settings → Appearance → Dark.

## Rules and limits

- An experiment belongs to its owner and lives in the owner's group. A **viewer** may look at it and export it. A **contributor** may also run it, change its settings and delete it. Only the **owner** decides who else may, renames it and hands it over.
- A **group lead** may open, run and stop any experiment in their group, but not delete it or decide its sharing. The **primary lead** alone appoints leads; the role is handed on, never dropped.
- Sharing is only with people in the experiment's group. Leaving a group ends the shares held in it.
- Someone who still owns experiments in the group cannot be removed until those are handed to someone else.
- A group has a seat limit and a storage limit, both set by a site administrator.
- Accounts are created by a lead (Add somebody) or an administrator; there is no sign-up. A forgotten password is reset by an administrator.
- Sharing, transferring, group membership, roles, account and preferences are the person's alone. The assistant does not open or change these pages.

## Questions people ask

**How do I let a colleague run my experiment?** Share it with them as Contributor from your Group page.

**Why can my lead see my experiment?** Leads reach every experiment in the group they lead, to run or stop it, but cannot delete or share it.

**Where is "share" on the experiment page?** The ownership line under the header links to it ("Share it from *group*").
