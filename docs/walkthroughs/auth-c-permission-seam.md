# Auth C — the permission seam

**Status:** implemented, awaiting your sign-off.
**You can now:** (nothing new — that is the point). Every route that reaches an
experiment now asks a policy object first, and the default policy says yes to
everything. A deployment that manages users supplies a different policy instead
of revisiting a dozen views.

Third group of the auth epic. **A** (the two exploitable path holes, `require_POST`,
the baked-in `SECRET_KEY`, media gating) shipped in `425c557`/`38a96f7`; **B** —
executing an upload during the web request — was subsumed by the
[model-environments epic](model-environments.md), which stopped executing
uploads at all. **D**–**F** follow.

---

## Why a seam and not just rules

This project is run locally and on private networks at least as often as it is
hosted. In those cases there are no users, no owners and nothing to authorize,
and the design commitment has been that auth is *transparent* there — not a
thing to configure off.

That commitment is what makes a seam necessary rather than optional. Without it,
"add auth later" means finding every place a view reaches an experiment and
adding a check, and the failure mode is silent: the one that was missed looks
exactly like the ones that were not. So the seam goes in **now**, while it is a
pure refactor with nothing to get wrong, and while the existing test suite can
prove it changed no behaviour.

## The two questions

They are separate because they fail differently.

**Which experiments exist, for this request?** A policy narrows a queryset, and
a miss is a **404**. An experiment you may not see has to be indistinguishable
from one that is not there — otherwise the URL space itself reports who has
what, and a stranger can enumerate the instance by reading status codes.

**May this request do X to this one?** Asked only about an experiment already
visible, so its existence is not a secret and a refusal is a plain **403**. This
is what group E needs for a shared experiment: visible to a colleague, not
runnable by them.

## What was built

**`ui/permissions.py`.**

- `OpenPolicy` — `experiments(request)` returns everything, `may(request, exp,
  action)` returns True. The default, and what a local install wants.
- `policy()` reads `settings.EXPERIMENT_POLICY` (default `ui.permissions.OpenPolicy`)
  and instantiates it. Cached **keyed on the setting string** rather than
  outright, so a test — or a later `access/` install — can point it elsewhere and
  be obeyed immediately.
- Five actions, named constants: `VIEW`, `RUN`, `EDIT`, `DELETE`, `EXPORT`. All-
  or-nothing would have forced group E to touch every view again; a vocabulary
  that exists from the start costs nothing while every answer is yes.
- `experiment_view(action)` — a decorator. `visible_experiments(request)` — for
  anything that enumerates rather than looks one up.

**The decorator changes the view signature**, deliberately:

```python
@experiment_view(VIEW)
def experiment_detail(request, exp):
    return render(request, "ui/experiment_detail.html", _detail_context(exp))
```

The view is handed the experiment *in place of* `pk`. That is the whole reason
this is a decorator rather than a helper the view calls: a view cannot
accidentally hold an unchecked `pk`, because it is never given one. Ten views
lost their `get_object_or_404(Experiment, pk=pk)` line; the `redirect(...,
pk=pk)` calls became `pk=exp.pk`.

**The two enumeration sites.** `ui/context_processors.py` builds the sidebar
from `visible_experiments(request)`, so the sidebar never lists something a page
would then 404 on. `ui/navigation.py` looks an experiment up the same way, so a
breadcrumb never names one straight out of the database on a page you cannot
open — the trail would otherwise be a directory of what exists. `_crumbs_for`
gained a `request` parameter; both callers already had one.

**Not converted:** `ui/services/modelenv.py` and `ui/services/run.py` reach
experiments by pk too, but they run in the worker with no request. A policy has
nothing to say there, and pretending otherwise would mean inventing a fake
request for a background task.

## The audit

A seam only holds if there is no second way in, and "remember the decorator" is
not a mechanism. `tests/ui/test_permissions.py` walks `ui.urls.urlpatterns`,
takes every route whose pattern contains `<int:pk>`, and fails on any whose
callback has no `experiment_action` attribute:

```python
undecorated = [p.name for p in _experiment_routes()
               if getattr(p.callback, "experiment_action", None) is None]
assert not undecorated
```

`functools.wraps` copies `__dict__`, so the attribute survives an outer
`require_POST` and the two orderings compose. A second test asserts the audit
finds routes at all — a pattern-matching mistake that matched nothing would
otherwise leave it passing vacuously forever. A third pins the action each route
declares, so reading the URLconf tells you what a route is *for*, and so a later
policy that refuses `RUN` refuses the right set.

**Checklist:** ✅ policy + five actions · ✅ decorator, experiment-not-pk ·
✅ ten views converted · ✅ sidebar and breadcrumbs through the policy ·
✅ `EXPERIMENT_POLICY` setting · ✅ URLconf audit, plus a guard against it going
vacuous · ✅ 404 vs 403 pinned · ✅ unknown action raises at decoration time,
because a typo would otherwise be a permission that is never checked inside a
decorator that looks like it is checking one.

## Verify

```bash
python -m pytest -m "not slow"        # 429 (416 + 13)
python manage.py check
```

The real assertion is one you can check with git: **no existing test was
edited.** A refactor that says "behaviour is unchanged" and comes with test
changes has not proved anything.

```bash
git show --stat <this commit>         # tests/ contains only the new file
```

The 13 new tests cover the audit, the default policy allowing everything, the
setting being honoured, and — with two deliberately restrictive stub policies
defined in the test file — that an invisible experiment 404s, stays out of the
sidebar and out of a breadcrumb, and that a refused action on a visible one
raises `PermissionDenied` rather than returning a hand-rolled 403 page.

## What this sets up

- **D (the login wall)** — an `access/` package that installs a
  `LoginRequiredMiddleware` subclass gated on `REQUIRE_LOGIN`. It needs nothing
  from this group, but it is where `EXPERIMENT_POLICY` starts pointing
  somewhere else.
- **E (ownership and sharing)** — an `OwnerPolicy`: `experiments()` filters to
  owned-or-shared, `may()` refuses `RUN`/`EDIT`/`DELETE` on someone else's
  shared experiment. That is the whole change, in one class, because of this
  group.
- **F (trusted uploads)** — the same shape, one level down: `_model_available`
  will check against `exp.owner`, and `execute_run` will check again in the
  worker, where a form gate cannot be bypassed.
