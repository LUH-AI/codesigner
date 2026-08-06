# Auth E — ownership and sharing

**Status:** implemented, awaiting your sign-off.
**You can now:** on an instance with accounts, an experiment belongs to whoever
created it. Other people cannot see it unless it is shared, and if it is shared
they can read and export it but not run, change or delete it. Without accounts,
nothing has changed.

Fifth group of the auth epic, on [the permission seam](auth-c-permission-seam.md)
and [the login wall](auth-d-login-wall.md).

---

## The whole change, because of group C

```python
class OwnerPolicy(OpenPolicy):
    def experiments(self, request): ...
    def may(self, request, experiment, action): ...
```

One class, forty lines. No view was touched to enforce any of it — the seam
already routes every route that names an experiment through
`policy().experiments(request)` and `policy().may(...)`, and the URLconf audit
guarantees there is no route that does not. That was the argument for building
the seam first, and this is the bill coming in low.

## Three kinds of experiment

**Yours.** Anything.

**Shared with you.** Readable and exportable; not runnable, editable or
deletable. This is the distinction that carries the most weight in the design:
*sharing is an invitation to look, not a transfer of control.* Reading
someone's results should not come with the ability to overwrite them — an
owner's incumbent should not change because a colleague pressed Run to see what
happened.

Export *is* allowed on a shared experiment. Refusing would be theatre: the file
carries only what the recipient's own page already shows them. Server paths are
stripped from every export anyway (below).

**Nobody's** (`owner IS NULL`). Everyone's, in full. These are the experiments
that existed before the instance had accounts. There is no owner whose wishes
are being overridden, and the alternative — hiding them — would silently swallow
an operator's existing work the moment they flipped the switch. An operator
assigns owners in the admin, after which the normal rules apply. The experiment
page says plainly that an unowned experiment is unowned, so this is not a state
you have to infer.

**Staff** are exempt from the action rules and see everything.

## Decisions worth naming

**One switch, still.** `EXPERIMENT_POLICY` now defaults to
`access.policy.OwnerPolicy` rather than `OpenPolicy`, and `OwnerPolicy` returns
`Experiment.objects.all()` and `may() → True` while `REQUIRE_LOGIN` is off. The
alternative — leaving the default as `OpenPolicy` and telling operators to also
set `EXPERIMENT_POLICY` — is the two-variable checklist that the login wall was
built to avoid. `OpenPolicy` remains as the base class and as the escape hatch
for an instance that wants accounts without ownership.

**`SET_NULL`, not `CASCADE`.** Removing a person from an instance must not
destroy results other people may be relying on. Their experiments become
nobody's, and an operator reassigns them.

**Anonymous gets nothing, not everything.** `experiments()` returns `.none()`
for an unauthenticated request on a hosted instance. The wall should have caught
that request already; not relying on it is what keeps one missing exemption from
becoming a data leak.

**404, not 403, for someone else's.** Group C's split, doing its job: an
experiment you may not see is indistinguishable from one that does not exist.
A 403 would let a stranger enumerate the instance by reading status codes.

**Checked at the route, not hidden in the template.** The buttons for actions
you may not take are not rendered, but the tests post to those routes directly
and assert 403 — a form can be submitted without the page that renders it.

## Export no longer names this server

`snapshot_from_experiment` emits absolute `dataset_path` and `model_path`. That
is right for its two internal callers — the detail page rebuilds the result from
it, and the run engine rebuilds the experiment from it — and wrong for the third:
a downloaded `.ihpo`. The paths name a machine that is not the recipient's, so
they are useless to them, and they describe how the instance is laid out.

Both are now blanked in `experiment_export`, beside the existing
`export_absolute_times` scrub — on the way out only, so nothing internal
changes. Unconditional, not a setting: a setting whose "on" position is a leak is
a footgun, and there is no case where the recipient benefits.

**One behaviour change to be aware of.** Export a `.ihpo` and re-import it with
the `import_ihpo` management command on the same machine, and the dataset is no
longer adopted automatically — the file no longer says where it was. The web
import view never adopted paths (you attach the dataset), so this only makes the
command consistent with it, and the original file is still where you left it.

## What was built

- `Experiment.owner` (nullable FK, `SET_NULL`, `related_name="experiments"`) and
  `Experiment.shared`; migration `0008`.
- `access/policy.py` — `OwnerPolicy`.
- `experiment_from_snapshot(..., owner=...)`, set by the create and import views.
  An `.ihpo` has no notion of who made it, so creation is the only place this
  could come from.
- `experiment_share` — a POST route, `@experiment_view(EDIT)`, so an experiment
  cannot be shared out from under its owner.
- The detail page gained an ownership line (the sharing checkbox for the owner,
  "shared with you by …" for a reader, "no owner" otherwise) and hides the
  Settings and Delete buttons for anyone who may not use them. All of it is
  `None` when the instance has no accounts, so the page is unchanged there.
- `owner`/`shared` in the admin, with `owner__username` searchable — the admin is
  the only place to change them for someone else.
- Three new strings, translated into de and es.
- README: a *Who sees what* table.

## Verify

```bash
python -m pytest -m "not slow"    # 466
python manage.py migrate
REQUIRE_LOGIN=True python manage.py runserver
```

20 new tests in `tests/access/test_ownership.py`, one per boundary: creation
assigning the owner, someone else's being a 404 and absent from the sidebar, a
shared one being readable but refusing run and delete at the route, only the
owner seeing the sharing control, sharing toggling both ways, an ownerless one
being fully usable, `SET_NULL` on user deletion, staff seeing everything, and
the export carrying no paths while the experiment itself still knows them.

Live-checked all three ownership states on the detail page: the owner gets the
checkbox and every button, a reader gets "Shared with you by ben — read only."
and Export alone, and an unowned experiment says so and keeps every button.

## Next

**F — trusted uploads and staff-only defaults.** `ALLOW_CUSTOM_MODELS` is
currently instance-wide, which on a hosted instance means every account or none.
The remaining work is a per-account permission checked at the form, the import
view, `_model_available` (against `exp.owner`) and — decisively — inside
`execute_run`, where a form gate cannot be bypassed. Plus the global default
experiment settings, which one user can currently change for everyone.
