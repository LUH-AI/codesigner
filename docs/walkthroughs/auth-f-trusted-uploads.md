# Auth F — trusted uploads and staff-only defaults

**Status:** implemented, awaiting your sign-off.
**You can now:** grant *Can upload and run custom models* to individual accounts
instead of turning the capability on for everyone at once, and keep the default
experiment settings — which every inheriting experiment follows — staff-only.
Without accounts, both are open, as before.

Last group of the auth epic, after
[the seam](auth-c-permission-seam.md), [the wall](auth-d-login-wall.md) and
[ownership](auth-e-ownership.md).

---

## The gap this closes

`ALLOW_CUSTOM_MODELS` is instance-wide. On a laptop that is exactly right. On an
instance hosted for a dozen industry partners it means **every account or none**,
for a capability whose honest description is *arbitrary code execution*. Neither
position is acceptable: off makes the product much less useful, on hands the
whole instance to whoever registers first.

So the flag stays as the floor and a per-account permission sits on top.

## Where it is checked, and which check counts

Four places, and they are not equal.

| Where | What it does |
|---|---|
| `NewExperimentForm` | The upload field and mounted-model dropdown are not built |
| `import_experiment` | An attached model `.py` is not adopted |
| The detail page | `can_run` is false, with the reason shown |
| **`execute_run`** | **The run fails, naming the account that was refused** |

The first three are the user interface being honest. The last one is the
decision. A form gate is advice: the form can be bypassed, an experiment can
change hands between upload and run, and a run is started by a **background task**
rather than by the request that rendered a page. `execute_run` is the process
that would actually execute the code, so that is where the answer has to be
final — and the tests go straight at it rather than through a page.

## Who is "the account behind" a run

The obvious answer — the experiment's owner — is wrong for one case the previous
group created: an **unowned** experiment has no owner to check. Refusing all of
them would mean an experiment predating accounts could never run a custom model;
allowing all of them would mean it always could. Both are wrong.

So `Run` gained `started_by` (nullable, `SET_NULL`), and the worker checks
`run.started_by or experiment.owner`. Recording who pressed Run is worth having
on a multi-user instance regardless, and it makes the question answerable rather
than approximated.

## Decisions worth naming

**The flag outranks the permission.** `ALLOW_CUSTOM_MODELS=False` means off for
everyone, including holders of the permission and including superusers. An
operator turning custom models off must not then have to audit who holds what.

**Staff is not trust.** `is_staff` means "can use the admin", not "may run
arbitrary code", so it does not confer the permission. A **superuser** does get
it, because Django grants superusers every permission by definition — that is
the framework's contract and overriding it would be more surprising than
following it.

**A refusal names the account.** `custom_model_refusal` returns a *string*, not
a boolean. This is the last thing standing between an upload and the
interpreter, and "denied" with no reason in a worker log is unactionable.

**The defaults are a second kind of global.** One person editing the default
experiment settings changes what every inheriting experiment on the instance
draws. There are two ways in — the settings page and the "Save settings as
default" button on an experiment's own page — and both now check
`may_change_defaults`, one raising `PermissionDenied` and the other hiding the
control that leads there.

## What was built

- `access/models.py` — an **unmanaged** model with `default_permissions = ()`,
  purely to host `use_custom_models`. No table is created; the `Permission` row
  is, so it appears in the admin's user and group editors and
  `user.has_perm("access.use_custom_models")` works. This is Django's documented
  way to declare a permission that is not about a row.
- Three new policy methods, on `OpenPolicy` (all permissive) and overridden in
  `OwnerPolicy`: `may_upload_models(request)`,
  `custom_model_refusal(experiment, user)`, `may_change_defaults(request)`.
- `Run.started_by`; `create_run(..., started_by=…)`; migration `0009`.
- `NewExperimentForm(may_upload_models=…)` — the form is **told**, rather than
  reading the flag itself, so it does not have to know whether the instance has
  accounts.
- A `capabilities` context processor for `may_change_defaults`, since the
  sidebar needs it outside any experiment's pages.
- The detail page shows the refusal instead of a Run form that is silently
  missing.
- Two new strings, translated into de and es. README: two new sections.

## Verify

```bash
python -m pytest -m "not slow"    # 481
python -m pytest -m slow          # 6
python manage.py migrate
REQUIRE_LOGIN=True python manage.py runserver
```

15 new tests in `tests/access/test_trusted_uploads.py`. The worker ones are the
substance: an untrusted owner's model refused with their username in the error,
a registry model never gated, and `started_by` deciding it for an unowned
experiment. Then the pages: no upload field for an untrusted user, one for a
trusted user, a POST with an upload anyway creating nothing, an import not
attaching the model, the reason shown on the page, and the flag still outranking
the permission. Then the defaults: 403 for a non-staff user by either route, the
link hidden, and both open with no accounts.

Live-checked all three account kinds on a hosted instance:

| | upload field | defaults link | defaults page |
|---|---|---|---|
| untrusted | no | no | 403 |
| trusted | **yes** | no | 403 |
| staff | no | **yes** | **200** |

— which is the intended matrix, including staff *not* getting the upload field.

## The auth epic, end to end

- **A** — the two exploitable path holes, `require_POST` on state-changing
  routes, the baked-in `SECRET_KEY`, media gating. `425c557`, `38a96f7`.
- **B** — executing an upload during the web request. Subsumed by
  [model environments](model-environments.md), which stopped executing uploads
  at all.
- **C** — [the permission seam](auth-c-permission-seam.md). One way in, plus a
  URLconf audit that keeps it that way.
- **D** — [the login wall](auth-d-login-wall.md). Default-closed, two exemptions.
- **E** — [ownership and sharing](auth-e-ownership.md). One policy class, no
  view changes.
- **F** — this. Per-account trust for the one capability that is genuinely
  dangerous.

`REQUIRE_LOGIN` remains the only switch. With it off, none of the above is
reachable and the app behaves exactly as it did before the epic started — which
was the requirement the whole design was built around.

## Still not done, deliberately

**Real sandboxing.** A trusted account's model still runs as the same OS user
with the same filesystem and network. The permission decides *who* may run code,
not *what* that code may do. `uv` gave dependency isolation; seccomp, bubblewrap,
a separate UID or network namespaces are a separate epic, and the docs must not
let "runs in its own environment" be read as "sandboxed".

**Per-user media.** `MEDIA_ROOT` is one flat directory. Serving it is already
disabled on a hosted instance, which is why this is not a live hole, but a
future feature that serves an uploaded file would need paths scoped per owner
first.

**Audit logging.** `started_by` records who ran what; nothing records who looked
at what, or who changed the defaults.
