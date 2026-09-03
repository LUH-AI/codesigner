# Auth D — the login wall

**Status:** implemented, awaiting your sign-off.
**You can now:** set `REQUIRE_LOGIN=True` and have every page require a
signed-in user, with accounts managed in the Django admin. Leave it off — the
default — and nothing about the app has changed.

Fourth group of the auth epic, on top of [the permission seam](auth-c-permission-seam.md).

---

## The shape of the switch

Two configurations, one boolean. Not two settings modules, not a separate
"hosted" image, not an `access` app that gets added to `INSTALLED_APPS` when you
want it. The package is **always installed and inert**, and `REQUIRE_LOGIN`
wakes it up.

That is a deliberate trade. A settings module that only the hosted deployment
loads is a file nobody runs locally, so it drifts, and the drift is discovered
in production. One boolean read by code that is always in the request path means
the two modes are the same code, and the difference is small enough to hold in
your head.

The cost is that the login page exists on a local install — reachable at
`/accounts/login/`, pointless, nothing links to it. That is the right way round:
a dead route beats a route that only appears in the configuration you test
least.

## Default closed

Django 5.1 added `LoginRequiredMiddleware`, which gates **every** view unless it
is marked `login_not_required`. That polarity is the whole reason to use it
rather than decorating views:

> a route added next year is protected by having been *forgotten about*, rather
> than exposed by it.

`access/middleware.py` subclasses it and adds one thing:

```python
class LoginWallMiddleware(LoginRequiredMiddleware):
    def process_view(self, request, view_func, view_args, view_kwargs):
        if not settings.REQUIRE_LOGIN:
            return None
        return super().process_view(request, view_func, view_args, view_kwargs)
```

The setting is read **per request**, not captured at startup. It costs nothing
and means a test — or an operator's `.env` — is never fighting a stale reading.

## The two exemptions, and why only two

**`/healthz/`.** The container runtime has no session. A healthcheck that 302s
to a login page reports a healthy instance as down, and the orchestrator
restarts it forever. Marked with `@login_not_required` in `ui/views.py`.

**The language switcher.** The login page carries it, so it has to work while
gated — otherwise the one page a stranger can reach is the one page they may not
be able to read. `set_language` is Django's view, so it could not be decorated
in place; `config/urls.py` now spells the route out instead of
`include("django.conf.urls.i18n")` — the same single route — and wraps it:

```python
path("i18n/setlang/", login_not_required(set_language), name="set_language"),
```

Django's own `LoginView` is exempt already (its `dispatch` carries the marker,
and `as_view()` copies it onto the view function). `LogoutView` is **not**, which
is correct: you have to be signed in to sign out.

A test asserts the exempt set is exactly `{healthz, set_language, login}`, by
walking the resolver. An exemption is always deliberate; this is what stops the
list growing quietly.

## What was built

- `access/` — `middleware.py`, `urls.py` (login and logout only), `apps.py`, and
  a standalone `login.html`.
- The login page **does not extend `base.html`**. There is no session yet, so
  the rail and experiment sidebar would render empty and offer links that bounce
  straight back. It keeps the language switcher and nothing else.
- `LOGIN_URL` / `LOGIN_REDIRECT_URL` / `LOGOUT_REDIRECT_URL`, set unconditionally
  so turning the switch on is one variable rather than a checklist.
- A sign-out control in the rail, rendered on `{% if user.is_authenticated %}`
  — so an install with no accounts sees no trace of it, without the template
  needing to know the setting exists.
- Six new strings, translated into de and es in the same commit.
- README: a *Hosting it for other people* section. `.env.example`: the variable.

**No password reset.** It needs a configured mail server, which is an operator
decision and a surface of its own. Until it is asked for, an operator resets a
password in the admin. **No self-registration** either — for an instance hosted
for a known set of people, `createsuperuser` plus the admin is the right amount
of machinery.

## What this does *not* do

Everyone who signs in still sees everything. There are no owners yet — that is
group E, and it is a policy class plus two fields, because of group C. This
group is only the wall.

Note also what `REQUIRE_LOGIN` already meant before this: media files stop being
served from the app (`MEDIA_ROOT` is one flat directory, so serving it would
hand every signed-in user every other user's data at a guessable URL), and a
model that would run in the application's own process is refused rather than
falling back. Those came from earlier groups; the README now collects all three
in one place.

## Verify

```bash
python -m pytest -m "not slow"     # 443
python manage.py check
REQUIRE_LOGIN=True python manage.py runserver
```

14 new tests in `tests/access/`: that the wall is invisible when off (the
primary case — a local install regressing into a login prompt would be the worst
outcome of this package existing at all), that the switch is read per request,
the redirect and the `?next=` round trip, a wrong password, POST-only logout,
both exemptions with the reason attached, the login page rendering in a chosen
language, and the exempt-set audit.

Live-checked signed out and signed in: the redirect carries `?next=`, the login
page renders standalone, and the rail shows the sign-out control with the
username in its tooltip.

**One thing the tests did not catch, found by looking.** The login template's
explanatory note was written as a multi-line `{# … #}`, which Django only
treats as a comment on a **single** line — so the note was rendering into the
page. Both instances are now `{% comment %}`, and
`tests/ui/test_template_hygiene.py` checks every template's source for the
mistake, since a rendered-output test only covers templates some test happens to
render. It also fails a new template that has text but no `{% translate %}` —
the i18n completeness check only sees strings that were marked, so a template
that forgets i18n entirely is invisible to it.

## Next

**E — ownership and sharing.** A nullable `owner` FK (`SET_NULL`, so deleting a
user does not delete their results), a `shared` boolean, and an `OwnerPolicy`
whose `experiments()` filters to owned-or-shared and whose `may()` refuses
`RUN`/`EDIT`/`DELETE` on someone else's. Plus blanking paths on export, since a
`.ihpo` currently names server-side paths.
