# Step 1 — Walking Skeleton

**Status:** implemented, awaiting your sign-off.
**What exists now:** a running Django site with the Codesigner page shell (sidebar + main area), an admin site, SQLite database, and environment-driven settings.

---

## 1. Django concepts introduced

### Project vs. app
Streamlit has one unit: the script. Django has two:

- A **project** (`config/`) — the deployment: settings, the root URL table, the
  WSGI/ASGI entry points servers use. It contains no features.
- **Apps** (`ui/`, later `core/`) — reusable feature packages. `ui` will hold
  all views/templates/forms. An app must be listed in `INSTALLED_APPS` before
  Django looks inside it.

We named the project `config` because that's all it is; the repo name
(codesigner) is the product name. (We also avoid `app`/`models` as names —
they'd shadow the packages we're porting from the old repo.)

### settings.py + django-environ
Streamlit needed no configuration; Django centralizes all of it in
[config/settings.py](../../config/settings.py). Anything secret or
machine-specific (SECRET_KEY, DEBUG, DATABASE_URL) is read from environment
variables via `django-environ`, with a `.env` file for local development
(gitignored; [.env.example](../../.env.example) documents the variables).
This is the "12-factor" pattern: the same code runs locally and deployed,
only the environment differs. Note `DATABASES` — switching to Postgres later
is literally one env var.

### The request path: URLconf → view → template
This replaces Streamlit's top-to-bottom rerun model. A request to `/`:

1. **Root URLconf** [config/urls.py](../../config/urls.py) — a table mapping
   URL patterns to handlers. It `include()`s [ui/urls.py](../../ui/urls.py),
   so each app owns its own routes.
2. **View** [ui/views.py](../../ui/views.py) — a plain function
   `home(request)` that returns a response. Where Streamlit re-runs your whole
   script on every interaction, Django runs *only* the view matched by the URL.
3. **Template** — `render()` fills an HTML template.
   [templates/base.html](../../templates/base.html) is the shared layout
   (declared in `TEMPLATES["DIRS"]`);
   [ui/templates/ui/home.html](../../ui/templates/ui/home.html) *extends*
   it and fills the `{% block content %}` hole. Template inheritance is
   Django's answer to "the sidebar renders on every page."

### Migrations
`python manage.py migrate` created `db.sqlite3` and the tables Django's
built-in apps need (sessions, admin, auth — present because the admin
requires them; we are **not** building auth features). In Step 3, when we
define our own `Experiment` model, `makemigrations` will generate the schema
change scripts and you'll see the full cycle.

### The admin
`/admin/` is a free, auto-generated database UI — Django's killer feature for
inspecting data. It becomes useful in Step 3 when our models exist. (Its login
is for the admin backdoor only; the app itself has no auth, per our plan.)

## 2. Feature checklist (from the Streamlit sources this step mirrors)

Extracted from `run.py`, `app/app.py`, `app/sidebar.py`, and
`utils/strings.py` in the old repo:

| Source behavior | Status |
|---|---|
| `st.set_page_config(page_title=..., layout="wide")` (`app.py:10`) | ✅ Implemented — `<title>Codesigner</title>`, two-column wide layout in `base.html` (title intentionally changed: rebrand) |
| Home page title + "Use the sidebar to create a new experiment." (`app.py:21-22`) | ✅ Implemented — `home.html` (rebranded heading) |
| Sidebar renders on every page (`app.py:14`) | ✅ Implemented — via template inheritance in `base.html` |
| Sidebar title "Experiments" (`sidebar.py:37`) | ✅ Implemented (static text for now) |
| Language selector 🌐 (`sidebar.py:20-35`) | ⏳ Deferred → Step 9 (visible placeholder in sidebar) |
| New-experiment button (`sidebar.py:41-44`) | ⏳ Deferred → Step 6 (disabled placeholder button) |
| Load-experiment button (`sidebar.py:45-47`) | ⏳ Deferred → Step 7 (disabled placeholder button) |
| Per-experiment buttons with active highlight (`sidebar.py:51-73`) | ⏳ Deferred → Step 4 |
| Running-experiment spinner (`sidebar.py:7-15,55-63`) | ⏳ Deferred → Step 8 |
| Session-state init: `experiments`/`active`/`creating`/`locale` (`run.py:29-33`) | 🔄 Intentionally changed — this state moves to the database (Step 3) and URLs (Step 4); nothing to port literally |
| `MODELS`/`OPTIMIZERS`/`METRICS` registries (`run.py:11-27`) | ⏳ Deferred → Step 6 (`ui/registry.py`) |
| Routing: creating → form, active → experiment, else home (`app.py:16-22`) | 🔄 Intentionally changed — becomes URLs: `/experiments/new/`, `/experiments/<id>/`, `/` (Steps 4/6) |

Every ⏳ item is tracked in [PARITY.md](../../PARITY.md) with its target step.

## 3. What is now possible

- `python manage.py runserver` serves the Codesigner shell at
  http://127.0.0.1:8000/ — sidebar with placeholders, branded home page.
- `/admin/` works (create a login with `python manage.py createsuperuser`).
- A SQLite database exists and migrations run.
- Settings are env-driven: copy `.env.example` → `.env`, and the same code is
  deployable elsewhere by setting real env vars.

## 4. Verify it yourself

```bash
cd ~/Downloads/python/projects/codesigner
.venv/bin/python manage.py runserver          # → http://127.0.0.1:8000/
# in the old repo, in parallel:
cd ~/Downloads/python/projects/InteractiveHPO && streamlit run run.py   # → :8501
```

Compare the shells side by side. Things worth poking at while you're there:
change the home heading in `ui/templates/ui/home.html` and reload (no
restart needed); delete `SECRET_KEY` from `.env` and see Django refuse to
start (env-driven settings are real); visit a URL that doesn't exist and read
the DEBUG-mode 404 page, which shows the URLconf patterns it tried.
