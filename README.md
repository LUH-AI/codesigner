# Codesigner

An interactive hyperparameter-optimization workbench: set up an experiment
(model, optimizer, dataset), run trials in the background, and explore the
results — best/selected configuration, hyperparameter importance, and the
incumbent's performance over trials. Experiments save to a portable `.ihpo`
file. A Django rebuild of the InteractiveHPO Streamlit app.

The interface is available in English, German, and Spanish (🌐 selector in the
sidebar).

## Architecture at a glance

- **web** — Django (templates + a little plotly.js), serving the UI. Static
  files are served by WhiteNoise, so no separate web server is required.
- **worker** — a [huey](https://huey.readthedocs.io) consumer
  (`manage.py run_huey`) that executes optimization runs out of band.
- Run state (status, cancellation, timestamps) lives in the database, so the
  web process only *enqueues* runs; polling and cancel are plain DB reads/writes.
- Storage is SQLite (`django-environ`, so a `DATABASE_URL` swaps in Postgres).

## Running locally

```bash
pip install -r requirements.txt
pip install -e .                   # registers the app version (pyproject.toml)
pip install -e ./model_sdk         # the model contract (core.models imports it)
cp .env.example .env               # set SECRET_KEY
python manage.py migrate
python manage.py compilemessages -l de -l es   # build the de/es catalogs
python manage.py runserver
```

By default (`DEBUG=True`) there is no consumer, so `runserver` alone is enough:
runs execute in-process on a background thread, and the page comes back as soon
as you press Run rather than waiting out the optimization. Stopping the server
mid-run leaves that run marked `running`; clear it with

```bash
python manage.py sweep_stale_runs
```

To exercise the real queue locally, set `HUEY_IMMEDIATE=false` and run the
consumer in a second terminal:

```bash
HUEY_IMMEDIATE=false python manage.py runserver     # terminal 1
HUEY_IMMEDIATE=false python manage.py run_huey       # terminal 2
```

## Running with Docker

```bash
cp .env.example .env               # set SECRET_KEY (ALLOWED_HOSTS/DEBUG handled by compose)
docker compose up --build
```

This starts two processes off one image — the gunicorn **web** server on
`:8000` and the huey **worker** — sharing a `data` volume (SQLite DB, huey
queue, uploaded media). The web service has a `/healthz/` healthcheck.

The worker runs `HUEY_WORKERS` jobs at once (default 4). It has to be more than
one because building a model's environment shares the queue with optimizations
and can spend minutes downloading; raise it if runs queue up behind each other,
bearing in mind that each concurrent run costs the memory of one model.

**Demo datasets and mounted models (volume workflow).** `./datasets` and
`./mounted_models` are bind-mounted into both containers. Drop a `*.csv` into
`datasets/` and it appears as a **demo dataset**; drop a `BaseModel` subclass
`*.py` into `mounted_models/` and it appears as a **mounted model** option —
no upload needed. (Mounted models are gated by `ALLOW_CUSTOM_MODELS`, below.)

## Custom models and their environments

A model you upload declares what it needs in a [PEP 723](https://peps.python.org/pep-0723/)
header, and runs in an environment built from exactly that — in its own process,
under an interpreter chosen for it:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["scikit-learn", "ConfigSpace", "numpy"]
# ///

from codesigner_model import BaseModel


class MyModel(BaseModel):
    name = "My Model"

    def get_config_space(self, seed: int = 0):
        ...                      # a ConfigSpace ConfigurationSpace

    def fit_predict(self, config, X_train, y_train, X_val, seed: int = 0):
        ...                      # one predicted label per row of X_val
```

You are never asked for a score. Codesigner keeps the validation labels back,
calls `fit_predict`, and computes every metric itself — so all models are
measured by the same code regardless of what they were built with.

The environment is resolved and locked **once**, when the experiment is created;
the page shows progress while that happens, and every later run of that
experiment uses the same pinned dependencies. Building it is also when the model
is first imported, so a file that does not work is reported there.

This needs [uv](https://docs.astral.sh/uv/). Without it a model is imported into
the application's own environment instead — which is what happened before any of
this existed, so a local install keeps working — and the experiment page says so.
A hosted instance (`REQUIRE_LOGIN=True`) refuses rather than falling back.

uv's cache of built environments can get large; one model needing torch is a few
gigabytes. In Docker it lives on its own `uv-cache` volume, separate from `data`
so it can be deleted safely. Reclaim space with:

```bash
python manage.py prune_model_envs
```

## Hosting it for other people

By default there are no accounts. Codesigner is run locally or on a trusted
private network at least as often as it is hosted, and in those cases nothing
about permissions should be in the way — so the default is no login, no users,
no ownership.

One variable changes that:

```bash
REQUIRE_LOGIN=True
```

Every page then requires a signed-in user. Two things stay outside the wall, and
only two: `/healthz/` (the container runtime has no session, and a healthcheck
that redirects to a login page reports a healthy instance as down) and the
language switcher (the login page carries it, and choosing a language you can
read should not require signing in first).

Accounts are created in the Django admin — there is no self-registration, which
is the right default for an instance hosted for a known set of people:

```bash
python manage.py createsuperuser     # then add the rest at /admin/
```

Password reset is not wired up; it needs a mail server, which is an operator
decision. Until it is asked for, an operator resets a password in the admin.

### Who sees what

With accounts, an experiment belongs to whoever created it. There are three
kinds:

| | Read | Run / edit / delete | Export |
|---|---|---|---|
| **Yours** | ✅ | ✅ | ✅ |
| **Shared with you** | ✅ | ❌ | ✅ |
| **Nobody's** (`owner` is empty) | ✅ | ✅ | ✅ |

Sharing is an invitation to look, not a transfer of control — a colleague can
read and download a shared experiment, and cannot run, rename or delete it. The
owner turns sharing on with a checkbox on the experiment page.

"Nobody's" is every experiment that existed before the instance had accounts.
They stay fully usable rather than disappearing when you flip the switch; assign
them owners in the admin if you want the normal rules to apply. Staff see and
can act on everything. Deleting a user does **not** delete their experiments —
they become nobody's.

Exported `.ihpo` files never carry server paths, whether or not this instance
has accounts: the paths name a machine that is not the recipient's, and they
describe how the instance is laid out.

### Other effects of the switch

Media files are no longer served from the app (`MEDIA_ROOT` is one flat
directory, so that would hand every signed-in user every other user's data at a
guessable URL), and a model that would run in the application's own process is
refused rather than falling back — see below.

## Custom / mounted models — trust model ⚠️

Beyond the built-in models, you can **upload** a model `.py` (a
`core.models.BaseModel` subclass) or pick one from the server-side
`mounted_models/` directory. **Loading either executes it** — arbitrary Python
running on the server, by design.

This is gated by `ALLOW_CUSTOM_MODELS` (env var), default **on** for local
single-user use. **Turn it off on any shared or public deployment:**

```bash
ALLOW_CUSTOM_MODELS=False
```

With it off, the upload field and mounted-model dropdown disappear, model files
in imported `.ihpo` experiments are not adopted, and a custom-model experiment
loads read-only. There is no sandboxing — the flag is the boundary. With the
task queue, this code executes in the **worker** process, not the web process.

**Model environments do not change this.** Giving a model its own environment
buys *dependency* isolation: it cannot be broken by, or break, what the
application has installed. It is not a security boundary — the model still runs
as the same user, with the same filesystem and the same network access. `uv`
solves "your model needs a library we don't have", not "your model is hostile".
Real sandboxing is separate work that has not been done.

## Tests

```bash
python -m pytest -m "not slow"     # fast suite
python -m pytest                    # includes slow SMAC end-to-end tests
```

The i18n catalogs are checked by `tests/ui/i18n` (every marked string must have a
complete, non-fuzzy de/es translation).
