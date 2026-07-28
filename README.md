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

**Demo datasets and mounted models (volume workflow).** `./datasets` and
`./mounted_models` are bind-mounted into both containers. Drop a `*.csv` into
`datasets/` and it appears as a **demo dataset**; drop a `BaseModel` subclass
`*.py` into `mounted_models/` and it appears as a **mounted model** option —
no upload needed. (Mounted models are gated by `ALLOW_CUSTOM_MODELS`, below.)

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

## Tests

```bash
python -m pytest -m "not slow"     # fast suite
python -m pytest                    # includes slow SMAC end-to-end tests
```

The i18n catalogs are checked by `tests/ui/i18n` (every marked string must have a
complete, non-fuzzy de/es translation).
