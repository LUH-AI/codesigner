# Codesigner

This is the prototype for a tool for iteratively running HPO, displaying analytics and interpretability metrics, and interactively manipulating aspects of the process i.e. Human-centered AutoML.  It is has many of the same features of deepCAVE and can be seen in some respects as a successor.

**NOTE:** This is still in its early phases of development.  Anticipate bugs.  Documentation to come.

## Running locally

Install once:

```bash
pip install -r requirements.txt
pip install -e .                   # registers the app version (pyproject.toml)
pip install -e ./model_sdk         # the model contract (core.models imports it)
cp .env.example .env               # set SECRET_KEY
```

Then `./run.sh`, which takes one argument per configuration — the same shape as
`docker-entrypoint.sh`, which dispatches the container's two roles:

```bash
./run.sh                    # transparent: no accounts, runs execute in-process
./run.sh auth               # the login wall, ownership and groups
./run.sh auth --demo        # ... plus a worked instance to sign in to
./run.sh queue              # the real queue instead of in-process runs
./run.sh docker             # web + worker in containers
./run.sh docker --hosted    # ... over Redis and Postgres (configuration C)
```

Every bare-metal mode applies migrations and clears runs orphaned by a previous
hard kill before starting. Both are idempotent, fast, and only ever noticed by
their absence — a missing column, or a row stuck at `running`. Trailing
arguments reach `runserver`, so `./run.sh 8001` moves the port.

`REQUIRE_LOGIN` arrives as a process variable rather than through `.env` on
purpose: `config/settings.py` reads `.env`, so a value left there is inherited
by the test suite and every account-agnostic test silently becomes a login-wall
test.

By default (`DEBUG=True`) there is no consumer: runs execute in-process on a
background thread, and the page comes back as soon as you press Run rather than
waiting out the optimization. `./run.sh queue` switches to the real queue, which
needs a consumer in a second terminal — the script prints the line.

By hand, if you would rather not use the script:

```bash
python manage.py migrate
python manage.py sweep_stale_runs                # after a hard kill
python manage.py runserver
REQUIRE_LOGIN=True python manage.py runserver    # with accounts
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
