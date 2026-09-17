# Codesigner

An interactive hyperparameter-optimization workbench: set up an experiment
(model, optimizer, dataset), run trials in the background, and explore the
results — best/selected configuration, hyperparameter importance, and the
incumbent's performance over trials. Experiments save to a portable `.ihpo`
file. A Django rebuild of the InteractiveHPO Streamlit app.

The interface is in English. German and Spanish catalogs exist under `locale/`
but are shelved while the interface is still moving: everything in them that did
not come from the InteractiveHPO original is an unreviewed draft, marked fuzzy
and not compiled, because a wrong translation is worse than an English one. The
strings stay marked for translation throughout, so re-enabling a language is
`LANGUAGES` in `config/settings.py` plus `compilemessages`, once someone who
speaks it has read the drafts.

## The .ihpo file

An experiment exports to one JSON file that carries enough to recreate it: the
seed, the dataset and model it ran on, how a trial was evaluated, every
optimizer setting, the trials themselves, and the history of runs that produced
them. It is a superset of SMAC's own output — `result` mirrors the runhistory
and embeds `scenario.json`, `intensifier.json`, `optimization.json` and
`configspace.json` verbatim.

One object per subject, each stating its subject once, and `result` last:

```jsonc
{ "format": 2, "version", "name", "seed",
  "dataset":     { "filename", "sha256", "rows", "columns", "column_names", … },
  "model":       { "kind", "name", "sha256", "dependencies", … },
  "evaluation":  { "scheme", "folds", "test_size", "stratified" },
  "metrics":     { "names", "current", "original" },
  "optimizer":   { "name", "params", "defaults_used" },
  "runs":        [ … ],
  "environment": { "codesigner", "python", "packages" },
  "result":      { … } }
```

Files written before `format` existed spelled all of this as one flat namespace;
they are lifted on the way in, so they still open. Files written now do not open
in a build from before this.

The dataset and any custom model are recorded by **SHA-256, not embedded**, so
the file stays a record rather than an archive. Importing with a dataset whose
digest disagrees is refused: trials measured on different data cannot be
compared with each other, and nothing downstream would notice. Importing with
*no* dataset is fine — the experiment loads browsable and unrunnable, and the
check happens when one is attached. Files exported before fingerprints existed
have nothing to disagree with and still open.

See [docs/walkthroughs/ihpo-provenance.md](docs/walkthroughs/ihpo-provenance.md).

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

**How a trial is evaluated** is chosen when the experiment is created: one 80/20
holdout, or k-fold cross-validation. It cannot change afterwards, because trials
evaluated different ways cannot be compared with each other, and an experiment's
own history has to be. Cross-validation costs k fits per trial and is the
default at five folds: the tables this is pointed at are small, and a single
split on a small table is noisy enough that a search can spend its budget
chasing the split rather than the model.

One consequence worth knowing if you are relying on the isolation. With a single
holdout the model is never sent a validation label at all. With k folds every
row trains in k−1 of them, so across one trial the union of what the model
receives is every label — it is never told which rows it is about to be scored
on, but a model deliberately caching what it was sent could reconstruct them.
Closing that would mean one process per fold, k times the memory for the whole
run. Like the rest of this: dependency isolation, not a sandbox.

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
language route (the login page carries the switcher when more than one language
is offered, and choosing a language you can read should not require signing in
first).

Accounts are created in the Django admin — there is no self-registration, which
is the right default for an instance hosted for a known set of people:

```bash
python manage.py createsuperuser     # then add the rest at /admin/
```

Password reset is not wired up; it needs a mail server, which is an operator
decision. Until it is asked for, an operator resets a password in the admin. A
signed-in user *can* change their own password — Settings → Account → Change
password — which needs no mail server because it asks for the current one.

That Account page is also where somebody can see what they are allowed to do
here, which is worth having now that the answer comes from four separate grants.

### Groups and the three roles

An instance hosted for several research groups exists so that the groups do
**not** see each other. An experiment's group is its owner's; nothing moves one
between groups except changing who owns it.

| | |
| --- | --- |
| **Member** | creates and runs experiments, and chooses for each whether it is shared with their group, with named colleagues, or with nobody |
| **Group lead** | a member who also manages their group's people, and can see and act on anything in it |
| **Site admin** | manages groups — their size, whether they are active, what they are using — and is shown no experiments at all |

What each can reach:

| | own | shared with the group | shared by name | the rest of their group | another group |
| --- | --- | --- | --- | --- | --- |
| Member | act | view | view | — | — |
| Group lead | act | act | act | act, in a panel of its own | — |
| Site admin | — | — | — | — | — |

Sharing **by name** deliberately crosses the group boundary: an invitation the
owner made person by person is an act of judgement rather than a property of a
group. Sharing with the *group* does not.

A group lead's own experiments stay on their Experiments tab; their colleagues'
are under **Group → Their experiments**. Being a lead should not quietly turn
the everyday list into everyone's.

#### Making people

```bash
python manage.py create_site_admin alice              # + a group of their own
python manage.py create_site_admin alice --users 0    # no group at all
```

`--users` is the seat limit of the group created for them, defaulting to 1 — a
site admin with somewhere to put an experiment of their own and no room to grow
a group underneath themselves by accident. **0 creates no group**, which is the
cleaner reading of the role.

Note it does **not** make them a Django superuser. A superuser can read every
experiment through `/admin/`, which is exactly what this role is meant not to
do; pass `--superuser` only if you want both.

After that, a group lead creates their own group's accounts under **Group →
People**, up to the seat limit a site admin set. There is still no registration
surface anywhere and no mail server.

#### The permissions underneath

`is_staff` is Django's flag for reaching `/admin/`, and it means only that.
Roles carry what codesigner asks for, and two permissions sit outside them:

| permission | grants |
| --- | --- |
| `access.manage_site` | the Site tab — groups, usage, jobs. This *is* being a site admin. |
| `access.use_custom_models` | may upload and run custom models — arbitrary code execution |
| `access.change_defaults` | may change the settings every inheriting experiment follows |
| `access.view_all_experiments` | **escape hatch.** Every experiment, across every group. |
| `access.manage_experiments` | **escape hatch.** Act on every experiment, across every group. |

The last two cut straight through the boundary everything else here draws. They
are granted to nobody, and are meant to be handed out for a support case and
taken back — not to describe a role. A group lead gets the same reach *inside
their own group* from their membership, which is the ordinary way.

`ALLOW_CUSTOM_MODELS=False` remains the floor under custom models: off means
off, for everyone.

#### An instance to look at

```bash
python manage.py seed_demo          # --reset to rebuild
```

Builds a site admin, two groups at different seat limits, a lead and members in
each, experiments at all three sharing levels, finished runs for the usage
totals, and one job left running so Stop has a target. Every password is the
username, so it refuses to run unless `DEBUG` is on.

### When somebody leaves

**Deactivate them; do not delete them.** Clear "Active" in the admin: they can no
longer sign in, and every experiment stays owned by the person who made it.

Deleting the account instead would set its experiments' owner to null, and an
ownerless experiment on this instance is *everyone's* — visible, runnable,
editable and deletable by any signed-in account. That rule exists for the
experiments that predate accounts, where there is no owner whose wishes are being
overridden; applied to somebody's unpublished work it is a quiet leak. The admin
therefore refuses to delete a user who still owns experiments, and says so on
their account page. Reassign the experiments first if the account really must go.

### TLS

Another variable, independent of `REQUIRE_LOGIN`:

```bash
SECURE_BEHIND_TLS=True
```

Off (the default) matches how this runs locally and in the reference
`docker-compose.yml` — plain HTTP, no reverse proxy. Turn it on only once
there is a TLS-terminating reverse proxy in front of the instance; it then
redirects HTTP to HTTPS, marks the session and CSRF cookies secure-only, and
enables HSTS. Turning it on without a proxy in front breaks the instance —
there is nothing to answer the HTTPS redirect.

Behind a proxy you also need the public origin named, or every form post is
refused as a CSRF failure — which reads like a broken page rather than a
missing setting:

```bash
CSRF_TRUSTED_ORIGINS=https://codesigner.example.org
```

It defaults to `https://` each `ALLOWED_HOSTS` entry, so an instance whose hosts
are already correct usually needs nothing here.

### Where the web server and the worker run

Runs execute in a separate `manage.py run_huey` consumer, not in the web
process. Whether the two can be on **different machines** is decided by three
settings, and by nothing else — each defaults to a file on local disk, which is
what ties them to one box:

| | one machine (default) | separate |
| --- | --- | --- |
| queue | `huey.SqliteHuey` — a file | `HUEY_CLASS=huey.RedisHuey` + `REDIS_URL` |
| database | SQLite — a file | `DATABASE_URL=postgres://…` |
| uploads | `MEDIA_ROOT` — a directory | `DEFAULT_FILE_STORAGE` → object storage |

This is the ordinary way to separate a web server from its background work — a
task queue over a network broker, a shared database, shared file storage — and
it needs no code, only the three drivers:

```bash
pip install -r requirements.txt -r requirements-hosted.txt
```

`docker-compose.hosted.yml` is a worked example, layered on the base file:

```bash
docker compose -f docker-compose.yml -f docker-compose.hosted.yml up
```

It adds Redis and Postgres, turns `REQUIRE_LOGIN` on and `ALLOW_CUSTOM_MODELS`
off. It does **not** turn on `SECURE_BEHIND_TLS`, for the reason above: there is
no proxy in it to answer an HTTPS redirect.

Two things do not travel over the broker and still need shared storage when the
processes are split: uploaded datasets and uploaded model `.py` files. Either
give both processes the same mount, or set `DEFAULT_FILE_STORAGE`. A worker that
cannot read a dataset fails the run rather than corrupting anything, but it
fails every run.

### What sharing actually grants

Who can reach an experiment is the table under *Groups and the three roles*
above. What they can then *do* with it is this:

| | Read | Run / edit / delete | Export |
|---|---|---|---|
| **Yours** | ✅ | ✅ | ✅ |
| **Shared with you**, by the group or by name | ✅ | ❌ | ✅ |
| **Anything in your group**, if you are its lead | ✅ | ✅ | ✅ |

Sharing is an invitation to look, not a transfer of control — a colleague can
read and download a shared experiment, and cannot run, rename or delete it. An
owner's results should not change because somebody else pressed Run. A group
lead is the exception, and deliberately: stopping a run that is going wrong
should not need its owner to be awake.

**An experiment with no owner is reachable by nobody.** It used to be
everyone's, which was right while the instance was one flat pool of accounts —
the only ownerless experiments were the ones predating them. With groups the
same rule is a leak by construction, so it is gone, and the migration that
introduced groups gave the existing ones an owner. A row can still end up
ownerless if an account is deleted, which is why the admin refuses to delete one
that owns anything — see *When somebody leaves*.

Exported `.ihpo` files never carry server paths, whether or not this instance
has accounts: the paths name a machine that is not the recipient's, and they
describe how the instance is laid out.

### Who may upload a model

Uploading a model is arbitrary code execution, so on a hosted instance
`ALLOW_CUSTOM_MODELS` alone is too blunt — it means every account or none. A
per-account permission sits on top of it:

> **Access permissions | Can upload and run custom models** — grant it in the
> admin, per user or via a group. It is the oldest of the permissions in
> *Groups and the three roles* above: the rest were split out of `is_staff`
> later, following the pattern this one set.

Without it the upload field and the mounted-model dropdown do not appear, an
imported `.ihpo`'s model file is not attached, and — the check that actually
matters — the worker refuses to run the model and says whose account was
refused. That last one is where the decision is made, because a run is started
by a background task rather than by the request that rendered a form.

`ALLOW_CUSTOM_MODELS=False` still outranks the permission: off means off for
everyone, so an operator turning custom models off never has to audit who holds
what. Being staff does **not** confer the permission — `is_staff` means "can use
the admin", not "trusted to run arbitrary code" — though a superuser has every
permission by definition.

### Who may change the defaults

The **default experiment settings** apply to every experiment that inherits
them, so one person changing them changes what everyone's pages draw. On a
hosted instance that page, and the "Save settings as default" button on an
experiment's own settings page, need `access.change_defaults` (see *Groups and
the three roles* above); without accounts both are open.

This used to be staff-only, which meant anyone who could reach `/admin/` could
also change what every page on the instance draws — two powers that have no
reason to arrive together.

### Other effects of the switch

Media files are no longer served from the app (`MEDIA_ROOT` is one flat
directory, so that would hand every signed-in user every other user's data at a
guessable URL), and a model that would run in the application's own process is
refused rather than falling back — see below.

## Running the optimizations on a cluster

A run does not have to execute where the web server is. With `RUN_BACKEND=slurm`
the consumer stages the experiment onto a cluster, submits a Slurm job, follows
it, and writes the result back — the page, the run history and the export cannot
tell the difference.

This is a separate choice from where the *consumer* runs. A consumer on your own
machine can submit to a cluster, and a consumer on another machine can run
in-process; the two compose.

First put the run half on the cluster:

```bash
CLUSTER_HOST=kisski ./cluster/deploy.sh
```

That copies `core/`, `model_sdk/` and `cluster/` — not `ui/`, not `config/`, no
database and no media — and builds a virtualenv there with
`cluster/requirements-cluster.txt`. Six packages: `core/` imports no Django, so
the cluster runs no web stack and needs neither a database nor a `MEDIA_ROOT`.
It is idempotent, and safe to run while jobs are in flight.

Then point the consumer at it:

```bash
RUN_BACKEND=slurm
CLUSTER_HOST=kisski            # a name ssh already understands
CLUSTER_ROOT=codesigner        # where deploy.sh put it
CLUSTER_PARTITION=kisski-inference
```

The host is resolved by `ssh` itself, so a key, a user and any `ProxyJump` stay
in `~/.ssh/config` where the rest of the system can see them too.

**What crosses, per run.** A directory on the cluster's filesystem holding the
experiment's snapshot, the dataset it was measured on, and what bounds the run;
the job writes its result back into the same directory. `cluster/job.sbatch` is
the script it becomes — edit that to change how work is submitted.

**While it runs**, the job rewrites a partial result every few seconds and the
consumer copies it into the experiment, so the figures move exactly as they do
for a local run.

**Cancelling asks rather than kills**: a file the run checks between trials, so
the trials already paid for are kept. "Give up on it" is the escalation, and
that one does `scancel` — losing whatever had not been written.

Note that a cluster is not automatically faster. For the models codesigner ships
— scikit-learn on a few thousand rows — staging and queueing cost more than the
run does, and the expensive half is SMAC's own search rather than the model. The
reason to do this is models that genuinely need a cluster, and keeping long runs
off your own machine.

## Custom / mounted models — trust model ⚠️

Beyond the built-in models, you can **upload** a model `.py` (a
`core.models.BaseModel` subclass) or pick one from the server-side
`mounted_models/` directory. **Loading either executes it** — arbitrary Python
running on the server, by design.

This is gated by `ALLOW_CUSTOM_MODELS` (env var), default **on** for local
single-user use, and on a hosted instance additionally by the per-account
*Can upload and run custom models* permission (above). **Turn the flag off on
any shared or public deployment:**

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
python -m pytest -m "not slow and not uv"   # fast suite, what CI runs per push
python -m pytest -m slow                     # real SMAC searches, ~10 minutes
python -m pytest -m uv                       # real environment building
```

The slow ones are real searches. They are what checks that a search reaches its
model rather than sampling throughout, that a resumed run picks up where it left
off, and that an experiment recreated from its `.ihpo` produces the same trials
— so run them before changing anything about how a search is configured. CI runs
them weekly and on demand rather than per push.

The i18n catalogs are checked by `tests/ui/i18n`, which pins that the shelved
German and Spanish drafts stay shelved and uncompiled — not that they are
complete.
