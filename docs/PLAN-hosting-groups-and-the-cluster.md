# Hosting codesigner: topology, a cluster, and who sees whose work

**Status: four bodies of work landed and green, all uncommitted on `main`.**
Suite at **1397 passed, 1 skipped** (from 1314 when this started). `manage.py
check` clean, no pending migrations.

This is a handoff. It records what was built, what was decided and why, what is
verified against reality versus only against tests, and what is left.

---

## Where the tree is

Branch `main`, last commit `04aa07c` (the analytics merge). Everything below is
**uncommitted** — 21 modified files, 24 new. Six unapplied-anywhere-but-here
migrations:

```
access/migrations/0002_alter_accesspermissions_options.py
access/migrations/0003_administrators_group.py          ← superseded by 0005
access/migrations/0004_group_…_membership.py
access/migrations/0005_groups_take_over.py
ui/migrations/0020_run_backend_run_job_id.py
ui/migrations/0021_experiment_shared_with.py
```

`0003` creates an `Administrators` auth group and `0005` removes it again. That
is not tidy, and it is deliberate: `0003` shipped in the middle of this work and
may already have been applied somewhere. Squashing them would be fine *if* it is
certain nobody has run `0003`; on this machine that is true.

---

## 1. SMAC is a plain dependency again

`requirements.txt` asks for `smac>=2.4.1` instead of a git URL.

**Why it is safe:** v2.4.1 *merged* the fork branch the pin existed for —
upstream PR #1325 — so the release contains commit `620e2876f` (what was
installed) by identical hash, along with `67694e9ef` "Fix random forest import
with scikit-learn 1.9", which was the actual bug. 2.4.1 is a strict superset.

`git` was dropped from the Dockerfile's apt list (it was there only to pip
install SMAC from source). `swig` and `build-essential` stay — **pyrfr still
ships no wheel**, checked.

**Not in 2.4.1:** `c3193d6` "Make the logarithmic weighting opt-in rather than
the default", from the fork branch `angrimson-output-constraints`. A
`requirements.txt` edit pointing at it was uncommitted and never installed, so
nothing has ever run with it. If that behaviour is wanted it needs its own route
upstream.

## 2. Topology is configuration

Three settings decide whether the web process and the huey consumer can be on
different machines. Each defaults to a file on local disk, which is one machine.

| | one machine (default) | separate |
| --- | --- | --- |
| queue | `huey.SqliteHuey` | `HUEY_CLASS=huey.RedisHuey` + `REDIS_URL` |
| database | SQLite | `DATABASE_URL=postgres://…` (already worked) |
| uploads | `MEDIA_ROOT` | `DEFAULT_FILE_STORAGE` |

The broker sets **either** `filename` **or** `url`, never both, so a
misconfigured Redis fails at startup rather than quietly enqueuing into a local
file the other machine cannot see — which would look like a hung worker rather
than a wrong setting.

`CSRF_TRUSTED_ORIGINS` was genuinely missing and would have refused every form
post on a hosted instance. It defaults to `https://` each `ALLOWED_HOSTS` entry
and correctly contributes nothing for `*`.

New: `requirements-hosted.txt` (redis, psycopg, django-storages — kept out of
the default install), `docker-compose.hosted.yml`.

**Unverified:** the Redis/Postgres stack has never actually been brought up.
`docker compose config` validates and the settings are unit-tested, but there is
no Docker daemon on this machine. That is the largest untested surface here.

## 3. Runs can execute on KISSKI

`RUN_BACKEND=slurm`. The consumer stages the experiment onto the cluster,
submits a Slurm job, follows it, and writes back the same rows a local run
writes. **Verified end to end against the real cluster**, several times.

Four things made it small, and all were checked rather than assumed:

1. The `.ihpo` snapshot is already a complete self-contained experiment —
   `snapshot_from_experiment` → `io.build_experiment` is an existing round trip.
2. **`core/` imports no Django.** Its third-party needs are six packages, so the
   cluster runs no web stack and has no database and no `MEDIA_ROOT`.
3. The three places a run touches the database are already *injected* — the
   `progress` callback, the `cancel_event`, and the final write — so the
   optimizer is called exactly as it is at home.
4. Live figures needed **no page changes**: `run_status` already polls and
   `poll.js` already redraws from `Experiment.result`.

```
cluster/requirements-cluster.txt   six packages
cluster/deploy.sh                  rsync core/ model_sdk/ cluster/ + uv venv
cluster/job.sbatch                 the job, with @@MARKERS@@
cluster/run_experiment.py          the remote runner: no Django, no DB
ui/services/cluster.py             ssh/sbatch/squeue/sacct, behind a transport seam
```

**Measured on the cluster** (`kisski01.cluster.uni-hannover.de`, login node
`svc2.kisski`): Slurm, partition `kisski-inference` (9 nodes × 4×H100, 3-day
limit), `/mnt/home` is NFS and shared with compute nodes, `uv` installed on both
login *and* compute nodes, and **compute nodes have PyPI egress** — so custom
model environments can be built on the node. The deployed copy lives at
`~/codesigner` and is current as of this writing.

**Proof it is the same computation:** a 12-trial wine run at seed 11 produced
bit-identical trials, configs and `best_score` on the desktop and on `gpu002`.
Cancel mid-run on a 200-trial job stopped in 8s and kept all 43 finished trials.

**Not done:** nothing prunes `~/codesigner/runs/run-N` on the cluster. One
directory per run, each holding a copy of the dataset. Needs a sweep command
before real use.

## 4. Groups, roles and user management

**Member / Group lead / Site admin** — named so only one says "admin".

| | |
| --- | --- |
| **Member** | creates and runs experiments; shares each with their group, with named people, or with nobody |
| **Group lead** | a member who also manages their group's people and can act on anything in it |
| **Site admin** | manages groups — size, active, usage, jobs — and is shown no experiments |

Models: `access.Group` (name, `user_limit`, `is_active`) and `access.Membership`
(user OneToOne, group FK, role). One group per person, because "which group is
this experiment in" has to have one answer — an experiment's group is its
owner's, and nothing moves it except changing the owner.

`GroupPolicy` (in `access/policy.py`) replaces `OwnerPolicy`, which is kept as an
alias so a pinned `EXPERIMENT_POLICY` still resolves. It has three querysets and
the distinction between them matters:

- `experiments()` — what may be **reached**. Authorization.
- `for_listing()` — what a page **lists**. Presentation.
- `group_experiments()` — a lead's colleagues', for their panel.

That split is what keeps a lead's Experiments tab from silently becoming
everyone's. `visible_experiments()` uses `for_listing`.

Sharing has two axes: `Experiment.shared` now means *my group* (it used to mean
everyone signed in, which was the same thing while the instance was one flat
pool), and `Experiment.shared_with` is a M2M of named people. **A named share
deliberately crosses the group boundary** — an invitation made person by person
is the owner's judgement, not a property of a group.

Surfaces: `ui/panels.py` plus `ui/templates/ui/panels/`, with rail tabs that
appear per role. Site panels show job id, elapsed, trials, who started it, their
email and group, and the experiment's opaque `identifier` — **no name, model,
metric or score**, asserted as an absence in the tests.

Also here: named permissions replacing `is_staff`'s overload, self-service
password change, an Account page saying what you may do, and an admin that
refuses to delete a user who owns experiments.

```
manage.py create_site_admin alice [--users N]   # 0 = no group; not a superuser
manage.py seed_demo [--reset]                   # a worked instance to sign in to
```

`seed_demo` builds a site admin, `vision-lab` (5 seats) and `nlp-group` (3), a
lead and members in each, experiments at all three sharing levels, finished runs
for the usage totals, and one job left running so Stop has a target. **Every
password is the username**, so it refuses unless `DEBUG` is on.

---

## Decisions worth not re-litigating

**A site admin's blindness is a UI decision, not a security boundary.** Anyone
with `/admin/` or database access can read the `Experiment` table. That is why
`create_site_admin` does *not* imply superuser or `is_staff`: a site admin who
should genuinely not read colleagues' work must have `manage_site` alone. Do not
"fix" this by making the role a superuser.

**`view_all_experiments` and `manage_experiments` are an escape hatch, not a
role.** They are instance-wide and cut straight through the group boundary.
Granted to nobody; meant to be handed out for a support case and taken back.

**`createsuperuser` is not overridden.** `django.contrib.auth` precedes `access`
in `INSTALLED_APPS` and `get_commands()` resolves so the earlier app wins — an
override would need `INSTALLED_APPS` reordered, which is a strange thing to do
to one list for one command.

**Ownerless means nobody's now.** It used to mean *everyone's*, which was right
while the only ownerless experiments predated accounts. With groups the same
rule is a leak by construction. `0005` gave the existing ones an owner.

**The URLconf audit was taught, not loosened.** `tests/ui/test_permissions.py`
has a `NOT_EXPERIMENTS` dict naming the two `<int:pk>` routes whose pk is not an
experiment (`group_remove_person`, `site_job_stop`), with a test that the list
cannot rot. A new `<int:pk>` route is still guilty until declared.

**One Slurm job per run, not a Dask worker pool.** SMAC 2.4.1 accepts a
`dask_client`, so fanning trials out to a Slurm-backed Dask cluster looks free.
It is not: codesigner drives SMAC in **ask/tell** mode with a deliberately
unreachable target function (`core/optimizers/smac_optimizer.py:868`), so SMAC's
`DaskParallelRunner` never executes anything. Parallelism would have to be
codesigner's own seam around its own loop. The evaluation call is kept
Executor-shaped so that stays possible.

---

## Gotchas found the hard way

**`.env` was silently changing what the tests tested.** `config/settings.py`
reads `.env`, so `REQUIRE_LOGIN=True` and `RUN_BACKEND=slurm` set for manual
testing leaked into the suite — every test that did not ask for accounts became
a login-wall test, and the run tests would have submitted **real Slurm jobs**.
`tests/conftest.py` now pins both. One subprocess-based settings probe re-reads
`.env` and cannot be reached that way, so it pins its own environment.

**`create_permissions` in a data migration.** `Permission` rows are created by a
`post_migrate` handler, so rows for permissions declared one migration ago do
not exist yet. The migration must call `create_permissions` itself — and must
hand it the **live** app registry, because the migration's historical app
configs are stubs with no `models_module`, which is what it checks first.

**Django's bulk delete is all-or-nothing.** `delete_selected` asks
`has_delete_permission` per object and refuses the whole selection if any is
refused, *before* `delete_queryset` runs. An override there is dead code.

**`AccessPermissions` is `managed = False`** — no table — so registering it in
the admin gives a changelist that errors.

**`scatter3d` rejects an array `marker.line.width`**, and an invalid attribute
takes the whole trace down. Relevant to the figures, found while fixing failure
marks; the cube drops `marker.line` for 3D.

---

## What is left

1. **Bring up configuration C for real.** Redis + Postgres via
   `docker-compose.hosted.yml`. Never run; needs a Docker host.
2. **Look at the group and site panels in a browser.** They are tested but have
   not been seen. `seed_demo` then sign in as `vera` (lead), `ana` (member),
   `site` (site admin).
3. **Prune cluster run directories.** One per run, each with a dataset copy.
4. **Commit.** Four coherent commits would be: the SMAC pin; topology; the
   cluster backend; groups and roles. Possibly squash `access/0003` into `0005`
   first — safe only if `0003` has not been applied anywhere else.
5. `docs/PLAN-scoring-and-smac-import.md` stages 5, 6 and half of 8 remain —
   pre-run metric usability, `auc`, and registering `rmse`/`logloss`. All
   *running*-side work, none reachable from an import.

## Running it

```bash
# transparent, everything local — the default, and what the suite runs under
python manage.py runserver

# local web, jobs on KISSKI  (.env)
RUN_BACKEND=slurm
CLUSTER_HOST=kisski
CLUSTER_ROOT=codesigner
CLUSTER_PARTITION=kisski-inference
HUEY_IMMEDIATE=false        # else the run dies with runserver's autoreload
#   terminal 1: manage.py runserver
#   terminal 2: manage.py run_huey
#   and ./cluster/deploy.sh after anything under core/

# accounts, groups and roles  (.env)
REQUIRE_LOGIN=True
#   manage.py create_site_admin alice
#   manage.py seed_demo
```

Full suite is ~26 minutes. **Do not edit the tree while it runs** — `harness.py`
is read per subprocess while `client.py` is already imported, so a mid-edit
collection reports failures that do not exist.
