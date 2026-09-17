"""
Django settings for the codesigner project.

Secrets and machine-specific values come from environment variables (or a
.env file at the repo root), read via django-environ.  See .env.example for
the full list.  Everything else in here is plain Django configuration.

https://docs.djangoproject.com/en/6.0/topics/settings/
"""

from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    # type, default — defaults are for local development only
    DEBUG=(bool, True),
)
# Values already present in the process environment win over .env entries.
environ.Env.read_env(BASE_DIR / ".env")

SECRET_KEY = env("SECRET_KEY")

DEBUG = env("DEBUG")

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["localhost", "127.0.0.1"])

# Whether this instance has accounts. Off is the primary case: run it locally
# or on a trusted private network and there is no login and no per-user
# anything. On, it is being hosted for several people — see the `access`
# package, and the README's hosting section.
REQUIRE_LOGIN = env.bool("REQUIRE_LOGIN", default=False)

# When on, users may upload a model .py that the app imports and executes to
# run experiments. That is arbitrary code execution by design, so it must be
# OFF on any shared or public deployment (see README).
ALLOW_CUSTOM_MODELS = env.bool("ALLOW_CUSTOM_MODELS", default=True)

# Who may see and do what to an experiment. The default enforces groups and
# ownership when REQUIRE_LOGIN is on and says "everyone, everything" when it is
# off, so that stays the only switch. Point this at ui.permissions.OpenPolicy to
# drop the boundary on an instance that has accounts, or at a policy of your own.
EXPERIMENT_POLICY = env.str("EXPERIMENT_POLICY", default="access.policy.GroupPolicy")

# Where the login wall sends an unauthenticated request, and where signing in
# and out land. Set even when REQUIRE_LOGIN is off, so turning it on is one
# variable rather than a checklist.
LOGIN_URL = "access:login"
LOGIN_REDIRECT_URL = "ui:home"
LOGOUT_REDIRECT_URL = "access:login"

# ── Model environments ────────────────────────────────────────────────────────
# A user's model declares its own dependencies with a PEP 723 header and runs in
# an environment built from them, in its own process. uv builds and caches those
# environments. Without uv the model is imported into this process instead —
# which is what happened before any of this existed, so a local install keeps
# working — but a hosted instance refuses rather than quietly doing that.
UV_BIN = env.str("UV_BIN", default="uv")

# The model contract, installed into every model environment. A built wheel is
# preferred: handing uv the source directory makes it fetch a build backend,
# which fails on a machine with no index access.
MODEL_SDK_WHEEL = env.str("MODEL_SDK_WHEEL", default=str(BASE_DIR / "wheels" / "sdk"))
MODEL_SDK_PATH = env.str("MODEL_SDK_PATH", default=str(BASE_DIR / "model_sdk"))

# Resolve against the cache only, and refuse to fetch an interpreter. Both
# belong on in an air-gapped or locked-down deployment.
MODEL_ENV_OFFLINE = env.bool("MODEL_ENV_OFFLINE", default=False)
MODEL_ENV_PYTHON_DOWNLOADS = env.bool("MODEL_ENV_PYTHON_DOWNLOADS", default=True)

# Preparing an environment can mean a large download, so this is generous. The
# per-trial and startup limits live in core.modelhost, which owns the process.
MODEL_ENV_PREPARE_TIMEOUT = env.float("MODEL_ENV_PREPARE_TIMEOUT", default=900.0)
MODEL_TRIAL_TIMEOUT = env.float("MODEL_TRIAL_TIMEOUT", default=600.0)

# The largest file a model may write. Not a memory limit — see core.modelhost.
MODEL_MAX_FILE_BYTES = env.int("MODEL_MAX_FILE_BYTES", default=1024 * 1024 * 1024)

# The largest dataset, model .py, or .ihpo a browser may upload — a different
# thing from MODEL_MAX_FILE_BYTES above, which bounds what a *running* model
# writes, not what a person hands the form. See ui/validators.py.
MAX_UPLOAD_BYTES = env.int("MAX_UPLOAD_BYTES", default=100 * 1024 * 1024)

# How much the importance analytics computed at run completion may spend, in
# shapiq coalition evaluations (2^hyperparameters x 3 games x metrics). Past it
# they are skipped and every metric carries the reason, which the experiment page
# already renders. A capacity limit for this machine, not a per-experiment
# preference — hence a deployment setting rather than a checkbox.
#
# Cost is 15-19ms per coalition and **exponential in hyperparameter count**,
# while being independent of trial count: a short run of a wide model is the
# expensive case. Measured over 4 metrics: 4 hyperparameters = 192 coalitions =
# 3.6s; 6 = 768 = 12.7s; 8 = 3072 = 45.6s. 10 = 12288 would be around 3 minutes.
#
# The 1024 default admits both registry models (Random Forest 4, SVM 6) and turns
# away the custom-upload pathology. Raise it if you routinely tune wide models and
# don't mind the wait; set it to 0 for no limit. See
# core.optimizers.base.BaseOptimizer.eager_analytics_budget_exceeded.
ANALYTICS_EAGER_MAX_COALITIONS = env.int("ANALYTICS_EAGER_MAX_COALITIONS", default=1024)

# Never passed to a model's process. The rest of the environment is inherited,
# because proxy, certificate and index settings are numerous and operator-
# specific; these are the ones that would matter if they leaked.
MODEL_ENV_DENYLIST = (
    "SECRET_KEY", "DATABASE_URL", "DJANGO_SETTINGS_MODULE", "HUEY_FILENAME",
    "MEDIA_ROOT", "ALLOWED_HOSTS", "REQUIRE_LOGIN", "ALLOW_CUSTOM_MODELS",
)

# ── Where a run executes ─────────────────────────────────────────────────────
# `local` runs the optimization in the consumer process, which is what every
# install has always done. `slurm` hands it to a cluster: the consumer stages the
# experiment, submits a job, watches it, and brings the result back. The web
# process is unaffected either way — it only ever enqueues.
#
# This is a different axis from where the *consumer* runs (the queue and database
# settings below decide that). A consumer on this machine can submit to a
# cluster, and a consumer on another machine can run in-process; the two choices
# compose.
RUN_BACKEND = env.str("RUN_BACKEND", default="local")

# How to reach the cluster, when RUN_BACKEND is `slurm`. The host is a name
# `ssh` already understands, so a ProxyJump, a key and a user stay in
# ~/.ssh/config where the rest of the system can see them too.
CLUSTER_HOST = env.str("CLUSTER_HOST", default="")
CLUSTER_ROOT = env.str("CLUSTER_ROOT", default="codesigner")
CLUSTER_PARTITION = env.str("CLUSTER_PARTITION", default="")
# Cores asked for, and told to the thread pools inside the job — see
# cluster/job.sbatch for why the second half matters.
CLUSTER_CPUS = env.int("CLUSTER_CPUS", default=8)
# The wall-clock ceiling on a job, in hours, when the run's own criteria do not
# imply a shorter one. A queue will refuse a job asking for longer than its
# partition allows, so this is a number an operator matches to their cluster.
CLUSTER_MAX_HOURS = env.int("CLUSTER_MAX_HOURS", default=24)
# How often the consumer asks the cluster what is happening. Each poll is an ssh
# round trip, and it is also how often a page's figures can move, so it trades
# liveness against load on the login node.
CLUSTER_POLL_SECONDS = env.float("CLUSTER_POLL_SECONDS", default=5.0)

# Background task queue. Runs execute in a separate `manage.py run_huey`
# consumer process; run state lives in the database, so the web process only
# enqueues (polling and cancellation are DB-based and unaffected). `immediate`
# runs tasks inline instead of via the consumer; it defaults to DEBUG so a lone
# `runserver` works in development, and is off in production where the consumer
# runs.
#
# **Which broker is the topology decision.** SqliteHuey is a file, so the
# consumer has to share a filesystem with the web process — one machine, which
# is the default and what development wants. RedisHuey is a network service,
# which is what lets the two run on different machines. That is the whole of the
# difference: a task queue over a network broker is the standard way to separate
# a web server from its background work, and it needs no code here beyond naming
# the class. The database has the same shape of switch already (`DATABASE_URL`).
#
# The consumer defaults to a single worker, which would mean one queue for two
# very different jobs: an optimization, and building a model's environment. The
# second can take minutes of downloading, and while it held the only worker
# nothing on the instance could run. Both spend nearly all their time waiting on
# a subprocess, so threads are the right kind of worker; the count is what needs
# to be greater than one.
HUEY_CLASS = env.str("HUEY_CLASS", default="huey.SqliteHuey")
HUEY = {
    "huey_class": HUEY_CLASS,
    "name": env.str("HUEY_NAME", default="codesigner"),
    "immediate": env.bool("HUEY_IMMEDIATE", default=DEBUG),
    "results": False,
    "consumer": {
        "workers": env.int("HUEY_WORKERS", default=4),
        "worker_type": "thread",
    },
}

# The broker's own argument, which differs by class: a path for SqliteHuey, a
# URL for RedisHuey. Only the one belonging to the chosen class is set, so a
# misconfigured Redis fails at startup rather than quietly falling back to a
# local file that the other machine cannot see — which would look like a queue
# that accepts work and never runs it.
if HUEY_CLASS.endswith("RedisHuey"):
    HUEY["url"] = env.str("REDIS_URL", default="redis://localhost:6379/0")
else:
    HUEY["filename"] = env.str("HUEY_FILENAME",
                               default=str(BASE_DIR / "huey.sqlite3"))

# Immediate mode executes a task in the caller — which, for a run launched from
# the page, is the request itself: the browser would wait out the whole
# optimization. When it is on, dispatch the inline task to a background thread
# so the request returns at once, the way it does with a real consumer. Tests
# turn this off (see tests/conftest.py) to keep inline execution synchronous.
RUN_IMMEDIATE_IN_THREAD = env.bool("RUN_IMMEDIATE_IN_THREAD", default=True)


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "huey.contrib.djhuey",
    "access",
    "ui",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # Serves static files directly from the app process (no separate web server
    # needed in the container); must sit right after SecurityMiddleware.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    # LocaleMiddleware must sit after SessionMiddleware and before CommonMiddleware.
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    # The login wall. Inert unless REQUIRE_LOGIN is on; must follow
    # AuthenticationMiddleware, which is what puts request.user there.
    "access.middleware.LoginWallMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # Project-wide templates (base layout) live in templates/; per-app
        # templates are found automatically via APP_DIRS.
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.template.context_processors.i18n",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "ui.context_processors.sidebar_experiments",
                "ui.context_processors.active_tab",
                "ui.context_processors.navigation",
                "ui.context_processors.capabilities",
                "ui.context_processors.offered_languages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"


# Database
# SQLite by default; set DATABASE_URL (e.g. postgres://...) to switch.

DATABASES = {
    "default": env.db_url("DATABASE_URL", default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}"),
}


# Password validation
# https://docs.djangoproject.com/en/6.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/6.0/topics/i18n/

LANGUAGE_CODE = "en"

# The languages offered in the switcher (labels shown in their own language).
#
# German and Spanish are shelved while the interface is still moving. Their
# catalogs are still in `locale/`, but every string that was not carried over
# from InteractiveHPO is marked fuzzy — a draft nobody has checked — and there
# are no compiled .mo files, so nothing of it can reach a user. An unreviewed
# translation is worse than an English one; a missing translation is not.
#
# Re-enabling is this list plus `compilemessages`, after the drafts have been
# read by someone who speaks the language. The `{% translate %}` markup stays
# throughout in the meantime, so nothing has to be re-marked.
LANGUAGES = [
    ("en", "English"),
]

#: What the rail's language selector offers, which is not the same list. The
#: selector is back on the page and deliberately inert — it shows the languages
#: the interface is *going* to have, so the shape of the page is settled before
#: the catalogs are, and picking one does nothing yet. `LANGUAGES` above is what
#: Django will actually serve, and stays at English alone until the drafts have
#: been reviewed: half a translation reaching a user is the thing this avoids.
OFFERED_LANGUAGES = [
    ("en", "English"),
    ("de", "Deutsch"),
    ("es", "Español"),
]

# Translation catalogs live here (locale/<lang>/LC_MESSAGES/django.{po,mo}).
LOCALE_PATHS = [BASE_DIR / "locale"]

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/6.0/howto/static-files/

STATIC_URL = "static/"
# `collectstatic` gathers files here for WhiteNoise to serve in production.
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    # Where uploads live. A filesystem by default, which is right whenever the
    # web process and the consumer share one; env-selectable because they do not
    # have to. Pointing this at object storage (django-storages' S3Storage, say)
    # is the third of the three switches that separate the two processes, beside
    # the broker and `DATABASE_URL` — and, like those, it is a setting rather
    # than a change here.
    "default": {"BACKEND": env.str(
        "DEFAULT_FILE_STORAGE",
        default="django.core.files.storage.FileSystemStorage")},
    # Non-manifest WhiteNoise storage: compresses files at collectstatic and
    # serves them with cache headers, but keeps plain filenames — so
    # {% static %} needs no manifest and works in tests / `runserver` without a
    # prior collectstatic (manifest/hashed storage errors there).
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

# Uploaded files (datasets, custom models) live under here. Env-configurable so
# the container can point it at a shared data volume (with the two SQLite files).
MEDIA_URL = "media/"
MEDIA_ROOT = env.str("MEDIA_ROOT", default=str(BASE_DIR / "media"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ── TLS-dependent hardening ───────────────────────────────────────────────────
# This runs locally or on a private network at least as often as it is hosted
# (see REQUIRE_LOGIN above), and in both of those cases there is no TLS: the
# instance talks plain HTTP directly, exactly like the reference
# docker-compose.yml. Defaulting SECURE_SSL_REDIRECT etc. to "on whenever
# DEBUG=False" would break that setup the moment anyone actually ran it — a
# redirect to HTTPS with nothing on the other end to answer it. So this is one
# more explicit switch in the REQUIRE_LOGIN/ALLOW_CUSTOM_MODELS mold: off
# leaves today's behaviour untouched; on is for an operator who has put a
# TLS-terminating reverse proxy in front and knows it. See the README's
# hosting section.
SECURE_BEHIND_TLS = env.bool("SECURE_BEHIND_TLS", default=False)

if SECURE_BEHIND_TLS:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    # A year, the usual HSTS starting point; subdomains/preload are only safe
    # once every subdomain is confirmed to be HTTPS-only too, which is an
    # operator decision this setting does not make for them.
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = False
    SECURE_HSTS_PRELOAD = False
    # TLS terminates at the reverse proxy, not at gunicorn — this is what
    # tells Django a request forwarded as plain HTTP was actually HTTPS on the
    # wire, so SECURE_SSL_REDIRECT doesn't loop and request.is_secure() agrees
    # with reality. Only trust this header from a proxy that overwrites it
    # rather than passing a client-supplied one through.
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Origins whose POSTs Django will accept. Django matches the Origin header
# against this list for any request it considers secure, and an empty list means
# every form on a hosted instance is rejected with "CSRF verification failed" —
# a failure that looks like a bug in the page rather than a missing setting.
# Set outside the block above because a proxy can terminate TLS without this
# instance being told to harden anything else, and defaulted from ALLOWED_HOSTS
# so the common case needs no second list saying the same names twice.
CSRF_TRUSTED_ORIGINS = env.list(
    "CSRF_TRUSTED_ORIGINS",
    default=[f"https://{host}" for host in ALLOWED_HOSTS if host != "*"],
)


# ── Logging ────────────────────────────────────────────────────────────────────
# Without this, everything reaches gunicorn's stdout only by accident (Django's
# own unconfigured-logging fallback). This formalizes that rather than changing
# it: console only, no file handlers — the container is the log store.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        "django": {"handlers": ["console"], "level": "INFO", "propagate": False},
        # A run's own failures already reach Run.error and the page; this is
        # for what happens around them (env builds, model processes).
        "ui": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "core": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
