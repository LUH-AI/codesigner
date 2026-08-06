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

# Who may see and do what to an experiment. The default says "everyone,
# everything", which is what an install with no accounts wants; a deployment
# that manages users points this at a policy of its own. See ui/permissions.py.
EXPERIMENT_POLICY = env.str("EXPERIMENT_POLICY", default="ui.permissions.OpenPolicy")

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

# Never passed to a model's process. The rest of the environment is inherited,
# because proxy, certificate and index settings are numerous and operator-
# specific; these are the ones that would matter if they leaked.
MODEL_ENV_DENYLIST = (
    "SECRET_KEY", "DATABASE_URL", "DJANGO_SETTINGS_MODULE", "HUEY_FILENAME",
    "MEDIA_ROOT", "ALLOWED_HOSTS", "REQUIRE_LOGIN", "ALLOW_CUSTOM_MODELS",
)

# Background task queue. Runs execute in a separate `manage.py run_huey`
# consumer process; run state lives in the database, so the web process only
# enqueues (polling and cancellation are DB-based and unaffected). SqliteHuey
# keeps everything on-box — no Redis. `immediate` runs tasks inline instead of
# via the consumer; it defaults to DEBUG so a lone `runserver` works in
# development, and is off in production where the consumer runs.
#
# The consumer defaults to a single worker, which would mean one queue for two
# very different jobs: an optimization, and building a model's environment. The
# second can take minutes of downloading, and while it held the only worker
# nothing on the instance could run. Both spend nearly all their time waiting on
# a subprocess, so threads are the right kind of worker; the count is what needs
# to be greater than one.
HUEY = {
    "huey_class": "huey.SqliteHuey",
    "name": "codesigner",
    "filename": env.str("HUEY_FILENAME", default=str(BASE_DIR / "huey.sqlite3")),
    "immediate": env.bool("HUEY_IMMEDIATE", default=DEBUG),
    "results": False,
    "consumer": {
        "workers": env.int("HUEY_WORKERS", default=4),
        "worker_type": "thread",
    },
}

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
LANGUAGES = [
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
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
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
