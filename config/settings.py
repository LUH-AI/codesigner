"""
`config.settings` reads secrets and machine-specific values from environment
variables or a `.env` file at the repo root. `.env.example` lists them all.
"""

from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    # Each value is (type, local-development default).
    DEBUG=(bool, True),
)
# Process environment values override `.env`.
environ.Env.read_env(BASE_DIR / ".env")

DEBUG = env.bool("DEBUG")

# Outside DEBUG, `SECRET_KEY` has no default, so a deployment missing it fails
# at startup instead of signing cookies with the repository's key.
if DEBUG:
    SECRET_KEY = env("SECRET_KEY", default="django-insecure-local-development-only")  # pyright: ignore[reportArgumentType]
else:
    SECRET_KEY = env("SECRET_KEY")

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["localhost", "127.0.0.1"])  # pyright: ignore[reportArgumentType]

#: `REQUIRE_LOGIN` turns on accounts and the `access` package for a shared
#: instance; local and trusted-network installs leave it off.
REQUIRE_LOGIN = env.bool("REQUIRE_LOGIN", default=False)

#: With accounts, `BIN_RETENTION_DAYS` is how long a deleted experiment stays in
#: its people's bins before moving to the site admins' bin, which never expires.
BIN_RETENTION_DAYS = env.int("BIN_RETENTION_DAYS", default=30)

#: `ALLOW_CUSTOM_MODELS` lets users upload model code that the app executes. It
#: defaults to off because, outside a container, that code runs with the host
#: account's files and ssh keys.
ALLOW_CUSTOM_MODELS = env.bool("ALLOW_CUSTOM_MODELS", default=False)

#: The default `GroupPolicy` stays open until `REQUIRE_LOGIN` is on, and
#: `access.policy.OpenPolicy` removes the group boundary even with accounts.
EXPERIMENT_POLICY = env.str("EXPERIMENT_POLICY", default="access.policy.GroupPolicy")

#: `LOGIN_URL` and the redirects stay set while `REQUIRE_LOGIN` is off, so
#: turning it on takes one variable.
LOGIN_URL = "ui:login"
LOGIN_REDIRECT_URL = "ui:home"
LOGOUT_REDIRECT_URL = "ui:login"

#: uv runs each model in its own process, in an environment built from its PEP
#: 723 header; without uv, a local instance imports the model in-process and a
#: hosted instance stops the run with an error (see `.resolve_runner`).
UV_BIN = env.str("UV_BIN", default="uv")

#: `MODEL_SDK_WHEEL` points at the built model contract; the `MODEL_SDK_PATH`
#: source fallback makes uv fetch a build backend, which fails offline.
MODEL_SDK_WHEEL = env.str("MODEL_SDK_WHEEL", default=str(BASE_DIR / "wheels" / "sdk"))
MODEL_SDK_PATH = env.str("MODEL_SDK_PATH", default=str(BASE_DIR / "model_sdk"))

#: An air-gapped deployment turns `MODEL_ENV_OFFLINE` on and Python downloads
#: off.
MODEL_ENV_OFFLINE = env.bool("MODEL_ENV_OFFLINE", default=False)
MODEL_ENV_PYTHON_DOWNLOADS = env.bool("MODEL_ENV_PYTHON_DOWNLOADS", default=True)

#: `MODEL_ENV_PREPARE_TIMEOUT` is generous because preparing an environment can
#: mean a large download.
MODEL_ENV_PREPARE_TIMEOUT = env.float("MODEL_ENV_PREPARE_TIMEOUT", default=900.0)
MODEL_TRIAL_TIMEOUT = env.float("MODEL_TRIAL_TIMEOUT", default=600.0)

#: Per-file write cap (`RLIMIT_FSIZE`) for a model.
MODEL_MAX_FILE_BYTES = env.int("MODEL_MAX_FILE_BYTES", default=1024 * 1024 * 1024)

#: `MAX_UPLOAD_BYTES` caps browser uploads; `MODEL_MAX_FILE_BYTES` caps what a
#: running model writes.
MAX_UPLOAD_BYTES = env.int("MAX_UPLOAD_BYTES", default=100 * 1024 * 1024)

#: `GROUP_STORAGE_LIMIT_BYTES` caps a group's disk use unless a site admin sets
#: the group's own limit; bundled demo datasets count against nobody.
GROUP_STORAGE_LIMIT_BYTES = env.int("GROUP_STORAGE_LIMIT_BYTES", default=20 * 1024 ** 3)

#: Usage share past which members see a warning banner.
STORAGE_WARNING_SHARE = env.float("STORAGE_WARNING_SHARE", default=0.9)

#: `ANALYTICS_EAGER_MAX_COALITIONS` caps the shapiq evaluations
#: (2^hyperparameters x 3 games x metrics) run at completion for importance
#: analytics. Each metric over the cap shows a skip reason. The cost grows
#: exponentially with hyperparameter count, and 0 lifts the cap.
ANALYTICS_EAGER_MAX_COALITIONS = env.int("ANALYTICS_EAGER_MAX_COALITIONS", default=1024)

#: A model's process inherits every variable except the listed ones, since proxy
#: and certificate settings vary by operator.
MODEL_ENV_DENYLIST = (
    "SECRET_KEY", "DATABASE_URL", "DJANGO_SETTINGS_MODULE", "HUEY_FILENAME",
    "MEDIA_ROOT", "ALLOWED_HOSTS", "REQUIRE_LOGIN", "ALLOW_CUSTOM_MODELS",
)

#: `local` runs the optimization in the consumer process; `slurm` has the
#: consumer submit it to a cluster. The queue and database settings below decide
#: separately where the consumer runs.
RUN_BACKEND = env.str("RUN_BACKEND", default="local")

#: `CLUSTER_HOST` is an `ssh` host name, so ProxyJump, key and user stay in
#: ~/.ssh/config.
CLUSTER_HOST = env.str("CLUSTER_HOST", default="")
CLUSTER_ROOT = env.str("CLUSTER_ROOT", default="codesigner")
CLUSTER_PARTITION = env.str("CLUSTER_PARTITION", default="")
#: `CLUSTER_CPUS` sets the job's cores and its thread-pool sizes
#: (remote/job.sbatch).
CLUSTER_CPUS = env.int("CLUSTER_CPUS", default=8)
#: `CLUSTER_MAX_HOURS` caps a job's wall-clock hours and should match the
#: partition limit, since `sbatch` fails to submit a longer job.
CLUSTER_MAX_HOURS = env.int("CLUSTER_MAX_HOURS", default=24)
#: Each poll is an ssh round trip and also paces page figures.
CLUSTER_POLL_SECONDS = env.float("CLUSTER_POLL_SECONDS", default=5.0)

#: SqliteHuey needs the `manage.py run_huey` consumer on the web process's
#: filesystem, and RedisHuey lets the two run on separate machines. A lone
#: `runserver` runs tasks inline because the `immediate` flag follows DEBUG. Four
#: thread workers keep a minutes-long environment build from blocking every run.
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

# A Redis setup never falls back to a SQLite file another machine cannot see,
# because each broker class only gets its own argument.
if HUEY_CLASS.endswith("RedisHuey"):
    HUEY["url"] = env.str("REDIS_URL", default="redis://localhost:6379/0")
else:
    HUEY["filename"] = env.str("HUEY_FILENAME",
                               default=str(BASE_DIR / "huey.sqlite3"))

#: Immediate mode runs a task inside the launching request, so
#: `RUN_IMMEDIATE_IN_THREAD` moves it to a background thread; tests/conftest.py
#: turns it off.
RUN_IMMEDIATE_IN_THREAD = env.bool("RUN_IMMEDIATE_IN_THREAD", default=True)


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "huey.contrib.djhuey",
    "access",
    "experiments",
    "ui",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # WhiteNoise serves static files from the app process and must follow
    # SecurityMiddleware.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    # After sessions, before CommonMiddleware.
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    # Inert without `REQUIRE_LOGIN`; needs `request.user` from
    # AuthenticationMiddleware.
    "access.middleware.LoginWallMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
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
                "ui.context_processors.theme",
                "ui.context_processors.storage",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"


# `DATABASE_URL` can name Postgres instead.

DATABASES = {
    "default": env.db_url("DATABASE_URL", default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}"),  # pyright: ignore[reportArgumentType]
}


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


LANGUAGE_CODE = "en"

#: `manage.py translations` compiles the catalogs, and `msgfmt` drops fuzzy
#: entries, so a string whose English changed falls back to English.
LANGUAGES = [
    ("en", "English"),
    ("de", "Deutsch"),
    ("es", "Español"),
]

#: `OFFERED_LANGUAGES` feeds the rail's language selector, and a test keeps its
#: codes equal to `LANGUAGES`.
OFFERED_LANGUAGES = [
    ("en", "English"),
    ("de", "Deutsch"),
    ("es", "Español"),
]

LOCALE_PATHS = [BASE_DIR / "locale"]

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    # `FileSystemStorage` needs the web process and consumer on one filesystem;
    # `DEFAULT_FILE_STORAGE` can name an object store instead.
    "default": {"BACKEND": env.str(
        "DEFAULT_FILE_STORAGE",
        default="django.core.files.storage.FileSystemStorage")},
    # `CompressedStaticFilesStorage` keeps plain filenames, so `{% static %}`
    # works in tests and `runserver` without collectstatic.
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

MEDIA_URL = "media/"
#: `MEDIA_ROOT` stores uploads, and the container points it at its shared data
#: volume.
MEDIA_ROOT = env.str("MEDIA_ROOT", default=str(BASE_DIR / "media"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


#: Local and private-network installs serve plain HTTP, so forcing HTTPS whenever
#: `DEBUG=False` would redirect them to a port nobody answers.
#: `SECURE_BEHIND_TLS` is for a TLS-terminating reverse proxy in front.
SECURE_BEHIND_TLS = env.bool("SECURE_BEHIND_TLS", default=False)

if SECURE_BEHIND_TLS:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    # One year is the usual HSTS start; subdomains and preload stay off until
    # the operator confirms every subdomain is HTTPS-only.
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = False
    SECURE_HSTS_PRELOAD = False
    # `SECURE_PROXY_SSL_HEADER` tells Django that a plain-HTTP request from the
    # TLS-terminating proxy was HTTPS, so the redirect never loops. The proxy
    # must overwrite any client-sent `X-Forwarded-Proto`.
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

#: Django checks a secure request's Origin against `CSRF_TRUSTED_ORIGINS`, so an
#: empty list fails every hosted form. The list defaults from `ALLOWED_HOSTS` and
#: applies without `SECURE_BEHIND_TLS`, since a proxy can still terminate TLS.
CSRF_TRUSTED_ORIGINS = env.list(
    "CSRF_TRUSTED_ORIGINS",
    default=[f"https://{host}" for host in ALLOWED_HOSTS if host != "*"],  # pyright: ignore[reportArgumentType]
)


#: `LOGGING` keeps Django's console-only fallback, since the container stores the
#: logs.
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
        # A run's failures reach `Run.error`; the app loggers cover env builds
        # and model processes.
        "ui": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "experiments": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "remote": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "core": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
