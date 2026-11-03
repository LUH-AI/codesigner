"""
`config.settings` reads its values from environment variables or the repo-root
`.env`; `.env.example` lists them.
"""

from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    DEBUG=(bool, True),
)
environ.Env.read_env(BASE_DIR / ".env")

DEBUG = env.bool("DEBUG")

# Outside DEBUG, `SECRET_KEY` has no default and startup fails without it.
if DEBUG:
    SECRET_KEY = env("SECRET_KEY", default="django-insecure-local-development-only")  # pyright: ignore[reportArgumentType]
else:
    SECRET_KEY = env("SECRET_KEY")

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["localhost", "127.0.0.1"])  # pyright: ignore[reportArgumentType]

#: `REQUIRE_LOGIN` turns on accounts and the `access` package for a shared
#: instance.
REQUIRE_LOGIN = env.bool("REQUIRE_LOGIN", default=False)

#: After `BIN_RETENTION_DAYS`, a deleted experiment leaves personal bins for the
#: site admins' bin.
BIN_RETENTION_DAYS = env.int("BIN_RETENTION_DAYS", default=30)

#: `ALLOW_CUSTOM_MODELS` lets users upload model code; outside a container, that
#: code runs with the host account's ssh keys.
ALLOW_CUSTOM_MODELS = env.bool("ALLOW_CUSTOM_MODELS", default=False)

#: `.GroupPolicy` stays open while `REQUIRE_LOGIN` is off; `.OpenPolicy` drops
#: group boundaries.
EXPERIMENT_POLICY = env.str("EXPERIMENT_POLICY", default="access.policy.GroupPolicy")

LOGIN_URL = "ui:login"
LOGIN_REDIRECT_URL = "ui:home"
LOGOUT_REDIRECT_URL = "ui:login"

#: uv (`UV_BIN`) runs each model in a PEP 723 environment; without uv, models
#: load in-process unless `REQUIRE_LOGIN` is on (see `.resolve_runner`).
UV_BIN = env.str("UV_BIN", default="uv")

#: `MODEL_SDK_WHEEL` points at the built model contract; the `MODEL_SDK_PATH`
#: source fallback needs network access.
MODEL_SDK_WHEEL = env.str("MODEL_SDK_WHEEL", default=str(BASE_DIR / "wheels" / "sdk"))
MODEL_SDK_PATH = env.str("MODEL_SDK_PATH", default=str(BASE_DIR / "model_sdk"))

#: Air-gapped hosts need offline on, downloads off.
MODEL_ENV_OFFLINE = env.bool("MODEL_ENV_OFFLINE", default=False)
MODEL_ENV_PYTHON_DOWNLOADS = env.bool("MODEL_ENV_PYTHON_DOWNLOADS", default=True)

#: Seconds, generous for large environment downloads.
MODEL_ENV_PREPARE_TIMEOUT = env.float("MODEL_ENV_PREPARE_TIMEOUT", default=900.0)
MODEL_TRIAL_TIMEOUT = env.float("MODEL_TRIAL_TIMEOUT", default=600.0)

#: Per-file `RLIMIT_FSIZE` for models.
MODEL_MAX_FILE_BYTES = env.int("MODEL_MAX_FILE_BYTES", default=1024 * 1024 * 1024)

#: `MAX_UPLOAD_BYTES` caps browser uploads.
MAX_UPLOAD_BYTES = env.int("MAX_UPLOAD_BYTES", default=100 * 1024 * 1024)

#: `GROUP_STORAGE_LIMIT_BYTES` is the default group quota; bundled demo datasets
#: count against nobody.
GROUP_STORAGE_LIMIT_BYTES = env.int("GROUP_STORAGE_LIMIT_BYTES", default=20 * 1024 ** 3)

#: Usage fraction that warns members.
STORAGE_WARNING_SHARE = env.float("STORAGE_WARNING_SHARE", default=0.9)

#: `ANALYTICS_EAGER_MAX_COALITIONS` caps the shapiq evaluations
#: (2^hyperparameters x 3 games x metrics) run at completion; 0 lifts the cap.
ANALYTICS_EAGER_MAX_COALITIONS = env.int("ANALYTICS_EAGER_MAX_COALITIONS", default=1024)

# ── The assistant ────────────────────────────────────────────────────────────
# Every page works unchanged while `ASSISTANT_ENABLED` is off or the model is
# unreachable.
ASSISTANT_ENABLED = env.bool("ASSISTANT_ENABLED", default=False)
# `ASSISTANT_LLM_BASE_URL` accepts any OpenAI-compatible chat server, such as
# vLLM or Ollama's /v1.
ASSISTANT_LLM_BASE_URL = env.str("ASSISTANT_LLM_BASE_URL",
                                 default="https://chat-ai.cluster.uni-hannover.de/v1")
ASSISTANT_LLM_MODEL = env.str("ASSISTANT_LLM_MODEL", default="qwen38-27b-fp8")
ASSISTANT_LLM_TOKEN = env.str("ASSISTANT_LLM_TOKEN", default="not-needed")
# Seconds per model call, and retries per failed call.
ASSISTANT_LLM_TIMEOUT = env.float("ASSISTANT_LLM_TIMEOUT", default=120.0)
ASSISTANT_LLM_RETRIES = env.int("ASSISTANT_LLM_RETRIES", default=2)
# `ASSISTANT_LLM_EXTRA_BODY` is a JSON object added to every request, such as
# {"chat_template_kwargs": {"enable_thinking": false}}.
ASSISTANT_LLM_EXTRA_BODY = env.json("ASSISTANT_LLM_EXTRA_BODY", default={})
# `ASSISTANT_EMBED_BASE_URL` accepts any OpenAI-compatible embeddings server;
# without one, library search matches words alone.
ASSISTANT_EMBED_BASE_URL = env.str("ASSISTANT_EMBED_BASE_URL", default="")
ASSISTANT_EMBED_MODEL = env.str("ASSISTANT_EMBED_MODEL", default="qwen3-embedding-8b")
ASSISTANT_EMBED_TOKEN = env.str("ASSISTANT_EMBED_TOKEN", default="not-needed")
# The prefix only precedes search queries, in Qwen3-Embedding's instruction
# format; models without one take an empty prefix.
ASSISTANT_EMBED_QUERY_PREFIX = env.str(
    "ASSISTANT_EMBED_QUERY_PREFIX",
    default="Instruct: Given a question about hyperparameter optimization, retrieve passages "
            "of papers that answer it\nQuery: ")
# `manage.py assistant_knowledge build` builds the library index here from
# open-access sources.
ASSISTANT_KNOWLEDGE_DIR = Path(env.str("ASSISTANT_KNOWLEDGE_DIR",
                                       default=str(BASE_DIR / "var" / "assistant-knowledge")))
# The model sees `ASSISTANT_SAMPLE_ROWS` example values per column; at 0 it only
# gets names, kinds and counts.
ASSISTANT_SAMPLE_ROWS = env.int("ASSISTANT_SAMPLE_ROWS", default=5)
# `ASSISTANT_DAILY_TOKENS` caps a person's daily tokens; each model call resends
# about 12 000 tokens, so the default covers some 250 calls.
ASSISTANT_DAILY_TOKENS = env.int("ASSISTANT_DAILY_TOKENS", default=3_000_000)
ASSISTANT_MAX_STEPS = env.int("ASSISTANT_MAX_STEPS", default=10)
ASSISTANT_MAX_TOOL_CALLS = env.int("ASSISTANT_MAX_TOOL_CALLS", default=20)
# A turn runs in a web-process thread, at most `ASSISTANT_TURN_THREADS` at once,
# and never takes a Huey worker.
ASSISTANT_TURN_THREADS = env.int("ASSISTANT_TURN_THREADS", default=4)
# Tests set `ASSISTANT_TURNS_INLINE` to run turns in-request.
ASSISTANT_TURNS_INLINE = env.bool("ASSISTANT_TURNS_INLINE", default=False)
# Seconds one event-stream connection stays open before reconnecting.
ASSISTANT_STREAM_SECONDS = env.float("ASSISTANT_STREAM_SECONDS", default=55.0)
# Seconds a page action waits for confirmation.
ASSISTANT_ACT_SECONDS = env.float("ASSISTANT_ACT_SECONDS", default=8.0)
# Seconds a page stays actionable after its last report.
ASSISTANT_PAGE_SECONDS = env.float("ASSISTANT_PAGE_SECONDS", default=90.0)

#: Models inherit all other variables, including proxy and certificate settings.
MODEL_ENV_DENYLIST = (
    "SECRET_KEY", "DATABASE_URL", "DJANGO_SETTINGS_MODULE", "HUEY_FILENAME",
    "MEDIA_ROOT", "ALLOWED_HOSTS", "REQUIRE_LOGIN", "ALLOW_CUSTOM_MODELS",
    "ASSISTANT_LLM_TOKEN", "ASSISTANT_LLM_BASE_URL",
    "ASSISTANT_EMBED_TOKEN", "ASSISTANT_EMBED_BASE_URL",
)

#: `local` runs the optimization in the Huey consumer; `slurm` submits it to a
#: cluster.
RUN_BACKEND = env.str("RUN_BACKEND", default="local")

#: An `ssh` alias, configured in ~/.ssh/config.
CLUSTER_HOST = env.str("CLUSTER_HOST", default="")
CLUSTER_ROOT = env.str("CLUSTER_ROOT", default="codesigner")
CLUSTER_PARTITION = env.str("CLUSTER_PARTITION", default="")
#: Job cores and thread-pool sizes (remote/job.sbatch).
CLUSTER_CPUS = env.int("CLUSTER_CPUS", default=8)
#: `CLUSTER_MAX_HOURS` should match the partition limit; `sbatch` fails to
#: submit longer jobs.
CLUSTER_MAX_HOURS = env.int("CLUSTER_MAX_HOURS", default=24)
#: Seconds between ssh polls, which refresh page figures.
CLUSTER_POLL_SECONDS = env.float("CLUSTER_POLL_SECONDS", default=5.0)

#: SqliteHuey needs the `run_huey` consumer on the web host; RedisHuey allows
#: separate machines. `immediate` follows DEBUG, so `runserver` alone runs tasks
#: inline.
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

if HUEY_CLASS.endswith("RedisHuey"):
    HUEY["url"] = env.str("REDIS_URL", default="redis://localhost:6379/0")
else:
    HUEY["filename"] = env.str("HUEY_FILENAME",
                               default=str(BASE_DIR / "huey.sqlite3"))

#: `RUN_IMMEDIATE_IN_THREAD` moves immediate-mode tasks out of the request into
#: a thread.
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
    "assistant",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # WhiteNoise must follow SecurityMiddleware.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    # After sessions, before CommonMiddleware.
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    # Needs `request.user` from AuthenticationMiddleware.
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
                "assistant.context_processors.assistant",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"


# Postgres via `DATABASE_URL`.

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

#: `msgfmt` drops fuzzy entries, so a string whose English changed shows
#: English.
LANGUAGES = [
    ("en", "English"),
    ("de", "Deutsch"),
    ("es", "Español"),
]

#: A test requires the same codes as `LANGUAGES`.
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
    # `FileSystemStorage` needs the web process and consumer on one filesystem.
    "default": {"BACKEND": env.str(
        "DEFAULT_FILE_STORAGE",
        default="django.core.files.storage.FileSystemStorage")},
    # Plain filenames let `{% static %}` work without collectstatic.
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

MEDIA_URL = "media/"
MEDIA_ROOT = env.str("MEDIA_ROOT", default=str(BASE_DIR / "media"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


#: Local installs serve plain HTTP, so `DEBUG=False` leaves HTTPS off;
#: `SECURE_BEHIND_TLS` turns it on behind a TLS-terminating proxy.
SECURE_BEHIND_TLS = env.bool("SECURE_BEHIND_TLS", default=False)

if SECURE_BEHIND_TLS:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    # Subdomains and preload stay off until every subdomain serves HTTPS.
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = False
    SECURE_HSTS_PRELOAD = False
    # The proxy must overwrite any client-sent `X-Forwarded-Proto`, since Django
    # trusts the header.
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

#: Django checks a secure request's Origin against `CSRF_TRUSTED_ORIGINS`, so an
#: empty list fails every hosted form. It defaults from `ALLOWED_HOSTS`, also
#: without `SECURE_BEHIND_TLS`, since a proxy can terminate TLS.
CSRF_TRUSTED_ORIGINS = env.list(
    "CSRF_TRUSTED_ORIGINS",
    default=[f"https://{host}" for host in ALLOWED_HOSTS if host != "*"],  # pyright: ignore[reportArgumentType]
)


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
        "ui": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "experiments": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "remote": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "core": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
