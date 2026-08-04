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

# When on, users may upload a model .py that the app imports and executes to
# run experiments. That is arbitrary code execution by design, so it must be
# OFF on any shared or public deployment (see README).
ALLOW_CUSTOM_MODELS = env.bool("ALLOW_CUSTOM_MODELS", default=True)

# Background task queue. Runs execute in a separate `manage.py run_huey`
# consumer process; run state lives in the database, so the web process only
# enqueues (polling and cancellation are DB-based and unaffected). SqliteHuey
# keeps everything on-box — no Redis. `immediate` runs tasks inline instead of
# via the consumer; it defaults to DEBUG so a lone `runserver` works in
# development, and is off in production where the consumer runs.
HUEY = {
    "huey_class": "huey.SqliteHuey",
    "name": "codesigner",
    "filename": env.str("HUEY_FILENAME", default=str(BASE_DIR / "huey.sqlite3")),
    "immediate": env.bool("HUEY_IMMEDIATE", default=DEBUG),
    "results": False,
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
