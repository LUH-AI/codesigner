FROM python:3.12-slim

# System deps: swig + a compiler to build SMAC's pyrfr (random-forest) backend,
# which ships no wheel; gettext for compilemessages; libgomp for
# numpy/scikit-learn's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential swig gettext libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# uv builds each model's own environment at runtime. Copied from its published
# image rather than curl-installed: no extra tooling in the image, the version is
# pinned by this reference and layer-cached, and uv stays independent of the
# application's own site-packages — which matters, because its job is managing
# other interpreters.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

# Where uv keeps downloads, built environments and any interpreters it fetches.
# Compose mounts a volume here; without one, a model needing torch re-downloads
# gigabytes after every rebuild.
ENV UV_CACHE_DIR=/uv-cache \
    UV_PYTHON_INSTALL_DIR=/uv-cache/python

# Install Python deps first for layer caching (pyrfr/SMAC build is slow).
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Register the project's own package metadata so importlib.metadata can read the
# app version (the single source in pyproject.toml). Deps are already installed
# above, so skip re-resolving them (avoids rebuilding pyrfr).
RUN pip install -e . --no-deps

# The model contract, as its own dependency-free distribution: core.models
# imports it here, and every model environment installs it. Built as a wheel as
# well, because handing uv a source directory makes it fetch a build backend from
# the index — which fails on a machine that has no access to one.
RUN pip install -e ./model_sdk --no-deps \
    && python -m pip wheel --no-deps -w /app/wheels/sdk ./model_sdk
ENV MODEL_SDK_WHEEL=/app/wheels/sdk

# Bake static + compiled translations into the image. A dummy SECRET_KEY lets
# these management commands run at build time; real secrets come in at runtime.
# ARG, not ENV: an ENV would persist into the final image, and settings read the
# process environment first — so a container started without a real SECRET_KEY
# would boot on this published one rather than failing.
ARG SECRET_KEY=build-only
ARG DEBUG=False
RUN SECRET_KEY=$SECRET_KEY DEBUG=$DEBUG python manage.py collectstatic --noinput \
    && SECRET_KEY=$SECRET_KEY DEBUG=$DEBUG python manage.py compilemessages -l de -l es

EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["web"]
