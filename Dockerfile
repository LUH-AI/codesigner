FROM python:3.12-slim

# System deps: swig + a compiler to build SMAC's pyrfr (random-forest) backend;
# git to pip-install SMAC from source; gettext for compilemessages; libgomp for
# numpy/scikit-learn's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential swig git gettext libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first for layer caching (pyrfr/SMAC build is slow).
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Register the project's own package metadata so importlib.metadata can read the
# app version (the single source in pyproject.toml). Deps are already installed
# above, so skip re-resolving them (avoids rebuilding the git-pinned SMAC).
RUN pip install -e . --no-deps

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
