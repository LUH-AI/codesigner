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

# Bake static + compiled translations into the image. A dummy SECRET_KEY lets
# these management commands run at build time; real secrets come in at runtime.
ENV SECRET_KEY=build-only DEBUG=False
RUN python manage.py collectstatic --noinput \
    && python manage.py compilemessages -l de -l es

EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["web"]
