# Codesigner

An interactive hyperparameter-optimization workbench: set up an experiment
(model, optimizer, dataset), run trials in the background, and explore the
results — best/selected configuration, hyperparameter importance, and the
incumbent's performance over trials. Experiments save to a portable `.ihpo`
file. A Django rebuild of the InteractiveHPO Streamlit app.

## Running locally

```bash
cp .env.example .env          # set SECRET_KEY
python manage.py migrate
python manage.py compilemessages   # build the de/es translation catalogs
python manage.py runserver
```

The interface is available in English, German, and Spanish (switch it from the
🌐 selector in the sidebar).

## Custom models — trust model ⚠️

Beyond the built-in models, you can upload your own model as a `.py` file
defining a `core.models.BaseModel` subclass. **Loading such a file executes it**
— it is arbitrary Python code running on the server, by design.

This is gated by the `ALLOW_CUSTOM_MODELS` setting (env var), which defaults to
**on** for local single-user use. **It must be turned off on any shared or
public deployment:**

```bash
ALLOW_CUSTOM_MODELS=False
```

With it off, the upload field disappears from the UI, uploaded model files in
imported `.ihpo` experiments are not adopted, and any experiment whose model is
a custom file cannot run (it loads read-only). There is no sandboxing of
uploaded code — the flag is the boundary.

Custom-model code currently executes in the web process; it moves to an
isolated worker process when the task queue lands (the deployment step).
