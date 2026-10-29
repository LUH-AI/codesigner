import copy
import importlib.util
import inspect
import os
import tempfile
from pathlib import Path
from typing import Any

from . import forecasting, tasks


def load_model_from_path(model_path: str) -> tuple:
    """Callers must pass only trusted paths (see `ALLOW_CUSTOM_MODELS`), because
    `load_model_from_path` executes the file. The result is ``(instance, None)``
    or ``(None, error)``, with untranslated errors.
    """
    from codesigner_model import BaseModel

    path = Path(model_path)
    if not path.is_file():
        return None, f"File not found: {model_path}"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="wb") as f:
        f.write(path.read_bytes())
        tmp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_custom_model", tmp_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            return None, f"Could not load file: {e}"

        candidates = [
            obj for obj in vars(module).values()
            if isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
        ]

        if not candidates:
            return None, "No BaseModel subclass found in the file."

        for cls in candidates:
            try:
                return cls(), None
            except TypeError:
                abstract = sorted(
                    name for name, val in inspect.getmembers(cls)
                    if getattr(val, "__isabstractmethod__", False)
                )
                return None, f"'{cls.__name__}' is missing required methods: {', '.join(abstract)}."
            except Exception as e:
                return None, f"Could not instantiate '{cls.__name__}': {e}"

        return None, "No valid BaseModel subclass could be instantiated."
    finally:
        os.unlink(tmp_path)


def give_context(model, attribute: str, value) -> None:
    """A model written before *attribute* existed may refuse it and stays as
    written, for numeric classification."""
    try:
        setattr(model, attribute, value)
    except AttributeError:
        pass


def resolve_model(snapshot: dict, evaluation: dict, task: str, available_models: dict,
                  model_name: str | None = None, load_file: bool = True) -> tuple[Any, str]:
    model_path = snapshot["model"].get("path", "")
    if snapshot["model"].get("kind") == "external":
        # The file of an external run holds the search and its scores, with no
        # model or dataset. `model = None` matches a read-only custom model, and
        # `_model_available` treats such an experiment as unrunnable.
        model = None
        return model, snapshot["model"].get("name") or "External"
    if model_path:
        if not load_file:
            # The caller supplies the model, usually in its own process, so
            # importing the file here would be waste.
            model = None
        else:
            model, err = load_model_from_path(model_path)
            if err:
                raise ValueError(f"custom model error: {err}")
            give_context(model, "task", task)
        return model, snapshot["model"]["name"]
    stored_model = model_name if model_name is not None else snapshot["model"]["name"]
    model_entry = (
        available_models.get(stored_model)
        or next((m for m in available_models.values()
                 if m.name == stored_model or stored_model in getattr(m, "aliases", ())),
                None)
    )
    if model_entry is None:
        raise ValueError(f"model '{stored_model}' is not available")
    if task not in tasks.supported(model_entry):
        raise ValueError(f"model '{stored_model}' does not do {task}")
    forecasts = evaluation["horizon"] > 0
    if tasks.forecaster(model_entry) and not forecasts:
        raise ValueError(f"model '{stored_model}' only forecasts")
    # The registry instance is shared, so each experiment gets a copy with
    # its own task; `core.forecasting` wraps a non-forecaster asked to
    # forecast.
    if forecasts and not tasks.forecaster(model_entry):
        model = forecasting.Reduced(model_entry, task)
    else:
        model = copy.copy(model_entry)
        model.task = task
    return model, stored_model
