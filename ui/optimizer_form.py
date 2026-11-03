"""`optimizer_form` renders the optimizer select and its settings panels.
"""

from __future__ import annotations

from django.utils.translation import gettext as _

from core.registry import OPTIMIZERS
from experiments.services import history

from . import run_form
from .optimizer_labels import LABELS, describe_all, grouped


def why_fixed(exp, name: str | None = None) -> str | None:
    """`why_fixed` returns why the optimizer or its setting *name* cannot
    change, or None. With trials, only the optimizer's `run_safe` settings
    change between runs.
    """
    if has_trials(exp):
        kind = OPTIMIZERS.get(exp.data.optimizer_name)
        if name is None or kind is None or name not in kind.run_safe:
            return _("This experiment has trials, and the next run continues their search.")
    if exp.is_running:
        return _("A run is under way with this optimizer.")
    return None


def has_trials(exp) -> bool:
    result = exp.data.result or {}
    return bool(result.get("data") or result.get("trials"))


def context(exp, locked=False) -> dict:
    data = exp.data
    selected = data.optimizer_name if data.optimizer_name in OPTIMIZERS else next(iter(OPTIMIZERS))
    panels = []
    for key, optimizer in OPTIMIZERS.items():
        if not optimizer.params_schema:
            continue
        stored = data.optimizer_params if key == data.optimizer_name else None
        panels.append({"key": key, "selected": key == selected,
                       "disabled": locked or key != selected,
                       **_param_context(optimizer, stored)})
    return {"optimizer_choices": list(OPTIMIZERS), "optimizer_selected": selected,
            "optimizer_panels": panels, "optimizer_locked": locked}


def save(request, exp, data=None, *, partial=False) -> str | None:
    """`save` stores the optimizer from *data* on *exp* and returns an error
    message or None. With *partial*, *data* carries only the changes.
    """
    data = request.POST if data is None else data
    if has_trials(exp):
        return _save_run_safe(request, exp, data, partial)
    refusal = why_fixed(exp)
    if refusal:
        return refusal
    name = data.get("optimizer_name") or (exp.data.optimizer_name if partial else None)
    chosen = OPTIMIZERS.get(name)
    if chosen is None:
        return _("Choose an optimizer.")
    base = None
    if partial:
        base = (chosen.known_params(exp.data.optimizer_params or {})
                if name == exp.data.optimizer_name else {})
    optimizer = chosen(**chosen.known_params(_posted_params(data, chosen, base)))
    return _store(request, exp, optimizer.name, optimizer.get_params())


def _save_run_safe(request, exp, data, partial) -> str | None:
    """`_save_run_safe` returns an error naming each posted locked setting that
    differs from its stored value.
    """
    if exp.is_running:
        return _("A run is under way with this optimizer.")
    chosen = OPTIMIZERS.get(exp.data.optimizer_name)
    name = data.get("optimizer_name")
    if chosen is None or not chosen.run_safe or (name and name != exp.data.optimizer_name):
        return why_fixed(exp)
    stored = {p.name: p.default for p in chosen.params_schema}
    stored.update(chosen.known_params(exp.data.optimizer_params or {}))
    posted = _posted_params(data, chosen, stored if partial else None)
    locked = [str(LABELS.get(p.name, (p.label, ""))[0]) for p in chosen.params_schema
              if p.name not in chosen.run_safe and f"{_OPTIMIZER_PREFIX}{p.name}" in data
              and posted[p.name] != stored[p.name]]
    if locked:
        return _("These settings stay as the first run had them: %(names)s.") % {
            "names": ", ".join(locked)}
    params = {**stored, **{n: posted[n] for n in chosen.run_safe}}
    return _store(request, exp, chosen.name, chosen(**chosen.known_params(params)).get_params())


def _store(request, exp, name, params) -> None:
    data = exp.data
    old = {"name": data.optimizer_name, "params": data.optimizer_params or {}}
    new = {"name": name, "params": params}
    data.optimizer_name, data.optimizer_params = name, params
    data.save(update_fields=["optimizer_name", "optimizer_params"])
    if new != old:
        history.record(exp, "optimizer_changed", user=request.user, old=old, new=new)
    return None


def next_run_context(exp) -> dict:
    """`next_run_context` lays out the run-safe settings for the run form's
    "Next run" fold.
    """
    kind = OPTIMIZERS.get(exp.data.optimizer_name)
    if kind is None or not kind.run_safe:
        return {}
    described = [p for p in describe_all(kind, exp.data.optimizer_params or {})
                 if p["name"] in kind.run_safe]
    layout, advanced = grouped(described)
    return {"next_run_layout": layout, "next_run_advanced": advanced}


#: `_OPTIMIZER_PREFIX` keeps setting names clear of `n_trials` and stopping
#: criteria.
_OPTIMIZER_PREFIX = "opt_"


def _posted_params(data, optimizer, base=None) -> dict:
    """`_posted_params` reads types and bounds from the optimizer's schema, so a
    newly declared setting needs no change here. Without *base*, an absent
    checkbox is off; with *base*, an omitted setting keeps its *base* value.
    """
    params = {}
    for param in optimizer.params_schema:
        key = f"{_OPTIMIZER_PREFIX}{param.name}"
        if base is not None and key not in data:
            params[param.name] = base.get(param.name, param.default)
            continue
        raw = data.get(key)
        if param.type == "bool":
            params[param.name] = (raw.strip().lower() not in ("", "false", "off", "0")
                                  if isinstance(raw, str) else bool(raw))
            continue
        raw = run_form.text(raw)
        if not raw:
            params[param.name] = None if param.default is None else param.default
            continue
        if param.type == "select":
            params[param.name] = raw if raw in param.choices else param.default
            continue
        try:
            value = (int if param.type == "int" else float)(raw)
        except ValueError:
            params[param.name] = param.default
            continue
        if param.min is not None:
            value = max(param.min, value)
        if param.max is not None:
            value = min(param.max, value)
        params[param.name] = value
    return params


def cleaned_params(optimizer, params: dict) -> dict:
    """`cleaned_params` returns what the form would store for *params*, with
    omitted settings at defaults."""
    posted = {f"{_OPTIMIZER_PREFIX}{name}": value for name, value in params.items()}
    return _posted_params(posted, optimizer, base={})


def _param_context(optimizer, stored=None) -> dict:
    layout, advanced = grouped(describe_all(optimizer, stored))
    return {
        "optimizer_layout": layout,
        "advanced_params": advanced,
        "has_advanced_params": bool(advanced),
    }
