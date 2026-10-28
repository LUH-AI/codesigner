"""`optimizer_form` renders the optimizer select and per-optimizer settings
panels for setup step 2 and for an experiment page before its first trial.
"""

from __future__ import annotations

from django.utils.translation import gettext as _

from core.registry import OPTIMIZERS
from experiments.services import history

from .optimizer_labels import describe_all, grouped


def why_fixed(exp) -> str | None:
    result = exp.data.result or {}
    if result.get("data") or result.get("trials"):
        return _("This experiment has trials, and the next run continues their search.")
    if exp.is_running:
        return _("A run is under way with this optimizer.")
    return None


def context(exp, locked=False) -> dict:
    """`context` fills the chosen panel from *exp*, and every control is
    disabled under *locked*."""
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


def save(request, exp) -> str | None:
    """`save` stores the posted optimizer on *exp* and returns a refusal reason
    or None. A change goes into the history."""
    refusal = why_fixed(exp)
    if refusal:
        return refusal
    chosen = OPTIMIZERS.get(request.POST.get("optimizer_name"))
    if chosen is None:
        return _("Choose an optimizer.")
    optimizer = chosen(**chosen.known_params(_posted_params(request, chosen)))
    data = exp.data
    old = {"name": data.optimizer_name, "params": data.optimizer_params or {}}
    new = {"name": optimizer.name, "params": optimizer.get_params()}
    data.optimizer_name, data.optimizer_params = new["name"], new["params"]
    data.save(update_fields=["optimizer_name", "optimizer_params"])
    if new != old:
        history.record(exp, "optimizer_changed", user=request.user, old=old, new=new)
    return None


#: The prefix keeps optimizer setting names clear of `n_trials` and stopping
#: criteria on the same form.
_OPTIMIZER_PREFIX = "opt_"


def _posted_params(request, optimizer) -> dict:
    """`_posted_params` reads types and bounds from the optimizer's own schema,
    so a newly declared setting needs no change here. A blank field takes the
    declared default.
    """
    params = {}
    for param in optimizer.params_schema:
        raw = request.POST.get(f"{_OPTIMIZER_PREFIX}{param.name}")
        if param.type == "bool":
            params[param.name] = bool(raw)
            continue
        raw = (raw or "").strip()
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


def _param_context(optimizer, stored=None) -> dict:
    layout, advanced = grouped(describe_all(optimizer, stored))
    return {
        "optimizer_layout": layout,
        "advanced_params": advanced,
        "has_advanced_params": bool(advanced),
    }
