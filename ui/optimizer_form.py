"""The optimizer and its settings, as a form: one select, and one settings
panel per optimizer that has settings.

Two pages offer it: step 2 of a draft's setup, and an experiment's page before
it has any trials. Once there are trials the search that found them is fixed —
the next run continues it — and while a run is under way it is the one in use.
"""

from __future__ import annotations

from django.utils.translation import gettext as _

from experiments.services import history

from .optimizer_labels import describe_all, grouped
from .registry import OPTIMIZERS


def why_fixed(exp) -> str | None:
    """Why *exp*'s optimizer can no longer change, or None if it can."""
    result = exp.data.result or {}
    if result.get("data") or result.get("trials"):
        return _("This experiment has trials, and the next run continues their search.")
    if exp.is_running:
        return _("A run is under way with this optimizer.")
    return None


def context(exp, locked=False) -> dict:
    """What `ui/_optimizer_form.html` needs: the optimizers to choose from,
    the chosen one, and a settings panel for each — the chosen one filled
    from *exp*, the others on their defaults. With *locked* every control is
    shown and none can be changed."""
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
    """Store the optimizer and settings *request* posted on *exp*. Returns why
    it could not, or None. A change is an entry in the experiment's history
    (a draft has none)."""
    refusal = why_fixed(exp)
    if refusal:
        return refusal
    chosen = OPTIMIZERS.get(request.POST.get("optimizer_name"))
    if chosen is None:
        return _("Choose an optimizer.")
    optimizer = type(chosen)(**type(chosen).known_params(_posted_params(request, chosen)))
    data = exp.data
    old = {"name": data.optimizer_name, "params": data.optimizer_params or {}}
    new = {"name": optimizer.name, "params": optimizer.get_params()}
    data.optimizer_name, data.optimizer_params = new["name"], new["params"]
    data.save(update_fields=["optimizer_name", "optimizer_params"])
    if new != old:
        history.record(exp, "optimizer_changed", user=request.user, old=old, new=new)
    return None


#: Optimizer settings are posted with this prefix, so a parameter can be named
#: whatever the optimizer calls it without colliding with `n_trials` or a
#: stopping criterion on the same form.
_OPTIMIZER_PREFIX = "opt_"


def _posted_params(request, optimizer) -> dict:
    """The optimizer's settings as submitted, read through its own schema.

    Types and bounds come from the declaration rather than a table here, which
    is the difference between this and `views.runs._posted_stopping`: an optimizer that
    declares a new setting gets it parsed without this function changing.

    A blank numeric field means "leave it to the strategy" and is stored as
    None, not dropped — the difference matters, because a stored value that is
    simply absent would be filled by the constructor default on the next read
    and the two would disagree.
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
    """What the settings partial needs, for whichever optimizer this is.

    Settings that belong to a group are handed over separately, keyed by name,
    so the group's own template can place each one rather than take them in
    declaration order.
    """
    layout, advanced = grouped(describe_all(optimizer, stored))
    return {
        "optimizer_layout": layout,
        "advanced_params": advanced,
        "has_advanced_params": bool(advanced),
    }
