"""The optimizer and its settings, as a form: one select, and one settings
panel per optimizer that has settings.

Two pages offer it: step 2 of a draft's setup, and an experiment's page before
it has any trials. Once there are trials the search that found them is fixed —
the next run continues it — and while a run is under way it is the one in use.
"""

from __future__ import annotations

from django.utils.translation import gettext as _


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
    from ..registry import OPTIMIZERS
    from ..views import _optimizer_param_context

    data = exp.data
    selected = data.optimizer_name if data.optimizer_name in OPTIMIZERS else next(iter(OPTIMIZERS))
    panels = []
    for key, optimizer in OPTIMIZERS.items():
        if not optimizer.params_schema:
            continue
        stored = data.optimizer_params if key == data.optimizer_name else None
        panels.append({"key": key, "selected": key == selected,
                       "disabled": locked or key != selected,
                       **_optimizer_param_context(optimizer, stored)})
    return {"optimizer_choices": list(OPTIMIZERS), "optimizer_selected": selected,
            "optimizer_panels": panels, "optimizer_locked": locked}


def save(request, exp, data, *, partial=False) -> str | None:
    """Store the optimizer and settings in *data* on *exp*, as *request*'s
    person. Returns why it could not, or None. A change is an entry in the
    experiment's history (a draft has none).

    *data* is the form as posted, `optimizer_name` and `opt_<setting>` keys.
    With *partial* it says only what changes: a missing `optimizer_name` keeps
    the current optimizer, and a missing setting keeps its current value."""
    from ..registry import OPTIMIZERS
    from ..views import _posted_optimizer_params
    from . import history

    refusal = why_fixed(exp)
    if refusal:
        return refusal
    name = data.get("optimizer_name")
    if partial and not name:
        name = exp.data.optimizer_name
    chosen = OPTIMIZERS.get(name)
    if chosen is None:
        return _("Choose an optimizer.")
    base = None
    if partial:
        base = (type(chosen).known_params(exp.data.optimizer_params or {})
                if name == exp.data.optimizer_name else {})
    optimizer = type(chosen)(**type(chosen).known_params(
        _posted_optimizer_params(data, chosen, base)))
    data = exp.data
    old = {"name": data.optimizer_name, "params": data.optimizer_params or {}}
    new = {"name": optimizer.name, "params": optimizer.get_params()}
    data.optimizer_name, data.optimizer_params = new["name"], new["params"]
    data.save(update_fields=["optimizer_name", "optimizer_params"])
    if new != old:
        history.record(exp, "optimizer_changed", user=request.user, old=old, new=new)
    return None
