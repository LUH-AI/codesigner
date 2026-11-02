"""When a choice is worth moving from its default, and what that costs: the
trade-off explainers in assistant/knowledge/explainers*/tradeoffs/, one per
SMAC setting or group of settings, the optimized metric and the validation
method, which the assistant proposes changes from.

A note is found by the form fields its `about` names (`opt_trial_cap`,
`optimize_metric`). A file in `explainers/` replaces the draft of the same id.
"""

from functools import cache

from . import explainers

VERDICTS = ("trade-off", "default is better", "change is usually better")


def _setting(field: str) -> str:
    return field.removeprefix("field:").removeprefix("opt_")


@cache
def _read() -> dict:
    found = {}
    for stem, explainer in explainers.read("tradeoffs").items():
        fields, sections = explainer["fields"], explainer["sections"]
        found[stem] = {
            "id": fields["id"],
            "title": fields["title"],
            "verdict": fields["verdict"],
            "fields": [f.removeprefix("field:") for f in fields["about"]],
            "trade_off": sections["The trade-off"],
            "stray_when": [line[2:].strip() for line in sections["When to stray"].splitlines()
                           if line.startswith("- ")],
        }
    return found


def notes(fields=None) -> dict:
    """The notes by name, each with the settings it `applies_to` (an optimizer
    setting by its own name); with *fields*, only the notes about one of
    those form fields."""
    out = {}
    for name, note in _read().items():
        if fields is not None and not set(note["fields"]) & set(fields):
            continue
        out[name] = {"title": note["title"], "verdict": note["verdict"],
                     "applies_to": [_setting(f) for f in note["fields"]],
                     "trade_off": note["trade_off"], "stray_when": note["stray_when"]}
    return out


def for_optimizer(optimizer, settings=None) -> dict:
    """The notes on *optimizer*'s settings, or on those of *settings*."""
    names = [p.name for p in optimizer.params_schema] if settings is None else settings
    return notes([f"opt_{name}" for name in names]) if names else {}
