"""`tradeoffs` reads the trade-off explainers in
`assistant/knowledge/explainers/tradeoffs/`.
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
    """`notes` filters by form *fields*; `applies_to` uses setting names."""
    out = {}
    for name, note in _read().items():
        if fields is not None and not set(note["fields"]) & set(fields):
            continue
        out[name] = {"title": note["title"], "verdict": note["verdict"],
                     "applies_to": [_setting(f) for f in note["fields"]],
                     "trade_off": note["trade_off"], "stray_when": note["stray_when"]}
    return out


def for_optimizer(optimizer, settings=None) -> dict:
    """*settings* overrides `params_schema` names."""
    names = [p.name for p in optimizer.params_schema] if settings is None else settings
    return notes([f"opt_{name}" for name in names]) if names else {}
