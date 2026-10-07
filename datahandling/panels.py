"""Data Handling's panels: the same kind of thing as a dashboard figure.

Every capability in `sections.py` is a panel here, on its section's tab. A
panel is a `ui.figures.Panel`, as a figure is, so the Data Handling page holds
them exactly as the dashboard holds its figures — a slot of the same grid, the
same shapes and widths, placed by `config/figure_layout.toml` — and a page that
gathers panels from both (the setup's steps) need not tell them apart.

Two kinds:

- `ProcessingStep`: one of the experiment's processing choices
  (`core.processing.CHOICES`), as a control, the experiment's model's own
  default among the options and chosen until something else is.
- `Planned`: a capability not built yet, as a card saying what it will do.
"""

from __future__ import annotations

from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy

from core.processing import AUTO, CHOICES, choices_of, model_defaults, unavailable
from ui.figures.base import CARD, DATA, Panel

from .sections import SECTIONS

#: What each processing choice is called on the page.
CHOICE_LABELS = {
    "impute": gettext_lazy("Fill in"),
    "keep": gettext_lazy("Leave for the model"),
    "standardize": gettext_lazy("Standardize"),
    "none": gettext_lazy("Leave as they are"),
    "one_hot": gettext_lazy("One column per label"),
    "categories": gettext_lazy("As categories"),
}

class DataPanel(Panel):
    """A panel on the Data Handling page."""

    page = DATA
    shape = CARD
    template_dir = "datahandling/panels"
    #: One template for every panel of a kind, rather than one per key.
    shared_template = ""
    #: What the panel is for, said under its heading.
    description = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.shared_template:
            cls.template = cls.shared_template


class ProcessingStep(DataPanel):
    """One processing choice, as a labelled select."""

    shared_template = "datahandling/panels/processing_step.html"
    #: Its key in `core.processing.CHOICES`.
    step = ""

    @classmethod
    def control(cls, exp):
        """The select for *exp*: one option per choice, the model's own
        default among them named as such — "Leave as they are (Random Forest
        default)" — and chosen until something else is. The default is posted
        as "auto", so it goes on following the model if the model changes.
        A choice the model cannot take is offered disabled."""
        model = _model_of(exp)
        chosen = choices_of(exp.data.processing)[cls.step]
        if model is None:
            return {"name": cls.step, "value": chosen,
                    "options": [{"value": v, "label": _("As the model needs") if v == AUTO
                                 else CHOICE_LABELS[v], "disabled": False}
                                for v in CHOICES[cls.step]]}
        default = model_defaults(model)[cls.step]
        options = []
        for choice in CHOICES[cls.step]:
            if choice == AUTO:
                continue
            is_default = choice == default
            options.append({
                "value": AUTO if is_default else choice,
                "label": (_("%(choice)s (%(model)s default)") % {
                    "choice": CHOICE_LABELS[choice], "model": model.name}
                          if is_default else str(CHOICE_LABELS[choice])),
                "disabled": unavailable(model, cls.step, choice),
            })
        return {"name": cls.step, "options": options,
                "value": AUTO if chosen in (AUTO, default) else chosen}


def _model_of(exp):
    """The registry model *exp* names, told its task, or None for an uploaded
    one (which is given the columns as they are) or none at all."""
    from ui.registry import MODELS

    model = MODELS.get(exp.data.model_name)
    if model is None or exp.data.model_file:
        return None
    model = model.__class__()
    model.task = exp.data.task
    return model


def describe(exp, step, choice, lower=True) -> str:
    """*choice* at *step* as a reader says it: "auto" as what it is for
    *exp*'s model — "standardize (SVM default)". Mid-sentence unless *lower*
    is false, when it reads as the card's option does."""
    model = _model_of(exp)
    if choice == AUTO:
        if model is None:
            text = str(_("As the model needs"))
        else:
            text = _("%(choice)s (%(model)s default)") % {
                "choice": str(CHOICE_LABELS[model_defaults(model)[step]]), "model": model.name}
    else:
        text = str(CHOICE_LABELS.get(choice, choice))
    return text[:1].lower() + text[1:] if lower else text


def is_default(exp, step, choice) -> bool:
    """Whether *choice* is what *exp*'s model is given at *step* anyway."""
    model = _model_of(exp)
    return choice == AUTO or (model is not None and model_defaults(model)[step] == choice)


class Planned(DataPanel):
    """A capability not built yet: what it will do."""

    shared_template = "datahandling/panels/planned.html"


def _panel(section, capability):
    base = ProcessingStep if capability.step else Planned
    return type(f"{base.__name__}_{capability.key}", (base,), {
        "key": capability.key,
        "label": capability.title,
        "description": capability.description,
        "home_tab": section.slug,
        "step": capability.step,
        "__module__": __name__,
    })


#: Every Data Handling panel, section by section, in the order `sections.py`
#: lists them.
DATA_PANELS = tuple(_panel(section, capability)
                    for section in SECTIONS for capability in section.planned)
DATA_PANELS_BY_KEY = {panel.key: panel for panel in DATA_PANELS}
PROCESSING_PANELS = tuple(p for p in DATA_PANELS if issubclass(p, ProcessingStep))
