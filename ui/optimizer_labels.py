"""What to call an optimizer's settings, and how to explain them.

The settings themselves are declared in `core.optimizers` — name, type, bounds,
default. This layer has the words. The split is the same one the stopping
criteria use: `core` has no Django and so cannot mark a string for translation,
and a plain-English label frozen into the domain layer would be untranslatable
wherever it surfaced.

Keyed by parameter name. A parameter with no entry falls back to the plain label
on its `OptimizerParam`, so adding a setting is never blocked on writing prose
for it first.
"""

from django.utils.translation import gettext_lazy as _

#: label, and a short line under it. Kept to one line: these sit in a form
#: someone is filling in, not a manual they are reading, and a paragraph under
#: every field turns the panel into a wall. An empty string means the label
#: already says it.
LABELS = {
    "search_strategy": (
        _("Search strategy"),
        _("Gaussian process for a few numeric settings; random forest for many, "
          "or categorical ones."),
    ),
    "exploration_ratio": (
        _("Exploration before modelling"),
        _("Share of the budget sampled before the model takes over."),
    ),
    "random_probability": (
        _("Random configurations"),
        _("Fraction of trials taken at random rather than from the model."),
    ),
    "use_default_config": (
        _("Try the model's own defaults first"),
        "",
    ),
    "initial_design": (
        _("How the exploration samples"),
        _("Sobol and Latin hypercube spread more evenly than chance."),
    ),
    "acquisition": (
        _("What makes a configuration worth trying"),
        _("Expected improvement weighs by how much; probability, only whether."),
    ),
    "acquisition_xi": (
        _("Improvement required"),
        _("Margin over the best so far. Higher explores more."),
    ),
    "challengers": (
        _("Candidates considered per trial"),
        _("More candidates, better pick, slower."),
    ),
    "local_search_iterations": (
        _("Candidates refined per trial"),
        _("How many of the best get a local search around them."),
    ),
    "retrain_after": (
        _("Trials between model refits"),
        _("Higher is faster and less informed."),
    ),
    "numeric_steps": (
        _("Grid steps per numeric hyperparameter"),
        _("Values per numeric range. Combinations multiply."),
    ),
}

#: The values of `select` parameters. Kept apart from `LABELS` because a choice
#: is named per parameter, not globally — "random" means something different
#: under a sampling scheme than it would elsewhere.
CHOICES = {
    "search_strategy": {
        "gp": _("Gaussian process — a few continuous settings"),
        "rf": _("Random forest — many settings, or categorical ones"),
    },
    "initial_design": {
        "sobol": _("Sobol — evenly spread"),
        "latin_hypercube": _("Latin hypercube"),
        "random": _("Uniformly random"),
        "default_only": _("Only the model's defaults"),
    },
    "acquisition": {
        "ei": _("Expected improvement"),
        "pi": _("Probability of improvement"),
    },
}


def described(param, value=None):
    """One `OptimizerParam` as the template needs it: words, and a value."""
    label, help_text = LABELS.get(param.name, (param.label, ""))
    return {
        "name": param.name,
        "label": label,
        "help": help_text,
        "type": param.type,
        "min": param.min,
        "max": param.max,
        "advanced": param.advanced,
        # An unset value renders empty, and the placeholder says the strategy
        # decides — which is true, and better than showing a number that is not
        # what will be used.
        "value": param.default if value is None else value,
        "checked": bool(param.default if value is None else value),
        "choices": [(v, CHOICES.get(param.name, {}).get(v, v)) for v in param.choices],
    }


def describe_all(optimizer, stored=None):
    """Every setting of *optimizer*, with the experiment's stored values applied."""
    stored = stored or {}
    return [described(p, stored.get(p.name)) for p in optimizer.params_schema]
