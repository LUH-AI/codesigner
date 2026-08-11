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

#: label, and a sentence saying what the setting is actually for. The help is
#: the part that matters: these are SMAC's knobs, and someone who already knew
#: what "acquisition function" meant would not be reading them here.
LABELS = {
    "search_strategy": (
        _("Search strategy"),
        _("How the optimizer models the objective. A Gaussian process is the "
          "stronger choice for a handful of numeric hyperparameters and a short "
          "budget; a random forest copes better with many hyperparameters, "
          "categorical choices, and settings that only apply sometimes."),
    ),
    "exploration_ratio": (
        _("Exploration before modelling"),
        _("The share of the run spent sampling the space before the model "
          "starts choosing. Too little and it builds its model from too few "
          "points; too much and it never gets to use what it learned."),
    ),
    "random_probability": (
        _("Random configurations"),
        _("How often to try a random configuration instead of the model's "
          "suggestion, as a fraction. Guards against the search settling into "
          "one region too early."),
    ),
    "use_default_config": (
        _("Try the model's own defaults first"),
        _("Evaluate the configuration the model ships with, so the search has a "
          "known reference point to beat."),
    ),
    "initial_design": (
        _("How the exploration samples"),
        _("Sobol and Latin hypercube cover the space more evenly than chance "
          "does, which is worth more the shorter the run."),
    ),
    "acquisition": (
        _("What makes a configuration worth trying"),
        _("Expected improvement weighs how much better a candidate might be; "
          "probability of improvement only asks whether it is likely to be "
          "better at all, and explores less."),
    ),
    "acquisition_xi": (
        _("Improvement required"),
        _("How much better than the best so far a candidate must promise to be "
          "before it is worth spending a trial on. Raising it pushes the search "
          "to explore."),
    ),
    "challengers": (
        _("Candidates considered per trial"),
        _("How many configurations the optimizer scores against its model "
          "before picking one. More is a better choice for more time between "
          "trials."),
    ),
    "local_search_iterations": (
        _("Candidates refined per trial"),
        _("How many of the best candidates are improved by a local search "
          "around them."),
    ),
    "retrain_after": (
        _("Trials between model refits"),
        _("Refitting every trial is the most informed and the slowest. Only "
          "worth raising once the model itself has become the expensive part."),
    ),
    "numeric_steps": (
        _("Grid steps per numeric hyperparameter"),
        _("How finely a numeric range is divided. Every combination is tried, "
          "so this multiplies out fast."),
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
