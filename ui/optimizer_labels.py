"""`core.optimizers` declares the settings, and `optimizer_labels` supplies
their translatable words, since `core` has no Django. A parameter without an
entry falls back to its `OptimizerParam` label.
"""

import json

from django.utils.translation import gettext_lazy as _

from .formatting import sigfigs

#: Every stopping criterion is named, since none is the default.
STOPPED_BY_LABELS = {
    "max_trials": _("all the requested trials ran"),
    "target_score": _("performance surpassed the target"),
    "max_seconds": _("the time limit was reached"),
    "max_trial_seconds": _("the compute budget was used up"),
    "no_improvement_trials": _("the score had stopped improving"),
    "incumbent_confidence": _("the search was confident nothing better remained"),
    "cancelled": _("it was interrupted"),
    "all_failing": _("too many trials were failing"),
    "exhausted": _("every configuration had been tried"),
}


#: Each entry pairs a short label with tooltip text, and "" means the label
#: needs no tooltip. "Search strategy" names the same thing everywhere,
#: placeholders included.
LABELS = {
    "search_strategy": (
        _("Search strategy"),
        _("Gaussian process suits a few numeric hyperparameters. Random forest "
          "suits many of them, or categorical ones."),
    ),
    "use_share_cap": (
        _("Use share cap"),
        _("Bound the initial points by a fraction of the run's budget, so a "
          "longer run explores for longer."),
    ),
    "share_cap": (
        _("Share cap"),
        _("Fraction of the run's trials spent sampling before the model takes "
          "over."),
    ),
    "use_trial_cap": (
        _("Use trial cap"),
        _("Bound the initial points by a fixed number, whatever the run's "
          "budget turns out to be."),
    ),
    "trial_cap": (
        _("Trial cap"),
        _("How many points to sample before the model takes over."),
    ),
    "initial_points_use_max": (
        _("Use the larger of the two"),
        _("With both caps in use, sample up to the larger rather than the "
          "smaller. SMAC itself only ever takes the smaller; the larger is "
          "worked out here and handed to it."),
    ),
    "random_probability": (
        _("Random trial rate"),
        _("How often a trial is drawn at random rather than from the model, "
          "once the model has taken over."),
    ),
    "use_default_config": (
        _("Include the model's defaults"),
        _("Spend one of the exploration trials on the configuration the model "
          "ships with, as a reference point to beat."),
    ),
    "initial_design": (
        _("Sampling method"),
        _("How the exploration trials are drawn. Sobol and Latin hypercube "
          "cover the space more evenly than chance does."),
    ),
    "acquisition": (
        _("Acquisition function"),
        _("How the model scores a candidate. Expected improvement weighs by how "
          "much better it might be; probability of improvement, only by whether."),
    ),
    "acquisition_xi": (
        _("Improvement margin"),
        _("How much better than the best so far a candidate must promise to be. "
          "Higher explores more."),
    ),
    "challengers": (
        _("Candidates per trial"),
        _("How many configurations the model scores before picking one. More is "
          "a better pick and a slower trial."),
    ),
    "local_search_iterations": (
        _("Local search iterations"),
        _("How far the search climbs from each promising candidate before "
          "settling on it."),
    ),
    "retrain_after": (
        _("Refit interval"),
        _("Trials between refits of the model. Higher is faster, and picks from "
          "staler information."),
    ),
    "numeric_steps": (
        _("Grid steps"),
        _("Values tried per numeric hyperparameter. Combinations multiply."),
    ),

    # Surrogate labels lead with "surrogate", since the tuned model can have
    # settings of the same names on the same page.
    "rf_trees": (
        _("Surrogate trees"),
        _("Trees in the forest the search fits over your trials — not in the "
          "model being tuned. More of them are better calibrated and slower, "
          "and the default of ten is not many."),
    ),
    "rf_max_depth": (
        _("Surrogate tree depth"),
        _("How deep the search's own trees may grow. Blank is no limit."),
    ),
    "rf_min_samples_split": (
        _("Surrogate split threshold"),
        _("Fewest trials at a node before the search's trees will split it."),
    ),
    "rf_min_samples_leaf": (
        _("Surrogate leaf size"),
        _("Fewest trials the search's trees will leave in a leaf."),
    ),
    "rf_feature_ratio": (
        _("Surrogate feature ratio"),
        _("Share of your hyperparameters the search considers at each split of "
          "its own trees."),
    ),
    "rf_bootstrapping": (
        _("Bootstrap the surrogate's trees"),
        _("Fit each of the search's trees to a resample of the trials. Off, the "
          "trees agree more, and the confidence the search reports narrows "
          "whether or not it should."),
    ),
    "gp_model_type": (
        _("Surrogate fitting"),
        _("Vanilla fits the process once, by maximum likelihood. MCMC samples "
          "the kernel's own hyperparameters and averages over the result: "
          "better calibrated, and about an order of magnitude slower."),
    ),
    "gp_restarts": (
        _("Surrogate fit restarts"),
        _("Attempts at fitting the process, from different starting points. "
          "More is a better fit and a slower trial."),
    ),
    "gp_normalize_y": (
        _("Normalise surrogate targets"),
        _("Centre and scale the scores before fitting the process to them."),
    ),
}

#: `select` values are named per parameter, since "random" differs between
#: parameters; tooltips explain them.
CHOICES = {
    "search_strategy": {
        "gp": _("Gaussian process"),
        "rf": _("Random forest"),
    },
    "initial_design": {
        "sobol": _("Sobol"),
        "latin_hypercube": _("Latin hypercube"),
        "random": _("Uniformly random"),
        "default_only": _("Defaults only"),
    },
    "acquisition": {
        "ei": _("Expected improvement"),
        "pi": _("Probability of improvement"),
    },
    "gp_model_type": {
        "vanilla": _("Vanilla"),
        "mcmc": _("MCMC (slow)"),
    },
}


#: Exceptions to the strategy-default placeholder, for settings without a
#: default.
PLACEHOLDERS = {
    "share_cap": _("0.25"),
    "trial_cap": _("10 per hyperparameter"),
    "rf_max_depth": _("no limit"),
}

GROUPS = {
    "initial_points": (
        _("Initial points"),
        _("How many configurations are sampled before the model takes over. "
          "SMAC bounds this two ways at once and uses the smaller of them."),
    ),
}

#: Fallback placeholder when the strategy's default cannot be read.
STRATEGY_DEFAULT = _("search strategy default")


def _number(value):
    """`_number` rounds placeholders to four significant figures, since SMAC
    defaults such as 0.08447232371720552 would widen the field. The exact value
    is used and recorded.
    """
    return sigfigs(value)


def placeholder_for(param, strategy_default=None):
    """An empty field means *strategy_default*, so the placeholder shows that
    number instead of naming the strategy.
    """
    if param.default is not None:
        return ""
    if param.name in PLACEHOLDERS:
        return PLACEHOLDERS[param.name]
    if strategy_default is None:
        return STRATEGY_DEFAULT
    return _number(strategy_default)


def range_for(param):
    """`range_for` returns the range as a phrase for the info bubble, since
    browsers report a `min`/`max` breach only after a wrong entry.
    """
    low, high = param.min, param.max
    if low is None and high is None:
        return ""
    if low is not None and high is not None:
        return f"{_number(low)}–{_number(high)}"
    return f"at least {_number(low)}" if low is not None else f"at most {_number(high)}"


def info_for(param, help_text: str) -> str:
    """The info bubble holds meaning and range, and the empty-field value lives
    in the placeholder instead.
    """
    accepted = range_for(param)
    if not accepted:
        return help_text
    accepted = _("Accepts %(range)s.") % {"range": accepted}
    return f"{help_text} {accepted}".strip()


def described(param, value=None, strategy_defaults=None, strategy_defaults_all=None):
    label, help_text = LABELS.get(param.name, (param.label, ""))
    return {
        "name": param.name,
        "label": label,
        "help": info_for(param, help_text),
        "type": param.type,
        "min": param.min,
        "max": param.max,
        "advanced": param.advanced,
        "group": param.group,
        "enabled_by": " ".join(param.enabled_by),
        "when": ("=".join(param.depends_on) if param.depends_on else ""),
        "placeholder": placeholder_for(param, (strategy_defaults or {}).get(param.name)),
        "defaults_by_strategy_json": json.dumps(
            {k: _number(v)
             for k, v in (strategy_defaults_all or {}).get(param.name, {}).items()
             if v is not None}),
        "defaults_by_strategy": {k: _number(v)
                                 for k, v in (strategy_defaults_all or {}).get(param.name, {}).items()
                                 if v is not None},
        "range": range_for(param),
        "value": param.default if value is None else value,
        "checked": bool(param.default if value is None else value),
        "choices": [(v, CHOICES.get(param.name, {}).get(v, v)) for v in param.choices],
    }


def describe_all(optimizer, stored=None):
    """Optimizers without `strategy_defaults()` get placeholders that name the
    strategy instead of quoting a value.
    """
    stored = stored or {}
    by_param = (optimizer.strategy_defaults()
                if hasattr(optimizer, "strategy_defaults") else {})
    selected = stored.get("search_strategy") or getattr(optimizer, "_search_strategy", None)
    current = {name: per_strategy.get(selected)
               for name, per_strategy in by_param.items()}
    return [described(p, stored.get(p.name), current, by_param)
            for p in optimizer.params_schema]


def grouped(described_params):
    """`grouped` places a group where its first member was declared, keeping a
    block of settings from drifting to the bottom.
    """
    layout, advanced, seen = [], [], set()
    for param in described_params:
        if param["advanced"] and not param["group"]:
            advanced.append(param)
            continue
        if not param["group"]:
            layout.append({"param": param})
            continue
        if param["group"] in seen:
            continue
        seen.add(param["group"])
        layout.append({"group": _group(param["group"], described_params)})
    return layout, advanced


def _group(name, described_params):
    label, help_text = GROUPS.get(name, (name, ""))
    return {
        "name": name,
        "label": label,
        "help": help_text,
        "template": f"ui/_group_{name}.html",
        "params": {p["name"]: p for p in described_params if p["group"] == name},
    }


def summary(optimizer_name, params) -> list[tuple[str, str]]:
    """`summary` leaves out other-strategy settings, settings whose switch is
    off, the switches themselves, false booleans, and values left to the
    strategy's default.
    """
    from django.utils.translation import gettext as _now

    from core.registry import optimizer_class

    optimizer = optimizer_class(optimizer_name)
    if optimizer is None:
        return [(name, sigfigs(value) if isinstance(value, float) else str(value))
                for name, value in (params or {}).items() if value is not None]

    params = params or {}
    schema = optimizer.params_schema

    def value_of(param):
        return params.get(param.name, param.default)

    by_name = {p.name: p for p in schema}
    switches = {name for p in schema for name in p.enabled_by}
    rows = []
    for param in schema:
        value = value_of(param)
        if param.name in switches or value is None:
            continue
        if param.depends_on and value_of(by_name[param.depends_on[0]]) != param.depends_on[1]:
            continue
        if not all(value_of(by_name[s]) for s in param.enabled_by if s in by_name):
            continue
        if param.type == "bool":
            if not value:
                continue
            text = _now("Yes")
        elif param.type == "select":
            text = str(CHOICES.get(param.name, {}).get(value, value))
        else:
            text = sigfigs(value)
        rows.append((str(LABELS.get(param.name, (param.label, ""))[0]), text))
    return rows
