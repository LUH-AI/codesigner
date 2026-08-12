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

#: label, and what it means. The label is a *name* — as short as the thing can
#: be called — and everything that explains it goes in the second string, which
#: the form shows as a tooltip rather than as a line of prose under the control.
#: An empty string means there is nothing to add to the name.
#:
#: Terminology is shared with the settings themselves: what the dropdown calls a
#: "search strategy" is called that everywhere, including the placeholder on the
#: fields whose default it supplies.
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

    # The surrogate. Every label leads with "surrogate" deliberately: the model
    # being tuned can have settings of the same names — a Random Forest tuned
    # over `max_depth` while the search's own forest has one too — and the two
    # appear on the same page.
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

#: The values of `select` parameters. Kept apart from `LABELS` because a choice
#: is named per parameter, not globally — "random" means something different
#: under a sampling scheme than it would elsewhere. Names only: what
#: distinguishes them belongs in the parameter's tooltip, said once.
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


#: What an empty field means, where it is not "whatever the search strategy
#: uses". Only a setting with no default of its own can be empty at all, and
#: most of those are the strategy's to decide — these are the exceptions.
PLACEHOLDERS = {
    "share_cap": _("0.25"),
    "trial_cap": _("10 per hyperparameter"),
    "rf_max_depth": _("no limit"),
}

#: A group's title, for the blocks the form lays out itself. Keyed by the
#: `group` on the parameters that belong to it.
GROUPS = {
    "initial_points": (
        _("Initial points"),
        _("How many configurations are sampled before the model takes over. "
          "SMAC bounds this two ways at once and uses the smaller of them."),
    ),
}

#: The rest of them.
STRATEGY_DEFAULT = _("search strategy default")


def placeholder_for(param):
    """What to show in *param*'s field while it is empty, if anything."""
    if param.default is not None:
        return ""
    return PLACEHOLDERS.get(param.name, STRATEGY_DEFAULT)


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
        "group": param.group,
        # Space-separated for the data attribute: every one of these has to be
        # checked for the field to be worth filling in.
        "enabled_by": " ".join(param.enabled_by),
        # "search_strategy=rf" — the form hides this field unless that setting
        # has that value. A string because it is going into a data attribute.
        "when": ("=".join(param.depends_on) if param.depends_on else ""),
        # A setting with a default of its own shows it, and needs no
        # placeholder; one without is empty, and the field says what filling it
        # in would displace.
        "placeholder": placeholder_for(param),
        "value": param.default if value is None else value,
        "checked": bool(param.default if value is None else value),
        "choices": [(v, CHOICES.get(param.name, {}).get(v, v)) for v in param.choices],
    }


def describe_all(optimizer, stored=None):
    """Every setting of *optimizer*, with the experiment's stored values applied."""
    stored = stored or {}
    return [described(p, stored.get(p.name)) for p in optimizer.params_schema]


def grouped(described_params):
    """The panel's layout: what to render, in the order it was declared.

    Each entry is either a single setting or a whole group, and a group sits
    where its first member was declared rather than after everything loose —
    otherwise a block of important settings would drift to the bottom of the
    panel because it happens to be a block.
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
        # Keyed by name so the group's own template can place each setting
        # deliberately rather than take them in declaration order.
        "params": {p["name"]: p for p in described_params if p["group"] == name},
    }
