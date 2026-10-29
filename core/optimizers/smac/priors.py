import logging

from ... import priors as priors_module

#: A prior that fails to apply must log instead of raising or staying silent.
logger = logging.getLogger(__name__)

#: `TabulatedPrior` coarsens the grid anyway, so 512 points only need to catch
#: the editor's narrowest peak, σ = 0.004.
PRIOR_GRID = 512

#: Per hyperparameter, because the open space grows with each one.
PRIOR_JUDGING_SAMPLES_PER_HYPERPARAMETER = 1000


def decay_beta(n_trials) -> float:
    """The search and the figure share `decay_beta`, so a caption cannot drift
    from the prior's decay. The factor comes from the budget, since a
    finished-trial count would grow and stop the decay.
    """
    return max(float(n_trials or 0) / 10.0, 1.0)


def stamped(report, smac):
    """`stamped` marks *report* with the trial count and time the prior took
    effect, and passes None through."""
    if report is None:
        return None
    from datetime import datetime, timezone

    return {**report, "at_trial": len(smac.runhistory),
            "at": datetime.now(timezone.utc).isoformat()}


def waits_for_the_design(priors) -> bool:
    """`apply_priors` puts every belief into one weight, so one waiting belief
    delays them all.
    """
    return any(bool((stated or {}).get("delay_decay")) for stated in (priors or {}).values())


def prior_grid(hyperparameter):
    """A prior is evaluated against vectorized values and stated on the unit
    interval, which coincide for a numerical hyperparameter. A categorical
    is evaluated only at its choices, placed evenly along the stated axis.
    """
    choices = getattr(hyperparameter, "choices", None)
    if choices:
        vectors = [float(hyperparameter.to_vector(c)) for c in choices]
        last = max(len(choices) - 1, 1)
        return vectors, [i / last for i in range(len(choices))]
    grid = [i / (PRIOR_GRID - 1) for i in range(PRIOR_GRID)]
    return grid, grid


def apply_priors(smac, scenario, config_space, priors, budget=None, acceptance=None):
    """*priors* maps hyperparameters to stated `kind`, `params` and `decay`,
    and `core.priors.density_from` evaluates each density here. All beliefs
    enter as one `TabulatedPrior` over the whole space, since
    `validate_against` needs every hyperparameter in column order and
    separate priors would multiply as an ensemble. A prior that fails is
    reported and skipped, and the returned report goes onto the result's
    metadata so a file records whether the prior took.
    """
    if not priors:
        return None
    try:
        from smac.acquisition.weight import TabulatedPrior, get_decay_schedule
    except ImportError:
        logger.warning("This SMAC has no TabulatedPrior, so stated priors are ignored. "
                       "codesigner needs the branch carrying the acquisition weight layer.")
        return {"applied": False,
                "reason": "this SMAC has no acquisition weight layer"}

    tables, shape, stated_beta, stated_ratio = {}, "none", None, None
    statements = []
    for name, stated in (priors or {}).items():
        if name not in config_space:
            continue
        # The figure gets its density from the same `core.priors` function
        # through `prior_density`.
        vectors, units = prior_grid(config_space[name])
        values = priors_module.density_from(
            (stated or {}).get("kind") or "uniform",
            (stated or {}).get("params") or {}, units) or []
        if len(values) >= 2:
            # The table uses vector positions, while the density came from
            # the unit interval.
            tables[name] = [[v, y] for v, (_, y) in zip(vectors, values)]
            statements.append(stated or {})

    # One weight carries one schedule, so the last named belief settles it.
    for stated in statements:
        decay = stated.get("decay")
        if isinstance(decay, dict):
            shape = decay.get("shape") or "none"
            # πBO and DynaBO state β as a budget ratio, and an absolute
            # `beta` still loads for older priors.
            if decay.get("beta_ratio") is not None:
                stated_ratio = float(decay["beta_ratio"])
            elif decay.get("beta") is not None:
                stated_beta = float(decay["beta"])
        else:
            shape = decay or "none"
    if not tables:
        return {"applied": False,
                "reason": "no stated prior named a hyperparameter of this "
                          "search space, or none described a curve"}

    # The decay factor comes from *budget*, the run's configured budget,
    # because a re-walk's scenario is sized to its replayed trials. A stated
    # ratio resolves against that budget, and `decay_beta` (N/10, as in πBO
    # and DynaBO) stands in otherwise.
    run_budget = (budget if budget is not None
                  else getattr(scenario, "n_trials", 0))
    if stated_ratio is not None:
        beta = max(float(run_budget or 0) * stated_ratio, 1e-6)
    elif stated_beta is not None:
        beta = stated_beta
    else:
        beta = decay_beta(run_budget)

    try:
        prior = TabulatedPrior(config_space, tables)
        prior.validate_against(config_space)
        # `add_prior` returns None when the acceptance policy refuses the
        # belief, and the default policy accepts everything.
        key = smac.add_prior(prior, key="codesigner",
                             decay=get_decay_schedule(shape, beta),
                             acceptance_policy=acceptance)
    except Exception as error:  # noqa: BLE001 — a prior must not cost a run
        logger.warning("Could not apply the stated prior, continuing without it: %s", error)
        return {"applied": False, "reason": str(error)}

    if key is None:
        # A refusal is the only failure that judges the reader's belief, so
        # the report marks it.
        return {"applied": False, "rejected": True,
                "hyperparameters": sorted(tables), "decay": shape, "beta": beta,
                "reason": "the surrogate scores this region worse than the "
                          "incumbent's neighbourhood by more than the "
                          "tolerance allows"}

    return {"applied": True, "key": key, "hyperparameters": sorted(tables),
            "decay": shape, "beta": beta}
