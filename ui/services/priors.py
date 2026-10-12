"""Stating a belief about one hyperparameter, and taking it back.

What the prior figure's controls do, as functions of the experiment, so the
page and anything else that states a belief store it the same way.
"""


def state(request, exp, hp_name, prior) -> dict:
    """Record *prior* about *hp_name* on *exp*, as *request*'s person, and
    return every hyperparameter's prior now. None withdraws it.

    *prior* is `{"kind", "params", "decay", "delay_decay"}` as the figure sends
    it. Uniform states nothing, so it is stored as nothing rather than as a
    flat table the optimizer would multiply by for no reason.
    """
    from ..views import _decay_record, _inside_initial_design, _rebuild_experiment
    from . import history

    priors = dict(exp.data.priors or {})
    if prior is None:
        priors.pop(hp_name, None)
    else:
        # The trial this belief was stated at. A decay schedule is an exponent
        # over the trials *since* a prior was supplied, not since the run began
        # — a belief stated at trial 120 arrives at full strength rather than
        # inheriting the near-flat exponent an older one has decayed to.
        built = _rebuild_experiment(exp)
        result_now = built["result"] if built else None
        at_trial = len(result_now.trials if result_now else [])
        previous = (exp.data.priors or {}).get(hp_name) or {}
        priors[hp_name] = {
            "kind": prior.get("kind") or "uniform",
            "params": prior.get("params") or {},
            "decay": _decay_record(prior.get("decay"), exp, at_trial),
            # Whether the decay waits for the initial design to finish. Only
            # ever true while there is a design still to finish: stated after
            # it, there is nothing to wait for and carrying the flag would
            # invite a later reader to think there was. The page offers the
            # choice in the same situation and defaults it on.
            "delay_decay": bool(prior.get("delay_decay")
                                and _inside_initial_design(result_now)),
            # Kept across an edit of the same shape: re-typing a sigma is not
            # restating the belief, and re-anchoring on every keystroke would
            # hold the decay at zero forever.
            "at_trial": (previous.get("at_trial")
                         if previous.get("kind") == prior.get("kind") else at_trial),
        }

    stated_before = (exp.data.priors or {}).get(hp_name)
    exp.data.priors = priors
    exp.data.save(update_fields=["priors"])
    history.record_prior(exp, hp_name, stated_before, priors.get(hp_name), user=request.user)
    return priors


def reset(request, exp, hp_name) -> bool:
    """Put *hp_name*'s belief back to what *exp*'s last run searched under, as
    *request*'s person, or to uniform when there is nothing to go back to.
    Returns whether there was something to go back to.

    The last run that got as far as recording anything: a pending one has
    searched under nothing yet, and going back to its priors would be going
    back to the very edits this is meant to undo.
    """
    from . import history

    last = (exp.runs.exclude(status="pending")
            .order_by("-id").values_list("priors", flat=True).first())
    restored = (last or {}).get(hp_name)

    priors = dict(exp.data.priors or {})
    stated_before = priors.get(hp_name)
    if restored:
        priors[hp_name] = restored
    else:
        priors.pop(hp_name, None)
    exp.data.priors = priors
    exp.data.save(update_fields=["priors"])
    if stated_before != priors.get(hp_name):
        history.record(exp, "prior_reset", user=request.user, hyperparameter=hp_name,
                       old=stated_before, new=priors.get(hp_name))
    return bool(restored)
