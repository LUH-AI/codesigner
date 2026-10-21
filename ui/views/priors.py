import json
import threading
from collections import OrderedDict

import numpy as np

from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import EDIT, RUN, VIEW, policy
from core import projected_prior
from core import priors as prior_math
from core.metrics import metric_for
from experiments.services import history
from experiments.services import priors as prior_service
from experiments.services import run as run_service
from experiments.services.settings import SETTING_BOUNDS
from experiments.services.snapshot import config_space_for, rebuild_experiment, rebuild_result

from .. import projected
from ..experiment_settings import SETTING_DEFAULTS, resolve_settings
from ..figures import acquisition_slice_plots
from ..formatting import sigfigs
from ..permissions import BOTH, experiment_view
from .common import _judges_priors, _model_available, _owner, _plain, _plot_json, logger


def _proposals(exp, metric):
    """The sampled configurations a proposal is picked from, and what the
    surrogate predicts of each for *metric* — the configuration cube's point
    cloud, out of the same cache. `(built, sample)`, or `(built, None)` when
    there is no sample to pick from."""
    built, _config_space, sample = _cube_sample(exp, metric)
    return built, sample


def _cube_sample(exp, metric):
    """`(built, config_space, sample)`: the configuration cube's shared sample
    for *metric*, with what it was drawn from — any of the last two None when
    there is nothing to draw from."""
    built = rebuild_experiment(exp)
    if not built or built["result"] is None or not built["result"].trials:
        return built, None, None
    config_space = config_space_for(built)
    if config_space is None:
        return built, None, None
    _surrogate, sample = _slice_model(exp, built, config_space, metric)
    return built, config_space, sample if sample and sample.get("configurations") else None


def _projected_exponents(exp, records, trials, inside_design):
    """How strongly each stated projected prior still counts — the same decay
    every prior has, see `experiments.services.priors.decayed`."""
    return [(prior_service.decayed(exp, record, trials, inside_design) or {}).get("exponent", 1.0)
            for record in records]


def _projected_weights(exp, built, config_space, X):
    """Every stated projected prior's weight at the vectorized rows *X*, as the
    page multiplies it in — rounded for the wire, or None with nothing stated."""
    records = list(exp.data.projected_priors or [])
    if not records or config_space is None or X is None or not len(X):
        return None
    result = built["result"]
    exponents = _projected_exponents(exp, records, len(result.trials),
                                     prior_math.inside_initial_design(result))
    weights = projected.stated_weights(records, exponents, config_space, np.asarray(X, float))
    return projected.rounded(weights) if weights is not None else None


def _projected_answer(exp, built, config_space, sample):
    """What the card and the cube are told after a projected prior changes: the
    stated priors as the card lists them, and their weight over the sample."""
    return {"stated": [projected.summary(r, config_space) for r in exp.data.projected_priors or []],
            "weights": _projected_weights(exp, built, config_space,
                                          projected.sample_vectors(sample) if sample else None)}


def _drawn(request, exp):
    """The posted drawing, or None unless it is a JSON object naming one of
    *exp*'s metrics."""
    try:
        body = json.loads(request.body or b"{}")
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("metric") not in exp.data.metric_names:
        return None
    return body


@require_POST
@experiment_view(VIEW)
def projected_prior_preview(request, exp):
    """A prior being drawn in the configuration cube, before it is stated: the
    record it would be, its weight over the cube's sample, and the shapes the
    cube draws for it. VIEW: looking at a belief changes nothing."""
    body = _drawn(request, exp)
    if body is None:
        return HttpResponseBadRequest("invalid request")
    built, config_space, sample = _cube_sample(exp, body["metric"])
    if sample is None:
        return JsonResponse({"error": _("There is no sample of this space to draw on.")}, status=400)
    try:
        record, Z = projected.draft_record(config_space, sample, body.get("draft") or {},
                                           body["metric"])
    except (projected_prior.PriorError, KeyError, TypeError, ValueError) as error:
        return JsonResponse({"error": str(error), "point": getattr(error, "point", None)},
                            status=400)
    return JsonResponse({
        "record": {k: v for k, v in record.items() if k != "map"},
        "weights": projected.rounded(projected_prior.density(Z, record)),
        "geometry": projected.geometry(record, Z, config_space),
        "profile": projected_prior.profile(record),
    })


@require_POST
@experiment_view(EDIT)
def state_projected_prior(request, exp):
    """State the prior drawn in the cube, or change a stated one's parameters
    and decay. Its points and projection are what it was drawn in and stay put:
    a different shape is a different belief."""
    body = _drawn(request, exp)
    if body is None:
        return HttpResponseBadRequest("invalid request")
    built, config_space, sample = _cube_sample(exp, body["metric"])
    if sample is None:
        return JsonResponse({"error": _("There is no sample of this space to draw on.")}, status=400)
    assert built is not None
    result = built["result"]
    records = list(exp.data.projected_priors or [])
    try:
        if body.get("id"):
            at = next((i for i, r in enumerate(records) if r.get("id") == body["id"]), None)
            if at is None:
                return HttpResponseBadRequest("no such prior")
            old = records[at]
            new = projected.with_params(old, body.get("params") or {})
        else:
            old = None
            new, _Z = projected.draft_record(config_space, sample, body.get("draft") or {},
                                             body["metric"])
            new["id"] = projected.new_id()
            new["at_trial"] = len(result.trials)
        new["decay"] = prior_math.decay_record(
            body.get("decay"), prior_service.run_budget(exp, int(new.get("at_trial") or 0)))
        new["delay_decay"] = bool(body.get("delay_decay") and prior_math.inside_initial_design(result))
        projected_prior.check(new, config_space)
    except (projected_prior.PriorError, KeyError, TypeError, ValueError) as error:
        return JsonResponse({"error": str(error)}, status=400)

    if old is None:
        records.append(new)
    else:
        records[at] = new  # pyright: ignore[reportPossiblyUnboundVariable]
    exp.data.projected_priors = records
    exp.data.save(update_fields=["projected_priors"])
    history.record_projected_prior(exp, new["id"], old, new, user=request.user)
    return JsonResponse(_projected_answer(exp, built, config_space, sample))


@require_POST
@experiment_view(EDIT)
def withdraw_projected_prior(request, exp):
    """Take a stated projected prior back."""
    body = _drawn(request, exp)
    if body is None:
        return HttpResponseBadRequest("invalid request")
    records = list(exp.data.projected_priors or [])
    old = next((r for r in records if r.get("id") == body.get("id")), None)
    if old is None:
        return HttpResponseBadRequest("no such prior")
    exp.data.projected_priors = [r for r in records if r is not old]
    exp.data.save(update_fields=["projected_priors"])
    history.record_projected_prior(exp, old["id"], old, None, user=request.user)
    built, config_space, sample = _cube_sample(exp, body["metric"])
    return JsonResponse(_projected_answer(exp, built, config_space, sample))


def _proposal_index(source):
    """`n` from *source* as an index into the sample, or None."""
    raw = source.get("n", "")
    return int(raw) if raw.isdigit() else None


def _run_refusal(request, exp):
    """Why this viewer cannot start a run of *exp* now, or None."""
    if not policy().may(request, exp, RUN):
        return _("You can view this experiment but not run it.")
    if exp.is_running:
        return _("A run is under way.")
    if not exp.data.has_dataset or not _model_available(exp):
        return _("This experiment cannot be run here.")
    return policy().custom_model_refusal(exp, request.user) or None


@experiment_view(VIEW)
def proposal_panel(request, exp):
    """The selected-config panel for a proposal: one of the configuration
    cube's sampled configurations, picked off its point cloud.

    A configuration nobody has run, so what it shows is the surrogate's
    prediction for it rather than a score, and it offers to run it.
    """
    metric = request.GET.get("metric", "")
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    n = _proposal_index(request.GET)
    built, sample = _proposals(exp, metric)
    if sample is None or n is None or n >= len(sample["configurations"]):
        return HttpResponseBadRequest("invalid proposal")
    assert built is not None

    result = built["result"]
    scoring = result.metric(metric)
    best = result.trials[result.best_index(metric)].scores[metric]
    mu, sigma = sample["mu"][n], sample["sigma"][n]
    return render(request, "ui/_selected_proposal_inner.html", {
        "proposal": {
            "metric": metric,
            "metric_title": scoring.title,
            "number": n + 1,
            "of": len(sample["configurations"]),
            "index": n,
            "mu": mu,
            "sigma": sigma,
            "delta": None if mu is None else mu - best,
            "delta_better": mu is not None and scoring.better(mu, best),
            "config": list(sample["configurations"][n].items()),
        },
        "experiment": exp,
        "run_refusal": _run_refusal(request, exp),
    })


@require_POST
@experiment_view(RUN)
def run_proposal(request, exp):
    """Run one proposal as a trial: a run of one, told which configuration to
    evaluate rather than asked to search for one, optimizing the metric the
    experiment already optimizes so that nothing else about the search moves.
    The run is recorded like any other, with the proposal it ran."""
    metric = request.POST.get("metric", "")
    n = _proposal_index(request.POST)
    if metric not in exp.data.metric_names or n is None or _run_refusal(request, exp):
        return redirect("ui:experiment_detail", pk=exp.pk)
    _built, sample = _proposals(exp, metric)
    if sample is None or n >= len(sample["configurations"]):
        return redirect("ui:experiment_detail", pk=exp.pk)

    number = n + 1
    run = run_service.create_run(
        exp, {"max_trials": 1}, exp.data.current_metric or metric,
        started_by=_owner(request),
        configurations=[{"config": dict(sample["configurations"][n]),
                         "origin": _("Proposal %(n)s") % {"n": number},
                         "proposal": number}])
    run_service.start_background_run(run.id)
    return redirect("ui:experiment_detail", pk=exp.pk)


#: Fitted slice surrogates, keyed by everything they depend on.
#:
#: A slice's model is a function of the experiment, the metric and the trials —
#: **not** of which hyperparameter is on the axis. Without this, clicking through
#: a six-hyperparameter model paid six identical fits, measured at roughly 300ms
#: for a Gaussian process and 750ms for a random forest over fifteen trials.
#:
#: Process-local and small rather than Django's cache: a fitted GP or forest is
#: derived data no other process needs, cheap to rebuild, and awkward to pickle.
#: The trial count is in the key, which is what makes a live run invalidate
#: itself as results arrive. Both figures ask for the same key as the page
#: opens, so the second waits on the first's fit rather than fitting again —
#: one lock per key, so nothing else waits.
_SLICE_SURROGATES = OrderedDict()


_SLICE_SURROGATE_CACHE = 8


_SLICE_LOCKS = {}


_SLICE_LOCKS_GUARD = threading.Lock()


def _slice_model(exp, built, config_space, metric):
    """The fitted surrogate for this (experiment, metric, trials), and an even
    sample of the whole space through it — computed once, and read by both
    figures that need them.

    Returns `(surrogate, sample)`, the sample being
    `BaseOptimizer.sample_prediction_space`'s answer. Neither depends on which
    hyperparameter the acquisition figure has on its axis, nor on which view
    the configuration cube is in, nor on the priors — a prior weights the
    acquisition, which both figures compute in the browser from the sample's
    mean and spread. So one fit and one sample serve the acquisition figure's
    cloud (`_sample_cloud`) and the cube's prediction layer alike.

    `(None, None)` when a surrogate could not be fitted, which leaves the caller
    to fit and to report why — a failure is not cached, so the reader gets the
    reason rather than silence.
    """
    result = built["result"]
    key = (exp.pk, metric, len(result.trials), built["seed"])
    with _SLICE_LOCKS_GUARD:
        lock = _SLICE_LOCKS.setdefault(key, threading.Lock())
    with lock:
        cached = _SLICE_SURROGATES.get(key)
        if cached is not None:
            _SLICE_SURROGATES.move_to_end(key)
            return cached

        optimizer = built["optimizer"]
        surrogate, _warning = optimizer.slice_surrogate(
            config_space, result.trials, metric, built["seed"])
        if surrogate is None:
            return None, None

        sample = optimizer.sample_prediction_space(
            config_space, result.trials, metric, seed=built["seed"], surrogate=surrogate)
        _SLICE_SURROGATES[key] = (surrogate, sample)
        while len(_SLICE_SURROGATES) > _SLICE_SURROGATE_CACHE:
            evicted, _ = _SLICE_SURROGATES.popitem(last=False)
            with _SLICE_LOCKS_GUARD:
                _SLICE_LOCKS.pop(evicted, None)
        return surrogate, sample


def _sample_cloud(sample):
    """The shared sample as the acquisition figure's cloud: `(rows, mu, sigma)`
    as arrays, with what could not be predicted or placed as NaN — see
    `compute_incumbent_slice`, which leaves those out per hyperparameter."""
    if not sample or not sample.get("vectors"):
        return None

    def array(values):
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    rows = np.array([[np.nan if v is None else v for v in row] for row in sample["vectors"]],
                    dtype=float)
    return rows, array(sample["mu"]), array(sample["sigma"])


@require_POST
@experiment_view(EDIT, drafts=BOTH)
def save_prior(request, exp):
    """Record what the reader believes about one hyperparameter.

    `EDIT` rather than `VIEW`: a prior is not a way of looking at a run, it is a
    statement that changes what the next one searches.

    The body carries the shape and its parameters *and* the density already
    evaluated on the grid. Only the second reaches SMAC — see
    `Experiment.priors` — and the first exists so the figure can put the
    controls back where they were left. The server does not evaluate a density
    itself, deliberately: two implementations of "a Normal with this σ" would
    have to agree forever, and the one in the browser is the one the reader saw.
    """
    try:
        body = json.loads(request.body or b"{}")
    except ValueError:
        return HttpResponseBadRequest("invalid json")

    hp_name = body.get("hp") or ""
    if not hp_name:
        return HttpResponseBadRequest("no hyperparameter named")

    priors = dict(exp.data.priors or {})
    prior = body.get("prior")
    if prior is None:
        # Uniform states nothing, so it is stored as nothing rather than as a
        # flat table the optimizer would multiply by for no reason.
        priors.pop(hp_name, None)
    else:
        # The trial this belief was stated at. A decay schedule is an exponent
        # over the trials *since* a prior was supplied, not since the run began
        # — a belief stated at trial 120 arrives at full strength rather than
        # inheriting the near-flat exponent an older one has decayed to.
        built = rebuild_experiment(exp)
        at_trial = len((built["result"].trials if built and built["result"] else []))
        previous = (exp.data.priors or {}).get(hp_name) or {}
        result_now = built["result"] if built else None
        priors[hp_name] = {
            "kind": prior.get("kind") or "uniform",
            "params": prior.get("params") or {},
            "decay": prior_math.decay_record(prior.get("decay"),
                                             prior_service.run_budget(exp, at_trial)),
            # Whether the decay waits for the initial design to finish. Only
            # ever true while there is a design still to finish: stated after
            # it, there is nothing to wait for and carrying the flag would
            # invite a later reader to think there was. The page offers the
            # choice in the same situation and defaults it on.
            "delay_decay": bool(prior.get("delay_decay")
                                and prior_math.inside_initial_design(result_now)),
            # Kept across an edit of the same shape: re-typing a sigma is not
            # restating the belief, and re-anchoring on every keystroke would
            # hold the decay at zero forever.
            "at_trial": (previous.get("at_trial")
                         if previous.get("kind") == prior.get("kind") else at_trial),
        }
        # No `knots`. The browser used to evaluate the density and send it, so
        # that what SMAC weighted by was literally what had been drawn; now the
        # server evaluates it for both, which makes that guarantee structural
        # rather than something two implementations had to keep agreeing on.

    stated_before = (exp.data.priors or {}).get(hp_name)
    exp.data.priors = priors
    exp.data.save(update_fields=["priors"])
    history.record_prior(exp, hp_name, stated_before, priors.get(hp_name), user=request.user)

    # The density for what was just stated, so the debounced save doubles as the
    # redraw. One round trip, and the curve the reader ends up looking at is the
    # one the optimizer will weight by, because a single function produced both.
    density = None
    if body.get("meta"):
        density = prior_math.prior_densities(priors.get(hp_name), body["meta"])
    return JsonResponse({"saved": sorted(priors), "density": density})


@experiment_view(EDIT, drafts=BOTH)
@require_POST
def reset_prior(request, exp):
    """Put one hyperparameter's belief back to what the last run searched under.

    The undo for an afternoon of dragging. A prior on this figure is edited
    live — every drag saves — so there is no other way back to the statement the
    trials on screen actually came out of, and "what I had before I started
    fiddling" is the thing a reader wants and cannot otherwise reconstruct.

    To uniform when there is nothing to go back to: no run yet, or a last run
    that stated nothing for this hyperparameter. Uniform *is* stating nothing —
    it multiplies the acquisition by a constant, which cannot change a ranking —
    so it is stored as absence, the same as withdrawing one by hand.

    `EDIT`, like `save_prior`, and for the same reason: this changes what the
    next run searches.
    """
    hp_name = (request.POST.get("hp") or "").strip()
    if not hp_name:
        return HttpResponseBadRequest("no hyperparameter named")

    # The last run that got as far as recording anything. A pending one has
    # searched under nothing yet, and going back to its priors would be going
    # back to the very edits this is meant to undo.
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

    # Handed straight back, so the figure redraws from the same answer every
    # other request gives it rather than recomputing the decay in the browser.
    # The curve too, where the page said where it wants it — as a save does.
    # Without it the restored knots would be drawn over the curve of the belief
    # they replaced.
    result = rebuild_result(exp)
    density = None
    try:
        meta = json.loads(request.POST.get("meta") or "null")
    except ValueError:
        meta = None
    if isinstance(meta, dict):
        density = prior_math.prior_densities(priors.get(hp_name), meta)
    return JsonResponse({
        "restored": bool(restored),
        "prior": prior_service.decayed_prior(exp, hp_name,
                                len(result.trials) if result else 0,
                                prior_math.inside_initial_design(result)),
        "density": density,
    })


@experiment_view(RUN, drafts=BOTH)
@require_POST
def evaluate_prior(request, exp):
    """Ask this run's optimizer whether the stated belief is worth acting on.

    DynaBO's safeguard, on demand. It builds the real facade, replays the
    recorded trials and offers the prior to `add_prior` behind a
    `ClimbingComparisonPolicy` — so the answer is the one the run itself would
    give, made by SMAC rather than by a lookalike here.

    *top_k* may be posted with the request, from the field beside the button;
    clamped to the setting's bounds, and the setting itself when it is absent
    or not a number.

    The tolerance is a share of the metric's scale (`experiments.services.priors.metric_scale`), so one
    setting means the same for an accuracy and an RMSE; SMAC is handed it in
    the metric's own units, which is what its threshold is compared in.

    `RUN` rather than `VIEW`, for `experiment_compute_analytics`' reason: it
    costs a model fit, thousands of acquisition evaluations and a local search
    per climb, and a diagnostic that anybody who can read an experiment can
    make it spend seconds on is a diagnostic that reading an experiment pays
    for.

    Nothing is written either way. A refused belief stays exactly where the
    reader left it — whether to keep it is their decision, and the page asks.
    """
    hp_name = (request.POST.get("hp") or "").strip()
    metric = exp.data.current_metric or (exp.data.metric_names or [None])[0]
    settings_now = resolve_settings(exp)
    low, high = SETTING_BOUNDS["prior_acceptance_tolerance"]
    tolerance = min(max(float(settings_now.get("prior_acceptance_tolerance", 0.15)), low), high)
    top_k = settings_now.get("prior_acceptance_top_k",
                             SETTING_DEFAULTS["prior_acceptance_top_k"])
    try:
        top_k = int(request.POST["top_k"])
    except (KeyError, ValueError):
        pass
    low, high = SETTING_BOUNDS["prior_acceptance_top_k"]
    top_k = min(max(top_k, low), high)

    def unjudged(message):
        """One shape for every answer, so the page has one thing to read."""
        return JsonResponse({"verdict": "unjudged", "message": message,
                             "tolerance": tolerance})

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    config_space = config_space_for(built)
    if result is None or not result.trials or config_space is None:
        return unjudged(_("Nothing has been run yet, so there is no model to "
                          "judge this against. A belief stated now is acted on "
                          "in full."))
    assert built is not None

    optimizer = built["optimizer"]
    if not hasattr(optimizer, "evaluate_prior"):
        return unjudged(_("This optimizer fits no model of the objective, so it "
                          "has no opinion about where the optimum is."))
    if not _judges_priors(exp):
        return unjudged(_("Needs the Gaussian process search strategy."))
    scale = prior_service.metric_scale(exp, metric, result)
    if scale is None:
        return unjudged(_("This metric has no scale to measure the tolerance against yet."))
    margin = float(tolerance) * scale

    outcome = optimizer.evaluate_prior(
        config_space, result.trials, metric, built["seed"],
        priors=exp.data.priors or None,
        projected_priors=exp.data.projected_priors or None,
        tolerance=margin,
        top_k=top_k,
        previous_result=result,
        # The same anchor the figure draws with, so the belief is judged at the
        # strength it is actually being applied at.
        budget=prior_service.run_budget(exp, int(((exp.data.priors or {}).get(hp_name) or {})
                                    .get("at_trial") or 0)))
    # The wording is made here, not in `core`, which has no Django and so no
    # catalog to translate against. What crosses from there is the verdict and a
    # reason for the log; what reaches the page is a sentence.
    verdict = outcome.get("verdict", "unjudged")
    # Said as a finding about the belief, because that is what a reader asked
    # for. What the model actually computes — a mean acquisition over samples
    # from the belief against samples from the incumbent's neighbourhood — is
    # the reason for the finding and not the finding itself, and leading with it
    # left the answer to the question buried at the end of a sentence about
    # regions and neighbourhoods.
    if verdict == "accepted":
        message = _("This prior is potentially informative: the model rates the "
                    "region it points at as promising as where the search is "
                    "already headed.")
    elif verdict == "rejected":
        message = _("This prior is unlikely to be helpful: the model rates the "
                    "region it points at more than %(margin)s worse than "
                    "where the search is already headed, so acting on it would "
                    "cost trials.") % {"margin": sigfigs(margin)}
    else:
        logger.info("A prior could not be judged: %s", outcome.get("reason", ""))
        message = _("This prior could not be judged, so it stands as stated.")
    return JsonResponse({"verdict": verdict, "message": message,
                         "tolerance": tolerance, "margin": margin, "top_k": top_k})


def _next_trials(exp, anchor_hp=None):
    """What this run's optimizer would ask for next, in its own rank order, as
    `(config_space, [(config, origin, walk), ...])` — or None where it cannot be
    asked. *walk* is the path its local search climbed to it, or None.

    Not a recomputation that resembles the search: `slice_challengers` builds the
    facade with `_facade`, replays the recorded trials, applies the stated
    priors and calls `ask()`. The first is the configuration the next trial
    would use.

    *anchor_hp* is the hyperparameter whose prior the budget is anchored at, so
    the walk weights by the same exponent that prior's figure is drawing. Left
    out, it is the most recently stated prior, or the run as it stands.
    """
    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return None
    assert built is not None
    config_space = config_space_for(built)
    if config_space is None:
        return None

    metric = exp.data.current_metric or (exp.data.metric_names or [None])[0]
    # Not every optimizer can be asked. Random and grid search have no
    # surrogate and no acquisition function, so there is no ranking to report —
    # and the reader gets told that rather than a 500.
    optimizer = built["optimizer"]
    if not hasattr(optimizer, "slice_challengers"):
        return None

    priors = exp.data.priors or {}
    if anchor_hp is not None:
        at_trial = (priors.get(anchor_hp) or {}).get("at_trial")
    else:
        anchors = [p.get("at_trial") for p in priors.values() if p and p.get("at_trial")]
        at_trial = max(anchors) if anchors else None
    asked = optimizer.slice_challengers(
        config_space, result.trials, metric, built["seed"],
        priors=priors or None, previous_result=result,
        projected_priors=exp.data.projected_priors or None,
        budget=prior_service.run_budget(exp, int(at_trial or len(result.trials))), with_walks=True)
    if not asked:
        return None
    configs, origins, walks = asked
    return config_space, list(zip(configs, origins, walks))


def _candidate(rank, config, origin, walk=None, config_space=None):
    """One proposed configuration as the page draws it: its rank, where it came
    from, every value, the whole of it pre-formatted for a hover — a point
    differs from the incumbent in every coordinate, not only the ones on show —
    and the path its local search climbed to get there, start first."""
    values = {k: _plain(v) for k, v in sorted(config.items())}
    return {"rank": rank, "origin": origin, "config": values,
            "label": "<br>".join(f"{k} = {sigfigs(v)}" for k, v in values.items()),
            # Where it sits on every hyperparameter's vectorized axis, so either
            # figure can place it on whichever axis it is drawing — the
            # acquisition figure's rug as well as the cube.
            "positions": _vector_positions(config_space, config),
            "walk": ([{k: _plain(v) for k, v in step.items()} for step in walk]
                     if walk else None)}


def _vector_positions(config_space, config):
    """*config* in ConfigSpace's vectorized form, keyed by hyperparameter, or
    None when there is no space or it will not encode. An inactive
    hyperparameter has no position and is left out."""
    if config_space is None:
        return None
    from ConfigSpace import Configuration

    try:
        vector = Configuration(config_space, values=config).get_array()
    except Exception:  # noqa: BLE001 — a candidate that will not encode has no place
        return None
    return {name: float(v) for name, v in zip(config_space.keys(), vector) if np.isfinite(v)}


@experiment_view(VIEW)
def next_trials(request, exp):
    """What this run's optimizer would run next, as whole configurations, under
    the priors as saved. Both the acquisition figure and the configuration cube
    draw the answer, each placing it on its own axes. *hp*, when given, is the
    hyperparameter whose prior the trial budget is anchored at — see
    `_next_trials`. Not cached: it rebuilds and asks the optimizer, which takes
    a few seconds, and only when someone asks for it.
    """
    hp_name = request.GET.get("hp") or None
    asked = _next_trials(exp, anchor_hp=hp_name)
    if asked is None:
        return JsonResponse({"candidates": None})
    config_space, proposals = asked
    return JsonResponse({"candidates": [_candidate(rank, config, origin, walk, config_space)
                                        for rank, (config, origin, walk)
                                        in enumerate(proposals, start=1)]})


#: Figures built but not sent. The surrogate panel is switched off — see the
#: acquisition template — and its traces are a few thousand points per request
#: that nothing draws. Still *built*, because the model behind it is fitted
#: either way (the acquisition curve is computed from the same mean and spread)
#: and because the figure's own behaviour is still under test; only the sending
#: is skipped. Putting the panel back is this set and the template's div.
_NOT_SHIPPED = frozenset({"surrogate"})


def _slice_figures(exp, built, config_space, hp_name, metric):
    """The three slice figures, their `meta`, and the slice they came from.

    Split out of the view so that what is *built* and what is *shipped* are two
    decisions in two places. The surrogate figure is no longer sent, and a test
    that asks whether its spread is stepped or its ghost band faint is asking
    about the machinery rather than about the payload — so it asks here.

    Returns `(None, None, sliced)` when the slice has no positions, which is the
    one case the caller answers differently.
    """
    surrogate, sample = _slice_model(exp, built, config_space, metric)
    cloud = _sample_cloud(sample)
    sliced = built["optimizer"].compute_incumbent_slice(
        config_space, built["result"].trials, metric, hp_name, seed=built["seed"],
        surrogate=surrogate, cloud=cloud)
    if not sliced.positions:
        return None, None, sliced

    figures, meta = acquisition_slice_plots(
        hp_name, sliced.positions, sliced.grid, sliced.mu, sliced.sigma, metric,
        eta=sliced.eta, incumbent=sliced.incumbent, cloud=sliced.cloud,
        higher_is_better=metric_for(metric).higher_is_better,
        kind="categorical" if hasattr(config_space[hp_name], "choices") else "continuous",
    )
    assert meta is not None
    meta["acquisitionFunction"] = _acquisition_function(built["optimizer"])
    # Priors drawn in the cube weigh whole configurations, so they are
    # evaluated at the configurations this figure predicted at: along the
    # slice, and over its cloud.
    slice_weights = _projected_weights(exp, built, config_space, sliced.slice_rows)
    meta["projectedWeights"] = (None if slice_weights is None else {
        "slice": slice_weights,
        "cloud": _projected_weights(exp, built, config_space, sliced.cloud_rows) or []})
    return figures, meta, sliced


def _acquisition_function(optimizer):
    """Which acquisition function the page should compute, and its margin.
    Expected improvement for an optimizer that names none."""
    return optimizer.acquisition_function() or {"kind": "ei", "xi": 0.0}


@experiment_view(VIEW, drafts=BOTH)
def acquisition_slice(request, exp):
    """The acquisition-and-priors figure for one (metric, hyperparameter).

    Fetched per hyperparameter rather than shipped, for `partial_dependence`'s
    reason exactly — it fits a surrogate and predicts across a grid, and only
    one hyperparameter is ever on screen.

    The prior and acquisition traces come back empty; the browser fills them
    (`ui/static/ui/acquisition.js`). Everything it needs to do that rides in the
    figure's own `layout.meta`, so the response is the figure plus a warning,
    the same shape `partial_dependence` returns.
    """
    metric = request.GET.get("metric", "")
    hp_name = request.GET.get("hp", "")

    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    config_space = config_space_for(built)
    if config_space is None:
        return JsonResponse({"figure": None, "warning": _(
            "The model this experiment used is not available here, so its "
            "search space cannot be rebuilt.")})
    assert built is not None

    # Checked against the search space rather than against a trial's
    # configuration. A prior is a statement about the space, and the space
    # exists before the search does — asked for on an experiment that has never
    # been run there is no trial to read a name off, and that is exactly when
    # somebody is most likely to be stating a belief.
    if hp_name not in config_space:
        return HttpResponseBadRequest("invalid hyperparameter")

    # Everything else on this figure is the surrogate's, and there is no
    # surrogate without trials. The prior panel is the one part that needs none.
    if (result is None or not result.trials) and not request.GET.get("prior_only"):
        return HttpResponseBadRequest("no result")

    # A prior is a statement about the search space, so the space is all it
    # takes to draw one. Asked for on its own, it skips the surrogate entirely —
    # which is what a reader waits on — so the panel they can actually edit is
    # there immediately, and stays there when "compute automatically" is off.
    if request.GET.get("prior_only"):
        grid, positions, kind = built["optimizer"].slice_axis(config_space, hp_name)
        if not positions:
            return JsonResponse({"figures": None, "meta": None, "warning": ""})
        zeros = [0.0] * len(positions)
        figures, meta = acquisition_slice_plots(
            hp_name, positions, grid, zeros, zeros, metric,
            eta=0.0, incumbent=None, cloud=None,
            higher_is_better=metric_for(metric).higher_is_better, kind=kind)
        assert figures is not None and meta is not None
        stated = (exp.data.priors or {}).get(hp_name)
        return JsonResponse({
            # Only the panel that needed nothing. The other two are what the
            # surrogate is for, and claiming them here with zeros would draw a
            # flat acquisition function as though it were a finding.
            "figures": {"prior": _plot_json(figures["prior"])},
            "meta": {k: v for k, v in meta.items() if k != "cloud"},
            "insideDesign": prior_math.inside_initial_design(result),
            "warning": "",
            # Zero trials when there has never been a run, which is the honest
            # step count: nothing has happened to this belief yet.
            "prior": prior_service.decayed_prior(exp, hp_name,
                                    len(result.trials) if result else 0,
                                    prior_math.inside_initial_design(result)),
            "betaRatioDefault": prior_math.BETA_RATIO_DEFAULT,
            "betaRatiosAblated": list(prior_math.BETA_RATIOS_ABLATED),
            "density": {"grid": prior_math.prior_density(stated, positions,
                                               meta.get("span") or [0.0, 1.0]),
                        "cloud": None},
        })

    assert result is not None
    figures, meta, sliced = _slice_figures(exp, built, config_space, hp_name, metric)
    if figures is None:
        return JsonResponse({"figure": None, "warning": sliced.warning})

    # Three figures and one `meta`, rather than the meta repeated on each: it
    # carries the sampled cloud, which is two thousand points, and all three are
    # drawn from the same numbers anyway.
    return JsonResponse({"figures": {name: _plot_json(fig)
                                     for name, fig in figures.items()
                                     if name not in _NOT_SHIPPED},
                         "meta": meta,
                         "warning": sliced.warning or _weighting_warning(exp),
                         "initialDesign": prior_math.initial_design_notice(result),
                         "insideDesign": prior_math.inside_initial_design(result),
                         "prior": prior_service.decayed_prior(exp, hp_name, len(result.trials),
                                                 prior_math.inside_initial_design(result)),
                         # What the β field should open on when nothing is
                         # stated yet: DynaBO's own initialisation.
                         "betaRatioDefault": prior_math.BETA_RATIO_DEFAULT,
                         "betaRatiosAblated": list(prior_math.BETA_RATIOS_ABLATED),
                         # So the first draw needs no second request.
                         "density": prior_math.prior_densities(
                             (exp.data.priors or {}).get(hp_name), meta)})


def _weighting_warning(exp):
    """Told to the reader, not just the log, when a stated prior cannot bite.

    A prior on a SMAC without the acquisition weight layer is drawn, saved, and
    ignored by both the search and the candidate walk. Nothing on the page would
    say so otherwise, and a belief that silently does nothing is worse than one
    that is refused.
    """
    if not (exp.data.priors or {}):
        return None
    try:
        import smac.acquisition.weight  # noqa: F401
    except ImportError:
        return _("This SMAC has no acquisition weight layer, so a stated prior "
                 "does not affect the search or the candidates below. Install "
                 "the version requirements.txt pins.")
    return None
