import json
import threading
from collections import OrderedDict

import numpy as np

from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import EDIT, RUN, VIEW, policy
from core import analytics
from core.jsonable import plain
from core.metrics import metric_for
from core import priors as prior_math
from experiments.models import account
from experiments.services import history
from experiments.services.settings import SETTING_BOUNDS
from experiments.services.snapshot import config_space_for, rebuild_experiment, rebuild_result
from experiments.services import priors as prior_service
from experiments.services import run as run_service

from ..experiment_settings import SETTING_DEFAULTS, resolve_settings
from ..figures import acquisition_slice_plots
from ..formatting import sigfigs
from ..permissions import BOTH, experiment_view
from .common import _judges_priors, _model_available, _plot_json, logger


def _cube_sample(exp, metric):
    """`_cube_sample` gives (built, config_space, sample) for *metric*, with the
    last two None when nothing can be drawn."""
    built = rebuild_experiment(exp)
    if not built or built["result"] is None or not built["result"].trials:
        return built, None, None
    config_space = config_space_for(built)
    if config_space is None:
        return built, None, None
    _surrogate, sample = _slice_model(exp, built, config_space, metric)
    return built, config_space, sample if sample and sample.get("configurations") else None


def _proposal_index(source):
    """`_proposal_index` reads index `n` from *source*, or None."""
    raw = source.get("n", "")
    return int(raw) if raw.isdigit() else None


def _run_refusal(request, exp):
    """`_run_refusal` explains why the viewer cannot run, or None."""
    if not policy().may(request, exp, RUN):
        return _("You can view this experiment but not run it.")
    if exp.is_running:
        return _("A run is under way.")
    if not exp.data.has_dataset or not _model_available(exp):
        return _("This experiment cannot be run here.")
    return policy().custom_model_refusal(exp, request.user) or None


@experiment_view(VIEW)
def proposal_panel(request, exp):
    """`proposal_panel` shows one sampled cube configuration with the
    surrogate's prediction and an offer to run it.
    """
    metric = request.GET.get("metric", "")
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    n = _proposal_index(request.GET)
    built, _config_space, sample = _cube_sample(exp, metric)
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
    """`run_proposal` runs one proposal as a single-trial run on the current
    metric, recorded with the proposal number."""
    metric = request.POST.get("metric", "")
    n = _proposal_index(request.POST)
    if metric not in exp.data.metric_names or n is None or _run_refusal(request, exp):
        return redirect("ui:experiment_detail", pk=exp.pk)
    _built, _config_space, sample = _cube_sample(exp, metric)
    if sample is None or n >= len(sample["configurations"]):
        return redirect("ui:experiment_detail", pk=exp.pk)

    number = n + 1
    run = run_service.create_run(
        exp, {"max_trials": 1}, exp.data.current_metric or metric,
        started_by=account(request.user),
        configurations=[{"config": dict(sample["configurations"][n]),
                         "origin": _("Proposal %(n)s") % {"n": number},
                         "proposal": number}])
    run_service.start_background_run(run.id)
    return redirect("ui:experiment_detail", pk=exp.pk)


#: `_SLICE_SURROGATES` caches fits by experiment, metric, trial count and seed,
#: since the axis hyperparameter changes nothing. The trial count lets a live
#: run invalidate entries, and a per-key lock makes a second figure wait for the
#: first fit.
_SLICE_SURROGATES = OrderedDict()


_SLICE_SURROGATE_CACHE = 8


_SLICE_LOCKS = {}


_SLICE_LOCKS_GUARD = threading.Lock()


def _slice_model(exp, built, config_space, metric):
    """`_slice_model` returns `(surrogate, sample)` for the experiment, metric
    and trials, shared by the acquisition cloud and the cube's prediction layer.
    Neither result depends on priors, because the browser applies priors to the
    acquisition alone. The cache skips a failed fit, which yields `(None, None)`
    for the caller to explain.
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

        sample = analytics.sample_prediction_space(
            optimizer, config_space, result.trials, metric, seed=built["seed"],
            surrogate=surrogate)
        _SLICE_SURROGATES[key] = (surrogate, sample)
        while len(_SLICE_SURROGATES) > _SLICE_SURROGATE_CACHE:
            evicted, _ = _SLICE_SURROGATES.popitem(last=False)
            with _SLICE_LOCKS_GUARD:
                _SLICE_LOCKS.pop(evicted, None)
        return surrogate, sample


def _sample_cloud(sample):
    """`_sample_cloud` turns the shared sample into `(rows, mu, sigma)` arrays,
    with NaN for missing values."""
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
    """`save_prior` stores one hyperparameter's prior shape and parameters under
    `EDIT`, since a prior changes what the next run searches. The server
    evaluates the density for both SMAC and the redraw.
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
        # Uniform states nothing, so it is stored as absence.
        priors.pop(hp_name, None)
    else:
        # Decay counts trials since the prior was stated, so a late belief
        # starts at full strength.
        built = rebuild_experiment(exp)
        at_trial = len((built["result"].trials if built and built["result"] else []))
        previous = (exp.data.priors or {}).get(hp_name) or {}
        result_now = built["result"] if built else None
        priors[hp_name] = {
            "kind": prior.get("kind") or "uniform",
            "params": prior.get("params") or {},
            "decay": prior_math.decay_record(prior.get("decay"),
                                             prior_service.run_budget(exp, at_trial)),
            # `delay_decay` holds only while the initial design is unfinished.
            "delay_decay": bool(prior.get("delay_decay")
                                and prior_math.inside_initial_design(result_now)),
            # An edited parameter keeps the anchor, or decay would stay at zero.
            "at_trial": (previous.get("at_trial")
                         if previous.get("kind") == prior.get("kind") else at_trial),
        }

    stated_before = (exp.data.priors or {}).get(hp_name)
    exp.data.priors = priors
    exp.data.save(update_fields=["priors"])
    history.record_prior(exp, hp_name, stated_before, priors.get(hp_name), user=request.user)

    # `prior_densities` produces both the returned curve and the optimizer's
    # weights, so the save doubles as the redraw.
    density = None
    if body.get("meta"):
        density = prior_math.prior_densities(priors.get(hp_name), body["meta"])
    return JsonResponse({"saved": sorted(priors), "density": density})


@experiment_view(EDIT, drafts=BOTH)
@require_POST
def reset_prior(request, exp):
    """`reset_prior` restores one hyperparameter's prior to what the last run
    searched under, undoing live edits. With nothing to restore, the prior
    becomes uniform and is stored as absence.
    """
    hp_name = (request.POST.get("hp") or "").strip()
    if not hp_name:
        return HttpResponseBadRequest("no hyperparameter named")

    # Pending runs are skipped, since their priors are the edits being undone.
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

    # The response carries the decayed prior and density for the redraw.
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
    """`evaluate_prior` replays the trials into SMAC and asks its `add_prior`
    safeguard whether the stated belief is worth acting on. The tolerance is a
    share of the metric's scale, and *top_k* is clamped to its bounds. `RUN`
    guards the view because each call costs seconds of fitting and search, and
    nothing is written.
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
        """`unjudged` gives every answer one response shape."""
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
        tolerance=margin,
        top_k=top_k,
        previous_result=result,
        # Judged at the same anchor the figure draws with.
        budget=prior_service.run_budget(exp, int(((exp.data.priors or {}).get(hp_name) or {})
                                    .get("at_trial") or 0)))
    # The wording lives here because `core` has no translation catalog.
    verdict = outcome.get("verdict", "unjudged")
    # The message leads with the finding about the belief rather than the
    # model's computation.
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
    """`_next_trials` returns `(config_space, [(config, origin, walk), ...])` in
    the optimizer's rank order from a real `ask()`, or None. *anchor_hp* picks
    the prior that anchors the budget, defaulting to the latest stated.
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
    # Random and grid search have no acquisition to rank by.
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
        budget=prior_service.run_budget(exp, int(at_trial or len(result.trials))), with_walks=True)
    if not asked:
        return None
    configs, origins, walks = asked
    return config_space, list(zip(configs, origins, walks))


def _candidate(rank, config, origin, walk=None, config_space=None):
    """`_candidate` gives one proposal's rank, origin, values, hover label,
    vector positions and climb path."""
    values = {k: plain(v) for k, v in sorted(config.items())}
    return {"rank": rank, "origin": origin, "config": values,
            "label": "<br>".join(f"{k} = {sigfigs(v)}" for k, v in values.items()),
            "positions": _vector_positions(config_space, config),
            "walk": ([{k: plain(v) for k, v in step.items()} for step in walk]
                     if walk else None)}


def _vector_positions(config_space, config):
    """`_vector_positions` maps *config* to vector positions by hyperparameter,
    or None when encoding fails; inactive ones drop out."""
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
    """`next_trials` returns the optimizer's next configurations under the saved
    priors for both figures. The view rebuilds on each call, which takes
    seconds.
    """
    hp_name = request.GET.get("hp") or None
    asked = _next_trials(exp, anchor_hp=hp_name)
    if asked is None:
        return JsonResponse({"candidates": None})
    config_space, proposals = asked
    return JsonResponse({"candidates": [_candidate(rank, config, origin, walk, config_space)
                                        for rank, (config, origin, walk)
                                        in enumerate(proposals, start=1)]})


#: The surrogate figure is still built for tests and the shared fit but stays
#: unsent.
_NOT_SHIPPED = frozenset({"surrogate"})


def _slice_figures(exp, built, config_space, hp_name, metric):
    """`_slice_figures` builds the three slice figures and `meta` apart from the
    view, and tests inspect unsent ones there. It returns `(None, None, sliced)`
    when the slice has no positions.
    """
    surrogate, sample = _slice_model(exp, built, config_space, metric)
    cloud = _sample_cloud(sample)
    sliced = analytics.compute_incumbent_slice(
        built["optimizer"], config_space, built["result"].trials, metric, hp_name,
        seed=built["seed"], surrogate=surrogate, cloud=cloud)
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
    return figures, meta, sliced


def _acquisition_function(optimizer):
    """`_acquisition_function` picks the acquisition function to compute,
    defaulting to EI."""
    return optimizer.acquisition_function() or {"kind": "ei", "xi": 0.0}


@experiment_view(VIEW, drafts=BOTH)
def acquisition_slice(request, exp):
    """`acquisition_slice` fetches one hyperparameter's figure on demand, since
    the fit is costly. The browser fills the prior and acquisition traces from
    `layout.meta` (`ui/static/ui/acquisition.js`).
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

    # The search space validates *hp*, since a prior can predate any trial.
    if hp_name not in config_space:
        return HttpResponseBadRequest("invalid hyperparameter")

    if (result is None or not result.trials) and not request.GET.get("prior_only"):
        return HttpResponseBadRequest("no result")

    # A prior-only request skips the surrogate and shows the editable panel at
    # once.
    if request.GET.get("prior_only"):
        grid, positions, kind = analytics.slice_axis(config_space, hp_name)
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
            # Zeros in the other two panels would fake a flat acquisition.
            "figures": {"prior": _plot_json(figures["prior"])},
            "meta": {k: v for k, v in meta.items() if k != "cloud"},
            "insideDesign": prior_math.inside_initial_design(result),
            "warning": "",
            # Zero trials with no run yet.
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

    # One `meta` for all three figures avoids repeating the two-thousand-point
    # cloud.
    return JsonResponse({"figures": {name: _plot_json(fig)
                                     for name, fig in figures.items()
                                     if name not in _NOT_SHIPPED},
                         "meta": meta,
                         "warning": sliced.warning or _weighting_warning(exp),
                         "initialDesign": prior_math.initial_design_notice(result),
                         "insideDesign": prior_math.inside_initial_design(result),
                         "prior": prior_service.decayed_prior(exp, hp_name, len(result.trials),
                                                 prior_math.inside_initial_design(result)),
                         # The β field's initial value, DynaBO's own.
                         "betaRatioDefault": prior_math.BETA_RATIO_DEFAULT,
                         "betaRatiosAblated": list(prior_math.BETA_RATIOS_ABLATED),
                         "density": prior_math.prior_densities(
                             (exp.data.priors or {}).get(hp_name), meta)})


def _weighting_warning(exp):
    """`_weighting_warning` tells the reader when SMAC lacks the acquisition
    weight layer, since a prior there is saved and then ignored.
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
