import json
from dataclasses import replace
from typing import cast

import numpy as np

from django.conf import settings
from django.http import Http404, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import RUN, VIEW
from core import analytics, forecasting, io, tasks
from core import priors as prior_math
from experiments.services import analytics as analytics_service
from experiments.services import priors as prior_service
from experiments.services import snapshot as snapshot_adapter
from experiments.services.snapshot import config_space_for, rebuild_experiment, rebuild_result

from ..experiment_settings import resolve_settings
from ..figures import (
    HP_GAME_FIELDS, UNCERTAINTY_SCALE, hyperparameter_ablation_plot,
    hyperparameter_importance_plot, forecast_plot, hyperparameter_progress_plot,
    local_effects_plot, partial_dependence_plot, slice_span,
)
from ..permissions import experiment_view
from ..registry import METRICS, MODELS, OPTIMIZERS
from .common import _baseline_index, _figure_plots, _plot_json, _shown_figures, _target_name
from .priors import _acquisition_function, _slice_model


@experiment_view(VIEW)
def trial_traceback(request, exp):
    """The stored traceback for one failed trial, as plain text.

    A file to open rather than a panel to read: a traceback is a dozen lines of
    paths and frames, which is the wrong shape for a sidebar and the right shape
    for something you scroll, search and paste elsewhere. Served from the stored
    result — the traceback was recorded when the trial failed, so this reads and
    never computes.

    404 for a trial that did not fail and for one that failed without a
    traceback: a timeout has none worth keeping, an old result predates them
    being recorded, and an export can have been asked to leave them out.
    """
    idx_raw = request.GET.get("idx", "")
    if not idx_raw.lstrip("-").isdigit():
        return HttpResponseBadRequest("invalid trial index")

    result = rebuild_result(exp)
    idx = int(idx_raw)
    if result is None or not (0 <= idx < len(result.trials)):
        return HttpResponseBadRequest("invalid trial index")

    trial = result.trials[idx]
    if not trial.traceback:
        raise Http404("no traceback for this trial")

    text = f"Trial {trial.trial} — {trial.failure}\n\n{trial.traceback}"
    response = HttpResponse(text, content_type="text/plain; charset=utf-8")
    # Inline, so a click opens it rather than downloading it. A reader wanting
    # the file can still save it from there.
    response["Content-Disposition"] = (
        f'inline; filename="{exp.identifier}-trial-{trial.trial}-traceback.txt"')
    return response


@experiment_view(VIEW)
def trial_ablation(request, exp):
    """The local-explanation (HyperSHAP ablation) figure for one (metric,
    trial index) — fetched lazily, only when the importance figure's game
    selector is on "Local" and the selection changes, since (unlike the
    other three games) this depends on which trial is selected rather than
    just the experiment's stored result, so it cannot be precomputed for
    every trial the way the rest of the page is.
    """
    metric = request.GET.get("metric", "")
    idx_raw = request.GET.get("idx", "")

    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    if not idx_raw.lstrip("-").isdigit():
        return HttpResponseBadRequest("invalid trial index")

    built = rebuild_experiment(exp)
    idx = int(idx_raw)
    if (built is None or built["result"] is None
            or not (0 <= idx < len(built["result"].trials))):
        return HttpResponseBadRequest("invalid trial index")
    try:
        baseline = _baseline_index(request, built["result"])
    except ValueError:
        return HttpResponseBadRequest("invalid baseline")

    figure, warning, ablation = _local_ablation_data(built, metric, idx, baseline)
    # What a trial has banked is a gain over the default, and so is what
    # tunability says was there to gain, so the two only subtract from there.
    rows, settled = (analytics.tuning_progress(built["result"], metric, ablation)
                     if TUNING_PROGRESS_ENABLED and baseline is None else ([], False))
    # The headroom renderings cannot be precomputed — they need this trial's
    # ablation — so they come back with it, all three at once, and switching
    # rendering afterwards costs no second request.
    remaining = {row["name"]: row["remaining"] for row in rows}
    return JsonResponse({
        "figure": figure,
        "warning": warning,
        "rows": rows,
        "settled": settled,
        # Two ways of showing the same pair. `split` sizes each hyperparameter as
        # the plain figure does and divides it into what has been banked and what
        # has not — the honest version of the default view. `headroom` sizes them
        # by what is left alone, which reorders the figure and is what the "still
        # to gain" box asks for.
        "split": {
            rendering: _plot_json(hyperparameter_progress_plot(rows, rendering))
            for rendering in ("pie", "bar")
        } if rows else {},
        "headroom": {
            rendering: _plot_json(hyperparameter_importance_plot(
                remaining, rendering, title=str(_("Still to gain"))))
            for rendering in ("pie", "bar")
        } if rows and not settled else {},
    })


@experiment_view(VIEW)
def partial_dependence(request, exp):
    """The partial-dependence (PDP/ICE) figure for one (metric,
    hyperparameter) — fetched lazily on every load and every hyperparameter
    switch, since which hyperparameter is being explained isn't one of a
    small precomputable set the rest of the page ships upfront (unlike the
    three global HyperSHAP games), and fitting a surrogate + predicting
    across a grid for every hyperparameter on every page load, for a figure
    that only ever shows one at a time, would be pure waste.
    """
    metric = request.GET.get("metric", "")
    hp_name = request.GET.get("hp", "")

    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    if (built is None or built["result"] is None or not built["result"].trials
            or hp_name not in built["result"].trials[0].config):
        return HttpResponseBadRequest("invalid hyperparameter")

    figure, warning = _partial_dependence_data(
        built, metric, hp_name, resolve_settings(exp)["ice_max_curves"])
    return JsonResponse({"figure": figure, "warning": warning})


@experiment_view(VIEW)
def forecast_figure(request, exp):
    """The Forecast figure for one trial: the series, and what the trial's
    configuration forecast at every backtest.

    Deferred like partial dependence, and for a stronger reason — it runs the
    model, once per backtest, as the trial did. *idx* names the trial (the
    selected one on the page), *series* which series to draw when there are
    several.
    """
    metric = request.GET.get("metric", "")
    if not exp.data.forecasts or metric not in exp.data.metric_names:
        return HttpResponseBadRequest("not a forecast")
    result = rebuild_result(exp)
    if result is None:
        return HttpResponseBadRequest("unknown trial")
    try:
        idx = int(request.GET.get("idx", ""))
        trial = result.trials[idx]
    except (TypeError, ValueError, IndexError, AttributeError):
        return HttpResponseBadRequest("unknown trial")
    if exp.data.model_file:
        return JsonResponse({"figure": None, "series": [], "warning": _(
            "The forecast is drawn for built-in models only: an uploaded model runs in "
            "its own environment, which this page does not start.")})
    try:
        built = io.build_experiment(snapshot_adapter.snapshot_from_experiment(exp),
                                    METRICS, MODELS, OPTIMIZERS, read_only=False)[1]
    except ValueError as exc:
        return JsonResponse({"figure": None, "series": [], "warning": str(exc)})

    splits, model = built["splits"], built["model"]
    labels = exp.data.task == tasks.CLASSIFICATION
    names = sorted({str(s) for s in splits.series}) if splits.series is not None else []
    chosen = request.GET.get("series") or (names[0] if names else "")
    rows = (np.arange(len(splits.y)) if not names
            else np.flatnonzero(np.asarray(splits.series).astype(str) == chosen))
    try:
        backtests = forecasting.backtests(model, trial.config, splits, built["times"], rows,
                                          seed=built["seed"], labels=labels)
    except Exception as exc:  # noqa: BLE001 — the model's failure is the figure's message
        return JsonResponse({"figure": None, "series": names, "warning": _(
            "This configuration failed to forecast: %(error)s") % {"error": exc}})
    figure = forecast_plot(built["times"][rows], splits.y[rows], backtests,
                           target=_target_name(exp), series=chosen, labels=labels)
    return JsonResponse({"figure": _plot_json(figure), "series": names, "warning": None})


@experiment_view(VIEW)
def surrogate_uncertainty(request, exp):
    """The uncertainty field under the configuration cube's axes view, for one
    (metric, x, y) — how much the surrogate's own trees disagree across the
    plane those two hyperparameters span.

    **The axes view only, and only at two axes.** The two projections have no
    way back: `core.projection.project` returns coordinates and discards the
    fitted PCA/PLS, so there is no inverse to map a grid in component space
    back to configurations — and even with one, the points it landed on need not
    be valid configurations. Three axes would be a volume through a point cloud,
    which is a worse picture than none. One is a strip, and a field under a strip
    has no second dimension to vary in.

    Fetched rather than shipped, like partial dependence and for the same
    reason: it is per *pair*, and a model with six hyperparameters has fifteen
    pairs of which the reader is looking at one.
    """
    metric = request.GET.get("metric", "")
    x_hp = request.GET.get("x", "")
    y_hp = request.GET.get("y", "")
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return HttpResponseBadRequest("no result")
    assert built is not None

    config_space = config_space_for(built)
    if config_space is None:
        return JsonResponse({
            "x": [], "y": [], "z": [],
            "warning": _("The model this experiment used is not available here, "
                         "so its search space cannot be rebuilt."),
        })

    x_grid, y_grid, z, warning = analytics.compute_surrogate_uncertainty(
        config_space, result.trials, metric, x_hp, y_hp, seed=built["seed"])
    return JsonResponse({
        "x": x_grid, "y": y_grid, "z": z, "warning": warning,
        # The ramp travels with the numbers, so what the colour *means* stays
        # declared once in `plots.py` beside every other colour on the page —
        # see UNCERTAINTY_SCALE for why it is neither green nor blue.
        "colorscale": UNCERTAINTY_SCALE,
    })


@experiment_view(VIEW)
def local_effects(request, exp):
    """The beeswarm of every sampled trial's local effect, for one metric.

    Fetched rather than shipped, and for a stronger reason than the other two
    deferred computations: this one is a whole ablation game per trial. Capped
    by the experiment's `local_effects_max_trials`, and (like every deferred
    computation now) it waits to be asked unless its autocompute setting says
    otherwise.
    """
    metric = request.GET.get("metric", "")
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    if built is None or built["result"] is None or not built["result"].trials:
        return HttpResponseBadRequest("no result")
    try:
        baseline = _baseline_index(request, built["result"])
    except ValueError:
        return HttpResponseBadRequest("invalid baseline")

    figure, warning = _local_effects_data(
        built, metric, resolve_settings(exp)["local_effects_max_trials"], baseline)
    return JsonResponse({"figure": figure, "warning": warning})


@experiment_view(VIEW)
def metric_figures(request, exp):
    """Every per-metric figure's plot payload for one metric.

    The page ships only the metric it opens on (see `_detail_context`), so this
    answers for the rest — once each, the first time one is selected. Unlike
    `trial_ablation` and `partial_dependence` this computes nothing new: the
    numbers behind these figures were all worked out at run completion and
    stored, so the cost here is building the Plotly objects and serializing
    them, which is exactly the cost the page used to pay for every metric on
    every load whether or not anyone looked.

    Honours the visibility settings through `_shown_figures`, so a figure
    switched off is absent here too rather than reachable by asking directly.
    """
    metric = request.GET.get("metric", "")
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return HttpResponseBadRequest("no result")

    per_metric = [f for f in _shown_figures(exp) if f.per_metric]
    return JsonResponse(_figure_plots(result, per_metric, metric,
                                      config_space=config_space_for(built)))


@experiment_view(VIEW)
def surrogate_prediction(request, exp):
    """What the surrogate predicts across the search space, for one metric: the
    configuration cube's prediction layer.

    One answer per metric, whichever view the cube is in. The sample, the
    predictions and every view's coordinates for it come back together, so
    switching view, dimensionality or axes redraws without asking again; only a
    different metric means a different model. See
    `core.analytics.sample_prediction_space`.
    """
    metric = request.GET.get("metric", "")
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return HttpResponseBadRequest("no result")
    assert built is not None
    config_space = config_space_for(built)
    if config_space is None:
        return JsonResponse({
            "values": [], "warning": _("The model this experiment used is not "
                                       "available here, so its search space "
                                       "cannot be rebuilt."),
        })

    optimizer = built["optimizer"]
    # The same sample the acquisition figure's cloud is drawn from, out of the
    # same cache, so whichever figure asks second costs nothing.
    _surrogate, shared = _slice_model(exp, built, config_space, metric)
    answer = dict(shared) if shared else analytics.sample_prediction_space(
        optimizer, config_space, result.trials, metric, seed=built["seed"])
    # The page draws the rounded `values`; the exact ones stay here, for
    # `run_proposal`.
    answer.pop("configurations", None)
    # Only an optimizer that fits a model of the objective has an acquisition
    # function to speak of; the stand-in forest for the others predicts, and
    # that is all.
    answer["acquisition"] = bool(getattr(optimizer, "fits_surrogate", False))
    answer["acquisition_function"] = _acquisition_function(optimizer)
    # Every stated prior, as the weight the acquisition is multiplied by: its
    # density over the same grid the acquisition figure draws it on, and the
    # exponent its decay has reached. The page interpolates each sample's own
    # position from these, and swaps one in as the acquisition figure edits it.
    priors = {}
    inside = prior_math.inside_initial_design(result)
    for name in config_space.keys():
        decayed = prior_service.decayed_prior(exp, name, len(result.trials), inside)
        if not decayed or decayed.get("kind") in (None, "uniform"):
            continue
        _grid, positions, kind = analytics.slice_axis(config_space, name)
        density = prior_math.prior_density(decayed, positions, slice_span(positions, kind))
        if density:
            priors[name] = {"positions": positions, "density": density,
                            "exponent": decayed["exponent"],
                            "categorical": kind == "categorical"}
    answer["priors"] = priors
    return JsonResponse(answer)


@experiment_view(VIEW)
def hypershap_figures(request, exp):
    """Every HyperSHAP game figure for one metric, measured from one trial.

    The stored games are gains over the config space's default, worked out at
    run completion. This answers for the sidebar's other baselines, the
    incumbent and the selected trial, which depend on the metric and on the
    click and so cannot be stored ahead of time. All three games at once, so
    the game selector switches between them without asking again; and as the
    same plots and warnings the page draws the stored games from, so it draws
    these the same way. `importance` carries the shares themselves, for the
    importance table's columns, which set every baseline side by side.
    """
    metric = request.GET.get("metric", "")
    idx_raw = request.GET.get("idx", "")
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    if not idx_raw.isdigit():
        return HttpResponseBadRequest("invalid trial index")

    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    idx = int(idx_raw)
    if result is None or idx >= len(result.trials):
        return HttpResponseBadRequest("invalid trial index")
    assert built is not None

    config_space = config_space_for(built)
    if config_space is None:
        reason = _("HyperSHAP needs the search space, which this experiment "
                   "does not carry and has no model to ask.")
        games = {game: ({metric: {}}, {metric: reason}, {metric: {}}, {metric: []}, {metric: 0.0})
                 for game in HP_GAME_FIELDS}
    else:
        games = analytics.compute_hp_games(
            config_space, result.trials, {metric: result.metric(metric)},
            seed=built["seed"], baseline=result.trials[idx].config,
            budget=settings.ANALYTICS_EAGER_MAX_COALITIONS or None)

    parts = ("importance", "warning", "interactions", "moebius")
    measured = replace(result, **{
        fields[part]: {metric: games[game][i][metric]}
        for game, fields in HP_GAME_FIELDS.items() for i, part in enumerate(parts)})
    figures = [f for f in _shown_figures(exp) if f.explains_games]
    return JsonResponse({
        "plots": _figure_plots(measured, figures, metric, config_space=config_space),
        "warnings": {game: games[game][1][metric] for game in HP_GAME_FIELDS},
        "importance": {game: games[game][0][metric] for game in HP_GAME_FIELDS},
    })


def _analytics_absent(result) -> bool:
    """True when a stored result has trials but no explanation games in it.

    Two ways to arrive here, both deliberate. A **cancelled** run skips them:
    pressing Cancel used to still buy the full analytics bill, so on a wide
    model you waited minutes for a run you had just stopped. A **partial** write
    (`experiments/services/run.py`) leaves them out for the same reason at finer grain —
    2^n_hp coalitions per game per metric is not something to pay per trial.

    Both used to be permanent: the numbers were computed once, at run
    completion, or never. `experiment_compute_analytics` is the way to ask for
    them afterwards, so declining to compute them eagerly no longer means
    declining to compute them at all.
    """
    return bool(result.trials) and not any(result.hyperparameter_importance.values())


@require_POST
@experiment_view(RUN)
def experiment_compute_analytics(request, exp):
    """Compute the explanation games for an experiment that has none.

    Synchronous, and bounded by the same budget the eager computation is held
    to: `eager_analytics_budget_exceeded` turns away anything wide enough to be
    a problem, which leaves this in the same range as the local-effects fetch
    the page already makes. Anything it declines is declined with a reason, on
    the page, rather than started and waited on.

    Written back onto the result rather than returned as a payload — which is
    where the three fetched figures differ, and why. Those answer a question
    that has no small precomputable set of answers (which trial, which
    hyperparameter), so each one is a fresh request. These are exactly the
    fields the result already has, computed from the same trials that are
    already stored, so filling them in makes the result what a completed run
    would have written — the page renders it with no live-update path of its
    own, the export carries it, and nobody pays for it twice.

    Refused while a run is in flight: it would race that run's own partial
    writes, and the run will compute them itself when it finishes.
    """
    if exp.is_running:
        return redirect("ui:experiment_detail", pk=exp.pk)

    analytics_service.compute_games(exp)
    return redirect("ui:experiment_detail", pk=exp.pk)


def _local_ablation_data(built, metric, idx, baseline=None):
    """The local-ablation figure + warning for one (metric, trial index) —
    HyperSHAP's `ablation` game, explaining that one trial's configuration
    against the trial at index *baseline*, or against the config space's
    default when that is None, rather than a global share like the other three
    games. `built` is `rebuild_experiment`'s full dict, since this needs the
    search space and the optimizer instance, neither of which the stored result
    carries.

    A trial explained against itself is not computed: every effect would be
    zero by construction, and a waterfall of zeros reads as a finding.

    Returns (figure_json_or_None, warning_or_None, raw_effects). The effects
    are the signed per-hyperparameter values the figure is drawn from, in the
    metric's own units, which `tuning_progress` needs and which cannot be read
    back out of the figure. The figure is None (with a warning) when there is no
    search space to be had — see `config_space_for` for the two places one can
    come from.
    """
    config_space = config_space_for(built)
    if config_space is None:
        return None, _("Local explanation needs the search space, which this "
                       "experiment does not carry and has no model to ask."), {}
    if baseline == idx:
        return None, _("This trial is the baseline."), {}
    result = built["result"]
    ablation, warning = analytics.compute_hp_ablation(
        config_space, result.trials, metric, result.trials[idx].config,
        seed=built["seed"],
        # From the result, not the registry: an imported run's objective is
        # declared inside its own file, and the explainer needs its direction
        # (see `_pair_trials_with_oriented_scores`).
        metric=result.metric(metric),
        baseline=None if baseline is None else result.trials[baseline].config)
    fig = hyperparameter_ablation_plot(ablation)
    return (json.loads(cast(str, fig.to_json())) if fig is not None else None), warning, ablation


#: Whether the page offers "still to gain" at all.
#:
#: Off: the measure is not ready to be read. The presentation problems were the
#: visible half, but the numbers are not established either — the same run,
#: same seed, gives tunability +0.096 in one process and +0.070 in another,
#: while being bit-identical within a process. `tuning_progress` subtracts one
#: of those quantities from another, so whatever moves between processes lands
#: in its result, and a reader would be comparing a difference against a
#: baseline that does not hold still. The tests below it check the arithmetic is
#: self-consistent, which is a weaker claim than the measure being sound.
#:
#: Everything stays in place and stops being reachable from the page, so the
#: switch back is this line — once the cross-process drift is understood.
TUNING_PROGRESS_ENABLED = False


def _partial_dependence_data(built, metric, hp_name, max_ice_curves=0):
    """The partial-dependence figure + warning for one (metric,
    hyperparameter) — same shape as `_local_ablation_data`, and for the same
    reason: fitting a surrogate and predicting across a grid needs the search
    space, not just the stored result, and depends on which hyperparameter is
    picked rather than being one of a small precomputable set.

    *max_ice_curves* is the experiment's `ice_max_curves` setting: how many
    trials get predicted and drawn, 0 for all of them. It caps the work, not
    just the picture — see `compute_partial_dependence`.

    Returns (figure_json_or_None, warning_or_None). The figure is None (with
    a warning) when there is no search space to be had.
    """
    config_space = config_space_for(built)
    if config_space is None:
        return None, _("Partial dependence needs the search space, which this "
                       "experiment does not carry and has no model to ask.")
    result = built["result"]
    grid, ice_lines, pdp, warning = analytics.compute_partial_dependence(
        config_space, result.trials, metric, hp_name, seed=built["seed"],
        max_ice_curves=max_ice_curves)
    fig = partial_dependence_plot(hp_name, grid, ice_lines, pdp)
    return (json.loads(cast(str, fig.to_json())) if fig is not None else None), warning


def _local_effects_data(built, metric, max_trials=0, baseline=None):
    """The beeswarm figure + warning for one metric — every sampled trial's
    local ablation, from `compute_local_effects`, against the trial at index
    *baseline* or against the config space's default when that is None.

    Same shape and same reasoning as `_partial_dependence_data`: it needs the
    search space, not just the stored result, and it is the one figure whose
    cost is per *trial*, which is what `max_trials` bounds.
    """
    config_space = config_space_for(built)
    if config_space is None:
        return None, _("Local effects need the search space, which this "
                       "experiment does not carry and has no model to ask.")
    result = built["result"]
    hp_names, rows, warning = analytics.compute_local_effects(
        config_space, result.trials, metric,
        seed=built["seed"], max_trials=max_trials, metric=result.metric(metric),
        baseline=None if baseline is None else result.trials[baseline].config)
    fig = local_effects_plot(hp_names, rows)
    return (json.loads(cast(str, fig.to_json())) if fig is not None else None), warning
