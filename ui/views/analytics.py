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
from core.registry import METRICS, MODELS, OPTIMIZERS
from core import priors as prior_math
from experiments.services.snapshot import config_space_for, rebuild_experiment, rebuild_result
from experiments.services import analytics as analytics_service
from experiments.services import priors as prior_service
from experiments.services import snapshot as snapshot_adapter

from ..experiment_settings import resolve_settings
from ..figures import (
    HP_GAME_FIELDS, UNCERTAINTY_SCALE, forecast_plot, hyperparameter_ablation_plot,
    hyperparameter_importance_plot, hyperparameter_progress_plot, local_effects_plot,
    partial_dependence_plot, slice_span,
)
from ..permissions import experiment_view
from .common import _baseline_index, _figure_plots, _plot_json, _shown_figures, _target_name
from .priors import _acquisition_function, _slice_model


@experiment_view(VIEW)
def trial_traceback(request, exp):
    """`trial_traceback` serves a failed trial's stored traceback as plain text,
    since a traceback suits a file better than a sidebar panel. A trial without
    a stored traceback gets a 404, as timeouts, old results and some exports
    carry none.
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
    response["Content-Disposition"] = (
        f'inline; filename="{exp.identifier}-trial-{trial.trial}-traceback.txt"')
    return response


@experiment_view(VIEW)
def trial_ablation(request, exp):
    """`trial_ablation` serves the local HyperSHAP ablation for one (metric,
    trial index) on demand, since the figure depends on the selected trial and
    cannot be precomputed.
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
    rows, settled = (analytics.tuning_progress(built["result"], metric, ablation)
                     if TUNING_PROGRESS_ENABLED and baseline is None else ([], False))
    remaining = {row["name"]: row["remaining"] for row in rows}
    return JsonResponse({
        "figure": figure,
        "warning": warning,
        "rows": rows,
        "settled": settled,
        # `split` sizes each hyperparameter as the plain figure does and divides
        # it into banked and unbanked gain. `headroom` sizes hyperparameters by
        # the gain still left, for the "still to gain" box.
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
    """`partial_dependence` fits per hyperparameter on demand, since
    precomputing every hyperparameter on each load would waste work on a figure
    that shows one.
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
    """`forecast_figure` reruns the model once per backtest for trial *idx*, so
    the figure is deferred. *series* picks the series when the file holds
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
    """`surrogate_uncertainty` maps how much the surrogate's trees disagree over
    the plane of two hyperparameters, in the axes view only. Projections have no
    inverse, since `core.projection.project` discards the fitted PCA/PLS. The
    field is fetched per pair, since six hyperparameters already make fifteen
    pairs.
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
        # The colour ramp travels with the numbers and stays declared in
        # `plots.py` (see `UNCERTAINTY_SCALE`).
        "colorscale": UNCERTAINTY_SCALE,
    })


@experiment_view(VIEW)
def local_effects(request, exp):
    """`local_effects` runs an ablation game per trial, capped by
    `local_effects_max_trials`, and waits to be asked unless its autocompute
    setting is on.
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
    """`metric_figures` builds plots for metrics beyond the one the page opened
    on, from numbers stored at run completion. `_shown_figures` keeps
    switched-off figures unreachable here too.
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
    """`surrogate_prediction` sends the sample, the predictions and every view's
    coordinates in one response, and a new request follows only a metric change.
    `core.analytics.sample_prediction_space` does the sampling.
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
    # `_slice_model` shares the acquisition figure's cached sample, so the
    # second asker pays nothing.
    _surrogate, shared = _slice_model(exp, built, config_space, metric)
    answer = dict(shared) if shared else analytics.sample_prediction_space(
        optimizer, config_space, result.trials, metric, seed=built["seed"])
    answer.pop("configurations", None)
    answer["acquisition"] = bool(getattr(optimizer, "fits_surrogate", False))
    answer["acquisition_function"] = _acquisition_function(optimizer)
    # `priors` maps each prior to a density over the acquisition grid plus a
    # decay exponent, and the page interpolates each sample's weight from these.
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
    """`hypershap_figures` computes all three games against trial *idx*, a
    baseline that depends on the click and cannot be stored. `importance`
    carries the raw shares for the importance table.
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
    """A cancelled run and a partial write both skip the explanation games on
    purpose, since 2^n_hp coalitions per game per metric are too costly per
    trial. `experiment_compute_analytics` computes the games later.
    """
    return bool(result.trials) and not any(result.hyperparameter_importance.values())


@require_POST
@experiment_view(RUN)
def experiment_compute_analytics(request, exp):
    """`experiment_compute_analytics` runs synchronously under the eager budget
    (`eager_analytics_budget_exceeded`). It writes the games back onto the
    result, so the page and export see what a completed run would have written.
    During a run the view returns a redirect without computing, since the run
    computes the games itself and a second write would race it.
    """
    if exp.is_running:
        return redirect("ui:experiment_detail", pk=exp.pk)

    analytics_service.compute_games(exp)
    return redirect("ui:experiment_detail", pk=exp.pk)


def _local_ablation_data(built, metric, idx, baseline=None):
    """`_local_ablation_data` explains trial *idx* against trial *baseline*, or
    the default when *baseline* is None, and needs `built` for the search space.
    A trial against itself gets a warning, since an all-zero waterfall reads as
    a finding. The raw effects are returned for `tuning_progress`, which cannot
    read them from the figure.
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
        # Direction comes from the result, since an imported run declares its
        # own objective.
        metric=result.metric(metric),
        baseline=None if baseline is None else result.trials[baseline].config)
    fig = hyperparameter_ablation_plot(ablation)
    return (json.loads(cast(str, fig.to_json())) if fig is not None else None), warning, ablation


#: `TUNING_PROGRESS_ENABLED` is off because tunability drifts between processes
#: (+0.096 versus +0.070 for one seed) while staying identical within one, and
#: `tuning_progress` subtracts such values. The code stays in place, so flipping
#: `TUNING_PROGRESS_ENABLED` re-enables the feature once the drift is
#: understood.
TUNING_PROGRESS_ENABLED = False


def _partial_dependence_data(built, metric, hp_name, max_ice_curves=0):
    """*max_ice_curves* caps how many trials are predicted and drawn, 0 for all,
    and limits the work too. The figure is None, with a warning, when no search
    space is available.
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
    """`_local_effects_data` explains every sampled trial against *baseline* or
    the default, and `max_trials` bounds its per-trial cost.
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
