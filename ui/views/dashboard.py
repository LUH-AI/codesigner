import json
from typing import Any

from django.conf import settings
from django.contrib import messages
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.translation import gettext_lazy, ngettext
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import DELETE, EDIT, EXPORT, RENAME, RUN, SHARE, VIEW, policy
from core.modelhost import deadline
from core.registry import MODELS, optimizer_class
from core import priors as prior_math
from experiments.models import Experiment
from experiments.services import model_export
from experiments.services.snapshot import config_space_for, rebuild_experiment, rebuild_result

from . import analytics
from .. import optimizer_form
from ..experiment_settings import resolve_settings
from ..figures import (
    ACCENT_COLOR, HP_GAME_FIELDS, HP_GAME_HELP, HP_GAME_LABELS, MARKER_COLOR, NEGATIVE_COLOR,
    PROPOSAL_COLOR, SELECTION_COLOR, autocompute_key, deferred_computations,
)
from ..figures.catalog import Trials
from ..navigation import safe_next
from ..optimizer_labels import STOPPED_BY_LABELS
from ..permissions import experiment_view
from .analytics import _analytics_absent
from .common import (
    _ACTIVE, _best_panels, _figure_plots, _model_available, _ownership, _poll_seconds,
    _prior_judging_context, _shown_figures, surpass_target,
)


@experiment_view(VIEW)
def experiment_detail(request, exp):
    """`experiment_detail` shows the saved experiment's dashboard."""
    return _dashboard(request, exp, _detail_context(request, exp))


def _dashboard(request, exp, context):
    """`_dashboard` shows figures after the first trial, else the pre-run page."""
    if context["trial_count"]:
        return render(request, "ui/experiment_detail.html", context)
    fixed = optimizer_form.why_fixed(exp)
    may_edit = policy().may(request, exp, EDIT)
    context.update(optimizer_form.context(exp, locked=bool(fixed) or not may_edit))
    context["optimizer_locked_reason"] = fixed
    return render(request, "ui/experiment_prerun.html", context)


@require_POST
@experiment_view(RENAME)
def experiment_rename(request, exp):
    """`experiment_rename` lets an empty name stand for "Untitled Experiment"."""
    exp.data.name = (request.POST.get("name") or "").strip()[:200]
    exp.data.save(update_fields=["name"])
    return redirect(safe_next(request) or reverse("ui:experiment_detail", args=[exp.pk]))


@require_POST
@experiment_view(EDIT)
def experiment_optimizer(request, exp):
    """`experiment_optimizer` saves the optimizer before the first trial."""
    refusal = optimizer_form.save(request, exp)
    if refusal:
        messages.error(request, refusal)
    else:
        messages.success(request, _("Saved."))
    return redirect("ui:experiment_detail", pk=exp.pk)


@experiment_view(VIEW)
def trial_panel(request, exp):
    """`trial_panel` renders the selected-config panel for a click on the
    performance figure, with *idx* indexing `result.trials`.
    """
    metric = request.GET.get("metric", "")
    idx_raw = request.GET.get("idx", "")

    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    if not idx_raw.lstrip("-").isdigit():
        return HttpResponseBadRequest("invalid trial index")

    result = rebuild_result(exp)
    idx = int(idx_raw)
    if result is None or not (0 <= idx < len(result.trials)):
        return HttpResponseBadRequest("invalid trial index")

    return render(request, "ui/_selected_config_inner.html",
                  {"sel": _selected_panel_data(result, metric, idx),
                   # The traceback link needs `exp` because the partial is also
                   # served alone.
                   "experiment": exp})


def _kept_trial_indices(exp, result):
    """`_kept_trial_indices` finds trials with kept models."""
    from experiments.models import TrialModel

    if result is None or not result.trials:
        return []
    kept = set(TrialModel.objects.filter(experiment=exp).values_list("trial", flat=True))
    return [i for i, trial in enumerate(result.trials) if trial.trial in kept]


def trial_models_reason(exp):
    """`trial_models_reason` explains why trials lack kept models."""
    from core.optimizers.trial import keeps_its_fit

    model = MODELS.get(exp.data.model_name)
    if exp.data.model_file or model is None or not keeps_its_fit(model):
        return _("Trial parameters are kept for the built-in models only: an uploaded model "
                 "is fitted in its own environment, which nothing is brought back from.")
    if not getattr(model, "has_parameters", True):
        return _("%(model)s has no fitted parameters to keep: its forecast is its history "
                 "replayed.") % {"model": model.name}
    best = resolve_settings(exp).get("keep_best_trial_models") or 0
    if not best and not resolve_settings(exp).get("keep_last_run_trial_models"):
        return _("This experiment keeps no trial models. Turn it on in Settings; a new run "
                 "keeps them from then on.")
    return ngettext(
        "This trial's model was not kept: a run keeps only the best %(n)s trial's model "
        "(and with the option on, its own), or its group had no room. Change it in Settings.",
        "This trial's model was not kept: a run keeps only the best %(n)s trials' models "
        "(and with the option on, its own), or its group had no room. Change it in Settings.",
        best) % {"n": best}


def _figure_options(figure, result, metric_names):
    """`_figure_options` gives the page script one figure's `absoluteScale` and
    `perMetric`. A per-metric scale is keyed by metric, and None hides the
    toggle.
    """
    scale = figure.absolute_scale
    if figure.per_metric and scale is not None:
        scale = {name: figure.absolute_scale_for(result.metric(name))
                 for name in metric_names}
        scale = {name: views for name, views in scale.items() if views} or None
    return {"absoluteScale": scale, "perMetric": figure.per_metric}


def _trial_rows(result, metric_names, hp_names, start: int = 0) -> list:
    """`_trial_rows` renders rows from *start* so the page build and the poll
    append share one partial.
    """
    return [
        {
            # `idx` is the result position every trial reference uses; `n` is
            # the displayed number.
            "idx": i,
            "n": t.trial,
            "config_values": [t.config.get(h) for h in hp_names],
            # `.get` tolerates older stored rows that miss a metric.
            "scores": [t.scores.get(m) for m in metric_names],
            "duration": t.duration,
            # The row flags a failed trial and its reason, since its 0.0 scores
            # look like a bad result.
            "failed": t.failed,
            "failure": t.failure,
        }
        for i, t in enumerate(result.trials)
        if i >= start
    ]


def metric_label(current_metric, original_metric):
    """`metric_label` returns "~" before a run commits a metric and
    "Inconsistent" when the current metric differs from the results' metric.
    """
    if original_metric is None:
        return "~"
    if current_metric != original_metric:
        return _("Inconsistent")
    return current_metric


def _selected_panel_data(result, metric, idx):
    """`_selected_panel_data` always includes the distance from the best trial,
    which is zero for the best itself with `is_best` set.
    """
    trials = result.trials
    best_idx = result.best_index(metric)
    trial = trials[idx]
    score, best = trial.scores[metric], trials[best_idx].scores[metric]
    return {
        "metric": metric,
        "trial_n": trial.trial,
        "score": score,
        "delta": score - best,
        # The metric decides which sign of `delta` is better, and the template
        # must avoid comparing it to zero.
        "delta_better": result.metric(metric).better(score, best),
        "is_best": idx == best_idx,
        "config": list(trial.config.items()),
        # A failed trial's score is a placeholder; `idx` is what the traceback
        # endpoint takes.
        "failed": trial.failed,
        "failure": trial.failure,
        "idx": idx,
        "has_traceback": bool(trial.traceback),
    }


def _prior_only_context(exp, built):
    """`_prior_only_context` gives the prior figure what it needs before any
    trial, without acquisition, candidates or incumbent. The figure fetches its
    density with a `prior_only` request, and an empty dict leaves the page
    unchanged.
    """
    figure = next((f for f in _shown_figures(exp) if f.key == "prior"), None)
    config_space = config_space_for(built)
    if figure is None or config_space is None:
        return {}
    return {
        "prior_only_figure": figure,
        # Taken from the space, since there are no trials.
        "hp_names": list(config_space.keys()),
        "beta_min": prior_math.BETA_RATIO_MIN,
        "beta_max": prior_math.BETA_RATIO_MAX,
        **_prior_judging_context(exp),
    }


#: Tab labels on the page.
TAB_LABELS = {
    "overview": gettext_lazy("Overview"),
    "hypershap": gettext_lazy("HyperSHAP"),
    "acquisition": gettext_lazy("Acquisition"),
    "misc": gettext_lazy("Misc"),
}


def _computed_version(exp, result):
    """`_computed_version` keys computed answers to the trials, the last run and
    the settings. The prior, selected trial and shown hyperparameter belong in
    each figure's own key.
    """
    import hashlib
    last = exp.runs.order_by("-id").values_list("pk", "status").first()
    digest = hashlib.sha1(json.dumps(resolve_settings(exp),
                                     sort_keys=True, default=str).encode()).hexdigest()[:12]
    return {"experiment": exp.pk,
            "version": f"{len(result.trials)}:{last}:{digest}"}


def _figure_tabs(layout, figures):
    """Tabs arranged per *layout*."""
    from ..layout import placed
    return [dict(tab, label=TAB_LABELS[tab["tab"]]) for tab in placed(layout, figures)]


@experiment_view(VIEW)
@require_POST
def save_layout(request, exp):
    """`save_layout` saves the reader's own arrangement under `VIEW`, since only
    the reader's row changes. A JSON `layout` or `reset` returns the normalised
    arrangement; the settings page's forms redirect to the experiment.
    """
    from .. import layout as layouts
    if request.POST.get("reset"):
        layouts.reset(exp, request.user)
        return redirect("ui:experiment_detail", pk=exp.pk)
    if request.POST.get("make_default"):
        layouts.save_user_default(request.user, layouts.layout_for(exp, request.user))
        return redirect("ui:experiment_detail", pk=exp.pk)
    if request.POST.get("reset_default"):
        layouts.reset_user_default(request.user)
        return redirect("ui:experiment_detail", pk=exp.pk)
    try:
        body = json.loads(request.body or b"{}")
    except ValueError:
        return HttpResponseBadRequest("not JSON")
    if not isinstance(body, dict):
        return HttpResponseBadRequest("not an object")
    if body.get("reset"):
        return JsonResponse({"layout": layouts.reset(exp, request.user)})
    if not isinstance(body.get("layout"), dict):
        return HttpResponseBadRequest("no layout")
    return JsonResponse({"layout": layouts.save(exp, request.user, body["layout"])})


#: Sentinel meaning rebuild here, unlike None.
_UNBUILT: Any = object()


def _prepared_summary(exp) -> str:
    """`_prepared_summary` describes the environment in one line."""
    meta = exp.env_meta or {}
    parts = []
    deps = meta.get("dependencies") or []
    if deps:
        parts.append(", ".join(deps))
    if meta.get("python"):
        parts.append(f"Python {meta['python']}")
    return " · ".join(parts)


def experiment_header(request, exp, result=_UNBUILT):
    """`experiment_header` builds the shared top of the dashboard, Data Handling
    and timeline, rebuilding *result* unless the caller passes one.
    """
    if result is _UNBUILT:
        result = rebuild_result(exp)
    last_run = exp.runs.order_by("-id").first()

    run_summary = None
    if last_run and last_run.status in ("done", "cancelled") and last_run.duration is not None:
        total = last_run.duration
        trials = last_run.trial_seconds or 0.0
        run_summary = {"total": total, "trials": trials, "overhead": max(0.0, total - trials),
                       "count": last_run.trial_count or 0,
                       "stopped_by": STOPPED_BY_LABELS.get(last_run.stopped_by)}

    has_result = result is not None and bool(result.trials)
    kept = _kept_trial_indices(exp, result)
    # Before any selection, the export buttons name the best trial on the viewed
    # metric.
    current = exp.data.current_metric or (exp.data.metric_names or [None])[0]
    best = (result.best_index(current)
            if has_result and result is not None and current in exp.data.metric_names else None)

    return {
        "experiment": exp,  # for `_run_status.html` URLs
        "summary": {
            "pk": exp.pk,
            "name": exp.title,
            "given_name": exp.name,
            "identifier": exp.identifier,
            "model_name": exp.data.model_name,
            "optimizer_name": exp.data.optimizer_name,
            "current_metric": exp.data.current_metric,
            "metric_label": metric_label(exp.data.current_metric, exp.data.original_metric),
            "seed": exp.data.seed,
            "evaluation": model_export.evaluation_label(exp),
        },
        # An empty result takes the template's "No results yet" branch.
        "has_result": has_result,
        # Per-action permissions for the buttons.
        "may": {action: policy().may(request, exp, action)
                for action in (RUN, EDIT, DELETE, EXPORT, SHARE, RENAME)},
        "ownership": _ownership(request, exp),
        "model_export_unavailable": model_export.unavailable(exp),
        # The parameters button links only for a selected trial whose model was
        # kept.
        "kept_trial_indices": kept,
        "trial_models_reason": trial_models_reason(exp),
        "export_default_kept": best is not None and best in kept,
        "run_error": last_run.error if (last_run and last_run.status == "error") else None,
        # Shown so the form explains its own refusal.
        "model_refusal": policy().custom_model_refusal(exp, request.user),
        "env": {
            "status": exp.env_status,
            "pending": exp.env_pending,
            "in_process": exp.env_in_process,
            "ready": exp.env_status == Experiment.ENV_READY,
            "failed": exp.env_status == Experiment.ENV_FAILED,
            "error": exp.env_error,
            "summary": _prepared_summary(exp),
        },
        "run_summary": run_summary,
    }


def _detail_context(request, exp):
    """`_detail_context` rebuilds the result read-only and builds panels and
    figures for the enabled figures; metrics switch in the browser.
    """
    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    metric_names = list(exp.data.metric_names)
    active_run = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    header = experiment_header(request, exp, result)
    model_refusal = header["model_refusal"]

    context = {
        **header,
        "metric_names": metric_names,
        "active_run": active_run,
        "can_run": (exp.data.has_dataset and _model_available(exp)
                    and policy().may(request, exp, RUN)
                    and not model_refusal),
        # Only a model-fitting optimizer supports the confidence criterion.
        "supports_confidence": getattr(
            optimizer_class(exp.data.optimizer_name), "supports_confidence_stopping", False),
        # Requery needs an optimizer with `slice_challengers`.
        "can_requery": hasattr(optimizer_class(exp.data.optimizer_name), "slice_challengers"),
        **_prior_judging_context(exp),
        "run_default_metric": exp.data.current_metric or (metric_names[0] if metric_names else None),
        # Deadline fields open on the deployment's `MODEL_TRIAL_TIMEOUT`.
        "default_trial_timeout": int(settings.MODEL_TRIAL_TIMEOUT),
        "default_trial_factor": deadline.DEFAULT_FACTOR,
        # The poll echoes the drawn trial count; 0 means no figure grid.
        "trial_count": 0,
        "poll_seconds": _poll_seconds(exp),
    }
    # A result without trials takes the same early return, since `io.parse`
    # accepts an empty `data`. Only the prior figure goes out, because a prior
    # needs the search space alone (`_prior_only_context`).
    if result is None or not result.trials:
        context.update(_prior_only_context(exp, built))
        return context

    # `run_status` compares the drawn trial count with the stored one to decide
    # whether to rebuild plots.
    context["trial_count"] = len(result.trials)
    # Missing explanations can be requested, except during a run, which computes
    # them at the end.
    context["analytics_absent"] = _analytics_absent(result) and not active_run

    # Switched-off figures are never built.
    figures = _shown_figures(exp)
    from ..layout import layout_for
    figure_layout = layout_for(exp, request.user)
    shown = resolve_settings(exp)
    hp_names = list(result.trials[0].config.keys()) if result.trials else []

    panels = _best_panels(result, metric_names)
    for panel in panels:
        m, best_idx = panel["metric"], panel["best_idx"]
        best = result.trials[best_idx]
        panel.update({
            # Defaults to the best trial.
            "selected": _selected_panel_data(result, m, best_idx),
            # Each HyperSHAP game gets its warning and table rows, keyed by game
            # for the page's game selector. The warning also covers the
            # interaction figures from the same call.
            "importance_by_game": {
                game: {
                    "warning": getattr(result, fields["warning"]).get(m),
                    # Each hyperparameter gets a row even without stored values,
                    # for the incumbent and selected columns.
                    "table": (sorted(getattr(result, fields["importance"]).get(m, {}).items(),
                                     key=lambda kv: kv[1], reverse=True)
                              or [(hp, None) for hp in hp_names]),
                }
                for game, fields in HP_GAME_FIELDS.items()
            },
            # The partial-dependence picker opens on the metric's most important
            # hyperparameter, else the first.
            "top_hp": (
                sorted(result.hyperparameter_importance.get(m, {}).items(),
                       key=lambda kv: kv[1], reverse=True)[0][0]
                if result.hyperparameter_importance.get(m)
                else (list(best.config.keys())[0] if best.config else "")
            ),
        })

    # Only the opening metric's plots are built; the browser fetches others on
    # first selection (`metric_figures`). Metric-independent figures are drawn
    # once here.
    opening_metric = context["run_default_metric"]
    config_space = config_space_for(built)
    metric_plots = {
        opening_metric: _figure_plots(
            result, [f for f in figures if f.per_metric], opening_metric,
            config_space=config_space),
    } if opening_metric else {}
    static_plots = _figure_plots(result, [f for f in figures if not f.per_metric],
                                 config_space=config_space)
    static_plots = {key: plot for key, plot in static_plots.items() if plot is not None}

    # The Run form's target starts at the incumbent, per metric for the
    # dropdown.
    incumbent_targets = {p["metric"]: surpass_target(p["best_score"]) for p in panels}
    context.update(
        result=result,
        panels=panels,
        metric_plots=metric_plots,
        static_plots=static_plots,
        figure_options={f.key: _figure_options(f, result, metric_names)
                        for f in figures},
        # The script reads view names here instead of guessing from payload
        # shape.
        figure_views={f.key: list(f.views) for f in figures if f.views},
        # Tab arrangement per `ui/layout.py`; sidebar figures sit outside the
        # tabs.
        figures=figures,
        figure_tabs=_figure_tabs(figure_layout, figures),
        # Labels for computed answers (`computed.js`).
        computed_version=_computed_version(exp, result),
        # The full arrangement, for layout.js to edit.
        figure_layout=figure_layout,
        sidebar_figures=[f for f in figures if f.in_sidebar],
        # Figures that take a trial click (`Figure.selects_trials`).
        selectable_figures=[f.key for f in figures if f.selects_trials],
        # `Figure.live` drives both this list and the `run_status` payloads.
        live_figures=[f.key for f in figures if f.live],
        # Figures that draw HyperSHAP games (`Figure.explains_games`).
        game_figures=[f.key for f in figures if f.explains_games],
        # Highlight colours come from the plot builders so the script restates
        # none. A 3D scene draws no marker outline, and the page draws its
        # failure mark from `failure`.
        selection_colors={"base": MARKER_COLOR, "selected": SELECTION_COLOR,
                          "comparison": ACCENT_COLOR,
                          "failure": NEGATIVE_COLOR, "proposal": PROPOSAL_COLOR},
        # Lazy labels resolve here because `json_script` cannot serialize them.
        explanation_games=[(game, str(label)) for game, label in HP_GAME_LABELS.items()],
        explanation_game_help={game: str(text) for game, text in HP_GAME_HELP.items()},
        hp_names=hp_names,
        # β field bounds, fixed constants.
        beta_min=prior_math.BETA_RATIO_MIN,
        beta_max=prior_math.BETA_RATIO_MAX,
        # With "still to gain" off, the markup goes too.
        tuning_progress=analytics.TUNING_PROGRESS_ENABLED,
        incumbent_targets=incumbent_targets,
        incumbent_target=incumbent_targets.get(context["run_default_metric"], ""),
        trial_rows=_trial_rows(result, metric_names, hp_names),
        # The table's page size comes from the figure.
        trials_page_size=Trials.page_size,
        trials_exhausted=(
            result.trials_limit is not None
            and len(result.trials) >= result.trials_limit
        ),
        # Deferred analytics allowed to fetch without a button.
        autocompute={name: shown[autocompute_key(name)]
                     for name, _label in deferred_computations()},
    )
    return context
