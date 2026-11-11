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

from access.policy import COMPUTE, DELETE, EDIT, EXPORT, RENAME, RUN, SHARE, VIEW, policy
from core.modelhost import deadline
from core.registry import MODELS, optimizer_class
from core import priors as prior_math
from experiments.models import Experiment
from experiments.services import model_export
from experiments.services.snapshot import config_space_for, rebuild_experiment, rebuild_result

from . import analytics
from .. import optimizer_form, run_form
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
    return _dashboard(request, exp, _detail_context(request, exp))


def _dashboard(request, exp, context):
    """`_dashboard` renders the pre-run page before trials."""
    if context["trial_count"]:
        return render(request, "ui/experiment_detail.html", context)
    fixed = optimizer_form.why_fixed(exp)
    may_edit = policy().may(request, exp, EDIT)
    context.update(optimizer_form.context(exp, locked=bool(fixed) or not may_edit,
                                          promoted=optimizer_form.PRERUN_PROMOTED))
    context["optimizer_locked_reason"] = fixed
    return render(request, "ui/experiment_prerun.html", context)


@require_POST
@experiment_view(RENAME)
def experiment_rename(request, exp):
    """An empty name shows `.UNTITLED`."""
    exp.data.name = (request.POST.get("name") or "").strip()[:200]
    exp.data.save(update_fields=["name"])
    return redirect(safe_next(request) or reverse("ui:experiment_detail", args=[exp.pk]))


@require_POST
@experiment_view(EDIT)
def experiment_optimizer(request, exp):
    refusal = optimizer_form.save(request, exp)
    if refusal:
        messages.error(request, refusal)
    else:
        messages.success(request, _("Saved."))
    return redirect("ui:experiment_detail", pk=exp.pk)


@experiment_view(VIEW)
def trial_panel(request, exp):
    """`trial_panel` renders the selected-config panel for `result.trials[idx]`.
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
                   # The standalone partial's traceback link needs `exp`.
                   "experiment": exp})


def _kept_trial_indices(exp, result):
    from experiments.models import TrialModel

    if result is None or not result.trials:
        return []
    kept = set(TrialModel.objects.filter(experiment=exp).values_list("trial", flat=True))
    return [i for i, trial in enumerate(result.trials) if trial.trial in kept]


def trial_models_reason(exp):
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
    """`_figure_options` returns `absoluteScale` and `perMetric`; None hides the
    toggle.
    """
    scale = figure.absolute_scale
    if figure.per_metric and scale is not None:
        scale = {name: figure.absolute_scale_for(result.metric(name))
                 for name in metric_names}
        scale = {name: views for name, views in scale.items() if views} or None
    return {"absoluteScale": scale, "perMetric": figure.per_metric}


def _trial_rows(result, metric_names, hp_names, start: int = 0) -> list:
    """`_trial_rows` skips rows before *start* for the poll append.
    """
    return [
        {
            # `idx` indexes the result; `n` is the displayed number.
            "idx": i,
            "n": t.trial,
            "config_values": [t.config.get(h) for h in hp_names],
            # Older rows can miss a metric.
            "scores": [t.scores.get(m) for m in metric_names],
            "duration": t.duration,
            # A failed trial's 0.0 scores look like a bad result.
            "failed": t.failed,
            "failure": t.failure,
        }
        for i, t in enumerate(result.trials)
        if i >= start
    ]


def metric_label(current_metric, original_metric):
    """`metric_label` returns "~" without an original metric and "Inconsistent"
    on a mismatch.
    """
    if original_metric is None:
        return "~"
    if current_metric != original_metric:
        return _("Inconsistent")
    return current_metric


def _selected_panel_data(result, metric, idx):
    """`_selected_panel_data` sets `delta` to zero and `is_best` for the best
    trial.
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
        # The metric decides which sign of `delta` is better.
        "delta_better": result.metric(metric).better(score, best),
        "is_best": idx == best_idx,
        "config": list(trial.config.items()),
        # A failed trial's score is a placeholder.
        "failed": trial.failed,
        "failure": trial.failure,
        "idx": idx,
        "has_traceback": bool(trial.traceback),
    }


def _prior_only_context(exp, built):
    """`_prior_only_context` serves the prior figure before any trial. The
    figure fetches its density with a `prior_only` request.
    """
    figure = next((f for f in _shown_figures(exp) if f.key == "prior"), None)
    config_space = config_space_for(built)
    if figure is None or config_space is None:
        return {}
    return {
        "prior_only_figure": figure,
        "hp_names": list(config_space.keys()),
        "beta_min": prior_math.BETA_RATIO_MIN,
        "beta_max": prior_math.BETA_RATIO_MAX,
        **_prior_judging_context(exp),
    }


TAB_LABELS = {
    "overview": gettext_lazy("Overview"),
    "hypershap": gettext_lazy("HyperSHAP"),
    "acquisition": gettext_lazy("Acquisition"),
    "misc": gettext_lazy("Misc"),
}


def _computed_version(exp, result):
    """`_computed_version` keys computed answers to the trial count, last run
    and settings. Each figure keys its own inputs.
    """
    import hashlib
    last = exp.runs.order_by("-id").values_list("pk", "status").first()
    digest = hashlib.sha1(json.dumps(resolve_settings(exp),
                                     sort_keys=True, default=str).encode()).hexdigest()[:12]
    return {"experiment": exp.pk,
            "version": f"{len(result.trials)}:{last}:{digest}"}


def _figure_tabs(layout, figures):
    from ..layout import placed
    return [dict(tab, label=TAB_LABELS[tab["tab"]]) for tab in placed(layout, figures)]


@experiment_view(VIEW)
@require_POST
def save_layout(request, exp):
    """`save_layout` only needs `VIEW`, since it only changes the reader's row.
    JSON requests get the normalised arrangement back; form posts redirect.
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


#: `_UNBUILT` requests a rebuild.
_UNBUILT: Any = object()


def _prepared_summary(exp) -> str:
    meta = exp.env_meta or {}
    parts = []
    deps = meta.get("dependencies") or []
    if deps:
        parts.append(", ".join(deps))
    if meta.get("python"):
        parts.append(f"Python {meta['python']}")
    return " · ".join(parts)


def experiment_header(request, exp, result=_UNBUILT):
    """`experiment_header` builds the top shared by the experiment pages.
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
    # Exports default to the viewed metric's best trial.
    current = exp.data.current_metric or (exp.data.metric_names or [None])[0]
    best = (result.best_index(current)
            if has_result and result is not None and current in exp.data.metric_names else None)

    return {
        "experiment": exp,  # `_run_status.html` URLs
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
        # An empty result shows "No results yet".
        "has_result": has_result,
        "may": {action: policy().may(request, exp, action)
                for action in (RUN, COMPUTE, EDIT, DELETE, EXPORT, SHARE, RENAME)},
        "ownership": _ownership(request, exp),
        "model_export_unavailable": model_export.unavailable(exp),
        # The parameters button only links trials with kept models.
        "kept_trial_indices": kept,
        "trial_models_reason": trial_models_reason(exp),
        "export_default_kept": best is not None and best in kept,
        "run_error": last_run.error if (last_run and last_run.status == "error") else None,
        # The form shows why a custom model cannot run.
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


def _budget_context(exp, done: int) -> dict:
    """`_budget_context` gives the run form its budget and opening trials."""
    return {"trial_budget": exp.data.trial_budget,
            "next_run_trials": run_form.opening_trials(exp.data.trial_budget, done)}


def _detail_context(request, exp):
    """`_detail_context` rebuilds the result read-only; the browser switches
    metrics.
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
        "supports_confidence": getattr(
            optimizer_class(exp.data.optimizer_name), "supports_confidence_stopping", False),
        "can_requery": hasattr(optimizer_class(exp.data.optimizer_name), "slice_challengers"),
        **_prior_judging_context(exp),
        "run_default_metric": exp.data.current_metric or (metric_names[0] if metric_names else None),
        "default_trial_timeout": int(settings.MODEL_TRIAL_TIMEOUT),
        "default_trial_factor": deadline.DEFAULT_FACTOR,
        # Polls echo the count; 0 means no grid.
        "trial_count": 0,
        "poll_seconds": _poll_seconds(exp),
        **_budget_context(exp, len(result.trials) if result is not None else 0),
    }
    # A result can lack trials, since `io.parse` accepts an empty `data`. The
    # prior figure needs the search space alone.
    if result is None or not result.trials:
        context.update(_prior_only_context(exp, built))
        return context

    # `run_status` rebuilds plots when the stored trial count differs.
    context["trial_count"] = len(result.trials)
    context.update(optimizer_form.next_run_context(exp) if policy().may(request, exp, EDIT)
                   else {})
    # An active run computes missing explanations when it ends.
    context["analytics_absent"] = _analytics_absent(result) and not active_run

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
            "selected": _selected_panel_data(result, m, best_idx),
            # The page's game selector reads `importance_by_game`; each warning
            # also covers interaction figures.
            "importance_by_game": {
                game: {
                    "warning": getattr(result, fields["warning"]).get(m),
                    # Each hyperparameter gets a row even without stored values.
                    "table": (sorted(getattr(result, fields["importance"]).get(m, {}).items(),
                                     key=lambda kv: kv[1], reverse=True)
                              or [(hp, None) for hp in hp_names]),
                }
                for game, fields in HP_GAME_FIELDS.items()
            },
            # The partial-dependence picker opens on the top hyperparameter.
            "top_hp": (
                sorted(result.hyperparameter_importance.get(m, {}).items(),
                       key=lambda kv: kv[1], reverse=True)[0][0]
                if result.hyperparameter_importance.get(m)
                else (list(best.config.keys())[0] if best.config else "")
            ),
        })

    # The browser fetches other metrics' plots on first selection
    # (`metric_figures`).
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

    # With a planned budget the Run form's target stays empty, since a near-best
    # target would end the run at its first improvement.
    incumbent_targets = ({} if exp.data.trial_budget else
                         {p["metric"]: surpass_target(p["best_score"]) for p in panels})
    context.update(
        result=result,
        panels=panels,
        metric_plots=metric_plots,
        static_plots=static_plots,
        figure_options={f.key: _figure_options(f, result, metric_names)
                        for f in figures},
        figure_views={f.key: list(f.views) for f in figures if f.views},
        # Sidebar figures sit outside the tabs.
        figures=figures,
        figure_tabs=_figure_tabs(figure_layout, figures),
        # `computed.js` version label.
        computed_version=_computed_version(exp, result),
        # For `layout.js` to edit.
        figure_layout=figure_layout,
        sidebar_figures=[f for f in figures if f.in_sidebar],
        selectable_figures=[f.key for f in figures if f.selects_trials],
        # `Figure.live` also drives `run_status` payloads.
        live_figures=[f.key for f in figures if f.live],
        game_figures=[f.key for f in figures if f.explains_games],
        # Colours come from the plot builders, so the script restates none. The
        # page draws failure marks from `failure`.
        selection_colors={"base": MARKER_COLOR, "selected": SELECTION_COLOR,
                          "comparison": ACCENT_COLOR,
                          "failure": NEGATIVE_COLOR, "proposal": PROPOSAL_COLOR},
        # `json_script` cannot serialize lazy labels.
        explanation_games=[(game, str(label)) for game, label in HP_GAME_LABELS.items()],
        explanation_game_help={game: str(text) for game, text in HP_GAME_HELP.items()},
        hp_names=hp_names,
        beta_min=prior_math.BETA_RATIO_MIN,
        beta_max=prior_math.BETA_RATIO_MAX,
        tuning_progress=analytics.TUNING_PROGRESS_ENABLED,
        incumbent_targets=incumbent_targets,
        incumbent_target=incumbent_targets.get(context["run_default_metric"], ""),
        trial_rows=_trial_rows(result, metric_names, hp_names),
        trials_page_size=Trials.page_size,
        trials_exhausted=(
            result.trials_limit is not None
            and len(result.trials) >= result.trials_limit
        ),
        # Analytics fetched without a button.
        autocompute={name: shown[autocompute_key(name)]
                     for name, _label in deferred_computations()},
    )
    return context
