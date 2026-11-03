from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import EDIT, RUN, VIEW, policy
from core.optimizers.base import FAILURE_CRITERIA
from experiments.models import account
from experiments.services import modelenv
from experiments.services.run_logic import decide_run, resolve_metric_change
from experiments.services.snapshot import config_space_for, rebuild_experiment
from experiments.services import run as run_service

from .common import (
    POLL_MAX_SECONDS, _ACTIVE, _best_panels, _figure_plots, _hp_names, _model_available,
    _poll_seconds, _shown_figures,
)
from .. import optimizer_form, run_form
from ..figures import FIGURES_BY_KEY
from ..permissions import experiment_view
from .dashboard import _dashboard, _detail_context, _trial_rows


@experiment_view(RUN)
def experiment_run(request, exp):
    """`experiment_run` confirms a metric change with a "new" or "old"
    `decision`.
    """
    if (request.method != "POST" or not exp.data.has_dataset or exp.is_running
            or not _model_available(exp)):
        return redirect("ui:experiment_detail", pk=exp.pk)

    chosen = request.POST.get("optimize_metric")
    stopping = run_form.stopping_from(request.POST, chosen or exp.data.current_metric or "")
    trial_timeout = run_form.trial_timeout_from(request.POST)
    decision = request.POST.get("decision")

    if not any(k not in FAILURE_CRITERIA for k in stopping):
        # Failure limits only catch broken runs.
        context = _detail_context(request, exp)
        context["run_error"] = _("Set at least one stopping criterion, so the "
                                 "run has something to end on.")
        return _dashboard(request, exp, context)

    if request.POST.get("next_run_settings") and policy().may(request, exp, EDIT):
        # Next-run settings apply whichever metric the run optimizes.
        refusal = optimizer_form.save(request, exp)
        if refusal:
            context = _detail_context(request, exp)
            context["run_error"] = refusal
            return _dashboard(request, exp, context)

    if decision:
        optimize_metric = resolve_metric_change(decision, exp.data.current_metric, chosen)
        if optimize_metric is None:
            return redirect("ui:experiment_detail", pk=exp.pk)
    else:
        action, optimize_metric = decide_run(exp.data.original_metric, exp.data.current_metric, chosen)
        if action == "warn":
            return render(request, "ui/metric_change.html", {
                "experiment": exp, "chosen": chosen,
                "stopping": stopping,
                "trial_timeout": trial_timeout,
            })

    run = run_service.create_run(exp, stopping, optimize_metric,
                                 started_by=account(request.user),
                                 trial_timeout=trial_timeout)
    run_service.start_background_run(run.id)
    return redirect("ui:experiment_detail", pk=exp.pk)


def _live_payloads(exp, since: int = 0):
    """`_live_payloads` builds every metric's live plots, since the page
    switches metric locally. Its keys match the page's
    `metric_plots[metric][key]` and `static_plots[key]`.
    """
    built = rebuild_experiment(exp)
    result = built["result"] if built else None
    if result is None or not result.trials:
        return {}

    shown = _shown_figures(exp)
    live = [f for f in shown if f.live]
    config_space = config_space_for(built)
    per_metric = [f for f in live if f.per_metric]
    static = [f for f in live if not f.per_metric]
    payload = {
        "metric_plots": {
            metric: _figure_plots(result, per_metric, metric, config_space=config_space)
            for metric in exp.data.metric_names
        } if per_metric else {},
        "static_plots": _figure_plots(result, static, None, config_space=config_space),
    }

    # The table only appends missing rows, so sort, paging and selection
    # survive. The page swaps best-configuration card contents in place.
    if any(f.key == "best_configuration" for f in shown):
        payload["best_html"] = render_to_string(
            "ui/figures/best_configuration.html",
            {"panels": _best_panels(result, exp.data.metric_names),
             "figure": FIGURES_BY_KEY["best_configuration"]})

    if any(f.key == "trials" for f in shown) and len(result.trials) > since:
        payload["rows_html"] = render_to_string(
            "ui/figures/_trial_rows.html",
            {"trial_rows": _trial_rows(result, exp.data.metric_names,
                                       _hp_names(result), start=since)})
    return payload


@experiment_view(VIEW)
def run_status(request, exp):
    """`run_status` serves the run poll with live figure payloads. Plots only
    rebuild when `?trials=` differs from the stored count; a page without trials
    reloads on the count. A finished run gets `HX-Refresh`.
    """
    run_service.sweep_orphaned_runs()
    active = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    if active is None:
        response = HttpResponse("")
        response["HX-Refresh"] = "true"
        return response

    stored = len((exp.data.result or {}).get("data") or [])
    try:
        seen = int(request.GET.get("trials", stored))
    except (TypeError, ValueError):
        seen = stored

    context = {"experiment": exp, "run": active, "trial_count": stored,
               "poll_seconds": (POLL_MAX_SECONDS if active.status == "submitted"
                                else _poll_seconds(exp))}
    if stored != seen:
        live = {"trials": stored}
        if seen > 0:
            live.update(_live_payloads(exp, since=seen))
        context["live"] = live
    return render(request, "ui/_run_status.html", context)


@experiment_view(VIEW)
def env_status(request, exp):
    if not exp.env_pending:
        response = HttpResponse("")
        response["HX-Refresh"] = "true"
        return response
    return render(request, "ui/_env_status.html", {"experiment": exp})


@require_POST
@experiment_view(RUN)
def prepare_env(request, exp):
    """`prepare_env` retries the build; a bad dependency name needs a new
    experiment.
    """
    if exp.data.model_file and not exp.is_running:
        modelenv.start_preparation(exp)
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(RUN)
def run_cancel(request, exp):
    """`run_cancel` sets a flag the optimizer polls between trials. Requiring
    POST stops a prefetcher from cancelling.
    """
    run_service.request_cancel(exp, account(request.user))
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(RUN)
def run_force_stop(request, exp):
    """`run_force_stop` fails a run that ignored a cancel. A live local worker
    can still overwrite the row.
    """
    run_service.force_stop(exp)
    return redirect("ui:experiment_detail", pk=exp.pk)
