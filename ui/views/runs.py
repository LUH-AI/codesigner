from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import RUN, VIEW
from core.metrics import metric_for
from core.modelhost import deadline
from core.optimizers.base import FAILURE_CRITERIA
from experiments.models import account
from experiments.services import modelenv
from experiments.services.run_logic import decide_run, resolve_metric_change
from experiments.services.snapshot import config_space_for, rebuild_experiment
from experiments.services import run as run_service

from ..figures import FIGURES_BY_KEY
from ..permissions import experiment_view
from .common import (
    POLL_MAX_SECONDS, _ACTIVE, _best_panels, _figure_plots, _hp_names, _model_available,
    _poll_seconds, _posted_number, _shown_figures,
)
from .dashboard import _dashboard, _detail_context, _trial_rows


#: Each Run-form criterion's parser and range.
_STOPPING_FIELDS = {
    "max_trials": (int, 1, 100_000),
    # `_posted_stopping` bounds the target by its metric's range.
    "target_score": (float, None, None),
    "max_seconds": (float, 1.0, None),
    "max_trial_seconds": (float, 1.0, None),
    "no_improvement_trials": (int, 1, None),
    "incumbent_confidence": (float, 0.0, 1.0),
    "max_failures": (int, 1, None),
    "max_consecutive_failures": (int, 1, None),
}


def _posted_stopping(request, metric_name: str = "") -> dict:
    """`_posted_stopping` drops blank and unparseable criteria instead of
    reading them as zero, since zero would stop the run at once. *metric_name*
    supplies the bounds of `target_score`.
    """
    stopping = {}
    for key, (parse, low, high) in _STOPPING_FIELDS.items():
        raw = (request.POST.get(key) or "").strip()
        if not raw:
            continue
        try:
            value = parse(raw)
        except ValueError:
            continue
        if key == "target_score":
            low, high = metric_for(metric_name).bounds
            # Clamping keeps an out-of-range target from vanishing and tripping
            # the no-criterion refusal.
            if low is not None:
                value = max(value, low)
        elif value < low:
            continue
        stopping[key] = min(value, high) if high is not None else value
    return stopping


def _posted_trial_timeout(request) -> dict:
    """`_posted_trial_timeout` reads the per-trial deadline apart from the
    criteria, since a timeout fails one trial and the run continues. A blank box
    falls back to `MODEL_TRIAL_TIMEOUT`, and only an explicit 0 turns the
    deadline off.
    """
    mode = request.POST.get("trial_timeout_mode")
    if mode not in deadline.MODES:
        mode = deadline.MODE_FIXED
    return {
        "mode": mode,
        "seconds": _posted_number(request, "trial_timeout_seconds",
                                  settings.MODEL_TRIAL_TIMEOUT, low=0.0),
        "factor": _posted_number(request, "trial_timeout_factor",
                                 deadline.DEFAULT_FACTOR, low=1.0),
    }


@experiment_view(RUN)
def experiment_run(request, exp):
    """`experiment_run` starts a background run, or first asks for confirmation
    when the optimized metric changes; the confirmation posts a `decision` of
    "new" or "old".
    """
    if (request.method != "POST" or not exp.data.has_dataset or exp.is_running
            or not _model_available(exp)):
        return redirect("ui:experiment_detail", pk=exp.pk)

    chosen = request.POST.get("optimize_metric")
    stopping = _posted_stopping(request, chosen or exp.data.current_metric or "")
    trial_timeout = _posted_trial_timeout(request)
    decision = request.POST.get("decision")

    if not any(k not in FAILURE_CRITERIA for k in stopping):
        # The failure limits only catch broken runs and cannot end a working
        # one.
        context = _detail_context(request, exp)
        context["run_error"] = _("Set at least one stopping criterion, so the "
                                 "run has something to end on.")
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
                # Carried through so the confirmation keeps the run's limits.
                "stopping": stopping,
                "trial_timeout": trial_timeout,
            })

    run = run_service.create_run(exp, stopping, optimize_metric,
                                 started_by=account(request.user),
                                 trial_timeout=trial_timeout)
    run_service.start_background_run(run.id)
    return redirect("ui:experiment_detail", pk=exp.pk)


def _live_payloads(exp, since: int = 0):
    """`_live_payloads` builds every metric's `Figure.live` plots, since the
    page switches metric without asking the server, plus the trial rows past
    *since*. The keys match the page's `metric_plots[metric][key]` and
    `static_plots[key]` maps.
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

    # The trials table only gets its missing rows, appended so sort, paging and
    # selection survive. The best-configuration cards render whole, and the page
    # swaps each card's contents in place.
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
    """`run_status` serves the HTMX run poll and carries live figure payloads in
    the same fragment. Plots only rebuild when `?trials=` differs from the
    stored count, and a page without trials gets the count alone and reloads
    (`experiment_prerun.html`). The response for a finished run is an
    `HX-Refresh`.
    """
    run_service.sweep_orphaned_runs()
    active = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    if active is None:
        response = HttpResponse("")
        response["HX-Refresh"] = "true"
        return response

    # The stored count comes straight off the JSON, since `rebuild_experiment`
    # costs too much per poll.
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
    """`env_status` mirrors `run_status` with the partial during the build and
    `HX-Refresh` afterwards.
    """
    if not exp.env_pending:
        response = HttpResponse("")
        response["HX-Refresh"] = "true"
        return response
    return render(request, "ui/_env_status.html", {"experiment": exp})


@require_POST
@experiment_view(RUN)
def prepare_env(request, exp):
    """`prepare_env` retries the build, which helps after a transient failure or
    a late uv install. A bad dependency name needs a new experiment.
    """
    if exp.data.model_file and not exp.is_running:
        modelenv.start_preparation(exp)
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(RUN)
def run_cancel(request, exp):
    """`run_cancel` sets a flag the optimizer polls between trials, so finished
    trials are kept. `run_cancel` only accepts POST, so a prefetcher cannot
    cancel a run.
    """
    run_service.request_cancel(exp, account(request.user))
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(RUN)
def run_force_stop(request, exp):
    """`run_force_stop` marks a run that ignored a cancel as failed, freeing the
    experiment. A local worker keeps running and may overwrite the row if alive.
    A cluster job also gets `scancel`, which loses its unwritten trials.
    """
    run_service.force_stop(exp)
    return redirect("ui:experiment_detail", pk=exp.pk)
