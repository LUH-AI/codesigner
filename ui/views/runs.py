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
from experiments.services import modelenv
from experiments.services import run as run_service
from experiments.services.run_logic import decide_run, resolve_metric_change
from experiments.services.snapshot import config_space_for, rebuild_experiment

from ..figures import FIGURES_BY_KEY
from ..permissions import experiment_view
from .common import (
    _ACTIVE, _figure_plots, _hp_names, _model_available, _owner, _poll_seconds,
    _posted_number, _shown_figures,
)
from .dashboard import _dashboard, _detail_context, _trial_rows


# How each stopping criterion is read off the Run form: its parser and the range
# it is held to. Every criterion is here, including the trial cap — the form has
# no required field, only a requirement that at least one be filled in.
_STOPPING_FIELDS = {
    "max_trials": (int, 1, 100_000),
    # None, None: a target is compared against a score, so its range is the
    # metric's and not a constant here — 0 to 1 is accuracy's range, not every
    # metric's. Resolved per submission; see `_posted_stopping`.
    "target_score": (float, None, None),
    "max_seconds": (float, 1.0, None),
    "max_trial_seconds": (float, 1.0, None),
    "no_improvement_trials": (int, 1, None),
    "incumbent_confidence": (float, 0.0, 1.0),
    "max_failures": (int, 1, None),
    "max_consecutive_failures": (int, 1, None),
}


def _posted_stopping(request, metric_name: str = "") -> dict:
    """The stopping criteria as submitted, ignoring the blanks.

    A criterion left empty is absent rather than zero: zero would mean "stop
    immediately", which is never what an empty box asks for. Unparseable input
    is dropped the same way — and if that leaves nothing at all, the caller
    refuses the run rather than starting one that cannot end.

    *metric_name* is the metric the run will optimize, which is what bounds
    `target_score`: a target is compared against a score, so what counts as out
    of range is whatever that metric's range is, and an unbounded metric bounds
    it not at all.
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
            # Clamped at both ends rather than dropped below the low one.
            # Dropping made an out-of-range target *absent*, so a run asking for
            # nothing else was then refused for having no criterion at all —
            # which names a different problem from the one the reader created.
            if low is not None:
                value = max(value, low)
        elif value < low:
            continue
        stopping[key] = min(value, high) if high is not None else value
    return stopping


def _posted_trial_timeout(request) -> dict:
    """How long one call to the model may take, as submitted.

    Read separately from `_posted_stopping` because it is not a stopping
    criterion: nothing here ends a run, it ends a trial, and the run carries on
    with that trial recorded as failed. Storing it among the criteria would also
    mean the collector silently dropped it, since it filters to
    `STOPPING_CRITERIA`.

    Unlike a criterion, this one always has a value — a blank or unreadable box
    falls back to the deployment's `MODEL_TRIAL_TIMEOUT` rather than being
    absent, because "no deadline at all" is a thing to ask for deliberately (0)
    and not a thing to arrive at by mistyping.
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
    """Launch a background run of an experiment (or confirm a metric change).

    The Run form supplies the stopping criteria and the metric to optimize. A
    run that would change the optimized metric first shows a confirmation; the
    confirmation posts back with a `decision` of "new" or "old".
    """
    if (request.method != "POST" or not exp.data.has_dataset or exp.is_running
            or not _model_available(exp)):
        return redirect("ui:experiment_detail", pk=exp.pk)

    chosen = request.POST.get("optimize_metric")
    stopping = _posted_stopping(request, chosen or exp.data.current_metric or "")
    trial_timeout = _posted_trial_timeout(request)
    decision = request.POST.get("decision")

    if not any(k not in FAILURE_CRITERIA for k in stopping):
        # A run has to be able to end. Back to the page with the reason rather
        # than a started run that never finishes. The failure limits do not
        # count: they are how a run notices it is broken, and a run that is
        # working would never reach them.
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
                # Carried through the confirmation, or answering it would
                # silently drop the limits the run was set up with.
                "stopping": stopping,
                "trial_timeout": trial_timeout,
            })

    run = run_service.create_run(exp, stopping, optimize_metric,
                                 started_by=_owner(request),
                                 trial_timeout=trial_timeout)
    run_service.start_background_run(run.id)
    return redirect("ui:experiment_detail", pk=exp.pk)


def _stored_trial_count(exp) -> int:
    """How many trials *exp*'s saved result holds, without rebuilding it.

    Read straight off the JSON rather than through `rebuild_experiment`,
    because this is polled every two seconds and the answer is usually "the same
    as last time" — the whole point of asking is to decide whether it is worth
    doing any of the expensive work at all.
    """
    return len((exp.data.result or {}).get("data") or [])


def _best_panels(result, metric_names):
    """Each metric's best trial so far, as the best-configuration cards show
    it. A metric no trial has scored has no card — see `_detail_context`."""
    panels = []
    for m in metric_names:
        best_idx = result.best_index(m)
        if best_idx is None:
            continue
        best = result.trials[best_idx]
        panels.append({"metric": m, "best_n": best.trial, "best_idx": best_idx,
                       "best_score": best.scores[m],
                       "best_config": list(best.config.items())})
    return panels


def _live_payloads(exp, since: int = 0):
    """Fresh plots for every live figure of *exp*, for every metric, and the
    table rows for the trials the page has not seen.

    Only the figures declaring `Figure.live` — the ones that read nothing but
    the trials. Every metric rather than only the one being viewed, because the
    poll has no way to know which that is and the page switches between them
    without asking the server (see `metric_figures`); a live figure is four
    cheap Plotly builds, against the surrogate fits this deliberately excludes.

    Shaped to match the page's own two payload maps, so the browser merges
    rather than translates: `metric_plots[metric][key]` and `static_plots[key]`.
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

    # The trials table is server-rendered HTML rather than a plot, so it cannot
    # ride in the payload maps above. It gets the rows it is missing instead —
    # rendered from the same partial the page built the table from, and appended
    # rather than swapped, so a sort, a page, a scroll position and a selected
    # row all survive an update that only adds to the end.
    # The best configuration is server-rendered too, and moves whenever a trial
    # beats it. Rendered whole, every metric's card, from the same partial the
    # page drew them from; the page swaps each card's contents in place, so the
    # one being viewed stays the one showing.
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
    """HTMX poll target: the current run's status, or a refresh when finished.

    Also the channel for figures that move while the run is still going. There
    was never a rendering constraint on those — the trial-based figures need
    nothing but the trials — only a persistence one: the result was written once,
    at the end. `experiments/services/run.py` now writes it periodically, and this hands
    the page whatever has arrived since it last said what it had.

    No new endpoint and no second poll loop: this fragment is already fetched
    every two seconds and already replaced wholesale, so the payload rides along
    inside it. `?trials=` is the page saying how many trials it has drawn, which
    is what keeps the cost down — build the plots only when that disagrees with
    what is stored, which under a five-second write interval is at most every
    third poll.

    A page with no trials yet has no figure grid to draw into, so it is sent
    the count alone and reloads itself once — see `experiment_prerun.html`.
    """
    run_service.sweep_orphaned_runs()
    active = exp.runs.filter(status__in=_ACTIVE).order_by("-id").first()
    if active is None:
        response = HttpResponse("")
        response["HX-Refresh"] = "true"
        return response

    stored = _stored_trial_count(exp)
    try:
        seen = int(request.GET.get("trials", stored))
    except (TypeError, ValueError):
        seen = stored

    context = {"experiment": exp, "run": active, "trial_count": stored,
               "poll_seconds": _poll_seconds(exp)}
    if stored != seen:
        live = {"trials": stored}
        # Nothing to merge into on a page that has no figures — it is about to
        # reload, so building payloads for it would be work thrown away.
        if seen > 0:
            live.update(_live_payloads(exp, since=seen))
        context["live"] = live
    return render(request, "ui/_run_status.html", context)


@experiment_view(VIEW)
def env_status(request, exp):
    """Poll target while a model environment is being built.

    Shaped exactly like `run_status`: the partial while there is something to
    wait for, then an empty response with `HX-Refresh` so the page reloads into
    whatever the outcome was. poll.js already understands both.
    """
    if not exp.env_pending:
        response = HttpResponse("")
        response["HX-Refresh"] = "true"
        return response
    return render(request, "ui/_env_status.html", {"experiment": exp})


@require_POST
@experiment_view(RUN)
def prepare_env(request, exp):
    """Build (or rebuild) this experiment's model environment.

    For the retry button. Helps when the cause was transient — no network, an
    index down — or when uv has since been installed. A model file with a bad
    dependency name needs a new experiment, since there is no way to replace the
    file in place.
    """
    if exp.data.model_file and not exp.is_running:
        modelenv.start_preparation(exp)
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(RUN)
def run_cancel(request, exp):
    """Request cancellation of the experiment's active run.

    Cooperative: this sets a flag the running optimizer polls between trials, so
    the trials already finished are kept and the result is still written. It
    needs something to be reading the flag — see `run_force_stop` for when
    nothing is.

    POST only: it changes something, and on GET a prefetcher or an <img> tag
    pointing here would cancel someone's run without CSRF ever being consulted.
    """
    run_service.request_cancel(exp, _owner(request))
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(RUN)
def run_force_stop(request, exp):
    """Give up on a run that is not answering, and say so on the row.

    Cancelling is a request to whatever is executing the run. When nothing is —
    the process was restarted, the consumer died — there is nobody to read it,
    and the experiment sits at "running" with no way out. This is the way out.

    For a run in this process it does not kill anything, and does not pretend
    to: the web process has no handle on the worker, which is in another thread
    or another process entirely. What it does is stop the experiment waiting on
    it. If the worker turns out to be alive after all, it finishes as it always
    would have — its last act is a filtered update of this same row, which will
    put back whatever actually happened.

    A run on a cluster is the exception, because there *is* a handle: the job
    id. Abandoning the row alone would leave the job running to its wall-clock
    limit with nobody listening, holding an allocation nobody wants — so it is
    cancelled with the scheduler too. That loses whatever it had not written,
    which is what this escalation always costs.

    Offered only once cancelling has been asked for and has not worked, so it is
    the escalation rather than the first thing to hand: a cooperative cancel
    keeps the trials that already ran, and this cannot.
    """
    run_service.force_stop(exp)
    return redirect("ui:experiment_detail", pk=exp.pk)
