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
    """Show a saved experiment: its config, results/figures, run state, Run form."""
    return _dashboard(request, exp, _detail_context(request, exp))


def _dashboard(request, exp, context):
    """The dashboard: its figures once there is a trial; before, the page for
    what can still be decided — the optimizer and the prior."""
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
    """Give the experiment a name, or take it away: an empty one shows as
    "Untitled Experiment". Back to the page it was renamed from."""
    exp.data.name = (request.POST.get("name") or "").strip()[:200]
    exp.data.save(update_fields=["name"])
    return redirect(safe_next(request) or reverse("ui:experiment_detail", args=[exp.pk]))


@require_POST
@experiment_view(EDIT)
def experiment_optimizer(request, exp):
    """Save the optimizer and its settings, before the experiment's first
    trial."""
    refusal = optimizer_form.save(request, exp)
    if refusal:
        messages.error(request, refusal)
    else:
        messages.success(request, _("Saved."))
    return redirect("ui:experiment_detail", pk=exp.pk)


@experiment_view(VIEW)
def trial_panel(request, exp):
    """Render the selected-config panel for one (metric, trial index).

    Backs click-to-select on the performance figure (curve 0 only, point index
    into result.trials) — the browser fetches this fragment and swaps it into
    the panel for the metric currently being viewed.
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
                   # The traceback link needs to know which experiment it is
                   # asking about; on the first render the page context has it,
                   # and this partial is also served on its own.
                   "experiment": exp})


def _kept_trial_indices(exp, result):
    """The indices into *result*'s trials of those whose model was kept."""
    from experiments.models import TrialModel

    if result is None or not result.trials:
        return []
    kept = set(TrialModel.objects.filter(experiment=exp).values_list("trial", flat=True))
    return [i for i, trial in enumerate(result.trials) if trial.trial in kept]


def trial_models_reason(exp):
    """Why a trial of *exp* has no kept model to export."""
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
    """One figure's display behaviour, for the page script.

    `absoluteScale` is the relayout its absolute/relative toggle pins to. For a
    per-metric figure it is keyed by metric first, because a range that is right
    for an accuracy is a fiction for a cost — and None when no metric offers one,
    which is how the toggle knows to stay hidden rather than appear and lie.
    """
    scale = figure.absolute_scale
    if figure.per_metric and scale is not None:
        scale = {name: figure.absolute_scale_for(result.metric(name))
                 for name in metric_names}
        scale = {name: views for name, views in scale.items() if views} or None
    return {"absoluteScale": scale, "perMetric": figure.per_metric}


def _trial_rows(result, metric_names, hp_names, start: int = 0) -> list:
    """The trials table's rows, from *start* onwards.

    Module-level and sliceable because it is rendered twice: the whole table
    when the page is built, and the tail of it when a run adds trials and the
    poll appends them — from the same partial, so the two cannot drift into
    formatting a duration or a failure differently.
    """
    return [
        {
            # Its position in the result, which is what everything that names a
            # trial uses — the panel endpoint, the local-ablation cache, every
            # plot's selection meta. `n` is the number shown.
            "idx": i,
            "n": t.trial,
            "config_values": [t.config.get(h) for h in hp_names],
            # .get, not [m]: a row stored before io.parse checked for it can be
            # missing a metric, and the table's job is to show what is there
            # rather than to be the thing that 500s the page.
            "scores": [t.scores.get(m) for m in metric_names],
            "duration": t.duration,
            # A trial that produced no measurement. Its scores are all 0.0 and
            # read as a bad trial rather than as no trial, so the row has to say
            # which it is — and the reason, since a table has room for it where
            # a figure does not.
            "failed": t.failed,
            "failure": t.failure,
        }
        for i, t in enumerate(result.trials)
        if i >= start
    ]


def metric_label(current_metric, original_metric):
    """The label shown for an experiment's metric.

    "~" when no metric has been committed by a run yet, "Inconsistent" when
    the current metric no longer matches the one the results were produced
    with, otherwise the metric name itself.
    """
    if original_metric is None:
        return "~"
    if current_metric != original_metric:
        return _("Inconsistent")
    return current_metric


def _selected_panel_data(result, metric, idx):
    """The selected-config panel's data for one (metric, trial index).

    The trial's score for *metric*, and its distance from the metric's best —
    always, including when the selected trial *is* the best one, where the
    distance is zero and `is_best` says why. The panel used to leave the row out
    in that case, which made it change height as you clicked from one trial to
    another; and "no difference" and "a difference of zero" are the same fact
    said two ways, only one of which you can read off the page.
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
        # Whether that difference is in the good direction, decided here rather
        # than by the template comparing it to zero. Against the *best* trial it
        # never is — but which sign means worse depends on the metric, and on
        # one where lower wins the sign is the other way round, so a template
        # reading `delta > 0` as "better" gets it exactly backwards.
        "delta_better": result.metric(metric).better(score, best),
        "is_best": idx == best_idx,
        "config": list(trial.config.items()),
        # A trial that produced no measurement: the score above is a
        # placeholder, and the panel says so rather than presenting a 0.0 as a
        # result. `idx`, not `trial_n`, because that is what the traceback
        # endpoint takes — the same index every figure's selection speaks in.
        "failed": trial.failed,
        "failure": trial.failure,
        "idx": idx,
        "has_traceback": bool(trial.traceback),
    }


def _prior_only_context(exp, built):
    """What the prior figure needs when there is nothing to acquire yet.

    The prior and no more: no acquisition curve, no candidates, no incumbent,
    because all three are the surrogate's and there are no trials to fit one
    to. The figure renders its own controls and asks the server for the
    density, the same `prior_only` request it already makes when the
    acquisition figure is waiting behind its Compute prompt.

    Empty when the figure is switched off or the space cannot be rebuilt, and
    the page then looks exactly as it did before — an empty dict updates
    nothing, so the caller needs no second branch.
    """
    figure = next((f for f in _shown_figures(exp) if f.key == "prior"), None)
    config_space = config_space_for(built)
    if figure is None or config_space is None:
        return {}
    return {
        "prior_only_figure": figure,
        # From the space rather than from a trial's configuration, which is the
        # whole point: there are no trials.
        "hp_names": list(config_space.keys()),
        "beta_min": prior_math.BETA_RATIO_MIN,
        "beta_max": prior_math.BETA_RATIO_MAX,
        **_prior_judging_context(exp),
    }


#: What each tab is called on the page.
TAB_LABELS = {
    "overview": gettext_lazy("Overview"),
    "hypershap": gettext_lazy("HyperSHAP"),
    "acquisition": gettext_lazy("Acquisition"),
    "misc": gettext_lazy("Misc"),
}


def _computed_version(exp, result):
    """The state of the results a computed figure's answer came from.

    Every figure behind a Compute button fits a model to this experiment's
    trials under its settings, so an answer is good for exactly as long as the
    trials, the run they came from and the settings stay as they were. Anything
    else — the reader's prior, the selected trial, the hyperparameter shown — is
    part of the question each figure asks, and is in its own key.
    """
    import hashlib
    last = exp.runs.order_by("-id").values_list("pk", "status").first()
    digest = hashlib.sha1(json.dumps(resolve_settings(exp),
                                     sort_keys=True, default=str).encode()).hexdigest()[:12]
    return {"experiment": exp.pk,
            "version": f"{len(result.trials)}:{last}:{digest}"}


def _figure_tabs(layout, figures):
    """The page's tabs, arranged per *layout* — see ui/layout.py."""
    from ..layout import placed
    return [dict(tab, label=TAB_LABELS[tab["tab"]]) for tab in placed(layout, figures)]


@experiment_view(VIEW)
@require_POST
def save_layout(request, exp):
    """Keep how this reader has arranged this experiment's page.

    `VIEW`, because arranging what you can see is not changing the experiment:
    it writes this reader's own row and nobody else's, and a viewer has as much
    use for a tidy page as its owner. From the page's script, a JSON body:

        {"layout": {...}}    keep this as mine
        {"reset": true}      forget mine, use my default

    answered with the arrangement now in force, normalised, so the page can
    redraw from what was actually kept. And from the settings page's three
    buttons — reset this one, make this my default, reset my default — ordinary
    forms, answered by going back to the experiment.
    """
    from .. import layout as layouts
    # The settings page's three buttons: ordinary forms, each answered by going
    # back to the experiment, where the result can be seen.
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


#: `experiment_header`'s "rebuild it yourself", told apart from a result of None.
_UNBUILT: Any = object()


def _prepared_summary(exp) -> str:
    """A short line about the environment, for the experiment page."""
    meta = exp.env_meta or {}
    parts = []
    deps = meta.get("dependencies") or []
    if deps:
        parts.append(", ".join(deps))
    if meta.get("python"):
        parts.append(f"Python {meta['python']}")
    return " · ".join(parts)


def experiment_header(request, exp, result=_UNBUILT):
    """What the top of every view of an experiment shows — its dashboard, its
    Data Handling and its timeline: what it is, what may be done with it and
    the buttons for that, whose it is, and anything wrong with it.

    *result* is the rebuilt OptimizationResult when the caller has one;
    otherwise it is rebuilt here.
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
    # The trial the export buttons name before anything is selected: the best
    # on the metric being viewed. Whether its fitted model was kept decides
    # whether Export trial parameters starts as a link.
    current = exp.data.current_metric or (exp.data.metric_names or [None])[0]
    best = (result.best_index(current)
            if has_result and result is not None and current in exp.data.metric_names else None)

    return {
        "experiment": exp,  # the _run_status.html include reverses URLs from experiment.pk
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
        # A result with no trials counts as no result: it is what the dashboard
        # stops building panels and plots for, so its template must take its
        # "No results yet" branch rather than render the figure script with
        # nothing for it to read.
        "has_result": has_result,
        # What this viewer may do, for the buttons. Read access got them here;
        # the rest depends on whose experiment it is.
        "may": {action: policy().may(request, exp, action)
                for action in (RUN, EDIT, DELETE, EXPORT, SHARE, RENAME)},
        "ownership": _ownership(request, exp),
        "model_export_unavailable": model_export.unavailable(exp),
        # Which trials' fitted models were kept, by their place in the result —
        # the button for their parameters follows the selected trial, and is
        # only a link for one of these.
        "kept_trial_indices": kept,
        "trial_models_reason": trial_models_reason(exp),
        "export_default_kept": best is not None and best in kept,
        "run_error": last_run.error if (last_run and last_run.status == "error") else None,
        # Why this viewer may not run this experiment's custom model, if they
        # may not. Shown rather than silently disabling the form.
        "model_refusal": policy().custom_model_refusal(exp, request.user),
        # The model's environment, for the notices under the heading.
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
    """Detail-page context: identity, run state, and the figures to draw.

    Rebuilds the OptimizationResult from the stored snapshot (read-only) and,
    when present, builds the panels and per-metric figures for whichever figures
    the settings have switched on; the browser switches metrics client-side.
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
        # Only an optimizer that fits a model of the objective can answer the
        # confidence criterion, so only then is it offered.
        "supports_confidence": getattr(
            optimizer_class(exp.data.optimizer_name), "supports_confidence_stopping", False),
        # And only one that can be asked what it would run next can offer the
        # configuration cube's Requery — see `next_trials`.
        "can_requery": hasattr(optimizer_class(exp.data.optimizer_name), "slice_challengers"),
        **_prior_judging_context(exp),
        "run_default_metric": exp.data.current_metric or (metric_names[0] if metric_names else None),
        # What the deadline fields open on. The deployment's number, so an
        # instance that has tuned `MODEL_TRIAL_TIMEOUT` for its own hardware
        # offers that rather than making everyone retype it.
        "default_trial_timeout": int(settings.MODEL_TRIAL_TIMEOUT),
        "default_trial_factor": deadline.DEFAULT_FACTOR,
        # How many trials this page has drawn, which the run-status poll sends
        # back so the server can tell whether anything has moved since. 0 here
        # and overwritten below: the early return means "no figure grid", and
        # that is exactly what 0 tells `run_status`.
        "trial_count": 0,
        # How long the run-status poll waits between asks. Follows the trials'
        # own durations — see `_poll_seconds`.
        "poll_seconds": _poll_seconds(exp),
    }
    # A result with no trials is the same story as no result at all — there is
    # nothing to plot and no best trial to describe — so it takes the same early
    # return. Not a hypothetical: `io.parse` accepts a snapshot whose result has
    # an empty `data`, and importing one used to 500 the detail page (an argmax
    # over an empty range). A run never stores one (`run.py` only writes a
    # result that has trials), which is why this went unnoticed.
    #
    # One figure goes out through this return rather than being stopped by it.
    # A prior is a statement about the *search space*, which exists before the
    # search does — and before the first run is exactly when somebody has
    # something to say about where the optimum might be. Everything else on
    # that card needs a surrogate and is left out; see `_prior_only_context`.
    if result is None or not result.trials:
        context.update(_prior_only_context(exp, built))
        return context

    # Past the early return, so there is a grid: say how much of the run it is
    # showing. `run_status` compares this against what is stored and only builds
    # fresh plots when the two disagree.
    context["trial_count"] = len(result.trials)
    # And whether the explanations are missing and could be asked for — a
    # cancelled run's are skipped by design, and a partial write leaves them out
    # for the same reason. Not while a run is in flight: it will compute them
    # itself when it ends, and asking now would race its own writes.
    context["analytics_absent"] = _analytics_absent(result) and not active_run

    # Which figures to draw. A figure that is switched off is not rendered and
    # its plot is not built, so nothing is computed or shipped to go unused.
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
            # No selection has been clicked yet, so it defaults to the best trial.
            "selected": _selected_panel_data(result, m, best_idx),
            # One entry per HyperSHAP game: its warning, and its "table"
            # rendering's rows (rendered straight from here rather than from a
            # plot, like best_config above, so it needs no JSON round-trip
            # through the page script). Keyed by game because the page's game
            # selector picks which applies, independently of the metric.
            #
            # The warning covers the interaction figures too. They are extracted
            # from the same HyperSHAP call, not a separate one, so whatever made
            # that call fail explains their emptiness as well — which is why
            # there is no second warning to keep in step with this one.
            "importance_by_game": {
                game: {
                    "warning": getattr(result, fields["warning"]).get(m),
                    # A row per hyperparameter even with nothing stored: the
                    # table also has the incumbent's and the selected trial's
                    # columns, which the page fills whether or not there is a
                    # default one.
                    "table": (sorted(getattr(result, fields["importance"]).get(m, {}).items(),
                                     key=lambda kv: kv[1], reverse=True)
                              or [(hp, None) for hp in hp_names]),
                }
                for game, fields in HP_GAME_FIELDS.items()
            },
            # The partial-dependence figure's hyperparameter picker defaults
            # to this metric's own top-tunability hyperparameter (falling
            # back to the first one in config order if importance isn't
            # available) — the same "most interesting thing first" instinct
            # `best_idx` already applies to which trial ablation explains by
            # default.
            "top_hp": (
                sorted(result.hyperparameter_importance.get(m, {}).items(),
                       key=lambda kv: kv[1], reverse=True)[0][0]
                if result.hyperparameter_importance.get(m)
                else (list(best.config.keys())[0] if best.config else "")
            ),
        })

    # Plots, keyed by figure, built straight off the catalog. Only the metric
    # the page opens on: building all of them was 72 `plot()` calls and 101ms
    # of a 151ms render on a 4-metric run, for six payloads that end up on
    # screen — the rest sat in the page as ~300KB of JSON (over a megabyte at
    # 500 trials) against the chance the metric selector was touched. The
    # browser fetches a metric the first time it is selected and keeps it, the
    # same lazy-fetch-and-cache shape partial dependence and local ablation
    # already use; see `metric_figures` and experiment_detail.html's `show`.
    # Metric-independent figures are still drawn once, here.
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

    # The Run form's target starts at what there is to beat. Per metric, so the
    # page can follow the metric dropdown; the one for the metric the form opens
    # on is what the field is filled with.
    incumbent_targets = {p["metric"]: surpass_target(p["best_score"]) for p in panels}
    context.update(
        result=result,
        panels=panels,
        metric_plots=metric_plots,
        static_plots=static_plots,
        # Each figure's declared display behavior, for the page script.
        figure_options={f.key: _figure_options(f, result, metric_names)
                        for f in figures},
        # Which figures have alternate views, and what they're called — only
        # for those, so the script can tell a plain payload from one keyed by
        # view without guessing from its shape.
        figure_views={f.key: list(f.views) for f in figures if f.views},
        # The tabs, each an ordered list of (figure, columns) — see
        # ui/layout.py — and the sidebar's, which are in none of them
        # (Figure.in_sidebar).
        figures=figures,
        figure_tabs=_figure_tabs(figure_layout, figures),
        # What the computed figures' answers are labelled with in the browser
        # — see ui/static/ui/computed.js.
        computed_version=_computed_version(exp, result),
        # The whole arrangement, figures switched off included, for layout.js
        # to edit and send back.
        figure_layout=figure_layout,
        sidebar_figures=[f for f in figures if f.in_sidebar],
        # Which of them take a click naming a trial and show which one is
        # selected (Figure.selects_trials), so the page's selection bus iterates
        # a declaration instead of naming figures.
        selectable_figures=[f.key for f in figures if f.selects_trials],
        # And which of them a running run can redraw from its trials alone
        # (Figure.live) — the same declaration `run_status` builds payloads for,
        # so the page and the poll cannot disagree about which figures move.
        live_figures=[f.key for f in figures if f.live],
        # And which of them draw the HyperSHAP games (Figure.explains_games),
        # the ones the sidebar's baseline measures from somewhere other than
        # the stored default.
        game_figures=[f.key for f in figures if f.explains_games],
        # The two colours the highlight is drawn in, from the builders that draw
        # everything else in them — the script used to restate them, which is
        # one place for the page and the plots to disagree about what "selected"
        # looks like.
        # `failure` for the one place the page has to draw a failure mark
        # itself: a 3D scene draws no marker outline, so the red cannot ride on
        # the glyph the way it does everywhere else. Sent rather than repeated
        # in the script, so the page restates no colour — see NEGATIVE_COLOR.
        selection_colors={"base": MARKER_COLOR, "selected": SELECTION_COLOR,
                          "comparison": ACCENT_COLOR,
                          "failure": NEGATIVE_COLOR, "proposal": PROPOSAL_COLOR},
        # The one game selector, and what each game asks — resolved here rather
        # than in the template because the labels are lazy translations and
        # `json_script` cannot serialize those.
        explanation_games=[(game, str(label)) for game, label in HP_GAME_LABELS.items()],
        explanation_game_help={game: str(text) for game, text in HP_GAME_HELP.items()},
        hp_names=hp_names,
        # Bounds for the acquisition figure's β field. Constants rather than
        # anything this experiment decided, so they are declarations.
        beta_min=prior_math.BETA_RATIO_MIN,
        beta_max=prior_math.BETA_RATIO_MAX,
        # Whether the importance figure offers "still to gain" at all. The
        # server already returns nothing for it when off, which would leave the
        # box and the table's three columns there offering an answer that never
        # arrives, so the markup goes too.
        tuning_progress=analytics.TUNING_PROGRESS_ENABLED,
        incumbent_targets=incumbent_targets,
        incumbent_target=incumbent_targets.get(context["run_default_metric"], ""),
        trial_rows=_trial_rows(result, metric_names, hp_names),
        # The table's own page size, so its field and the script that reads it
        # start from the figure's number rather than each restating one.
        trials_page_size=Trials.page_size,
        trials_exhausted=(
            result.trials_limit is not None
            and len(result.trials) >= result.trials_limit
        ),
        # Which on-request analytics may fetch themselves as soon as their figure
        # needs them, and which wait for a button. Keyed by the computation's name
        # (see Figure.deferred), for the page script.
        autocompute={name: shown[autocompute_key(name)]
                     for name, _label in deferred_computations()},
    )
    return context
