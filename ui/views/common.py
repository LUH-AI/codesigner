import json
import logging
import statistics

from django.conf import settings
from django.contrib import messages
from django.urls import reverse
from django.utils.translation import ngettext
from django.utils.translation import gettext as _

from access.policy import groups_to_choose_from
from core import datasets
from core.registry import MODELS, optimizer_class
from experiments.models import Experiment
from experiments.services import storage
from experiments.services.settings import SETTING_BOUNDS

from ..experiment_settings import resolve_settings
from ..figures import FIGURES
from ..formatting import human


_ACTIVE = ["pending", "running"]


#: Reaches the console handler configured in settings.LOGGING. Used only where
#: something went wrong that the reader cannot be told about, because the thing
#: they asked for succeeded anyway.
logger = logging.getLogger(__name__)


def _model_available(exp):
    """Whether *exp*'s model can be built to run it.

    Registry models are always available. A custom model needs its .py present,
    the feature enabled, and either a prepared environment or permission to be
    imported into this process. Reads `env_status` — a column — so rendering a
    page never shells out.
    """
    if exp.data.model_name in MODELS:
        return True
    if not (exp.data.model_file and settings.ALLOW_CUSTOM_MODELS):
        return False
    if exp.env_status == Experiment.ENV_READY:
        return True
    return not exp.env_pending and not settings.REQUIRE_LOGIN


def _group_by_pk(groups, pk):
    return next((g for g in groups if g.pk == pk), None)


def _chosen_group(request, context):
    """The group an import goes in, or an error, for the pages with no Form.

    `(group, None)` when it was chosen, or there was nothing to choose — then
    `Experiment.save` files it in the one group there is. `(None, message)`
    when a choice was needed and none of this person's groups was given.
    """
    groups = groups_to_choose_from(request.user)
    context["groups"] = groups
    if not groups:
        return None, None
    try:
        group = _group_by_pk(groups, int(request.POST.get("group", "")))
    except ValueError:
        group = None
    if group is None:
        return None, _("Choose which group this experiment is in.")
    return group, None


def _ownership(request, exp):
    """Who this experiment belongs to, or None when the instance has no
    accounts and there is nothing to say."""
    if not settings.REQUIRE_LOGIN:
        return None
    from access.policy import membership_in, share_of
    # Where sharing is done, for somebody who can open it — a member of the
    # experiment's group. Anybody else would be linked to a 404.
    in_group = membership_in(request.user, exp.group_id) is not None
    return {"owner": exp.owner, "mine": exp.owner_id == request.user.pk,
            "shared_with": exp.shares.count(),
            "level": share_of(request.user, exp),
            "group": exp.group if in_group else None,
            "group_url": (reverse("ui:group_detail", args=[exp.group_id])
                          + f"#exp-{exp.pk}") if in_group else None}


def surpass_target(score: float, places: int = 4) -> str:
    """The incumbent's score, as a target for the next run to beat.

    Rounded for the field it goes in, but never *below* the score itself: the
    criterion is a strict improvement, so a target a hair under what is already
    in hand would be met before the run started and end it after one trial.
    """
    shown = round(score, places)
    if shown < score:
        shown = round(shown + 10 ** -places, places)
    return f"{shown:g}"


def _posted_number(request, key: str, default: float, *, low: float) -> float:
    raw = (request.POST.get(key) or "").strip()
    try:
        return max(low, float(raw.replace(",", ".")))
    except ValueError:
        return float(default)


def _judges_priors(exp):
    """Whether the experiment's optimizer, as configured now, can judge a
    prior — which decides whether "Evaluate Prior" is offered."""
    optimizer = optimizer_class(exp.data.optimizer_name)
    return bool(optimizer) and optimizer.judges_priors(exp.data.optimizer_params or {})


def _prior_judging_context(exp):
    """What the prior figure's "Evaluate Prior" controls open on."""
    return {"judges_priors": _judges_priors(exp),
            "prior_top_k": resolve_settings(exp)["prior_acceptance_top_k"],
            "prior_top_k_bounds": SETTING_BOUNDS["prior_acceptance_top_k"]}


def _best_panels(result, metric_names):
    """Each metric's best trial so far, as the best-configuration cards show
    it."""
    panels = []
    for m in metric_names:
        best_idx = result.best_index(m)
        # No trial carries a score for this metric, so there is no best trial to
        # describe and no selection to open on — `best_index` documents None as
        # the answer a caller handles. The panel is left out rather than built
        # around a trial that isn't there, which is the same decline the
        # per-metric figures make through `has_every_score`. The page script
        # keys its maps by metric and the Run form's incumbent target is built
        # from these panels, so both simply have nothing for this one.
        if best_idx is None:
            continue
        best = result.trials[best_idx]
        panels.append({
            "metric": m,
            "best_n": best.trial,
            # The array index (as opposed to best_n, the trial's own number)
            # of the metric's best trial — what the local-explanation fetch
            # asks for by default, before any click. The same index
            # click-to-select already works in (Plotly's pointIndex).
            "best_idx": best_idx,
            "best_score": best.scores[m],
            "best_config": list(best.config.items()),
        })
    return panels


def _settle_storage(request, exp) -> bool:
    """Make room in *exp*'s group for what creating it stored, or undo it.

    Datasets and model files come first: kept trial models are deleted, the
    stalest first, until the group is within its limit, and the person is told
    how many. Only when the datasets and model files alone are over is the
    experiment refused — deleted again, files and all. See
    experiments/services/storage.py.
    """
    group = exp.group
    if group is None or (not exp.data.dataset and not exp.data.model_file):
        return True
    within, deleted = storage.make_room(group)
    now = storage.usage(group)
    numbers = {"group": group.name, "used": human(now["total"]),
               "limit": human(now["limit"])}
    if not within:
        storage.delete_experiment_files(exp)
        messages.error(request, _(
            "%(group)s has no room for this: its datasets and model files alone would take "
            "%(used)s of its %(limit)s. Delete experiments it no longer needs, or ask a site "
            "administrator for more.") % numbers)
        return False
    if deleted:
        messages.warning(request, ngettext(
            "To make room in %(group)s, %(n)s kept trial model was deleted, the stalest "
            "first. It now uses %(used)s of %(limit)s.",
            "To make room in %(group)s, %(n)s kept trial models were deleted, the stalest "
            "first. It now uses %(used)s of %(limit)s.", deleted) % dict(numbers, n=deleted))
    return True


def _plot_json(plot):
    return json.loads(plot.to_json()) if plot is not None else None


def _target_name(exp):
    """The dataset's target column, for the forecast's axis."""
    try:
        return datasets.target_of(exp.data.dataset_path())[0]
    except Exception:  # noqa: BLE001 — an axis title is not worth an error
        return ""


def _baseline_index(request, result):
    """The `baseline` query parameter as a trial index, or None when absent.

    The trial every HyperSHAP explanation on the page is measured from, when it
    is not the config space's default. Raises ValueError for a value naming no
    trial, which the caller answers with a 400 like any other bad index.
    """
    raw = request.GET.get("baseline", "")
    if raw == "":
        return None
    if not raw.isdigit() or int(raw) >= len(result.trials):
        raise ValueError(raw)
    return int(raw)


#: How often the run-status fragment is polled, in seconds. The floor is what a
#: poll can usefully be — the run writes at most every
#: `PARTIAL_RESULT_SECONDS`, so anything faster is a request that can only
#: answer "nothing yet". Above it the interval follows the trials themselves:
#: polling ten times per trial is ten times the traffic for one new row, and on
#: a real dataset a trial is minutes rather than milliseconds. The ceiling keeps
#: a long-trial run from feeling stopped.
POLL_MIN_SECONDS = 5


POLL_MAX_SECONDS = 30


def _poll_seconds(exp) -> int:
    """How long the page should wait before asking again.

    Read off the trials already stored — their own recorded durations — so it
    adapts as a run goes and needs nothing measured on the page. The median of
    the last few rather than the mean of all of them: a run's first trials are
    not like its later ones, and one pathological trial should not slow the poll
    for the rest of the run.

    Rendered into the fragment's own `hx-trigger`, which poll.js re-reads off
    each replacement — so the interval adapts per swap with no state anywhere
    and no second mechanism.
    """
    data = (exp.data.result or {}).get("data") or []
    recent = [entry.get("time") or 0.0 for entry in data[-10:]]
    typical = statistics.median(recent) if recent else 0.0
    return int(min(POLL_MAX_SECONDS, max(POLL_MIN_SECONDS, round(typical))))


def _shown_figures(exp):
    """The figures *exp*'s settings switch on, in catalog order.

    Shared by the detail page and the `metric_figures` endpoint so a figure
    switched off is absent from both — the endpoint must not become a way to
    fetch what the page deliberately did not build.
    """
    shown = resolve_settings(exp)
    return [figure for figure in FIGURES if shown[figure.setting_key]
            and figure.applies(exp.data)]


def _hp_names(result) -> list:
    """The hyperparameters a result's trials carry, in config order."""
    return list(result.trials[0].config.keys()) if result.trials else []


def _figure_plots(result, figures, metric=None, config_space=None):
    """`{figure key: plot JSON}` for *figures* at *metric*.

    A figure with declared views gets a dict of view key -> plot JSON (or None)
    instead of a single plot, one entry per view — the browser picks which to
    show; see experiment_detail.html's `payloadFor`.

    A figure whose `plot()` returns None gets None, which `draw()` reads as "no
    data yet" — that is how the three fetched figures (partial dependence, local
    explanation, local effects) end up here without being computed.

    Module-level, and taking the metric rather than closing over one, because
    `metric_figures` answers with exactly this for a metric the page did not
    ship — see `_detail_context` for why it only ships one.
    """
    def fig_json(plot):
        return json.loads(plot.to_json()) if plot is not None else None

    def one(figure):
        extra = {"config_space": config_space} if figure.needs_config_space else {}
        if figure.views:
            return {view: fig_json(figure.plot(result, metric, view=view, **extra))
                    for view in figure.views}
        return fig_json(figure.plot(result, metric, **extra))

    return {figure.key: one(figure) for figure in figures}
