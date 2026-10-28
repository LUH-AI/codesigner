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


logger = logging.getLogger(__name__)


def _model_available(exp):
    """Registry models always run. A custom model needs its .py and
    `ALLOW_CUSTOM_MODELS`, plus a ready environment or, with `REQUIRE_LOGIN` off
    and no environment pending, in-process import.
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
    """`_chosen_group` returns `(None, None)` when there is no group to choose,
    leaving `Experiment.save` to pick one. A needed but missing choice gives
    `(None, message)`.
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
    """`_ownership` returns ownership details, or None without accounts."""
    if not settings.REQUIRE_LOGIN:
        return None
    from access.policy import membership_in, share_of
    # Only group members get the group link, since others would hit a 404.
    in_group = membership_in(request.user, exp.group_id) is not None
    return {"owner": exp.owner, "mine": exp.owner_id == request.user.pk,
            "shared_with": exp.shares.count(),
            "level": share_of(request.user, exp),
            "group": exp.group if in_group else None,
            "group_url": (reverse("ui:group_detail", args=[exp.group_id])
                          + f"#exp-{exp.pk}") if in_group else None}


def surpass_target(score: float, places: int = 4) -> str:
    """`surpass_target` rounds the incumbent's score up, since a target under
    the score would meet the strict-improvement criterion at once.
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
    optimizer = optimizer_class(exp.data.optimizer_name)
    return bool(optimizer) and optimizer.judges_priors(exp.data.optimizer_params or {})


def _prior_judging_context(exp):
    return {"judges_priors": _judges_priors(exp),
            "prior_top_k": resolve_settings(exp)["prior_acceptance_top_k"],
            "prior_top_k_bounds": SETTING_BOUNDS["prior_acceptance_top_k"]}


def _best_panels(result, metric_names):
    panels = []
    for m in metric_names:
        best_idx = result.best_index(m)
        # `best_index` returns None when no trial scores a metric, and such a
        # metric gets no panel, as with `has_every_score` in the per-metric
        # figures.
        if best_idx is None:
            continue
        best = result.trials[best_idx]
        panels.append({
            "metric": m,
            "best_n": best.trial,
            # `best_idx` is the array index, unlike `best_n`, and matches
            # Plotly's pointIndex for the local-explanation fetch.
            "best_idx": best_idx,
            "best_score": best.scores[m],
            "best_config": list(best.config.items()),
        })
    return panels


def _settle_storage(request, exp) -> bool:
    """`_settle_storage` deletes kept trial models, stalest first, until *exp*'s
    group fits its limit. The experiment is refused and its files deleted only
    when datasets and model files alone exceed the limit.
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
    try:
        return datasets.target_of(exp.data.dataset_path())[0]
    except Exception:  # noqa: BLE001 — an axis title is not worth an error
        return ""


def _baseline_index(request, result):
    """`_baseline_index` returns None when `baseline` is absent and raises
    ValueError for an unknown trial, which callers answer with a 400.
    """
    raw = request.GET.get("baseline", "")
    if raw == "":
        return None
    if not raw.isdigit() or int(raw) >= len(result.trials):
        raise ValueError(raw)
    return int(raw)


#: The floor matches the run's write interval, `PARTIAL_RESULT_SECONDS`, and
#: `POLL_MAX_SECONDS` keeps a long-trial run from looking stopped. Between them
#: the interval follows trial durations.
POLL_MIN_SECONDS = 5


POLL_MAX_SECONDS = 30


def _poll_seconds(exp) -> int:
    """`_poll_seconds` uses the median duration of the last ten stored trials,
    so a single outlier trial cannot slow the poll. poll.js re-reads the
    interval from the fragment's `hx-trigger` on each swap.
    """
    data = (exp.data.result or {}).get("data") or []
    recent = [entry.get("time") or 0.0 for entry in data[-10:]]
    typical = statistics.median(recent) if recent else 0.0
    return int(min(POLL_MAX_SECONDS, max(POLL_MIN_SECONDS, round(typical))))


def _shown_figures(exp):
    """`_shown_figures` serves the detail page and `metric_figures`, so the
    endpoint never returns a figure the settings switch off.
    """
    shown = resolve_settings(exp)
    return [figure for figure in FIGURES if shown[figure.setting_key]
            and figure.applies(exp.data)]


def _hp_names(result) -> list:
    return list(result.trials[0].config.keys()) if result.trials else []


def _figure_plots(result, figures, metric=None, config_space=None):
    """Each view of a figure gets its own plot JSON, and None tells `draw()`
    there is no data yet. `metric_figures` reuses `_figure_plots` for a metric
    missing from the page.
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
