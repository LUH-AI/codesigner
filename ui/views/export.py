from pathlib import Path

from django.http import Http404, HttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.utils import timezone
from django.utils.http import content_disposition_header  # pyright: ignore[reportAttributeAccessIssue]

from access.policy import EXPORT, VIEW
from experiments.services import model_export, timestamps
from experiments.services import snapshot as snapshot_adapter
from experiments.services.snapshot import rebuild_result

from .. import times
from ..permissions import experiment_view
from .dashboard import experiment_header, trial_models_reason


#: The value a ticked box posts, for both questions.
KEEP = "keep"


@experiment_view(EXPORT)
def experiment_export(request, exp):
    """`experiment_export` asks per file which form trial times take and whether
    tracebacks stay, since both describe a person or machine and the answer
    depends on the recipient. `snapshot_adapter.export` blanks server paths and
    applies the answers.
    """
    return _export(request, exp)


@experiment_view(VIEW)
def experiment_timeline(request, exp):
    """`experiment_timeline` lists changes to how *exp* searches, newest first."""
    from .. import timeline

    entries = timeline.build(exp)
    return render(request, "ui/experiment_timeline.html", {
        **experiment_header(request, exp),
        "scheduled": [e for e in entries if e["scheduled"]],
        "entries": [e for e in entries if not e["scheduled"]],
        "time_basis": exp.data.time_basis})


@experiment_view(EXPORT)
def experiment_export_model(request, exp):
    """`experiment_export_model` sends the model as a runnable script tuned to
    the trial `_picked_trial` names.
    """
    reason = model_export.unavailable(exp)
    if reason:
        raise Http404(reason)
    picked = _picked_trial(request, exp)
    if isinstance(picked, HttpResponse):
        return picked
    result, trial, metric = picked
    body = model_export.build(exp, trial, metric, result)
    response = HttpResponse(body, content_type="text/x-python; charset=utf-8")
    response["Content-Disposition"] = (
        f'attachment; filename="{model_export.filename(exp, trial.trial)}"')
    return response


def _picked_trial(request, exp):
    """`_picked_trial` resolves `?metric=&idx=` to (result, trial, metric) and
    falls back to the best trial on the viewed metric. A bad query gets a 400
    response."""
    result = rebuild_result(exp)
    if result is None or not result.trials:
        raise Http404("no trials to export")
    metric = request.GET.get("metric") or exp.data.current_metric or exp.data.metric_names[0]
    if metric not in exp.data.metric_names:
        return HttpResponseBadRequest("unknown metric")
    idx_raw = request.GET.get("idx")
    if idx_raw is None:
        idx = result.best_index(metric)
    elif idx_raw.lstrip("-").isdigit() and 0 <= int(idx_raw) < len(result.trials):
        idx = int(idx_raw)
    else:
        return HttpResponseBadRequest("invalid trial index")
    return result, result.trials[idx], metric


@experiment_view(EXPORT)
def experiment_export_trial_model(request, exp):
    """`experiment_export_trial_model` sends a zip of the parameters that the
    run kept for the selected trial after one fold's training.
    """
    import io as bytes_io
    import zipfile

    from experiments.models import TrialModel

    picked = _picked_trial(request, exp)
    if isinstance(picked, HttpResponse):
        return picked
    _result, trial, _metric = picked
    kept = TrialModel.objects.filter(experiment=exp, trial=trial.trial).first()
    folder = Path(kept.directory) if kept else None
    if folder is None or not folder.is_dir():
        raise Http404(trial_models_reason(exp))
    assert kept is not None

    buffer = bytes_io.BytesIO()
    stem = model_export.filename(exp, trial.trial).removesuffix(".py") + "-parameters"
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                archive.write(path, f"{stem}/{path.name}")
    TrialModel.objects.filter(pk=kept.pk).update(last_used_at=timezone.now())
    response = HttpResponse(buffer.getvalue(), content_type="application/zip")
    response["Content-Disposition"] = f'attachment; filename="{stem}.zip"'
    return response


def _export(request, exp, cancel_url=None):
    if request.method != "POST":
        # Counting from the stored result skips rebuilding an
        # `OptimizationResult`.
        entries = ((exp.data.result or {}).get("data") or [])
        count = sum(1 for e in entries
                    if (e.get("additional_info") or {}).get("traceback"))
        return render(request, "ui/export_confirm.html",
                      {"experiment": exp, "has_tracebacks": bool(count),
                       "traceback_count": count, "cancel_url": cancel_url,
                       "timestamp_modes": _timestamp_choices(exp),
                       "timestamp_mode": times.for_export(exp, request.user)})

    # "keep" is a legacy answer; a missing answer means none.
    posted = request.POST.get("timestamps")
    mode = (timestamps.ABSOLUTE if posted == KEEP
            else posted if posted in timestamps.MODES else timestamps.NONE)
    body = snapshot_adapter.export(exp, mode, keep_tracebacks=request.POST.get("tracebacks") == KEEP)
    response = HttpResponse(body, content_type="application/octet-stream")
    # The identifier fallback stops unnamed files becoming dotfiles.
    response["Content-Disposition"] = content_disposition_header(
        True, f"{exp.name or exp.identifier}.ihpo")
    return response


def _timestamp_choices(exp):
    """`_timestamp_choices` labels the time forms allowed for *exp*."""
    return [(mode, *times.LABELS[mode]) for mode in timestamps.allowed(exp)]
