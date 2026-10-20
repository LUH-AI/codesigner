from pathlib import Path

from django.http import Http404, HttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.utils import timezone
from django.utils.http import content_disposition_header

from access.policy import EXPORT, VIEW
from experiments.services import model_export, timestamps
from experiments.services import snapshot as snapshot_adapter
from experiments.services.snapshot import rebuild_result

from .. import times
from ..permissions import experiment_view
from .dashboard import experiment_header, trial_models_reason


#: What a ticked box on the export page posts. One value for both of its
#: questions: they are asked the same way and answered the same way, and a
#: second constant would only be a second place to change it.
KEEP = "keep"


@experiment_view(EXPORT)
def experiment_export(request, exp):
    """Download a saved experiment as an .ihpo file, having asked what goes in it.

    Server paths are always blanked. Two things are worth a question, because
    both are facts about a person or a machine rather than about a search, and
    the file is complete as a record of the search without either:

    - per-trial `starttime`/`endtime`, which say what time of day someone was
      working and on which days. The durations that make the timing figures
      readable survive without them.
    - a failed trial's stored traceback, which names absolute paths on this
      machine, the packages installed on it and their versions. The failure
      itself survives without it, with its one-line reason.

    Asked at the point of export rather than kept as a setting. A setting is
    answered once, by whoever set it up, for every file the instance ever sends
    — and the answer depends on the file and on who is about to receive it. It
    also has to be found and understood before the question has been raised,
    which is the wrong order for a question about what you are handing over.

    Scrubbed here, not in the adapter, so the detail page's own reconstruction
    and the run engine are unaffected; deserialize ignores the keys on
    re-import.
    """
    return _export(request, exp)


@experiment_view(VIEW)
def experiment_timeline(request, exp):
    """Everything that changed how this experiment searches, newest first —
    see ui/timeline.py."""
    from .. import timeline

    entries = timeline.build(exp)
    return render(request, "ui/experiment_timeline.html", {
        **experiment_header(request, exp),
        "scheduled": [e for e in entries if e["scheduled"]],
        "entries": [e for e in entries if not e["scheduled"]],
        "time_basis": exp.data.time_basis})


@experiment_view(EXPORT)
def experiment_export_model(request, exp):
    """Download the experiment's model as a runnable script, tuned to one trial.

    The trial is the one the page has selected, named the way every figure
    names it — `?metric=&idx=`, the index into the result's trials. Without
    them it is the best trial on the metric being viewed. See
    core/model_export.py for what the file holds.
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
    """The trial a download names — `?metric=&idx=`, the index into the
    result's trials, as every figure names one; without them the best on the
    metric being viewed — as (result, trial, metric), or the response refusing
    the request."""
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
    """Download the selected trial's fitted model — its parameters, in its
    library's own format, with a note of what it is — as a zip.

    Only for a trial whose model the run kept (see experiments/services/trial_models.py):
    nothing is trained here. What comes back is partially trained, fitted on
    one fold's training rows, and its note says so.
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
        # How many tracebacks there are to ask about, and so whether to ask.
        # Counted off the stored result rather than rebuilt through the
        # optimizer: this needs one key per trial, not an OptimizationResult.
        entries = ((exp.data.result or {}).get("data") or [])
        count = sum(1 for e in entries
                    if (e.get("additional_info") or {}).get("traceback"))
        return render(request, "ui/export_confirm.html",
                      {"experiment": exp, "has_tracebacks": bool(count),
                       "traceback_count": count, "cancel_url": cancel_url,
                       "timestamp_modes": _timestamp_choices(exp),
                       "timestamp_mode": times.for_export(exp, request.user)})

    # "keep" is what the question used to post, when the choice was all or
    # nothing; no answer at all is still nothing.
    posted = request.POST.get("timestamps")
    mode = (timestamps.ABSOLUTE if posted == KEEP
            else posted if posted in timestamps.MODES else timestamps.NONE)
    body = snapshot_adapter.export(exp, mode, keep_tracebacks=request.POST.get("tracebacks") == KEEP)
    response = HttpResponse(body, content_type="application/octet-stream")
    # Its name when it has one, else its identifier: an unnamed experiment's
    # file is not a dotfile.
    response["Content-Disposition"] = content_disposition_header(
        True, f"{exp.name or exp.identifier}.ihpo")
    return response


def _timestamp_choices(exp):
    """The forms *exp*'s times can leave in, each with what it means."""
    return [(mode, *times.LABELS[mode]) for mode in timestamps.allowed(exp)]
