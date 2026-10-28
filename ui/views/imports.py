import json

from django.core.exceptions import ValidationError
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from access.policy import groups_to_choose_from, policy
from core import io, smac_import
from experiments.models import account
from experiments.services import snapshot as snapshot_adapter

from ..forms import ImportForm
from ..validators import validate_size
from .common import _chosen_group, _settle_storage


def import_experiment(request):
    """`import_experiment` creates an experiment from an .ihpo upload; an
    attached dataset and model make it runnable.
    """
    may_upload = policy().may_upload_models(request)
    context = {"allow_custom_models": may_upload,
               "groups": groups_to_choose_from(request.user)}
    if request.method == "POST" and request.FILES.getlist("smac_dir"):
        return _import_smac_directory(request, context)
    if request.method == "POST" and request.FILES.get("file") is not None:
        form = ImportForm(request.POST, request.FILES, allow_model=may_upload)
        if not form.is_valid():
            context["error"] = " ".join(str(message) for messages in form.errors.values()
                                        for message in messages)
            return render(request, "ui/import.html", context)
        group, error = _chosen_group(request, context)
        if error:
            context["error"] = error
            return render(request, "ui/import.html", context)

        upload = form.cleaned_data["file"]
        dataset_upload = form.cleaned_data.get("dataset")
        model_upload = form.cleaned_data.get("model")
        snapshot = form.snapshot
        exp = snapshot_adapter.experiment_from_snapshot(
            snapshot, dataset_file=dataset_upload, model_file=model_upload,
            owner=account(request.user), group=group,
        )
        if not _settle_storage(request, exp):
            return render(request, "ui/import.html", context)
        snapshot_adapter.finish_import(exp, source="ihpo", filename=upload.name,
                                       user=request.user)
        return redirect("ui:experiment_detail", pk=exp.pk)

    return render(request, "ui/import.html", context)


def _import_smac_directory(request, context):
    """`_import_smac_directory` creates a read-only experiment from a SMAC run
    directory, since a SMAC scenario records nothing to check a dataset against.
    """
    files = {}
    for upload in request.FILES.getlist("smac_dir"):
        try:
            validate_size(upload)
        except ValidationError as error:
            context["error"] = error.messages[0]
            return render(request, "ui/import.html", context)
        # Django keeps only basenames, so two run directories collide and the
        # importer refuses them.
        name = upload.name
        if not name.endswith(".json"):
            continue
        if name in files:
            context["error"] = _(
                "This looks like more than one run — two files here are called "
                "%(name)s. Choose a single run directory, the one holding its "
                "runhistory.json.") % {"name": name}
            return render(request, "ui/import.html", context)
        try:
            files[name] = json.loads(upload.read().decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            context["error"] = _("Could not read %(name)s.") % {"name": name} + f" ({exc})"
            return render(request, "ui/import.html", context)

    try:
        snapshot = io.parse(json.dumps(
            smac_import.snapshot_from_smac(files)).encode("utf-8"))
    except ValueError as exc:
        context["error"] = _("This is not a readable SMAC run.") + f" ({exc})"
        return render(request, "ui/import.html", context)

    group, error = _chosen_group(request, context)
    if error:
        context["error"] = error
        return render(request, "ui/import.html", context)
    exp = snapshot_adapter.experiment_from_snapshot(snapshot, owner=account(request.user),
                                                    group=group)
    snapshot_adapter.finish_import(exp, source="smac", filename=snapshot.get("name") or "",
                                   user=request.user)
    return redirect("ui:experiment_detail", pk=exp.pk)
