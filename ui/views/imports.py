import json

from django.core.exceptions import ValidationError
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from access.policy import groups_to_choose_from, policy
from core import io, smac_import
from experiments.models import account
from experiments.services import history, modelenv
from experiments.services import snapshot as snapshot_adapter

from ..forms import ImportForm
from ..validators import validate_size
from .common import _chosen_group, _settle_storage


def import_experiment(request):
    """Upload an .ihpo file to create a saved experiment.

    A dataset and — when custom models are enabled — a model .py may be
    attached to make the imported experiment runnable; without them it loads
    read-only.
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
        history.record(exp, "imported", user=request.user, source="ihpo",
                       filename=upload.name, dataset_attached=exp.data.has_dataset)
        # An imported model is re-locked here rather than trusting pins chosen by
        # whoever exported it.
        modelenv.start_preparation(exp)
        return redirect("ui:experiment_detail", pk=exp.pk)

    return render(request, "ui/import.html", context)


def _import_smac_directory(request, context):
    """Create a read-only experiment from an uploaded SMAC output directory.

    Its own branch rather than a third optional file on the `.ihpo` form: this
    reads a *set* of files with no `.ihpo` among them, and neither a dataset nor
    a model may be attached to it. A SMAC scenario records nothing about what
    was being optimized, so there is nothing here to check an attached dataset
    against — see `core/smac_import.py`, and `import_experiment` on why a
    dataset that cannot be checked is not accepted.
    """
    files = {}
    for upload in request.FILES.getlist("smac_dir"):
        try:
            validate_size(upload)
        except ValidationError as error:
            context["error"] = error.messages[0]
            return render(request, "ui/import.html", context)
        # Django's uploader keeps only the basename, so a directory picker's
        # relative paths do not survive the post — which is why the importer
        # matches on basenames, and why two run directories at once cannot be
        # told apart. Refused rather than blended: SMAC writes one run per
        # `<name>/<seed>` directory, and a runhistory read against another
        # run's config space is not a run that happened.
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
    history.record(exp, "imported", user=request.user, source="smac",
                   filename=snapshot.get("name") or "", dataset_attached=False)
    return redirect("ui:experiment_detail", pk=exp.pk)
