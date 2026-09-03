"""Upload guards shared by the create form and the import view.

`NewExperimentForm`'s `FileField`s hang the `validate_*` functions here as
validators. `import_experiment` reads `request.FILES` directly, with no Form
to hang a validator off, so it calls the plain `*_error` functions below
instead — the same checks, called the way a view calls things rather than the
way a form does, so neither path can grow a rule the other quietly lacks.
"""

from pathlib import Path

from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _


def oversized(upload) -> str:
    """"" if *upload* is within `MAX_UPLOAD_BYTES`, else why it was refused."""
    if upload.size > settings.MAX_UPLOAD_BYTES:
        mb = settings.MAX_UPLOAD_BYTES / (1024 * 1024)
        return _("%(name)s is larger than the %(mb)s MB upload limit.") % {
            "name": upload.name, "mb": f"{mb:g}"}
    return ""


def wrong_extension(upload, extension: str) -> str:
    """"" if *upload*'s name ends in *extension* (e.g. "csv"), else why not."""
    if Path(upload.name).suffix.lower() != f".{extension}":
        return _("%(name)s must be a .%(ext)s file.") % {
            "name": upload.name, "ext": extension}
    return ""


def dataset_upload_error(upload) -> str:
    """The checks a dataset upload must pass, on both the create form and
    the import view."""
    return wrong_extension(upload, "csv") or oversized(upload)


def model_upload_error(upload) -> str:
    """The checks a model-file upload must pass, on both the create form and
    the import view."""
    return wrong_extension(upload, "py") or oversized(upload)


def validate_dataset_upload(upload) -> None:
    """Form-field validator wrapping `dataset_upload_error`."""
    message = dataset_upload_error(upload)
    if message:
        raise ValidationError(message)


def validate_model_upload(upload) -> None:
    """Form-field validator wrapping `model_upload_error`."""
    message = model_upload_error(upload)
    if message:
        raise ValidationError(message)
