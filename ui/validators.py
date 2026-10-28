"""`validators` guards uploads."""

from pathlib import Path

from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _


def validate_size(upload) -> None:
    """`validate_size` enforces `MAX_UPLOAD_BYTES`."""
    if upload.size > settings.MAX_UPLOAD_BYTES:
        mb = settings.MAX_UPLOAD_BYTES / (1024 * 1024)
        raise ValidationError(_("%(name)s is larger than the %(mb)s MB upload limit.") % {
            "name": upload.name, "mb": f"{mb:g}"})


def _validate_extension(upload, extension: str) -> None:
    if Path(upload.name).suffix.lower() != f".{extension}":
        raise ValidationError(_("%(name)s must be a .%(ext)s file.") % {
            "name": upload.name, "ext": extension})


def validate_dataset_upload(upload) -> None:
    _validate_extension(upload, "csv")
    validate_size(upload)


def validate_model_upload(upload) -> None:
    _validate_extension(upload, "py")
    validate_size(upload)
