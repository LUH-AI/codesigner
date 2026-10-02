"""Experiments made from a demo dataset name it instead of keeping a copy.

Every experiment made from a demo used to store its own copy of the file. A
copy byte-for-byte equal to a bundled demo is replaced by the demo's name and
deleted; anything else — an upload, or a demo since changed — keeps its file.
"""

import hashlib
from pathlib import Path

from django.db import migrations


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def forwards(apps, schema_editor):
    from core.io import demo_datasets

    demos = {}
    for name, path in demo_datasets().items():
        try:
            demos[_digest(path)] = name
        except OSError:
            continue
    ExperimentData = apps.get_model("ui", "ExperimentData")
    for data in ExperimentData.objects.exclude(dataset="").exclude(dataset__isnull=True):
        try:
            stored = Path(data.dataset.path)
            name = demos.get(_digest(stored)) if stored.is_file() else None
        except (OSError, ValueError):
            continue
        if name is None:
            continue
        others = ExperimentData.objects.filter(dataset=data.dataset.name).exclude(pk=data.pk)
        data.demo_dataset = name
        data.dataset = None
        data.save(update_fields=["demo_dataset", "dataset"])
        if not others.exists():
            stored.unlink(missing_ok=True)


class Migration(migrations.Migration):

    dependencies = [("ui", "0036_storage")]

    operations = [migrations.RunPython(forwards, migrations.RunPython.noop)]
