import os
import tempfile

from django import forms
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from core.io import demo_datasets, load_model_from_path

from .models import Experiment
from .registry import MODELS, OPTIMIZERS


class NewExperimentForm(forms.Form):
    """Set up an experiment: name, model, optimizer, dataset, seed.

    The model is either a registry choice or — when ALLOW_CUSTOM_MODELS is on —
    an uploaded ``.py`` file defining a BaseModel subclass. A dataset comes from
    either the demo dropdown or an upload; exactly one is required. Creating an
    experiment does not run it — the metric to optimize and the number of trials
    are chosen per-run on the detail page. All metrics are always scored.
    Optimizer parameters use their defaults here (editing them is a later step).
    """

    name = forms.CharField(label=_("Experiment name"), max_length=200)
    model_name = forms.ChoiceField(label=_("Model"), required=False)
    model_file = forms.FileField(label=_("…or upload a model .py"), required=False)
    optimizer_name = forms.ChoiceField(label=_("Optimizer"))
    demo_dataset = forms.ChoiceField(label=_("Demo dataset"), required=False)
    dataset_file = forms.FileField(label=_("…or upload a CSV (last column = target)"), required=False)
    seed = forms.IntegerField(label=_("Seed (negative = random)"), initial=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["model_name"].choices = [("", _("— select —"))] + [(k, k) for k in MODELS]
        self.fields["optimizer_name"].choices = [(k, k) for k in OPTIMIZERS]
        demos = demo_datasets()
        self.fields["demo_dataset"].choices = [("", _("— none —"))] + [(p, k) for k, p in demos.items()]
        if not settings.ALLOW_CUSTOM_MODELS:
            del self.fields["model_file"]

    def clean_name(self):
        name = self.cleaned_data["name"]
        if Experiment.objects.filter(name=name).exists():
            raise forms.ValidationError(_("An experiment named '%(name)s' already exists.") % {"name": name})
        return name

    def clean(self):
        cleaned = super().clean()

        # Model: an uploaded .py (validated by loading it) takes precedence;
        # otherwise a registry choice is required. A valid custom file resolves
        # the stored model_name to the model's own .name.
        upload = cleaned.get("model_file")
        if upload:
            resolved, err = self._load_uploaded_model(upload)
            if err:
                self.add_error("model_file", err)
            else:
                cleaned["model_name"] = resolved
        elif not cleaned.get("model_name"):
            self.add_error("model_name", _("Choose a model or upload a model .py file."))

        if not cleaned.get("demo_dataset") and not cleaned.get("dataset_file"):
            raise forms.ValidationError(_("Choose a demo dataset or upload a CSV file."))
        return cleaned

    @staticmethod
    def _load_uploaded_model(upload):
        """Validate an uploaded model file by loading it; return (name, error).

        The upload is written to a temp file so load_model_from_path can import
        it, then rewound so the view can still persist it to storage.
        """
        data = upload.read()
        upload.seek(0)
        tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
        try:
            tmp.write(data)
            tmp.close()
            model, err = load_model_from_path(tmp.name)
        finally:
            os.unlink(tmp.name)
        return (None, err) if err else (model.name, None)
