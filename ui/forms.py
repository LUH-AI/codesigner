from pathlib import Path

from django import forms
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from core.io import demo_datasets, mounted_models
from core.model_source import inspect_model_source

from .figures import FIGURES
from .registry import MODELS, OPTIMIZERS


class NewExperimentForm(forms.Form):
    """Set up an experiment: name, model, optimizer, dataset, seed.

    The model is either a registry choice or — when the person filling the form
    is allowed to bring one — an uploaded ``.py`` file defining a BaseModel
    subclass. That permission is decided by the policy and passed in, so the
    form does not have to know whether the instance has accounts. A dataset comes from
    either the demo dropdown or an upload; exactly one is required. How a trial
    is scored — one holdout or k folds — is settled here too, because it cannot
    change later without making the experiment's own trials incomparable.
    Creating an experiment does not run it — the metric to optimize and the number of trials
    are chosen per-run on the detail page. All metrics are always scored.
    Optimizer parameters use their defaults here (editing them is a later step).
    """

    name = forms.CharField(label=_("Experiment name"), max_length=200)
    model_name = forms.ChoiceField(label=_("Model"), required=False)
    model_file = forms.FileField(label=_("…or upload a model .py"), required=False)
    mounted_model = forms.ChoiceField(label=_("…or a mounted model .py"), required=False)
    optimizer_name = forms.ChoiceField(label=_("Optimizer"))
    demo_dataset = forms.ChoiceField(label=_("Demo dataset"), required=False)
    dataset_file = forms.FileField(label=_("…or upload a CSV (last column = target)"), required=False)
    seed = forms.IntegerField(label=_("Seed (negative = random)"), initial=0)
    # Fixed for the experiment's life, so it is asked here rather than per run:
    # trials scored k-fold and trials scored on one holdout are not comparable,
    # and an experiment's own history has to be.
    cv_folds = forms.ChoiceField(
        label=_("How each trial is scored"), initial="0", required=False,
        choices=[
            ("0", _("One 80/20 split — fastest")),
            ("3", _("3-fold cross-validation")),
            ("5", _("5-fold cross-validation — steadier on a small dataset")),
            ("10", _("10-fold cross-validation")),
        ],
    )

    def __init__(self, *args, may_upload_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        if may_upload_models is None:
            may_upload_models = settings.ALLOW_CUSTOM_MODELS
        self.fields["model_name"].choices = [("", _("— select —"))] + [(k, k) for k in MODELS]
        self.fields["optimizer_name"].choices = [(k, k) for k in OPTIMIZERS]
        demos = demo_datasets()
        self.fields["demo_dataset"].choices = [("", _("— none —"))] + [(p, k) for k, p in demos.items()]

        mounted = mounted_models() if may_upload_models else {}
        if mounted:
            self.fields["mounted_model"].choices = [("", _("— none —"))] + [(p, k) for k, p in mounted.items()]
        else:
            del self.fields["mounted_model"]
        if not may_upload_models:
            del self.fields["model_file"]

    def clean(self):
        cleaned = super().clean()

        # Model: a custom .py takes precedence over a registry choice, and is
        # read rather than run — see core.model_source. Its declared name and
        # dependencies are carried through for the view to record; whether it
        # actually imports is settled later, in its own environment.
        upload = cleaned.get("model_file")
        mounted = cleaned.get("mounted_model")
        if upload:
            source = upload.read()
            upload.seek(0)
            self._read_model_source(cleaned, source, "model_file")
        elif mounted:
            self._read_model_source(cleaned, Path(mounted).read_bytes(), "mounted_model")
        elif not cleaned.get("model_name"):
            self.add_error("model_name", _("Choose a model or upload a model .py file."))

        if not cleaned.get("demo_dataset") and not cleaned.get("dataset_file"):
            raise forms.ValidationError(_("Choose a demo dataset or upload a CSV file."))
        return cleaned

    def _read_model_source(self, cleaned, source: bytes, field: str) -> None:
        """Inspect a custom model's source, recording its name or an error."""
        info, err = inspect_model_source(source)
        if err:
            self.add_error(field, err)
            return
        cleaned["model_name"] = info.name
        cleaned["model_source"] = info


_EXPORT_ABS_LABEL = _("Include absolute timestamps in exported .ihpo files")


class ExperimentSettingsFields(forms.Form):
    """The experiment settings themselves.

    Both settings pages show exactly these, so both forms inherit them: the
    defaults page edits the template new experiments follow, the per-experiment
    page edits one experiment's own copy. One field per figure (`show_<key>`)
    comes from the catalog, so a newly declared figure gets its checkbox on both
    pages without touching this class.
    """

    export_absolute_times = forms.BooleanField(label=_EXPORT_ABS_LABEL, required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for figure in FIGURES:
            self.fields[figure.setting_key] = forms.BooleanField(
                label=figure.label, required=False,
            )

    @property
    def figure_fields(self):
        """The figure checkboxes, in catalog order — for the template to loop."""
        return [self[figure.setting_key] for figure in FIGURES]


class DefaultExperimentSettingsForm(ExperimentSettingsFields):
    """The defaults every inheriting experiment uses."""


class ExperimentSettingsForm(ExperimentSettingsFields):
    """One experiment's settings, plus whether it just inherits the defaults."""

    use_default_settings = forms.BooleanField(
        label=_("Use default experiment settings"), required=False)
