from pathlib import Path

from django import forms
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from core.io import DEFAULT_TEST_SIZE, demo_datasets, mounted_models
from core.splits import MIN_FOLDS
from core.model_source import inspect_model_source

from .figures import FIGURES, autocompute_key, deferred_computations
from .registry import MODELS, OPTIMIZERS
from .validators import validate_dataset_upload, validate_model_upload


#: The two ways a trial can be evaluated, and what the number beside the
#: selector means under each. One place, so the form, the template and the
#: script cannot disagree about the bounds or what to call it.
EVALUATION_KFOLD = "kfold"
EVALUATION_HOLDOUT = "holdout"
EVALUATION_SCHEMES = {
    EVALUATION_KFOLD: {
        "label": _("Folds"),
        "min": MIN_FOLDS, "max": 20, "step": "1", "default": 5,
        "help": _("Every row is validated against exactly once. Costs one model "
                  "fit per fold."),
    },
    EVALUATION_HOLDOUT: {
        "label": _("Share held out for validation"),
        "min": 0.05, "max": 0.5, "step": "0.05", "default": DEFAULT_TEST_SIZE,
        "help": _("One division of the data. Cheapest, and noisier on a small "
                  "table — the search can end up chasing the split."),
    },
}


class NewExperimentForm(forms.Form):
    """Set up an experiment: name, model, optimizer, dataset, seed.

    The model is either a registry choice or — when the person filling the form
    is allowed to bring one — an uploaded ``.py`` file defining a BaseModel
    subclass. That permission is decided by the policy and passed in, so the
    form does not have to know whether the instance has accounts. A dataset comes from
    either the demo dropdown or an upload; exactly one is required. How a trial
    is evaluated — one holdout or k folds — is settled here too, because it
    cannot change later without making the experiment's own trials incomparable.
    Creating an experiment does not run it — the metric to optimize and what
    stops the run are chosen per-run on the detail page. Every metric is always
    computed; the primary one is only the one the search optimizes.
    """

    name = forms.CharField(label=_("Experiment name"), max_length=200)
    model_name = forms.ChoiceField(label=_("Model"), required=False)
    model_file = forms.FileField(label=_("…or upload a model .py"), required=False,
                                  validators=[validate_model_upload])
    mounted_model = forms.ChoiceField(label=_("…or a mounted model .py"), required=False)
    optimizer_name = forms.ChoiceField(label=_("Optimizer"))
    demo_dataset = forms.ChoiceField(label=_("Demo dataset"), required=False)
    dataset_file = forms.FileField(label=_("…or upload a CSV (last column = target)"), required=False,
                                    validators=[validate_dataset_upload])
    seed = forms.IntegerField(
        label=_("Seed"), initial=0,
        help_text=_("Negative picks one at random. Drives every stochastic part "
                    "of the experiment: how the data is divided, the model's own "
                    "randomness, which configurations the search tries, and the "
                    "surrogates behind partial dependence and the explanations."))
    # Fixed for the experiment's life, so it is asked here rather than per run:
    # trials scored k-fold and trials scored on one holdout are not comparable,
    # and an experiment's own history has to be.
    # Two fields rather than one dropdown of preset combinations. The scheme and
    # the number are separate decisions — how the data is divided, and how
    # finely — and folding them into a fixed menu meant 7-fold or a 70/30 split
    # were simply not offerable. Cross-validation by default: a single split on
    # a small table is noisy enough that a search can spend its budget chasing
    # the split rather than the model, and the tables this is pointed at are
    # small.
    evaluation_scheme = forms.ChoiceField(
        label=_("Validation method"), initial=EVALUATION_KFOLD, required=False,
        help_text=_("Fixed once the experiment exists: trials evaluated "
                    "different ways cannot be compared with each other. "
                    "Cross-validation costs one fit per fold."),
        choices=[
            (EVALUATION_KFOLD, _("Cross-validation")),
            (EVALUATION_HOLDOUT, _("Dataset split")),
        ],
    )
    #: The number the scheme needs: folds under cross-validation, the share held
    #: out under a single split. One field because only one of them applies at a
    #: time, and two would leave whichever is inactive sitting there inviting a
    #: value that goes nowhere. Its label, bounds and default follow the scheme —
    #: see EVALUATION_SCHEMES and new_experiment.html's script.
    evaluation_value = forms.FloatField(
        label=_("Folds"), required=False,
        initial=EVALUATION_SCHEMES[EVALUATION_KFOLD]["default"])

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

    def _resolve_evaluation(self, cleaned) -> None:
        """Fold the scheme and its number into what the experiment stores.

        `cv_folds` stays the single source of truth for which scheme is in use —
        0 is a split, 2 or more is cross-validation — because everything
        downstream already reads it that way, and a second flag would be a
        second thing to keep in step. `test_size` is carried under either
        scheme, so switching one later has somewhere to start from.

        Out of range is clamped rather than refused: these are two ends of one
        continuum and every value between them is meaningful, so there is no
        typo here worth failing a form over.
        """
        scheme = cleaned.get("evaluation_scheme") or self.fields["evaluation_scheme"].initial
        spec = EVALUATION_SCHEMES[scheme]
        value = cleaned.get("evaluation_value")
        if value is None:
            value = spec["default"]
        value = max(spec["min"], min(spec["max"], value))

        if scheme == EVALUATION_KFOLD:
            cleaned["cv_folds"] = str(int(round(value)))
            cleaned["test_size"] = DEFAULT_TEST_SIZE
        else:
            cleaned["cv_folds"] = "0"
            cleaned["test_size"] = float(value)
        cleaned["evaluation_value"] = value

    def clean(self):
        cleaned = super().clean()
        self._resolve_evaluation(cleaned)

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


_ICE_LABEL = _("Trials drawn on the partial-dependence figure")
_LOCAL_EFFECTS_LABEL = _("Trials explained on the local-effects figure")


class ExperimentSettingsFields(forms.Form):
    """The experiment settings themselves.

    Both settings pages show exactly these, so both forms inherit them: the
    defaults page edits the template new experiments follow, the per-experiment
    page edits one experiment's own copy. One field per figure (`show_<key>`) and
    one per deferred computation (`autocompute_<name>`) come from the catalog, so
    a newly declared figure gets its checkboxes on both pages without touching
    this class.
    """

    ice_max_curves = forms.IntegerField(
        label=_ICE_LABEL, required=False, min_value=0, max_value=10_000,
        help_text=_("0 draws every trial. The curve is the mean of whatever is "
                    "drawn, so a smaller number is faster and lighter but a "
                    "coarser average."))
    local_effects_max_trials = forms.IntegerField(
        label=_LOCAL_EFFECTS_LABEL, required=False, min_value=0, max_value=10_000,
        help_text=_("0 explains every trial. Each one costs its own explanation, "
                    "so this is the setting that decides how long the figure takes."))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for figure in FIGURES:
            self.fields[figure.setting_key] = forms.BooleanField(
                label=figure.label, required=False,
            )
        for name, label in deferred_computations():
            self.fields[autocompute_key(name)] = forms.BooleanField(
                label=label, required=False,
            )

    @property
    def figure_fields(self):
        """The figure checkboxes, in catalog order — for the template to loop."""
        return [self[figure.setting_key] for figure in FIGURES]

    @property
    def autocompute_fields(self):
        """The deferred-computation checkboxes, in catalog order."""
        return [self[autocompute_key(name)] for name, _label in deferred_computations()]


class DefaultExperimentSettingsForm(ExperimentSettingsFields):
    """The defaults every inheriting experiment uses."""


class ExperimentSettingsForm(ExperimentSettingsFields):
    """One experiment's settings, plus whether it just inherits the defaults."""

    use_default_settings = forms.BooleanField(
        label=_("Use default experiment settings"), required=False)
