from pathlib import Path

from django import forms
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from core import tasks
from core.io import (
    DEFAULT_TEST_SIZE, demo_datasets, forecast_problem, mounted_models, orderable_columns,
    read_dataset, target_of, time_order_problem,
)
from core.splits import MIN_FOLDS
from core.model_source import inspect_model_source

from .figures import FIGURES, autocompute_key, deferred_computations
from .models import UNTITLED
from .registry import MODELS
from .services import timestamps
from .validators import validate_dataset_upload, validate_model_upload


#: The ways a trial can be evaluated, and what the number beside the selector
#: means under each. One place, so the form, the template and the script cannot
#: disagree about the bounds or what to call it.
EVALUATION_KFOLD = "kfold"
EVALUATION_HOLDOUT = "holdout"
#: A forecast: each fold trains on a series' past and predicts its next steps
#: (`core.splits.backtest`). Either task can be forecast — the next labels or
#: the next numbers — so it is a way of evaluating, not a task.
EVALUATION_BACKTEST = "backtest"
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
    EVALUATION_BACKTEST: {
        "label": _("Backtests"),
        "min": 1, "max": 20, "step": "1", "default": 3,
        "help": _("Each backtest trains on everything before an origin and "
                  "forecasts the horizon after it, the latest origin one horizon "
                  "before the end and each earlier one a horizon before that. "
                  "Costs one model fit per backtest."),
    },
}


class NewExperimentForm(forms.Form):
    """Set up an experiment: name, model, dataset, seed. The optimizer is a
    step of its own (ui/services/optimizer_form.py).

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

    # Optional: an experiment is told apart by its identifier, and one left
    # unnamed is shown as "Untitled Experiment" (`Experiment.title`).
    name = forms.CharField(label=_("Experiment name"), max_length=200, required=False,
                           widget=forms.TextInput(attrs={"placeholder": UNTITLED}))
    model_name = forms.ChoiceField(label=_("Model"), required=False)
    model_file = forms.FileField(label=_("…or upload a model .py"), required=False,
                                  validators=[validate_model_upload])
    mounted_model = forms.ChoiceField(label=_("…or a mounted model .py"), required=False)
    demo_dataset = forms.ChoiceField(label=_("Demo dataset"), required=False)
    dataset_file = forms.FileField(label=_("…or upload a CSV (last column = target)"), required=False,
                                    validators=[validate_dataset_upload])
    #: Asked, never guessed: a column of whole numbers is classes to one
    #: person and a quantity to the next, and which models are offered follows
    #: from the answer.
    task = forms.ChoiceField(
        label=_("Task"),
        help_text=_("What the experiment predicts: one of a set of labels, or a "
                    "number. Only the models that can do it are offered. Fixed "
                    "once the experiment exists. To predict a series' next labels "
                    "or numbers, choose backtests as the validation method."),
        choices=[("", _("— select —")),
                 (tasks.CLASSIFICATION, _("Classification")),
                 (tasks.REGRESSION, _("Regression"))],
    )
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
                    "Cross-validation costs one fit per fold. Backtests make the "
                    "experiment a forecast: each predicts the next steps of a "
                    "series from its past."),
        choices=[
            (EVALUATION_KFOLD, _("Cross-validation")),
            (EVALUATION_HOLDOUT, _("Dataset split")),
            (EVALUATION_BACKTEST, _("Backtests")),
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

    #: Empty is a random division, as before. Named rather than chosen from a
    #: list because the columns are the dataset's, and the dataset is chosen on
    #: the same form; the page suggests the ones that can be ordered by.
    time_column = forms.CharField(
        label=_("Order by"), required=False, max_length=255,
        help_text=_("A date or number column to divide the rows in time order by: "
                    "every trial trains on earlier rows and is validated on later "
                    "ones, as it would be used. Leave empty to divide at random. "
                    "Rows without a value in this column are left out."))
    time_gap = forms.IntegerField(
        label=_("Gap (rows)"), required=False, min_value=0, initial=0,
        help_text=_("Only with an order: rows left out between training and "
                    "validation, for labels that arrive late."))

    #: Backtests only. The validation method's number is then how many
    #: backtests there are, each one horizon later than the last.
    horizon = forms.IntegerField(
        label=_("Horizon (steps)"), required=False, min_value=1, initial=12,
        help_text=_("How many time steps ahead each forecast predicts. Each "
                    "backtest forecasts this many steps from an origin one "
                    "horizon later than the last."))
    series_column = forms.CharField(
        label=_("Series column"), required=False, max_length=255,
        help_text=_("A column naming which series each row belongs to, when the "
                    "file holds several — one per store, say. Each is forecast "
                    "from its own history. Leave empty for one series."))
    season = forms.IntegerField(
        label=_("Season length (steps)"), required=False, min_value=1,
        help_text=_("How many steps make a season — 12 for monthly data, 7 for "
                    "daily. Leave empty to infer it from the order column's "
                    "spacing."))

    def __init__(self, *args, may_upload_models=None, groups=(), current_dataset=None,
                 current_model=None, **kwargs):
        """*current_dataset* (a path) and *current_model* (a custom model's
        source, bytes) are what a draft being set up already has: given
        nothing new, the form keeps them, and checks against them."""
        super().__init__(*args, **kwargs)
        self.current_dataset = current_dataset
        self.current_model = current_model
        # Only when there is a choice to make: somebody in one group has their
        # experiment filed there without being asked. Required when present —
        # a guess would file work under a boundary nobody chose.
        if groups:
            self.fields["group"] = forms.TypedChoiceField(
                label=_("Group"), coerce=int,
                choices=[("", _("— select —"))] + [(g.pk, g.name) for g in groups])
        if may_upload_models is None:
            may_upload_models = settings.ALLOW_CUSTOM_MODELS
        self.fields["model_name"].choices = [("", _("— select —"))] + [(k, k) for k in MODELS]
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
        cleaned["evaluation_scheme"] = scheme
        spec = EVALUATION_SCHEMES[scheme]
        value = cleaned.get("evaluation_value")
        if value is None:
            value = spec["default"]
        value = max(spec["min"], min(spec["max"], value))

        if scheme in (EVALUATION_KFOLD, EVALUATION_BACKTEST):
            # Backtests are counted in the same place as folds: one backtest
            # is stored as 1, which reads as no cross-validation everywhere
            # that does not forecast and as one backtest where it does.
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
        elif not cleaned.get("model_name") and self.current_model is not None:
            # A draft's uploaded model, kept.
            self._read_model_source(cleaned, self.current_model, "model_name")
            cleaned["keep_model"] = True
        elif not cleaned.get("model_name"):
            self.add_error("model_name", _("Choose a model or upload a model .py file."))

        if not cleaned.get("demo_dataset") and not cleaned.get("dataset_file"):
            if self.current_dataset is None:
                raise forms.ValidationError(_("Choose a demo dataset or upload a CSV file."))
            # A draft's dataset, kept.
            cleaned["keep_dataset"] = Path(self.current_dataset)
        self._resolve_task(cleaned)
        self._resolve_time_order(cleaned)
        return cleaned

    def _resolve_task(self, cleaned) -> None:
        """Check that the target and the model can both do the task asked for,
        evaluated the way asked."""
        task = cleaned.get("task")
        if not task:
            return
        upload = cleaned.get("dataset_file")
        try:
            if upload:
                source = upload.read()
                upload.seek(0)
            else:
                source = Path(cleaned.get("demo_dataset") or cleaned["keep_dataset"])
            column, y = target_of(source)
        except Exception as exc:  # noqa: BLE001 — any unreadable file is the same problem
            self.add_error(None, _("The dataset could not be read: %(error)s") % {"error": exc})
            return

        if tasks.target_problem(y, task):
            self.add_error("task", _("The target column, “%(column)s”, is not numbers, so it "
                                     "cannot be regressed on. Choose classification, or put "
                                     "the column to predict last.") % {"column": column})
            return

        info = cleaned.get("model_source")
        name = cleaned.get("model_name")
        supported = (info.tasks if info is not None
                     else tasks.supported(MODELS[name]) if name in MODELS else None)
        field = ("model_file" if cleaned.get("model_file") else
                 "mounted_model" if cleaned.get("mounted_model") else "model_name")
        if supported is not None and task not in supported:
            self.add_error(field, _("%(model)s does not do %(task)s.") % {
                "model": name, "task": TASK_LABELS[task]})
            return
        # Every built-in forecasts through `core.forecasting`; an upload runs
        # in its own process, so it forecasts only if it says it does.
        backtests = cleaned.get("evaluation_scheme") == EVALUATION_BACKTEST
        forecaster = (info.forecaster if info is not None
                      else tasks.forecaster(MODELS[name]) if name in MODELS else None)
        if forecaster and not backtests:
            self.add_error(field, _("%(model)s only forecasts: choose backtests as the "
                                    "validation method.") % {"model": name})
            return
        if info is not None and backtests and not forecaster:
            self.add_error(field, _("%(model)s does not forecast, so it cannot be "
                                    "backtested. A model file that does says so with "
                                    "forecaster = True.") % {"model": name})
            return

    def _resolve_time_order(self, cleaned) -> None:
        """Check the rows can be divided in time order the way the evaluation
        asks, by the column named."""
        column = (cleaned.get("time_column") or "").strip()
        cleaned["time_column"] = column
        cleaned["time_gap"] = int(cleaned.get("time_gap") or 0) if column else 0
        forecasting = cleaned.get("evaluation_scheme") == EVALUATION_BACKTEST
        cleaned["horizon"] = int(cleaned.get("horizon") or 0) if forecasting else 0
        cleaned["series_column"] = (cleaned.get("series_column") or "").strip() if forecasting else ""
        cleaned["season"] = int(cleaned.get("season") or 0) if forecasting else 0
        if forecasting and not column:
            self.add_error("time_column", _("A forecast needs the column its rows are "
                                            "ordered in time by."))
            return
        if forecasting and not cleaned["horizon"]:
            self.add_error("horizon", _("A forecast needs a horizon of at least one step."))
            return
        if not column:
            return
        upload = cleaned.get("dataset_file")
        try:
            if upload:
                frame = read_dataset(upload.read())
                upload.seek(0)
            else:
                frame = read_dataset(Path(cleaned.get("demo_dataset") or cleaned["keep_dataset"]))
        except Exception:  # noqa: BLE001 — `_resolve_task` reports an unreadable file
            return
        if forecasting:
            problem = forecast_problem(frame, column, cleaned["series_column"],
                                       cleaned["horizon"], int(cleaned.get("cv_folds") or 0),
                                       cleaned["time_gap"])
        else:
            problem = time_order_problem(frame, column, int(cleaned.get("cv_folds") or 0),
                                         cleaned.get("test_size") or DEFAULT_TEST_SIZE,
                                         cleaned["time_gap"])
        if problem:
            orderable = orderable_columns(frame)
            self.add_error("time_column", _("This dataset cannot be ordered that way: "
                                            "%(problem)s. Columns it can be ordered by: "
                                            "%(columns)s.") % {
                "problem": problem, "columns": ", ".join(orderable) or _("none")})

    def _read_model_source(self, cleaned, source: bytes, field: str) -> None:
        """Inspect a custom model's source, recording its name or an error."""
        info, err = inspect_model_source(source)
        if err:
            self.add_error(field, err)
            return
        cleaned["model_name"] = info.name
        cleaned["model_source"] = info


#: What to call each task where a sentence names one.
TASK_LABELS = {tasks.CLASSIFICATION: _("classification"), tasks.REGRESSION: _("regression")}


_ICE_LABEL = _("Trials drawn on the partial-dependence figure")
_LOCAL_EFFECTS_LABEL = _("Trials explained on the local-effects figure")
_PRIOR_TOLERANCE_LABEL = _("Prior acceptance tolerance")


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
        help_text=_("0 draws every trial."))
    local_effects_max_trials = forms.IntegerField(
        label=_LOCAL_EFFECTS_LABEL, required=False, min_value=0, max_value=10_000,
        help_text=_("0 explains every trial."))
    prior_acceptance_tolerance = forms.FloatField(
        label=_PRIOR_TOLERANCE_LABEL, required=False, min_value=0.0,
        max_value=1_000.0,
        help_text=_("How much worse a prior's region may score before it is refused."))
    keep_best_trial_models = forms.IntegerField(
        label=_("Trial models kept for export"), required=False, min_value=0, max_value=100,
        help_text=_("0 keeps none."))
    keep_last_run_trial_models = forms.BooleanField(
        label=_("Also keep every trial of the last run"), required=False,
        help_text=_("Kept until the next run starts."))
    export_timestamps = forms.ChoiceField(
        label=_("Time"), required=False,
        choices=[("", _("The exporter's preference"))]
                + [(mode, timestamps.LABELS[mode][0]) for mode in timestamps.MODES],
        help_text=_("What the export page preselects."))

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
