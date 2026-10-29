from pathlib import Path

from django import forms
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from core.datasets import (
    DEFAULT_TEST_SIZE, demo_datasets, forecast_problem, mounted_models, orderable_columns,
    read_dataset, target_last, time_order_problem,
)
from core import io, provenance, tasks
from core.model_source import inspect_model_source
from core.registry import MODELS
from core.splits import MIN_FOLDS
from experiments.models import UNTITLED
from experiments.services import timestamps

from . import times
from .figures import FIGURES, autocompute_key, deferred_computations
from .validators import validate_dataset_upload, validate_model_upload, validate_size


#: `EVALUATION_SCHEMES` keeps each scheme's bounds and label in one place for
#: the form, template and script.
EVALUATION_KFOLD = "kfold"
EVALUATION_HOLDOUT = "holdout"
#: Backtests train each fold on a series' past and predict its next steps
#: (`core.splits.backtest`). Either task can be forecast, so backtesting is an
#: evaluation scheme.
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


class ImportForm(forms.Form):
    """`ImportForm` takes an `.ihpo` plus an optional dataset and, for readers
    allowed to upload, a model."""

    file = forms.FileField(validators=[validate_size])
    dataset = forms.FileField(required=False, validators=[validate_dataset_upload])
    model = forms.FileField(required=False, validators=[validate_model_upload])

    def __init__(self, *args, allow_model=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not allow_model:
            del self.fields["model"]

    def clean_file(self):
        upload = self.cleaned_data["file"]
        try:
            self.snapshot = io.parse(upload.read())
        except ValueError as exc:
            raise forms.ValidationError(
                _("Invalid or unreadable experiment file.") + f" ({exc})") from exc
        return upload

    def clean(self):
        # An attached dataset must match the one recorded in the file, since
        # nothing downstream could detect trials added to the wrong history.
        cleaned = super().clean()
        dataset = cleaned.get("dataset")
        if dataset is not None and hasattr(self, "snapshot"):
            mismatch = provenance.dataset_mismatch(
                self.snapshot.get("dataset"), provenance.sha256_stream(dataset))
            if mismatch:
                raise forms.ValidationError(mismatch)
        return cleaned


class NewExperimentForm(forms.Form):
    """`NewExperimentForm` sets up the name, model, dataset, seed and
    evaluation, while ui/optimizer_form.py handles the optimizer. The policy
    decides whether a `.py` model upload is allowed.
    """

    name = forms.CharField(label=_("Experiment name"), max_length=200, required=False,
                           widget=forms.TextInput(attrs={"placeholder": UNTITLED}))
    model_name = forms.ChoiceField(label=_("Model"), required=False)
    model_file = forms.FileField(label=_("…or upload a model .py"), required=False,
                                  validators=[validate_model_upload])
    mounted_model = forms.ChoiceField(label=_("…or a mounted model .py"), required=False)
    demo_dataset = forms.ChoiceField(label=_("Dataset (demo)"), required=False)
    dataset_file = forms.FileField(label=_("Dataset (.csv)"), required=False,
                                    validators=[validate_dataset_upload])
    #: The picker's × clears the draft's dataset, so step 1 keeps it no longer.
    clear_dataset = forms.BooleanField(required=False, widget=forms.HiddenInput)
    #: The target column is typed rather than listed, since the columns belong
    #: to whichever dataset is chosen above.
    target_column = forms.CharField(
        label=_("Target"), required=False, max_length=255,
        help_text=_("The column to predict. Empty is the dataset's last column."))
    #: The task is asked because whole numbers may be classes or quantities, and
    #: the answer filters the models.
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
    #: The scheme is fixed for the experiment's life, since k-fold and holdout
    #: scores are incomparable. Cross-validation is the default because one split
    #: on a small table is noisy.
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
    #: `evaluation_value` is the number the scheme needs, and its label, bounds
    #: and default follow `EVALUATION_SCHEMES`.
    evaluation_value = forms.FloatField(
        label=_("Folds"), required=False,
        initial=EVALUATION_SCHEMES[EVALUATION_KFOLD]["default"])

    #: Empty divides at random; the page suggests orderable columns.
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

    #: Backtests only, where the validation number counts the backtests.
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
        """*current_dataset* and *current_model* are the draft's existing files,
        kept when nothing new is posted."""
        super().__init__(*args, **kwargs)
        self.current_dataset = current_dataset
        self.current_model = current_model
        # The group field only appears with a choice to make, and is required
        # then, since a guess would misfile work.
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
        """`cv_folds` stays the one record of the scheme, with 0 a split and 2
        or more cross-validation. Out-of-range values are clamped, since every
        value in range is meaningful.
        """
        scheme = cleaned.get("evaluation_scheme") or self.fields["evaluation_scheme"].initial
        cleaned["evaluation_scheme"] = scheme
        spec = EVALUATION_SCHEMES[scheme]
        value = cleaned.get("evaluation_value")
        if value is None:
            value = spec["default"]
        value = max(spec["min"], min(spec["max"], value))

        if scheme in (EVALUATION_KFOLD, EVALUATION_BACKTEST):
            # Backtests are counted in `cv_folds` too, and one backtest stored
            # as 1 reads as no cross-validation outside forecasting.
            cleaned["cv_folds"] = str(int(round(value)))
            cleaned["test_size"] = DEFAULT_TEST_SIZE
        else:
            cleaned["cv_folds"] = "0"
            cleaned["test_size"] = float(value)
        cleaned["evaluation_value"] = value

    def clean(self):
        cleaned = super().clean()
        self._resolve_evaluation(cleaned)

        # A custom .py beats a registry choice and is read, never run (see
        # core.model_source).
        upload = cleaned.get("model_file")
        mounted = cleaned.get("mounted_model")
        if upload:
            source = upload.read()
            upload.seek(0)
            self._read_model_source(cleaned, source, "model_file")
        elif mounted:
            self._read_model_source(cleaned, Path(mounted).read_bytes(), "mounted_model")
        elif not cleaned.get("model_name") and self.current_model is not None:
            self._read_model_source(cleaned, self.current_model, "model_name")
            cleaned["keep_model"] = True
        elif not cleaned.get("model_name"):
            self.add_error("model_name", _("Choose a model or upload a model .py file."))

        if not cleaned.get("demo_dataset") and not cleaned.get("dataset_file"):
            if self.current_dataset is None or cleaned.get("clear_dataset"):
                raise forms.ValidationError(_("Choose a demo dataset or upload a CSV file."))
            cleaned["keep_dataset"] = Path(self.current_dataset)
        self._resolve_task(cleaned)
        self._resolve_time_order(cleaned)
        return cleaned

    def _resolve_task(self, cleaned) -> None:
        """`_resolve_task` checks target and model against the task and scheme."""
        cleaned["target_column"] = (cleaned.get("target_column") or "").strip()
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
            frame = read_dataset(source)
        except Exception as exc:  # noqa: BLE001 — any unreadable file is the same problem
            self.add_error(None, _("The dataset could not be read: %(error)s") % {"error": exc})
            return
        try:
            frame = target_last(frame, cleaned["target_column"] or None)
        except ValueError:
            self.add_error("target_column", _("The dataset has no column named “%(column)s”.")
                           % {"column": cleaned["target_column"]})
            return
        column, y = str(frame.columns[-1]), frame.iloc[:, -1].to_numpy()

        if tasks.target_problem(y, task):
            self.add_error("task", _("The target column, “%(column)s”, is not numbers, so it "
                                     "cannot be regressed on. Choose classification, or "
                                     "another target.") % {"column": column})
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
        # `core.forecasting` serves every built-in model, and an upload
        # forecasts only if it says so.
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
        """`_resolve_time_order` checks the time-order column against the
        evaluation."""
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
            target = cleaned.get("target_column") or None
            if upload:
                frame = read_dataset(upload.read(), target)
                upload.seek(0)
            else:
                frame = read_dataset(Path(cleaned.get("demo_dataset") or cleaned["keep_dataset"]),
                                     target)
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
        info, err = inspect_model_source(source)
        if err:
            self.add_error(field, err)
            return
        assert info is not None
        cleaned["model_name"] = info.name
        cleaned["model_source"] = info


TASK_LABELS = {tasks.CLASSIFICATION: _("classification"), tasks.REGRESSION: _("regression")}


_ICE_LABEL = _("Trials drawn on the partial-dependence figure")
_LOCAL_EFFECTS_LABEL = _("Trials explained on the local-effects figure")
_PRIOR_TOLERANCE_LABEL = _("Prior acceptance tolerance")
_PRIOR_TOP_K_LABEL = _("Prior acceptance top k")


class ExperimentSettingsFields(forms.Form):
    """Both settings pages inherit `ExperimentSettingsFields`. The catalog
    supplies the figure and deferred-computation checkboxes, and a new figure
    reaches both pages without changes here.
    """

    ice_max_curves = forms.IntegerField(
        label=_ICE_LABEL, required=False, min_value=0, max_value=10_000,
        help_text=_("0 draws every trial."))
    local_effects_max_trials = forms.IntegerField(
        label=_LOCAL_EFFECTS_LABEL, required=False, min_value=0, max_value=10_000,
        help_text=_("0 explains every trial."))
    prior_acceptance_tolerance = forms.FloatField(
        label=_PRIOR_TOLERANCE_LABEL, required=False, min_value=0.0,
        max_value=1.0,
        help_text=_("Share of the metric's scale a prior may fall short by."))
    prior_acceptance_top_k = forms.IntegerField(
        label=_PRIOR_TOP_K_LABEL, required=False, min_value=1, max_value=100,
        help_text=_("Best configurations compared on each side."))
    keep_best_trial_models = forms.IntegerField(
        label=_("Trial models kept for export"), required=False, min_value=0, max_value=100,
        help_text=_("0 keeps none."))
    keep_last_run_trial_models = forms.BooleanField(
        label=_("Also keep every trial of the last run"), required=False,
        help_text=_("Kept until the next run starts."))
    export_timestamps = forms.ChoiceField(
        label=_("Time"), required=False,
        choices=[("", _("The exporter's preference"))]
                + [(mode, times.LABELS[mode][0]) for mode in timestamps.MODES],
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
        return [self[figure.setting_key] for figure in FIGURES]

    @property
    def autocompute_fields(self):
        return [self[autocompute_key(name)] for name, _label in deferred_computations()]


class ExperimentSettingsForm(ExperimentSettingsFields):

    use_default_settings = forms.BooleanField(
        label=_("Use default experiment settings"), required=False)
