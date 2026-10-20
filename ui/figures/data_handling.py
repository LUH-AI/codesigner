"""Data Handling: looking at, cleaning, reshaping and augmenting an
experiment's dataset before it is tuned on.

The page renders from `SECTIONS` and from nothing else: its tabs, and the
capabilities on each. Each capability is a panel on its section's tab, a
`Panel` like a dashboard figure and placed by `ui/figure_layout.toml` the same
way. Most are planned, shown as a card saying what they will do. One with a
`step` is built: it is one of the experiment's processing choices
(`core.processing`), shown as a control. A draft's setup gathers those on an
Overview of its own — what can be decided before anything runs — while a
created experiment's Overview describes its dataset.

How the pieces are meant to fit, once they are built: Data Handling produces a
*recipe* — an ordered list of steps (choose the target, set a column's kind,
impute, encode, drop duplicates, oversample, ...) applied to the uploaded CSV
before it is split. The raw file is never rewritten. The recipe would travel in
the ``.ihpo`` with the rest of the experiment, and into an exported model, so
that predictions made outside Codesigner see exactly the transforms the tuned
model was scored on. Steps that only make sense on training rows —
augmentation, resampling — would apply inside each fold rather than before the
split, so they never leak into what a model is validated on.
"""

from __future__ import annotations

from dataclasses import dataclass

from django.utils.translation import gettext_lazy as _

from core.processing import AUTO, CHOICES, choices_of, model_defaults, unavailable
from experiments.services.processing import model_of

from .base import CARD, DATA, Panel

@dataclass(frozen=True)
class Capability:
    #: Stable identifier: the key of the panel this becomes, named in
    #: `ui/figure_layout.toml`.
    key: str
    title: str
    description: str
    #: The processing step this capability sets (a key of
    #: `core.processing.CHOICES`), or "" while it is only planned.
    step: str = ""


@dataclass(frozen=True)
class Section:
    slug: str
    title: str
    planned: tuple[Capability, ...]


SECTIONS = (
    Section(
        "overview", _("Overview"),
        (
            Capability("profile", _("Profile"), _("Rows, columns, size in memory, and how the file was read "
                                       "(separator, encoding, header).")),
            Capability("column_kinds", _("Column kinds"), _("Each column as numeric, categorical, datetime, text or "
                                            "identifier, as detected.")),
            Capability("where_values_are_missing", _("Where values are missing"), _("How much of each column is empty, and where.")),
            Capability("target_distribution", _("Target distribution"), _("Class counts for classification, a histogram for "
                                                   "regression, the series itself for forecasting.")),
            Capability("sample", _("Sample"), _("The first rows, a random sample, and the rows that look "
                                      "unusual.")),
        ),
    ),
    Section(
        "columns", _("Columns"),
        (
            Capability("target", _("Target"), _("Choose the column to predict, rather than it having to be "
                                      "the last.")),
            Capability("kinds", _("Kinds"), _("Override a detected kind: numeric, categorical, datetime, "
                                     "free text, identifier, or ignored.")),
            Capability("rename_and_drop", _("Rename and drop"), _("Give columns readable names; leave out the ones that "
                                               "should not be features.")),
            Capability("roles", _("Roles"), _("Mark the time column, the series identifier for many series "
                                     "in one file, and the group column for grouped splits.")),
            Capability("known_in_the_future", _("Known in the future"), _("For forecasting: which features are known ahead of "
                                                   "time (a holiday calendar) and which only up to "
                                                   "now (yesterday's sales).")),
        ),
    ),
    Section(
        "quality", _("Quality"),
        (
            Capability("missing_value_patterns", _("Missing-value patterns"), _("Whether gaps are random or cluster in certain rows, "
                                                      "columns or classes.")),
            Capability("duplicates", _("Duplicates"), _("Exact and near-duplicate rows, and duplicates that "
                                          "disagree on the target.")),
            Capability("constant_columns", _("Constant columns"), _("Columns with one value, or nearly so, which carry no "
                                                "information.")),
            Capability("outliers", _("Outliers"), _("Values far from the rest of their column, per column and "
                                        "jointly.")),
            Capability("class_imbalance", _("Class imbalance"), _("How rare the rarest class is, and what that does to "
                                               "accuracy as a metric.")),
            Capability("label_noise", _("Label noise"), _("Rows whose target disagrees with their nearest "
                                           "neighbours'.")),
            Capability("target_leakage", _("Target leakage"), _("Features that predict the target suspiciously well — an "
                                              "identifier, or something recorded after the fact.")),
            Capability("train_and_test_overlap", _("Train and test overlap"), _("The same row on both sides of a split, which "
                                                      "flatters every score.")),
        ),
    ),
    Section(
        "cleaning", _("Cleaning"),
        (
            Capability("missing_values", _("Missing values"), _("Fill each gap in — a number with its column's median, a "
                                              "label with its commonest label — or leave it for the "
                                              "model. A model that cannot take a gap always has its "
                                              "gaps filled."), step="missing"),
            Capability("imputation", _("Imputation"), _("Fill gaps with a column's mean, a constant, or from the "
                                          "nearest rows.")),
            Capability("drop_rows_or_columns", _("Drop rows or columns"), _("Leave out rows or columns over a threshold of "
                                                    "missing values, or by hand.")),
            Capability("de_duplication", _("De-duplication"), _("Keep one of each duplicate, or none of those that "
                                              "disagree.")),
            Capability("outlier_handling", _("Outlier handling"), _("Clip to a percentile range, or remove.")),
            Capability("type_coercion", _("Type coercion"), _("Read numbers written as text, dates in a given format, "
                                             "decimal commas.")),
            Capability("value_mapping", _("Value mapping"), _("Merge spellings of one category; group rare categories "
                                             "into \"other\".")),
        ),
    ),
    Section(
        "features", _("Features"),
        (
            Capability("label_columns", _("Label columns"), _("One column per label, or the labels as categories of their "
                                             "own for a model that takes them so — codes for a booster, "
                                             "words for CatBoost."), step="labels"),
            Capability("scaling", _("Scaling"), _("Standardize the numbers — zero mean, unit variance — for a "
                                       "model that measures distances, or leave them as they "
                                       "are."), step="scale"),
            Capability("encodings", _("Encodings"), _("Ordinal, target or frequency encoding for categorical "
                                         "columns.")),
            Capability("other_scaling", _("Other scaling"), _("Min-max or robust scaling.")),
            Capability("transforms", _("Transforms"), _("Log, Box-Cox or Yeo-Johnson; binning a numeric column "
                                          "into ranges.")),
            Capability("calendar_features", _("Calendar features"), _("Year, month, weekday, hour, holidays from a datetime "
                                                 "column.")),
            Capability("interactions", _("Interactions"), _("Products, ratios and polynomial features of chosen "
                                            "columns.")),
            Capability("lags_and_windows", _("Lags and windows"), _("Earlier values and rolling statistics, for data "
                                                "ordered in time.")),
            Capability("feature_selection", _("Feature selection"), _("Keep the most informative columns by importance, "
                                                 "correlation or mutual information.")),
        ),
    ),
    Section(
        "augmentation", _("Augmentation"),
        (
            Capability("resampling", _("Resampling"), _("Oversample rare classes or undersample common ones.")),
            Capability("smote", _("SMOTE"), _("Synthetic rows between neighbours of a rare class.")),
            Capability("noise_injection", _("Noise injection"), _("Small perturbations of numeric features.")),
            Capability("mixup", _("Mixup"), _("Blends of pairs of rows and their targets.")),
            Capability("synthetic_data", _("Synthetic data"), _("New rows drawn from a model of the dataset's joint "
                                              "distribution, such as a Gaussian copula.")),
            Capability("time_series_augmentation", _("Time-series augmentation"), _("Jitter, scaling and window slicing of a series.")),
        ),
    ),
    Section(
        "splits", _("Splits"),
        (
            Capability("scheme", _("Scheme"), _("Holdout, k-fold, stratified, grouped, time-ordered, or "
                                      "rolling-origin backtesting.")),
            Capability("final_test_set", _("Final test set"), _("Rows set aside from tuning entirely, scored once at "
                                              "the end.")),
            Capability("fold_preview", _("Fold preview"), _("Each fold's size, its class balance or target range, "
                                            "and its time span.")),
        ),
    ),
    Section(
        "sources", _("Sources"),
        (
            Capability("replace", _("Replace"), _("Upload a new version of the dataset, checked against the "
                                       "columns the old one had.")),
            Capability("append", _("Append"), _("Add rows from another file with the same columns.")),
            Capability("join", _("Join"), _("Add columns from another table on a shared key.")),
            Capability("subsample", _("Subsample"), _("Tune on a sample of a large dataset, stratified or by "
                                         "time.")),
        ),
    ),
    Section(
        "versions", _("Versions"),
        (
            Capability("history", _("History"), _("Every version of the dataset, with its fingerprint and "
                                       "when it was made.")),
            Capability("recipe", _("Recipe"), _("The steps applied to the uploaded file, in order, and "
                                      "what each one changed.")),
            Capability("runs_per_version", _("Runs per version"), _("Which runs were tuned on which version.")),
            Capability("differences", _("Differences"), _("Rows and columns added, removed or changed between two "
                                           "versions.")),
            Capability("export", _("Export"), _("Download the recipe, or the dataset with the recipe "
                                      "applied.")),
        ),
    ),
)

BY_SLUG = {s.slug: s for s in SECTIONS}


#: What each processing choice is called on the page.
CHOICE_LABELS = {
    "impute": _("Fill in"),
    "keep": _("Leave for the model"),
    "standardize": _("Standardize"),
    "none": _("Leave as they are"),
    "one_hot": _("One column per label"),
    "categories": _("As categories"),
}


class DataPanel(Panel):
    """A panel on the Data Handling page."""

    page = DATA
    shape = CARD
    template_dir = "ui/data_handling/panels"
    #: One template for every panel of a kind, rather than one per key.
    shared_template = ""
    #: What the panel is for, said under its heading.
    description = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.shared_template:
            cls.template = cls.shared_template


class ProcessingStep(DataPanel):
    """One processing choice, as a labelled select."""

    shared_template = "ui/data_handling/panels/processing_step.html"
    #: Its key in `core.processing.CHOICES`.
    step = ""

    @classmethod
    def control(cls, exp):
        """The select for *exp*: one option per choice, the model's own
        default among them named as such — "Leave as they are (Random Forest
        default)" — and chosen until something else is. The default is posted
        as "auto", so it goes on following the model if the model changes.
        A choice the model cannot take is offered disabled."""
        model = model_of(exp)
        chosen = choices_of(exp.data.processing)[cls.step]
        if model is None:
            return {"name": cls.step, "value": chosen,
                    "options": [{"value": v, "label": _("As the model needs") if v == AUTO
                                 else CHOICE_LABELS[v], "disabled": False}
                                for v in CHOICES[cls.step]]}
        default = model_defaults(model)[cls.step]
        options = []
        for choice in CHOICES[cls.step]:
            if choice == AUTO:
                continue
            is_default = choice == default
            options.append({
                "value": AUTO if is_default else choice,
                "label": (_("%(choice)s (%(model)s default)") % {
                    "choice": CHOICE_LABELS[choice], "model": model.name}
                          if is_default else str(CHOICE_LABELS[choice])),
                "disabled": unavailable(model, cls.step, choice),
            })
        return {"name": cls.step, "options": options,
                "value": AUTO if chosen in (AUTO, default) else chosen}


def describe(exp, step, choice, lower=True) -> str:
    """*choice* at *step* as a reader says it: "auto" as what it is for
    *exp*'s model — "standardize (SVM default)". Mid-sentence unless *lower*
    is false, when it reads as the card's option does."""
    model = model_of(exp)
    if choice == AUTO:
        if model is None:
            text = str(_("As the model needs"))
        else:
            text = _("%(choice)s (%(model)s default)") % {
                "choice": str(CHOICE_LABELS[model_defaults(model)[step]]), "model": model.name}
    else:
        text = str(CHOICE_LABELS.get(choice, choice))
    return text[:1].lower() + text[1:] if lower else text


def is_default(exp, step, choice) -> bool:
    """Whether *choice* is what *exp*'s model is given at *step* anyway."""
    model = model_of(exp)
    return choice == AUTO or (model is not None and model_defaults(model)[step] == choice)


class Planned(DataPanel):
    """A capability not built yet: what it will do."""

    shared_template = "ui/data_handling/panels/planned.html"


def _panel(section, capability):
    base = ProcessingStep if capability.step else Planned
    return type(f"{base.__name__}_{capability.key}", (base,), {
        "key": capability.key,
        "label": capability.title,
        "description": capability.description,
        "home_tab": section.slug,
        "step": capability.step,
        "__module__": __name__,
    })


#: Every Data Handling panel, section by section, in the order `SECTIONS`
#: lists them.
DATA_PANELS = tuple(_panel(section, capability)
                    for section in SECTIONS for capability in section.planned)
DATA_PANELS_BY_KEY = {panel.key: panel for panel in DATA_PANELS}
PROCESSING_PANELS = tuple(p for p in DATA_PANELS if issubclass(p, ProcessingStep))
