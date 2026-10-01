"""What Data Handling is to do, section by section.

The pages render from this list and from nothing else: the sidebar's tabs, each
page's heading and purpose, and the capabilities it lists as planned. None of
them is built. Reshaping a section — adding an idea, dropping one, moving one
somewhere it fits better — is an edit here.
"""

from dataclasses import dataclass

from django.utils.translation import gettext_lazy as _


@dataclass(frozen=True)
class Capability:
    title: str
    description: str


@dataclass(frozen=True)
class Section:
    slug: str
    title: str
    purpose: str
    planned: tuple[Capability, ...]


SECTIONS = (
    Section(
        "overview", _("Overview"),
        _("What the dataset is, at a glance: enough to tell whether it was read "
          "the way it was meant to be before anything is tuned on it."),
        (
            Capability(_("Profile"), _("Rows, columns, size in memory, and how the file was read "
                                       "(separator, encoding, header).")),
            Capability(_("Column kinds"), _("Each column as numeric, categorical, datetime, text or "
                                            "identifier, as detected.")),
            Capability(_("Missing values"), _("How much of each column is empty, and where.")),
            Capability(_("Target distribution"), _("Class counts for classification, a histogram for "
                                                   "regression, the series itself for forecasting.")),
            Capability(_("Sample"), _("The first rows, a random sample, and the rows that look "
                                      "unusual.")),
        ),
    ),
    Section(
        "columns", _("Columns"),
        _("What each column is and what it is for. Today the last column is the "
          "target and every other column is a feature; this is where that stops "
          "being a convention."),
        (
            Capability(_("Target"), _("Choose the column to predict, rather than it having to be "
                                      "the last.")),
            Capability(_("Kinds"), _("Override a detected kind: numeric, categorical, datetime, "
                                     "free text, identifier, or ignored.")),
            Capability(_("Rename and drop"), _("Give columns readable names; leave out the ones that "
                                               "should not be features.")),
            Capability(_("Roles"), _("Mark the time column, the series identifier for many series "
                                     "in one file, and the group column for grouped splits.")),
            Capability(_("Known in the future"), _("For forecasting: which features are known ahead of "
                                                   "time (a holiday calendar) and which only up to "
                                                   "now (yesterday's sales).")),
        ),
    ),
    Section(
        "quality", _("Quality"),
        _("Problems worth knowing about before tuning, found rather than "
          "fixed: each one links to where it would be fixed."),
        (
            Capability(_("Missing-value patterns"), _("Whether gaps are random or cluster in certain rows, "
                                                      "columns or classes.")),
            Capability(_("Duplicates"), _("Exact and near-duplicate rows, and duplicates that "
                                          "disagree on the target.")),
            Capability(_("Constant columns"), _("Columns with one value, or nearly so, which carry no "
                                                "information.")),
            Capability(_("Outliers"), _("Values far from the rest of their column, per column and "
                                        "jointly.")),
            Capability(_("Class imbalance"), _("How rare the rarest class is, and what that does to "
                                               "accuracy as a metric.")),
            Capability(_("Label noise"), _("Rows whose target disagrees with their nearest "
                                           "neighbours'.")),
            Capability(_("Target leakage"), _("Features that predict the target suspiciously well — an "
                                              "identifier, or something recorded after the fact.")),
            Capability(_("Train and test overlap"), _("The same row on both sides of a split, which "
                                                      "flatters every score.")),
        ),
    ),
    Section(
        "cleaning", _("Cleaning"),
        _("Fixing what Quality found, as steps in the dataset's recipe — the "
          "uploaded file itself is never changed."),
        (
            Capability(_("Imputation"), _("Fill gaps with a column's mean, median or most frequent "
                                          "value, a constant, or from the nearest rows.")),
            Capability(_("Drop rows or columns"), _("Leave out rows or columns over a threshold of "
                                                    "missing values, or by hand.")),
            Capability(_("De-duplication"), _("Keep one of each duplicate, or none of those that "
                                              "disagree.")),
            Capability(_("Outlier handling"), _("Clip to a percentile range, or remove.")),
            Capability(_("Type coercion"), _("Read numbers written as text, dates in a given format, "
                                             "decimal commas.")),
            Capability(_("Value mapping"), _("Merge spellings of one category; group rare categories "
                                             "into \"other\".")),
        ),
    ),
    Section(
        "features", _("Features"),
        _("New columns made from the existing ones. Some can be tuned along "
          "with the model rather than fixed in advance."),
        (
            Capability(_("Encodings"), _("One-hot, ordinal, target or frequency encoding for "
                                         "categorical columns.")),
            Capability(_("Scaling"), _("Standard, min-max or robust scaling.")),
            Capability(_("Transforms"), _("Log, Box-Cox or Yeo-Johnson; binning a numeric column "
                                          "into ranges.")),
            Capability(_("Calendar features"), _("Year, month, weekday, hour, holidays from a datetime "
                                                 "column.")),
            Capability(_("Interactions"), _("Products, ratios and polynomial features of chosen "
                                            "columns.")),
            Capability(_("Lags and windows"), _("Earlier values and rolling statistics, for data "
                                                "ordered in time.")),
            Capability(_("Feature selection"), _("Keep the most informative columns by importance, "
                                                 "correlation or mutual information.")),
        ),
    ),
    Section(
        "augmentation", _("Augmentation"),
        _("More, or more balanced, training data. Applied to training rows "
          "only, inside each fold, so it never reaches what a model is scored "
          "on."),
        (
            Capability(_("Resampling"), _("Oversample rare classes or undersample common ones.")),
            Capability(_("SMOTE"), _("Synthetic rows between neighbours of a rare class.")),
            Capability(_("Noise injection"), _("Small perturbations of numeric features.")),
            Capability(_("Mixup"), _("Blends of pairs of rows and their targets.")),
            Capability(_("Synthetic data"), _("New rows drawn from a model of the dataset's joint "
                                              "distribution, such as a Gaussian copula.")),
            Capability(_("Time-series augmentation"), _("Jitter, scaling and window slicing of a series.")),
        ),
    ),
    Section(
        "splits", _("Splits"),
        _("How rows are divided into what a model learns from and what it is "
          "scored on — and a look at the result before any time is spent "
          "tuning against it."),
        (
            Capability(_("Scheme"), _("Holdout, k-fold, stratified, grouped, time-ordered, or "
                                      "rolling-origin backtesting.")),
            Capability(_("Final test set"), _("Rows set aside from tuning entirely, scored once at "
                                              "the end.")),
            Capability(_("Fold preview"), _("Each fold's size, its class balance or target range, "
                                            "and its time span.")),
        ),
    ),
    Section(
        "sources", _("Sources"),
        _("Where the data comes from, and changing it after the experiment "
          "exists."),
        (
            Capability(_("Replace"), _("Upload a new version of the dataset, checked against the "
                                       "columns the old one had.")),
            Capability(_("Append"), _("Add rows from another file with the same columns.")),
            Capability(_("Join"), _("Add columns from another table on a shared key.")),
            Capability(_("Subsample"), _("Tune on a sample of a large dataset, stratified or by "
                                         "time.")),
        ),
    ),
    Section(
        "versions", _("Versions"),
        _("What the data was when each run used it, so a result can always be "
          "traced back to the rows that produced it."),
        (
            Capability(_("History"), _("Every version of the dataset, with its fingerprint and "
                                       "when it was made.")),
            Capability(_("Recipe"), _("The steps applied to the uploaded file, in order, and "
                                      "what each one changed.")),
            Capability(_("Runs per version"), _("Which runs were tuned on which version.")),
            Capability(_("Differences"), _("Rows and columns added, removed or changed between two "
                                           "versions.")),
            Capability(_("Export"), _("Download the recipe, or the dataset with the recipe "
                                      "applied.")),
        ),
    ),
)

BY_SLUG = {s.slug: s for s in SECTIONS}
