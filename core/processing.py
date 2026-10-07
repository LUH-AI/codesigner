"""How an experiment's columns are processed before its model sees them.

Three choices, each made once for the experiment and recorded in its `.ihpo`:

* **missing** — fill a missing value in (``"impute"``: a number with its
  column's median, a label with its commonest label) or leave it as NaN
  (``"keep"``);
* **scale** — standardize the numbers (``"standardize"``: zero mean, unit
  variance) or leave them (``"none"``);
* **labels** — a label column as one column per label (``"one_hot"``), or as
  categories of its own (``"categories"``) in the form the model takes them:
  codes for a booster, words for CatBoost, plain numbers for a model with no
  notion of a category.

Each defaults to ``"auto"``: what the model needs (see
`codesigner_model.OutputBaseModel`'s `accepts_missing`, `scale_sensitive` and
`native_categories`). Left to the defaults an SVM gets its gaps filled, its
numbers scaled and its labels one-hot; a forest one-hot labels and nothing
else; a booster its columns as they are. A choice the model cannot take — gaps
left for a model that refuses NaN — is overruled, and `resolve` says so.

Processing is *fitted*: the medians, the scale and the labels it has seen come
from the rows it is fitted on, and in a trial those are the fold's training
rows only — the validation rows never inform how they themselves are scaled.

Imports nothing from this package, so an exported model carries this module as
its processing step.
"""

import numpy as np

AUTO = "auto"

#: Every choice, in the order the Data Handling page offers them.
CHOICES = {
    "missing": (AUTO, "impute", "keep"),
    "scale": (AUTO, "standardize", "none"),
    "labels": (AUTO, "one_hot", "categories"),
}

#: An experiment that has said nothing: every step as its model needs it.
DEFAULTS = {step: AUTO for step in CHOICES}


def choices_of(processing):
    """*processing* (as stored: possibly empty or partial) with every step
    present and every value one the step knows."""
    processing = processing or {}
    return {step: (processing.get(step) if processing.get(step) in options else AUTO)
            for step, options in CHOICES.items()}


def model_defaults(model) -> dict:
    """What *model* is given at each step when nothing is chosen: the choice
    "auto" stands for, as one of the others."""
    return {
        "missing": "keep" if getattr(model, "accepts_missing", True) else "impute",
        "scale": "standardize" if getattr(model, "scale_sensitive", False) else "none",
        "labels": "categories" if getattr(model, "native_categories", None) else "one_hot",
    }


def unavailable(model, step, choice) -> bool:
    """Whether *model* cannot take *choice* at *step* — one that cannot take
    a missing value has its gaps filled whatever is asked."""
    return step == "missing" and choice == "keep" and not getattr(model, "accepts_missing", True)


def resolve(processing, model):
    """The plan *model* is given under *processing*: (plan, overruled).

    *plan* has a concrete value for every step — ``"auto"`` replaced by what
    the model needs, and ``labels`` as ``"one_hot"``, ``"codes"``, ``"text"``
    or ``"numbers"`` (categories for a model with no notion of them). *overruled*
    names the steps whose choice the model cannot take.
    """
    choices = choices_of(processing)
    accepts_missing = bool(getattr(model, "accepts_missing", True))
    native = getattr(model, "native_categories", None)
    plan, overruled = {}, []

    missing = choices["missing"]
    if missing == AUTO:
        missing = "keep" if accepts_missing else "impute"
    elif missing == "keep" and not accepts_missing:
        missing = "impute"
        overruled.append("missing")
    plan["missing"] = missing

    scale = choices["scale"]
    if scale == AUTO:
        scale = "standardize" if getattr(model, "scale_sensitive", False) else "none"
    plan["scale"] = scale

    labels = choices["labels"]
    if labels == AUTO:
        labels = "categories" if native else "one_hot"
    plan["labels"] = labels if labels == "one_hot" else (native or "numbers")
    return plan, overruled


class Processing:
    """The processing a plan (see `resolve`) describes, for columns of *kinds*.

    A scikit-learn transformer: `fit` learns the medians, the scale and the
    labels from the rows it is given, `transform` applies them. `kinds_out`
    is the kind of each column it returns — label columns one-hot are numbers,
    label columns kept are still labels — or None when it cannot say how many
    there will be before it is fitted (one-hot), in which case every column it
    returns is a number.
    """

    def __init__(self, plan=None, kinds=()):
        self.plan = dict(plan or {})
        self.kinds = tuple(kinds)

    def get_params(self, deep=True):
        return {"plan": self.plan, "kinds": self.kinds}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ── what it does ────────────────────────────────────────────────────

    def _parts(self):
        """The fitted-to-be transformer, or None when the columns pass as they
        are."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        plan, kinds = self.plan, self.kinds
        numeric = [i for i, kind in enumerate(kinds) if kind != "categorical"]
        categorical = [i for i, kind in enumerate(kinds) if kind == "categorical"]
        impute = plan.get("missing") == "impute"
        scale = plan.get("scale") == "standardize"
        labels = plan.get("labels", "one_hot")

        def one_hot():
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        if not impute and not scale:
            # The columns as they are, but for the labels: one-hot in front of
            # everything else, or as words.
            if labels == "one_hot" and categorical:
                return ColumnTransformer([("labels", one_hot(), categorical)],
                                         remainder="passthrough", sparse_threshold=0)
            if labels == "text" and categorical:
                return TextCategories(tuple(categorical))
            return None

        parts = []
        if numeric:
            steps = ([SimpleImputer(strategy="median")] if impute else []) + \
                    ([StandardScaler()] if scale else [])
            parts.append(("numbers", make_pipeline(*steps) if steps else "passthrough", numeric))
        if categorical:
            steps = ([SimpleImputer(strategy="most_frequent")] if impute else []) + \
                    ([one_hot()] if labels == "one_hot" else
                     [TextCategories()] if labels == "text" else [])
            parts.append(("labels", make_pipeline(*steps) if steps else "passthrough", categorical))
        return ColumnTransformer(parts, sparse_threshold=0)

    def kinds_out(self):
        """The kind of each column `transform` returns, or None — every column
        a number, how many not known until fitted."""
        kinds, plan = self.kinds, self.plan
        categorical = [kind for kind in kinds if kind == "categorical"]
        if plan.get("labels", "one_hot") == "one_hot" and categorical:
            return None
        if plan.get("labels") == "numbers":
            return ("numeric",) * len(kinds)
        if plan.get("missing") != "impute" and plan.get("scale") != "standardize":
            return kinds
        numeric = [kind for kind in kinds if kind != "categorical"]
        return tuple(["numeric"] * len(numeric) + ["categorical"] * len(categorical))

    def fit(self, X, y=None):
        self.transformer_ = self._parts()
        if self.transformer_ is not None:
            self.transformer_.fit(X, y)
        return self

    def transform(self, X):
        return X if self.transformer_ is None else self.transformer_.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def is_identity(self):
        """Whether it leaves every column as it is."""
        return self._parts() is None


class TextCategories:
    """Label codes as text, for a learner that takes a category as a word.

    CatBoost takes a category as text or a whole number and refuses NaN in
    one, so a label's code becomes its digits and a missing label the word
    "missing" — a category of its own, which is how CatBoost treats it anyway.
    Numbers stay numbers, NaN and all. *columns* are the label columns; empty,
    every column given is one (as inside a column transformer's label part).
    """

    def __init__(self, columns=()):
        self.columns = columns

    def get_params(self, deep=True):
        return {"columns": self.columns}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pandas as pd

        frame = pd.DataFrame(X)
        for i in (self.columns or range(frame.shape[1])):
            if i < frame.shape[1]:
                frame[i] = [("missing" if pd.isna(v) else str(int(v))) for v in frame[i]]
        return frame

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
