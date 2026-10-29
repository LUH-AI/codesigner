"""`CHOICES` maps each data preparation step (missing values, scaling and
label encoding) to the options an experiment can select for it. An experiment
saves its selected options in its `.ihpo` file. Every step starts as "auto",
which `model_defaults` resolves to the option the model needs, and `resolve`
overrules an option the model cannot take. `Processing` only fits on a fold's
training rows, so validation rows never shape their own scaling. An exported
model carries `core.processing` as its processing step, since the module
imports nothing from codesigner.
"""

from sklearn.base import BaseEstimator

AUTO = "auto"

#: Data Handling page order.
CHOICES = {
    "missing": (AUTO, "impute", "keep"),
    "scale": (AUTO, "standardize", "none"),
    "labels": (AUTO, "one_hot", "categories"),
}

DEFAULTS = {step: AUTO for step in CHOICES}


def choices_of(processing):
    """`choices_of` fills missing or unknown steps with "auto"."""
    processing = processing or {}
    return {step: (processing.get(step) if processing.get(step) in options else AUTO)
            for step, options in CHOICES.items()}


def model_defaults(model) -> dict:
    """The concrete choice "auto" stands for."""
    return {
        "missing": "keep" if getattr(model, "accepts_missing", True) else "impute",
        "scale": "standardize" if getattr(model, "scale_sensitive", False) else "none",
        "labels": "categories" if getattr(model, "native_categories", None) else "one_hot",
    }


def unavailable(model, step, choice) -> bool:
    """`unavailable` flags `keep` for a model whose `accepts_missing` is False."""
    return step == "missing" and choice == "keep" and not getattr(model, "accepts_missing", True)


def resolve(processing, model):
    """`resolve` returns ``(plan, overruled)``, where *plan* replaces "auto" and
    gives ``labels`` as "one_hot", "codes", "text" or "numbers". `resolve` puts
    the steps whose choice the model cannot take in *overruled*.
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


class Processing(BaseEstimator):
    """`Processing` is a scikit-learn transformer that learns medians, scale and
    labels in `fit`. `kinds_out` returns None when one-hot makes the column
    count unknown before fitting.
    """

    def __init__(self, plan=None, kinds=()):
        self.plan = dict(plan or {})
        self.kinds = tuple(kinds)


    def _parts(self):
        """`_parts` returns None when the columns pass unchanged."""
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
            # Only the labels change here, into leading one-hot columns or text.
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
        """`is_identity` is true when nothing changes."""
        return self._parts() is None


class TextCategories(BaseEstimator):
    """CatBoost raises a `CatBoostError` for NaN in a category, so
    `TextCategories` writes each label code as digits and a missing label as
    "missing". An empty *columns* makes every given column a label.
    """

    def __init__(self, columns=()):
        self.columns = columns

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
