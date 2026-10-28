"""An exported model can carry `parts` unchanged because `parts` imports nothing
from `core.models`.
"""

import numpy as np
from sklearn.base import BaseEstimator


def categorical_mask(data, most_labels):
    """`categorical_mask` returns the columns a booster splits as categories, or
    None. A column over *most_labels* labels stays numeric and splits by code.
    """
    mask = np.array([kind == "categorical" for kind in data["kinds"]], dtype=bool)
    for column in np.flatnonzero(mask):
        largest = data["max_code"][column]
        if largest is not None and largest > most_labels - 1:
            mask[column] = False
    return mask if mask.any() else None


def categorical_columns(data):
    return [i for i, kind in enumerate(data["kinds"]) if kind == "categorical"]


class LabelCoded(BaseEstimator):
    """`LabelCoded` encodes labels to 0..k-1 for its classifier and decodes
    predictions back."""

    def __init__(self, classifier=None):
        self.classifier = classifier

    def fit(self, X, y, **fit_args):
        from sklearn.preprocessing import LabelEncoder

        self.labels_ = LabelEncoder().fit(y)
        self.classes_ = self.labels_.classes_
        assert self.classifier is not None
        self.classifier.fit(X, self.labels_.transform(y), **fit_args)
        return self

    def predict(self, X):
        assert self.classifier is not None
        return self.labels_.inverse_transform(self.classifier.predict(X).astype(int))

    def predict_proba(self, X):
        assert self.classifier is not None
        return self.classifier.predict_proba(X)
