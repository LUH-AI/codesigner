"""What more than one built-in model's `build` needs, written once.

* `categorical_mask`, `categorical_columns` — which columns a booster should
  split as categories;
* `LabelCoded` — a classifier that only knows 0..k-1, answering in the
  dataset's own labels (XGBoost).

How the columns are prepared — filled in, scaled, one-hot — is not here: that
is the experiment's processing (`core.processing`), done before the model.

Imports nothing from this package, so an exported model can carry this module
as it is.
"""

import numpy as np


def categorical_mask(data, most_labels):
    """Which columns a booster should split as categories, as a boolean mask,
    or None when there are none.

    Codes, not one-hot: the booster groups labels itself, which is better than
    one-hot on a column with many of them. A column with more labels than the
    booster can take (*most_labels*) is left numeric, codes and all — still
    usable, just split by code.
    """
    mask = np.array([kind == "categorical" for kind in data["kinds"]], dtype=bool)
    for column in np.flatnonzero(mask):
        largest = data["max_code"][column]
        if largest is not None and largest > most_labels - 1:
            mask[column] = False
    return mask if mask.any() else None


def categorical_columns(data):
    """The indices of *data*'s label columns."""
    return [i for i, kind in enumerate(data["kinds"]) if kind == "categorical"]


class LabelCoded:
    """A classifier that knows only the labels 0..k-1, answering in the
    dataset's own labels: they are encoded on the way in and decoded on the
    way out. Sample weights pass straight through."""

    def __init__(self, classifier=None):
        self.classifier = classifier

    def get_params(self, deep=True):
        return {"classifier": self.classifier}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y, **fit_args):
        from sklearn.preprocessing import LabelEncoder

        self.labels_ = LabelEncoder().fit(y)
        self.classes_ = self.labels_.classes_
        self.classifier.fit(X, self.labels_.transform(y), **fit_args)
        return self

    def predict(self, X):
        return self.labels_.inverse_transform(self.classifier.predict(X).astype(int))

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
