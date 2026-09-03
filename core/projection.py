"""Where the search went, projected onto a subspace worth looking at.

This is the same operation the configuration cube performs, not a different
one. Both keep two or three linear coordinates of a p-dimensional space and drop
the rest. Picking axes is picking a *coordinate* subspace; PCA picks any
subspace. So the cube is this problem with the search restricted to the axes —
and the difference between them is a matter of degree, of two kinds.

**How much is lost.** On standardised data every hyperparameter carries one unit
of variance, so any two axes retain exactly two of the p units. PCA's first two
components retain λ1 + λ2, which is at least two and equal to two only when the
standardised hyperparameters are entirely uncorrelated. It is the same choice,
made over a larger set of candidates, so it can only do better — and on a search
whose hyperparameters move together it does much better.

**Whether a distance survives.** The cube plots raw values, so the length of a
line between two of its points mixes `n_estimators` with `min_samples_split` and
is not a length at all. Here every column is standardised before the projection
(see `encode_configurations`), which is what makes the underlying distance
coherent, and the subspace is then chosen to lose as little of it as possible —
so a plotted distance approximates a real one, with the discarded variance as
the size of the error. That is two changes and only the second is about PCA;
standardising the cube's axes would fix half of it, at the cost of the thing the
cube is for, which is that a position on an axis is a hyperparameter value you
can read.

**And what is being maximised.** PCA minimises what is lost from the
configurations, unsupervised. PLS does not: it maximises covariance with the
*score*, so it will drop a high-variance direction the metric does not care
about and keep a low-variance one it does. Its first component is the direction
through configuration space that moves the metric most, which is a different
question from where the search spread out — and usually the one being asked.

Deliberately not in `core/optimizers/`: none of this needs an optimizer, a
surrogate or a refit — only the trials that were run and the space they came
from — and it is worth being testable without any of them.

DeepCAVE ships a third point on the same line as its separate *Footprint*
plugin, which projects with MDS: no linear map at all, distances fitted
directly, so nothing survives that says which hyperparameters made them. PCA and
PLS are linear, so a component is a weighted sum of hyperparameters and can in
principle be read.
"""

import math
from typing import List, Tuple

#: How many components a projection offers at most. Three, because three is
#: what can be drawn — a fourth would be computed for nobody.
MAX_COMPONENTS = 3

#: The projections on offer, and what each is called where a reader sees it.
METHODS = ("pca", "pls")


def log_hyperparameters(config_space, hp_names) -> list:
    """Which of *hp_names* are searched on a log scale, in the order given.

    Empty when there's no config space to ask (a custom model viewed read-only,
    where nothing was imported), which puts every one of them back on the linear
    default — the same graceful-degradation rule the rest of the analytics
    follow when something isn't available.
    """
    if config_space is None:
        return []
    return [h for h in hp_names
            if h in config_space and getattr(config_space[h], "log", False)]


def _is_numeric(value) -> bool:
    """Whether a hyperparameter value is a number to be measured with.

    `bool` is a subclass of `int` in Python and is excluded on purpose: True and
    False are two categories, and treating them as 1 and 0 would put them a unit
    apart on a scale where nothing else is.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def encode_configurations(config_space, trials) -> Tuple[List[List[float]], List[str]]:
    """*trials*' configurations as a numeric matrix, one row each.

    Returns `(rows, feature_names)`. Three things happen to a configuration on
    the way in, each for its own reason:

    - a **log-scaled** hyperparameter goes on as `log10`. `C` on the SVM model
      is sampled log-uniformly over 0.01 to 100, so on a linear scale three
      quarters of the runs sit within a hundredth of each other and the
      projection would report the handful of large values as the only structure
      there is. Parallel coordinates recodes the same values for the same
      reason.
    - a **categorical or boolean** one becomes one column per value, holding 0
      or 1. Parallel coordinates codes categories to integers because it needs
      one column per axis to draw; here that would invent an ordering and a
      distance between kernels — and inventing distances is precisely what this
      module must not do, since distance is the thing it draws.
    - every column is then **standardised** to zero mean and unit variance.
      Without it PCA reports whichever hyperparameter happens to have the widest
      raw range: `n_estimators` over 10–500 would drown `min_samples_split` over
      2–10 whatever either did to the score. A column that never varied is left
      at zero rather than divided by zero.
    """
    names = list(trials[0].config.keys()) if trials else []
    logs = set(log_hyperparameters(config_space, names))

    columns: List[List[float]] = []
    labels: List[str] = []
    for name in names:
        raw = [t.config.get(name) for t in trials]
        if all(_is_numeric(v) for v in raw):
            if name in logs and all(v > 0 for v in raw):
                columns.append([math.log10(v) for v in raw])
                labels.append(f"log10({name})")
            else:
                columns.append([float(v) for v in raw])
                labels.append(name)
            continue
        for value in sorted({str(v) for v in raw}):
            columns.append([1.0 if str(v) == value else 0.0 for v in raw])
            labels.append(f"{name}={value}")

    return [list(row) for row in zip(*(_standardised(c) for c in columns))], labels


def _standardised(column: List[float]) -> List[float]:
    """One column at zero mean and unit variance; all zeros if it never varied."""
    mean = sum(column) / len(column)
    centred = [v - mean for v in column]
    spread = math.sqrt(sum(v * v for v in centred) / len(centred))
    return centred if spread == 0 else [v / spread for v in centred]


def project(config_space, trials, scores, method: str):
    """*trials* projected to at most `MAX_COMPONENTS` dimensions.

    Returns `(coordinates, labels, warning)` — one row of coordinates per trial,
    one label per component, and a warning instead of both when there is not
    enough to say anything.

    *method* is `"pca"` or `"pls"`. PLS is fitted against *scores*, which is
    what makes it the supervised one: its first component is the direction that
    moves the metric, not the direction the search happened to spread out in.

    A projection to *k* dimensions needs at least *k* directions to project
    from and enough points to find them — `min(n_trials - 1, n_features)` — so
    fewer are returned when that is all there is, and the caller draws what it
    gets. PCA's labels carry each component's share of the variance, since that
    is the one number that says whether the picture is worth reading. PLS has no
    comparable single figure and its labels say only which component they are.

    No seed: both are solved exactly rather than sampled, so there is nothing
    here for one to control. See `_pca` for the one place that could have gone
    otherwise.
    """
    if len(trials) < 3:
        return [], [], "Too few trials to project."

    if method not in METHODS:
        return [], [], f"No such projection: {method}."

    rows, features = encode_configurations(config_space, trials)
    if len(features) < 2:
        return [], [], "A projection needs more than one hyperparameter."
    # Every column standardised to nothing means every column never varied,
    # which means every trial ran the same configuration. There is no direction
    # to find, and asking for one gets a plane of NaNs rather than an error.
    if not any(any(value for value in row) for row in rows):
        return [], [], "Every trial ran the same configuration."

    wanted = min(MAX_COMPONENTS, len(trials) - 1, len(features))
    try:
        if method == "pls":
            return _pls(rows, scores, wanted)
        return _pca(rows, wanted)
    except Exception as exc:  # noqa: BLE001 — a failed projection is a caption
        return [], [], f"Could not project these trials: {exc}"


def _pca(rows, components: int):
    from sklearn.decomposition import PCA

    # An exact decomposition rather than sklearn's `auto`, which switches to a
    # randomized solver on a large enough matrix — and a randomized solver takes
    # a seed, which this would then have to be given and thread down from the
    # experiment. Exact is deterministic, and at the sizes here (a few hundred
    # features at the outside) `auto` would choose it anyway.
    model = PCA(n_components=components, svd_solver="full")
    coordinates = model.fit_transform(rows)
    labels = [f"PC {i + 1} ({share:.0%})"
              for i, share in enumerate(model.explained_variance_ratio_)]
    return [[float(v) for v in row] for row in coordinates], labels, None


def _pls(rows, scores, components: int):
    from sklearn.cross_decomposition import PLSRegression

    model = PLSRegression(n_components=components)
    model.fit(rows, list(scores))
    coordinates = model.transform(rows)
    return ([[float(v) for v in row] for row in coordinates],
            [f"PLS {i + 1}" for i in range(components)], None)
