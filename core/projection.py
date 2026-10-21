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
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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


class Column(NamedTuple):
    """One column of the encoded matrix, and what it takes to read it back:
    which hyperparameter it came from, and the value it stands for when it is
    one of a categorical's one-hot columns (None for a numeric one)."""

    label: str
    name: str
    value: Optional[str]
    log: bool
    mean: float
    spread: float


def encode_configurations(config_space, trials) -> Tuple[List[List[float]], List[Column]]:
    """*trials*' configurations as a numeric matrix, one row each.

    Returns `(rows, columns)`. Three things happen to a configuration on
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
    specs: List[Tuple[str, str, Optional[str], bool]] = []
    for name in names:
        raw = [t.config.get(name) for t in trials]
        if all(_is_numeric(v) for v in raw):
            if name in logs and all(v > 0 for v in raw):
                columns.append([math.log10(v) for v in raw])
                specs.append((f"log10({name})", name, None, True))
            else:
                columns.append([float(v) for v in raw])
                specs.append((name, name, None, False))
            continue
        for value in sorted({str(v) for v in raw}):
            columns.append([1.0 if str(v) == value else 0.0 for v in raw])
            specs.append((f"{name}={value}", name, value, False))

    standardised, described = [], []
    for column, spec in zip(columns, specs):
        values, mean, spread = _standardised(column)
        standardised.append(values)
        described.append(Column(*spec, mean, spread))
    return [list(row) for row in zip(*standardised)], described


def encode_column(column: "Column", values):
    """One encoded column for many configurations — the rule `Fitted.transform`
    encodes by, in one place, for whichever form the values arrive in.

    *values* holds the hyperparameter's own values: a float array for a numeric
    column (NaN where it is inactive or not a number), anything with a `str`
    for a categorical's one-hot column. A numeric value goes on logged when the
    column is, and at the trials' mean when it is missing or cannot be logged;
    a categorical column is 1 where the value is the one it stands for. Then
    the trials' standardisation.
    """
    import numpy as np

    if column.value is not None:
        raw = np.array([str(v) == column.value for v in values], dtype=float)
    else:
        numbers = np.asarray(values, dtype=float)
        usable = np.isfinite(numbers) & ((numbers > 0) if column.log else True)
        with np.errstate(divide="ignore", invalid="ignore"):
            measured = np.log10(numbers) if column.log else numbers
        raw = np.where(usable, measured, column.mean)
    if not column.spread:
        return np.zeros(len(raw))
    return (raw - column.mean) / column.spread


def _numeric_or_raw(column, values):
    """Configuration values as `encode_column` takes them: floats (NaN for
    anything that is not a number) for a numeric column, as they are for a
    categorical one."""
    if column.value is not None:
        return values
    return [float(v) if _is_numeric(v) else float("nan") for v in values]


def encode_vectors(config_space, columns, X):
    """Configurations in ConfigSpace's vectorized form — SMAC's arrays, one row
    each and one column per hyperparameter in the space's order — encoded the
    way `Fitted.transform` encodes dicts, through the same `encode_column`.

    A vectorized cell goes back to the hyperparameter's own value by
    ConfigSpace's own transform; a NaN cell is an inactive hyperparameter,
    which a dict would simply not have.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    order = list(config_space.keys())
    values: Dict[str, object] = {}
    encoded = []
    for column in columns:
        if column.name not in values:
            if column.name not in config_space:
                values[column.name] = [None] * len(X)
            else:
                hp = config_space[column.name]
                cells = X[:, order.index(column.name)]
                active = np.isfinite(cells)
                if column.value is not None:
                    decoded = np.full(len(cells), None, dtype=object)
                    if active.any():
                        decoded[active] = hp.to_value(cells[active])
                    values[column.name] = decoded
                else:
                    decoded = np.full(len(cells), np.nan)
                    if active.any():
                        decoded[active] = np.asarray(hp.to_value(cells[active]), dtype=float)
                    values[column.name] = decoded
        encoded.append(encode_column(column, values[column.name]))
    return np.column_stack(encoded) if encoded else np.zeros((len(X), 0))


def _standardised(column: List[float]) -> Tuple[List[float], float, float]:
    """One column at zero mean and unit variance — all zeros if it never varied
    — with the mean and spread it was standardised by."""
    mean = sum(column) / len(column)
    centred = [v - mean for v in column]
    spread = math.sqrt(sum(v * v for v in centred) / len(centred))
    return (centred if spread == 0 else [v / spread for v in centred]), mean, spread


def project(config_space, trials, scores, method: str):
    """*trials* projected to at most `MAX_COMPONENTS` dimensions.

    Returns `(coordinates, labels, warning)` — one row of coordinates per trial,
    one label per component, and a warning instead of both when there is not
    enough to say anything. `fit` with the fitted projection dropped.
    """
    fitted, warning = fit(config_space, trials, scores, method)
    if fitted is None:
        return [], [], warning
    return fitted.coordinates, fitted.labels, None


def fit(config_space, trials, scores, method: str):
    """*trials* projected to at most `MAX_COMPONENTS` dimensions, keeping what
    it takes to go back.

    Returns `(Fitted, None)`, or `(None, warning)` when there is not enough to
    say anything.

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
        return None, "Too few trials to project."

    if method not in METHODS:
        return None, f"No such projection: {method}."

    rows, columns = encode_configurations(config_space, trials)
    if len(columns) < 2:
        return None, "A projection needs more than one hyperparameter."
    # Every column standardised to nothing means every column never varied,
    # which means every trial ran the same configuration. There is no direction
    # to find, and asking for one gets a plane of NaNs rather than an error.
    if not any(any(value for value in row) for row in rows):
        return None, "Every trial ran the same configuration."

    wanted = min(MAX_COMPONENTS, len(trials) - 1, len(columns))
    try:
        if method == "pls":
            coordinates, labels, model = _pls(rows, scores, wanted)
        else:
            coordinates, labels, model = _pca(rows, wanted)
    except Exception as exc:  # noqa: BLE001 — a failed projection is a caption
        return None, f"Could not project these trials: {exc}"
    return Fitted(coordinates, labels, model, columns, config_space), None


class Fitted(NamedTuple):
    """A projection that remembers how it was made, so that configurations
    other than the trials it was fitted on can be put in the same picture."""

    coordinates: List[List[float]]
    labels: List[str]
    model: Any
    columns: List[Column]
    config_space: Any

    def transform(self, configurations) -> List[List[float]]:
        """*configurations* (dicts) in this projection's coordinates.

        Encoded exactly as the trials were — the same columns, logged, one-hot
        and standardised by the trials' own mean and spread — so a
        configuration lands where a trial running it would have. A value the
        trials never took in a categorical sets none of its columns, and a
        numeric hyperparameter the configuration leaves inactive sits at the
        trials' mean.
        """
        return [[float(v) for v in row]
                for row in self.model.transform(self._encoded(configurations))]

    def _encoded(self, configurations):
        """*configurations* (dicts) as the standardised rows the model takes."""
        import numpy as np

        by_name: Dict[str, list] = {}
        for column in self.columns:
            if column.name not in by_name:
                by_name[column.name] = [c.get(column.name) for c in configurations]
        encoded = [encode_column(column, _numeric_or_raw(column, by_name[column.name]))
                   for column in self.columns]
        return np.column_stack(encoded) if encoded else np.zeros((len(configurations), 0))

    def linear_map(self) -> dict:
        """This projection as plain numbers, so the page can place a
        configuration itself — a candidate the optimizer has just proposed,
        say — exactly where `transform` would.

        `columns` says how each value is encoded (the hyperparameter, the
        category it stands for or None, whether it is logged, and the mean and
        spread it is standardised by); `matrix` and `offset` are the affine map
        from that standardised row to the components. Both methods are linear,
        so this is the projection, not an approximation of it.
        """
        import numpy as np

        width = len(self.columns)
        offset = np.asarray(self.model.transform([[0.0] * width])[0])
        units = np.asarray(self.model.transform(np.eye(width)))
        return {
            "columns": [{"name": c.name, "value": c.value, "log": c.log,
                         "mean": c.mean, "spread": c.spread} for c in self.columns],
            "matrix": (units - offset).T.tolist(),
            "offset": offset.tolist(),
        }

    def outline(self, dimensions: int):
        """Where a legal configuration can land, in the first *dimensions*
        components: the image of the search space's box.

        The box stays a box through the encoding (a log scale maps an interval
        to an interval, and standardising is a shift and a stretch), and a
        linear projection takes a box to a convex shape — an interval, a
        polygon, a polyhedron — whose edge is the edge of the legal space. A
        categorical contributes the points its choices project to rather than
        an interval. Built as a running Minkowski sum, one hyperparameter at a
        time, keeping only the hull, so it stays small however many there are.

        Returns `[low, high]` in one dimension, the polygon's vertices in order
        in two, and `{"x", "y", "z", "i", "j", "k"}` — the hull as a triangle
        mesh — in three. None when there is no search space to bound.
        """
        import numpy as np

        if self.config_space is None:
            return None
        width = len(self.columns)
        origin = np.asarray(self.model.transform([[0.0] * width])[0][:dimensions])

        def image(unit_row):
            return np.asarray(self.model.transform([unit_row])[0][:dimensions]) - origin

        points = np.array([origin])
        for generators in self._generators().values():
            images = [image(encoded) for encoded in generators]
            sums = (points[:, None, :] + np.array(images)[None, :, :]).reshape(-1, dimensions)
            points = _hull_points(sums)

        if dimensions == 1:
            return [float(points.min()), float(points.max())]
        if dimensions == 2:
            return [[float(x), float(y)] for x, y in points]
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points, qhull_options="QJ")
        return {"x": points[:, 0].tolist(), "y": points[:, 1].tolist(), "z": points[:, 2].tolist(),
                "i": hull.simplices[:, 0].tolist(), "j": hull.simplices[:, 1].tolist(),
                "k": hull.simplices[:, 2].tolist()}

    def _generators(self) -> Dict[str, List[List[float]]]:
        """Each hyperparameter's extreme encoded rows, as offsets from the
        all-zero row: a numeric one's two ends, a categorical's one-hot rows
        (and the row with none set, for a value the trials never took)."""
        width = len(self.columns)
        grouped: Dict[str, List[int]] = {}
        for index, column in enumerate(self.columns):
            grouped.setdefault(column.name, []).append(index)

        generators = {}
        for name, indices in grouped.items():
            first = self.columns[indices[0]]
            if first.value is None:
                hp = self.config_space[name] if name in self.config_space else None
                lower, upper = getattr(hp, "lower", None), getattr(hp, "upper", None)
                if lower is None or upper is None:
                    continue
                rows = []
                for end in (lower, upper):
                    raw = math.log10(end) if first.log and end > 0 else float(end)
                    row = [0.0] * width
                    row[indices[0]] = (raw - first.mean) / first.spread if first.spread else 0.0
                    rows.append(row)
                generators[name] = rows
                continue
            rows = []
            # No column of the group set, then each of them on in turn.
            for chosen in [None] + list(range(len(indices))):
                row = [0.0] * width
                for position, i in enumerate(indices):
                    column = self.columns[i]
                    raw = 1.0 if position == chosen else 0.0
                    row[i] = (raw - column.mean) / column.spread if column.spread else 0.0
                rows.append(row)
            if not self._has_unseen_choice(name, len(indices)):
                rows = rows[1:]
            generators[name] = rows
        return generators

    def _has_unseen_choice(self, name, seen: int) -> bool:
        """Whether *name* can take a value no trial took — which encodes as no
        column of its group set."""
        if name not in self.config_space:
            return False
        hp = self.config_space[name]
        choices = getattr(hp, "choices", None) or getattr(hp, "sequence", None)
        return choices is not None and len(choices) > seen


def _hull_points(points):
    """*points*' convex hull vertices: all of them in one dimension's two ends,
    in order around the hull in two, as a set in three."""
    import numpy as np

    dimensions = points.shape[1]
    if dimensions == 1:
        return np.array([[points.min()], [points.max()]])
    unique = np.unique(np.round(points, 12), axis=0)
    if len(unique) <= dimensions:
        return unique
    from scipy.spatial import ConvexHull
    try:
        return unique[ConvexHull(unique, qhull_options="QJ").vertices]
    except Exception:  # noqa: BLE001 — degenerate: everything on a line or a plane
        return unique


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
    return [[float(v) for v in row] for row in coordinates], labels, model


def _pls(rows, scores, components: int):
    from sklearn.cross_decomposition import PLSRegression

    model = PLSRegression(n_components=components)
    model.fit(rows, list(scores))
    coordinates = model.transform(rows)
    return ([[float(v) for v in row] for row in coordinates],
            [f"PLS {i + 1}" for i in range(components)], model)
