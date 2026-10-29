"""`core.projection` projects trials onto two or three linear components, as the
configuration cube does with two raw axes. On standardised data any two axes
keep two of p units of variance, while PCA's first two components keep λ1 + λ2,
at least two. Standardising also makes a plotted distance approximate a real one
with the discarded variance as the error, and PLS keeps the direction that moves
the score most. The module only needs trials and their space, so it lives
outside `core/optimizers/` and tests without an optimizer.
"""

import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

#: Three, since only three can be drawn.
MAX_COMPONENTS = 3

#: The projections on offer.
METHODS = ("pca", "pls")


def log_hyperparameters(config_space, hp_names) -> list:
    """`log_hyperparameters` returns an empty list without a config space, as
    for a read-only custom model, which leaves every hyperparameter linear.
    """
    if config_space is None:
        return []
    return [h for h in hp_names
            if h in config_space and getattr(config_space[h], "log", False)]


def _is_numeric(value) -> bool:
    """`_is_numeric` excludes `bool`, because True and False are two categories
    and should never sit a unit apart.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Column(NamedTuple):
    """`Column` records the source hyperparameter, the one-hot value it stands
    for (None if numeric), and its standardisation."""

    label: str
    name: str
    value: Optional[str]
    log: bool
    mean: float
    spread: float


def encode_configurations(config_space, trials) -> Tuple[List[List[float]], List[Column]]:
    """`encode_configurations` returns `(rows, columns)`. A log-scaled
    hyperparameter enters as `log10`, since a log-uniform `C` over 0.01 to 100
    would otherwise crowd most trials together. A categorical or boolean becomes
    one-hot columns, because integer codes would invent the distances the
    projection draws. Every column is then standardised, so the widest raw range
    cannot dominate PCA, and a constant column stays at zero.
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
    """`encode_column` is the single encoding rule behind `Fitted.transform`. A
    numeric value is logged when the column is, and a missing or unloggable one
    takes the trials' mean; a one-hot column is 1 where the value matches.
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
    if column.value is not None:
        return values
    return [float(v) if _is_numeric(v) else float("nan") for v in values]


def encode_vectors(config_space, columns, X):
    """`encode_vectors` runs SMAC's vectorized rows through `encode_column`,
    after ConfigSpace's own transform decodes each cell. A NaN cell is an
    inactive hyperparameter.
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
    mean = sum(column) / len(column)
    centred = [v - mean for v in column]
    spread = math.sqrt(sum(v * v for v in centred) / len(centred))
    return (centred if spread == 0 else [v / spread for v in centred]), mean, spread


def project(config_space, trials, scores, method: str):
    """`project` returns `(coordinates, labels, warning)`, which is `fit`
    without the fitted model.
    """
    fitted, warning = fit(config_space, trials, scores, method)
    if fitted is None:
        return [], [], warning
    return fitted.coordinates, fitted.labels, None


def fit(config_space, trials, scores, method: str):
    """`fit` returns `(Fitted, None)`, or `(None, warning)` when the trials are
    too few or too alike. PLS fits against *scores*, so its first component
    moves the metric. The component count drops to `min(n_trials - 1,
    n_features)` when fewer directions exist, and only PCA's labels carry a
    variance share. Both methods have closed-form solutions and need no seed
    (see `_pca`).
    """
    if len(trials) < 3:
        return None, "Too few trials to project."

    if method not in METHODS:
        return None, f"No such projection: {method}."

    rows, columns = encode_configurations(config_space, trials)
    if len(columns) < 2:
        return None, "A projection needs more than one hyperparameter."
    # Identical configurations everywhere would yield a plane of NaNs instead of
    # an error.
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
    """`Fitted` keeps what it needs to place new configurations in the same
    picture."""

    coordinates: List[List[float]]
    labels: List[str]
    model: Any
    columns: List[Column]
    config_space: Any

    def transform(self, configurations) -> List[List[float]]:
        """`transform` encodes configurations as the trials were, so a
        configuration lands where its trial would. An unseen categorical value
        sets no column, and an inactive numeric sits at the trials' mean.
        """
        return [[float(v) for v in row]
                for row in self.model.transform(self._encoded(configurations))]

    def _encoded(self, configurations):
        import numpy as np

        by_name: Dict[str, list] = {}
        for column in self.columns:
            if column.name not in by_name:
                by_name[column.name] = [c.get(column.name) for c in configurations]
        encoded = [encode_column(column, _numeric_or_raw(column, by_name[column.name]))
                   for column in self.columns]
        return np.column_stack(encoded) if encoded else np.zeros((len(configurations), 0))

    def linear_map(self) -> dict:
        """`linear_map` exports the projection as numbers, so the page can place
        a proposed configuration where `transform` would. `matrix` and `offset`
        give the affine map from a standardised row to the components, with no
        approximation since both methods are linear.
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
        """`outline` returns the image of the search space's box, which stays a
        box through the encoding and becomes a convex shape under a linear map.
        The method builds a running Minkowski sum one hyperparameter at a time,
        only keeping the hull. The shape comes back as `[low, high]` in one
        dimension, as polygon vertices in two and as a triangle mesh in three,
        and `outline` returns None without a space.
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
        """`_generators` returns each hyperparameter's extreme encoded rows as
        offsets from the all-zero row."""
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
            # The empty row first, then each column on.
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
        """`_has_unseen_choice` detects a value that encodes as no column set."""
        if name not in self.config_space:
            return False
        hp = self.config_space[name]
        choices = getattr(hp, "choices", None) or getattr(hp, "sequence", None)
        return choices is not None and len(choices) > seen


def _hull_points(points):
    """`_hull_points` returns hull vertices, only ordered in two dimensions."""
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

    # `svd_solver="full"` stays deterministic, while `auto` may switch to a
    # seeded randomized solver on a large matrix.
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
