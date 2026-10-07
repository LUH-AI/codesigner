"""A tuned model as a file someone can take away and use.

What Codesigner delivers at the end of a search is a model and the
hyperparameters that suited this dataset. The exported file is exactly that,
in three parts, and nothing of the search:

    # /// script                  what it needs installed
    a docstring                   what it predicts; where the hyperparameters
                                  came from
    # ── The model ──             the model as Codesigner runs it: its
                                  delivered class (codesigner_model's
                                  OutputBaseModel and the model's own class)
    # ── The hyperparameters ──   HYPERPARAMETERS = {...}, the selected trial's
    # ── The data processing ──   Process: the dataset's columns read as
                                  Codesigner read them, then processed as the
                                  experiment says, fitted on the training rows
    # ── Running it ──            train on a CSV, predict another

`uv run model.py train.csv --predict new.csv` builds an environment with what
the header lists and runs it; Codesigner is not involved. No search space, no
SDK, no trial machinery: those are how the hyperparameters were found, not
part of what they are for.

The model's code is copied from where it runs here — the SDK's two classes and
the model's own module — so the delivered model is the same code that was
tuned, not a second rendering of it. A model uploaded by a person is their
own code: it is delivered as they wrote it, its search space taken out
(`export_upload`).
"""

from __future__ import annotations

import ast
import inspect
import math
import pprint
import re
import sys
import textwrap

from codesigner_model import ModelBase, OptimizableBaseModel, OutputBaseModel

from . import encoding, forecasters, processing
from .model_source import _PEP723_BLOCK
from .models import parts

#: The imports the delivered model's own code needs, before any module's.
_PREAMBLE_IMPORTS = ("from __future__ import annotations", "",
                     "from abc import ABC, abstractmethod", "from typing import Any", "",
                     "import numpy as np")

#: Top-level packages whose imports an exported file drops: it defines what it
#: needs of them itself, or (the search space) does not need it at all.
_DEFINED_HERE = ("core", "codesigner_model", "ConfigSpace")


def plain(value):
    """*value* as a Python literal `pprint` writes back readably: numpy scalars
    as the Python number or string they hold."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    return value


def configuration(config: dict) -> dict:
    """*config* as plain values, without inactive hyperparameters.

    A conditional hyperparameter that does not apply is absent from a
    configuration or stored as NaN, depending on where the configuration came
    from. Absent is what a model reading it with `.get` expects.
    """
    out = {}
    for key, value in dict(config).items():
        value = plain(value)
        if isinstance(value, float) and math.isnan(value):
            continue
        out[key] = value
    return out


# ── Collecting the delivered model's code ──────────────────────────────────

def output_class(model) -> type:
    """The class *model* is delivered as: the one its tuned class adds a search
    space to (`core.models.base.Tunable`)."""
    for cls in type(model).__mro__:
        if (issubclass(cls, OutputBaseModel) and not issubclass(cls, OptimizableBaseModel)
                and cls is not OutputBaseModel):
            return cls
    raise ValueError(f"{type(model).__name__} has no delivered class")


def _is_local_import(node) -> bool:
    return isinstance(node, ast.ImportFrom) and bool(
        node.level or (node.module or "").split(".")[0] in _DEFINED_HERE)


def _module_body(module, only=None) -> tuple[list[str], str]:
    """*module*'s source as (its imports, the rest), without its docstring or
    any import of what the exported file defines itself — and, given *only*,
    without any definition not named in it."""
    text = inspect.getsource(module)
    tree, lines = ast.parse(text), text.splitlines()
    imports, keep = [], []
    for node in tree.body:
        chunk = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if node is tree.body[0] and isinstance(node, ast.Expr) \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not _is_local_import(node):
                imports.append(chunk)
            continue
        if only is not None and getattr(node, "name", None) not in only:
            continue
        start = node.lineno - 1
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        keep.append("\n".join(lines[start:node.end_lineno]))
    return imports, "\n\n\n".join(keep)


def _names_in(source: str) -> set[str]:
    """Every name *source* refers to."""
    return {node.id for node in ast.walk(ast.parse(textwrap.dedent(source)))
            if isinstance(node, ast.Name)}


def _class_and_constants(cls) -> tuple[list[str], str]:
    """*cls*'s source with the module-level constants it names, and its
    module's imports."""
    module_text = inspect.getsource(sys.modules[cls.__module__])
    tree, lines = ast.parse(module_text), module_text.splitlines()
    source = textwrap.dedent(inspect.getsource(cls))
    named = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    constants, imports = [], []
    for node in tree.body:
        chunk = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in named for t in node.targets):
            start = node.lineno - 1
            while start > 0 and lines[start - 1].lstrip().startswith("#:"):
                start -= 1
            constants.append("\n".join(lines[start:node.end_lineno]))
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and not _is_local_import(node):
            imports.append(chunk)
    return imports, "\n\n".join(constants + [source])


def _sdk_classes() -> str:
    """The SDK's root and delivered-model classes, as the exported file's own."""
    return "\n\n\n".join(textwrap.dedent(inspect.getsource(cls))
                         for cls in (ModelBase, OutputBaseModel))


def _header(dependencies) -> str:
    listed = ", ".join(f'"{d}"' for d in dict.fromkeys(dependencies))
    return ("# /// script\n"
            '# requires-python = ">=3.11"\n'
            f"# dependencies = [{listed}]\n"
            "# ///\n")


def _literal(name: str, value) -> str:
    return f"{name} = {pprint.pformat(value, sort_dicts=False)}"


def _section(title: str) -> str:
    return f"# ── {title} " + "─" * max(3, 74 - len(title)) + "\n\n"


def _docstring(about: str) -> str:
    return f'"""{about.strip()}\n"""\n\n' if about else ""


# ── Exporting a built-in ────────────────────────────────────────────────────

def export_model(model, *, config: dict, task: str, seed: int = 0, columns: dict | None = None,
                 plan: dict | None = None, about: str = "", forecast: dict | None = None,
                 time_column: str = "", series_column: str = "") -> str:
    """The runnable file for built-in *model* tuned to *config*.

    *model* is the model as the registry has it — a tuned built-in, or one
    forecasting by reduction (`core.forecasting.Reduced`). *columns* is how the
    dataset's columns are read (`core.encoding.fit`; None for a dataset of
    numbers), *plan* the experiment's processing as resolved for this model
    (`core.processing.resolve`). *about* is the docstring: what it predicts and
    where the hyperparameters came from. A forecast (*forecast* given, see
    `core.forecasting.context`) gets the forecasting runner.
    """
    from .forecasting import PREFIX, Reduced

    reduced = isinstance(model, Reduced)
    inner = model.model if reduced else model
    cls = output_class(inner)
    hyperparameters = configuration(config)
    lags = int(hyperparameters.pop(f"{PREFIX}lags", 3)) if reduced else None

    # The processing's two classes only: choosing a plan is the experiment's
    # business, and the plan is written out below.
    modules = [(processing, ("Processing", "TextCategories"))]
    used = _names_in(inspect.getsource(cls))
    shared = [name for name, _ in inspect.getmembers(parts) if name in used
              and getattr(getattr(parts, name), "__module__", "") == parts.__name__]
    if shared:
        modules.append((parts, tuple(shared)))
    if forecast is not None:
        modules.append((forecasters, None))
    class_imports, class_source = _class_and_constants(cls)
    imports, bodies = list(class_imports), []
    for module, only in modules:
        more, body = _module_body(module, only)
        imports += more
        bodies.append(body)
    imports = [line for line in dict.fromkeys(imports) if line != "import numpy as np"]

    dependencies = ["numpy", "pandas", "scikit-learn"] + list(getattr(inner, "dependencies", ()))
    out = [_header(dependencies), _docstring(about),
           "\n".join(_PREAMBLE_IMPORTS + tuple(imports)) + "\n\n\n"]

    out.append(_section("The model"))
    out.append(_sdk_classes() + "\n\n\n")
    for body in bodies[1:]:
        out.append(body + "\n\n\n")
    out.append(class_source + "\n\n")

    out.append(_section("The hyperparameters"))
    out.append(_literal("HYPERPARAMETERS", hyperparameters) + "\n")
    if reduced:
        out.append("#: How many of a series' last values each forecast is made from.\n"
                   f"LAGS = {lags}\n")
    out.append("\n\n")

    out.append(_section("The data processing"))
    out.append("# The dataset's columns read as Codesigner read them (text as label codes,\n"
               "# dates as calendar parts), then processed as the experiment said, fitted\n"
               "# on the training rows only.\n\n")
    out.append(_literal("COLUMNS", columns if columns and not encoding.is_plain(columns) else None)
               + "\n")
    out.append(_literal("PROCESSING", dict(plan or {})) + "\n")
    if forecast is not None:
        out.append(_literal("FORECAST", dict(forecast)) + "\n")
        out.append(f"TIME_COLUMN = {time_column!r}\nSERIES_COLUMN = {series_column!r}\n")
    out.append("\n\n" + bodies[0] + "\n\n\n")
    out.append(textwrap.dedent(inspect.getsource(encoding.encode)) + "\n")
    out.append(PROCESS.replace("{forecast}", "True" if forecast is not None else "False"))

    out.append("\n\n" + _section("Running it"))
    runner = FORECAST_RUNNER if forecast is not None else RUNNER
    build = (REDUCED_FORECASTER if reduced else SERIES_FORECASTER) if forecast is not None else ""
    out.append(runner.replace("{build}", build).replace("{cls}", cls.__name__)
               .replace("{task}", repr(task)).replace("{seed}", repr(int(seed))))
    return "".join(out)


#: The third part: the dataset's columns as the model was tuned on them.
PROCESS = '''

def _kinds(columns, n):
    """The kind of each column `encode` returns for *columns*: a label's code
    is categorical, everything else — a calendar part included — numeric."""
    if columns is None:
        return ("numeric",) * n
    kinds = []
    for column in columns["columns"]:
        kinds += (["numeric"] * len(column["parts"]) if column["kind"] == "datetime"
                  else [column["kind"]])
    return tuple(kinds)


class Process:
    """The data processing these hyperparameters were tuned with.

    `fit` learns it from the training rows — the medians, the scale, the
    labels — and `transform` applies it to any rows with the same columns,
    matched by name. `kinds` is what each processed column is, for the model.
    """

    forecast = {forecast}

    def _columns(self, frame):
        if COLUMNS is None:
            return frame.to_numpy(dtype=float)
        return encode(COLUMNS, frame)

    def fit(self, frame):
        X = self._columns(frame)
        kinds = _kinds(COLUMNS, X.shape[1])
        # A forecast's own columns are a series' times: what is processed is
        # the examples the forecaster makes from them (ReducedForecaster).
        self.steps = Processing({} if self.forecast else PROCESSING, kinds).fit(X)
        self.kinds = self.steps.kinds_out() or ("numeric",) * self.steps.transform(X).shape[1]
        return self

    def transform(self, frame):
        return self.steps.transform(self._columns(frame))
'''

_READ_CSV = '''
def _read_csv(path):
    """A CSV read the way Codesigner read it: `;` or `,`, whichever the first
    lines use more of."""
    import pandas as pd

    with open(path, "rb") as f:
        sample = f.read(2048).decode("utf-8", errors="replace")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(path, sep=sep)
'''

#: Train on a CSV, predict another.
RUNNER = _READ_CSV + '''

def main(argv=None):
    import argparse
    import sys

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0]
                                     + " Trains on the whole of TRAIN.")
    parser.add_argument("train", help="CSV to train on; the last column is the target")
    parser.add_argument("--predict", metavar="CSV", required=True,
                        help="CSV to predict. Its last column is ignored if it has as "
                             "many columns as TRAIN (the target), used otherwise.")
    parser.add_argument("--out", metavar="CSV",
                        help="where to write the predictions (default: standard output)")
    args = parser.parse_args(argv)

    train = _read_csv(args.train)
    train = train[train.iloc[:, -1].notna()]
    process = Process().fit(train.iloc[:, :-1])
    model = {cls}()
    model.task = {task}
    fitted = model.fit(HYPERPARAMETERS, process.transform(train.iloc[:, :-1]),
                       train.iloc[:, -1].to_numpy(), seed={seed}, kinds=process.kinds)

    new = _read_csv(args.predict)
    if new.shape[1] == train.shape[1]:
        new = new.iloc[:, :-1]
    predictions = model.predict(fitted, process.transform(new))
    pd.DataFrame({train.columns[-1]: list(predictions)}).to_csv(args.out or sys.stdout,
                                                              index=False)


if __name__ == "__main__":
    main()
'''

#: The forecaster for a model that forecasts by itself (ETS, Seasonal Naive).
SERIES_FORECASTER = '''

def forecaster(forecast, kinds):
    model = {cls}()
    model.task = {task}
    model.forecast = forecast
    return model.build(HYPERPARAMETERS, {"kinds": kinds}, seed={seed})
'''

#: The forecaster for a model that forecasts by reduction.
REDUCED_FORECASTER = '''

def forecaster(forecast, kinds):
    model = {cls}()
    model.task = {task}
    model.feature_kinds = kinds
    return ReducedForecaster(model=model, hyperparameters=HYPERPARAMETERS, lags=LAGS,
                             forecast=forecast, plan=PROCESSING, task={task}, seed={seed})
'''

#: Read a history, forecast what comes after it.
FORECAST_RUNNER = _READ_CSV + '''

def _times(values):
    import pandas as pd

    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _next_times(times, horizon):
    """The *horizon* times after the last of *times*, at their spacing."""
    import pandas as pd

    times = pd.Series(times).dropna().drop_duplicates().sort_values()
    if pd.api.types.is_numeric_dtype(times):
        step = times.diff().median() if len(times) > 1 else 1
        return [times.iloc[-1] + step * (k + 1) for k in range(horizon)]
    index = pd.DatetimeIndex(times)
    frequency = pd.infer_freq(index) if len(index) >= 3 else None
    if frequency:
        return list(pd.date_range(index[-1], periods=horizon + 1, freq=frequency)[1:])
    step = index.to_series().diff().median()
    return [index[-1] + step * (k + 1) for k in range(horizon)]
{build}

def main(argv=None):
    import argparse
    import sys

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0]
                                     + " Learns from all of HISTORY and forecasts what "
                                       "comes after it.")
    parser.add_argument("history", help="CSV of the series so far; the last column is "
                                        "the target, " + repr(TIME_COLUMN) + " its time")
    parser.add_argument("--horizon", type=int, default=FORECAST["horizon"],
                        help="how many steps ahead to forecast (default: the tuned %(default)s)")
    parser.add_argument("--out", metavar="CSV",
                        help="where to write the forecast (default: standard output)")
    args = parser.parse_args(argv)

    history = _read_csv(args.history)
    history = history[history.iloc[:, -1].notna()]
    when = _times(history[TIME_COLUMN])
    history = history[when.notna()].iloc[when[when.notna()].argsort(kind="stable")]
    target = history.columns[-1]

    # Columns other than the time and the series are not known in the future:
    # they are left empty, and the forecast is made without them.
    groups = (history.groupby(SERIES_COLUMN, sort=False) if SERIES_COLUMN
              else [(None, history)])
    future = []
    for key, rows in groups:
        for moment in _next_times(_times(rows[TIME_COLUMN]), args.horizon):
            row = {column: None for column in history.columns[:-1]}
            row[TIME_COLUMN] = moment
            if SERIES_COLUMN:
                row[SERIES_COLUMN] = key
            future.append(row)
    future = pd.DataFrame(future, columns=history.columns[:-1])

    process = Process().fit(history.iloc[:, :-1])
    # From the end of the history: nothing is skipped between it and the forecast.
    model = forecaster(dict(FORECAST, horizon=args.horizon, gap=0), process.kinds)
    model.fit(process.transform(history.iloc[:, :-1]), history.iloc[:, -1].to_numpy())
    forecast = model.predict(process.transform(future))

    out = future[[TIME_COLUMN] + ([SERIES_COLUMN] if SERIES_COLUMN else [])].copy()
    out[target] = list(forecast)
    out.to_csv(args.out or sys.stdout, index=False)


if __name__ == "__main__":
    main()
'''


# ── Exporting an uploaded model ─────────────────────────────────────────────

#: Where `BaseModel` comes from in an exported upload: the SDK when it is
#: installed, and nothing when it is not, since nothing outside Codesigner
#: looks at the base class.
SHIM = '''\
try:
    from codesigner_model import BaseModel
except ImportError:  # Outside Codesigner the model needs no base class.
    BaseModel = object
'''

#: The import line the shim replaces, the one every uploaded model is told to write.
_BASE_IMPORT = re.compile(r"(?m)^from codesigner_model import BaseModel[ \t]*$\n?")

#: `from __future__` imports, which have to stay the first statement of the file.
_FUTURE = re.compile(r"(?m)^from __future__ import [^\n]*\n")

UPLOAD_RUNNER = _READ_CSV + '''

def _features(frame):
    if COLUMNS is None:
        return frame.to_numpy()
    return encode(COLUMNS, frame)


def main(argv=None):
    import argparse
    import sys

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0]
                                     + " Trains on the whole of TRAIN.")
    parser.add_argument("train", help="CSV to train on; the last column is the target")
    parser.add_argument("--predict", metavar="CSV", required=True,
                        help="CSV to predict. Its last column is ignored if it has as "
                             "many columns as TRAIN (the target), used otherwise.")
    parser.add_argument("--out", metavar="CSV",
                        help="where to write the predictions (default: standard output)")
    args = parser.parse_args(argv)

    train = _read_csv(args.train)
    train = train[train.iloc[:, -1].notna()]
    X, y = _features(train.iloc[:, :-1]), train.iloc[:, -1].to_numpy()
    model = {cls}()
    model.task = {task}
    model.feature_kinds = _kinds(COLUMNS, X.shape[1])

    new = _read_csv(args.predict)
    if new.shape[1] == train.shape[1]:
        new = new.iloc[:, :-1]
    predictions = model.fit_predict(HYPERPARAMETERS, X, y, _features(new), {seed})
    pd.DataFrame({train.columns[-1]: list(predictions)}).to_csv(args.out or sys.stdout,
                                                              index=False)


if __name__ == "__main__":
    main()
'''


def _without_search_space(body: str, class_name: str) -> str:
    """*body* with *class_name*'s `get_config_space` taken out."""
    tree = ast.parse(body)
    lines = body.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "get_config_space":
                    start = (item.decorator_list[0].lineno if item.decorator_list
                             else item.lineno) - 1
                    del lines[start:item.end_lineno]
                    return "".join(lines)
    return body


def export_upload(source: str, *, class_name: str, task: str, config: dict, seed: int = 0,
                  columns: dict | None = None, about: str = "") -> str:
    """The runnable file for a person's own model, tuned to *config*: their
    code as they wrote it, its search space taken out, the hyperparameters and
    how the columns are read beside it."""
    match = _PEP723_BLOCK.search(source)
    if match:
        header, body = source[:match.end()] + "\n", source[match.end():]
        if not re.search(r"""["']pandas["'\s<>=!~\[;]""", header):
            if re.search(r"(?m)^#\s*dependencies\s*=\s*\[", header):
                header = re.sub(r"(?m)^(#\s*dependencies\s*=\s*\[)", r'\1"pandas", ', header,
                                count=1)
            else:
                header = header.replace("# ///\n\n", '# dependencies = ["pandas"]\n# ///\n\n', 1)
    else:
        header, body = _header(["pandas", "numpy"]), source
    body = _BASE_IMPORT.sub("", body)
    futures = "".join(_FUTURE.findall(body))
    body = _without_search_space(_FUTURE.sub("", body), class_name)
    # The docstring first: it may come before `from __future__`, and is only
    # the module's docstring if it does.
    return (header + _docstring(about) + futures + "\n"
            + SHIM + "\n" + body.strip("\n") + "\n\n\n"
            + _section("The hyperparameters")
            + _literal("HYPERPARAMETERS", configuration(config)) + "\n\n\n"
            + _section("The data processing")
            + _literal("COLUMNS", columns if columns and not encoding.is_plain(columns) else None)
            + "\n\n\n" + textwrap.dedent(inspect.getsource(encoding.encode))
            + PROCESS.split("class Process:")[0]
            + "\n" + _section("Running it")
            + UPLOAD_RUNNER.replace("{cls}", class_name).replace("{task}", repr(task))
            .replace("{seed}", repr(int(seed))))
