"""A tuned model as a file someone can take away and use.

`export_model` turns a model's source and one trial's configuration into a
single runnable script:

    # /// script                 the model's own PEP 723 header, plus pandas
    # ...
    # ///
    TASK = "classification"      what Codesigner tuned it for
    SEED = 0
    CONFIG = {...}               the selected trial's hyperparameters
    <the model's source>         unchanged but for where BaseModel comes from
    <a command-line runner>      train on a CSV, predict another, save the fit

`uv run model.py train.csv --predict new.csv` reads the header, builds an
environment with what the model needs and runs it — no Codesigner involved. The
file is still a Codesigner model too: the class still inherits `BaseModel`
where the SDK is installed, so the same file can be uploaded again.

Built-in models export from their own module source, which is why each of
those modules is one self-contained class with a header of its own: what is
exported is exactly the code that ran here.
"""

from __future__ import annotations

import math
import pprint
import re

from .model_source import _PEP723_BLOCK

#: Where `BaseModel` comes from in an exported file: the SDK when it is
#: installed, so the class is still a Codesigner model, and nothing when it is
#: not, since nothing outside Codesigner looks at the base class.
SHIM = '''\
try:
    from codesigner_model import BaseModel
except ImportError:  # Outside Codesigner the model needs no base class.
    BaseModel = object
'''

#: The import lines the shim replaces: a built-in's relative one, and the one
#: every uploaded model is told to write.
_BASE_IMPORT = re.compile(
    r"(?m)^from (?:\.base|\.models\.base|codesigner_model) import BaseModel[ \t]*$\n?")

#: `from __future__` imports, which have to stay the first statement of the file.
_FUTURE = re.compile(r"(?m)^from __future__ import [^\n]*\n")

#: What the runner needs beyond the model's own dependencies.
RUNNER_DEPENDENCIES = ("pandas",)

RUNNER = '''

# ── Running it ───────────────────────────────────────────────────────────────
#
# Reads CSVs the way Codesigner did: `;` or `,`, whichever the first lines use
# more of, with the last column as the target.

def _read_csv(path):
    import pandas as pd

    with open(path, "rb") as f:
        sample = f.read(2048).decode("utf-8", errors="replace")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(path, sep=sep)


def _name():
    name = getattr(MODEL_CLASS, "name", None)
    return name if isinstance(name, str) else MODEL_CLASS.__name__


def main(argv=None):
    import argparse
    import pickle
    import sys

    parser = argparse.ArgumentParser(
        description=f"{_name()}, with the configuration Codesigner tuned "
                    f"for {TASK}. Trains on the whole of TRAIN.")
    parser.add_argument("train", help="CSV to train on; the last column is the target")
    parser.add_argument("--predict", metavar="CSV",
                        help="CSV to predict. Its last column is ignored if it has as "
                             "many columns as TRAIN (the target), used otherwise.")
    parser.add_argument("--out", metavar="CSV",
                        help="where to write the predictions (default: standard output)")
    parser.add_argument("--save", metavar="FILE",
                        help="pickle the fitted model to FILE")
    args = parser.parse_args(argv)
    if not (args.predict or args.save):
        parser.error("nothing to do: give --predict, --save, or both")

    train = _read_csv(args.train)
    X, y = train.iloc[:, :-1].to_numpy(), train.iloc[:, -1].to_numpy()
    model = MODEL_CLASS()
    model.task = TASK

    if args.save:
        if not hasattr(model, "fit"):
            sys.exit(f"{_name()} has no fit(config, X, y, seed) method, so "
                     f"there is no fitted model to save. --predict still works.")
        with open(args.save, "wb") as f:
            pickle.dump(model.fit(CONFIG, X, y, SEED), f)

    if args.predict:
        new = _read_csv(args.predict)
        if new.shape[1] == train.shape[1]:
            new = new.iloc[:, :-1]
        predictions = model.fit_predict(CONFIG, X, y, new.to_numpy(), SEED)
        import pandas as pd

        frame = pd.DataFrame({train.columns[-1]: list(predictions)})
        frame.to_csv(args.out or sys.stdout, index=False)


if __name__ == "__main__":
    main()
'''


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
    for key, value in config.items():
        value = plain(value)
        if isinstance(value, float) and math.isnan(value):
            continue
        out[key] = value
    return out


def _with_dependencies(source: str, needed) -> str:
    """*source* with *needed* among its header's dependencies, adding a header
    if it has none."""
    match = _PEP723_BLOCK.search(source)
    if match is None:
        listed = ", ".join(f'"{d}"' for d in needed)
        header = ("# /// script\n"
                  '# requires-python = ">=3.11"\n'
                  f"# dependencies = [{listed}]\n"
                  "# ///\n")
        return header + source

    block = match.group(0)
    missing = [d for d in needed
               if not re.search(r'["\']%s(?:["\'\s<>=!~\[;])' % re.escape(d), block)]
    if not missing:
        return source
    listed = "".join(f'"{d}", ' for d in missing)
    if re.search(r"(?m)^#\s*dependencies\s*=\s*\[", block):
        updated = re.sub(r"(?m)^(#\s*dependencies\s*=\s*\[)", lambda m: m.group(1) + listed,
                         block, count=1)
    else:
        updated = block[:-len("# ///")] + f"# dependencies = [{listed.rstrip(', ')}]\n# ///"
    return source[:match.start()] + updated + source[match.end():]


def _split_header(source: str) -> tuple[str, str]:
    """(the PEP 723 header and anything before it, everything after it)."""
    match = _PEP723_BLOCK.search(source)
    if match is None:
        return "", source
    end = match.end()
    if source[end:end + 1] == "\n":
        end += 1
    return source[:end], source[end:]


def export_model(source: str, *, class_name: str, task: str, config: dict,
                 seed: int = 0, description: str = "") -> str:
    """The runnable file for *source*'s *class_name*, tuned to *config*.

    *description* is a line or two said at the top of the file — which
    experiment and trial this came from.
    """
    source = _with_dependencies(source, RUNNER_DEPENDENCIES)
    header, body = _split_header(source)
    body = _BASE_IMPORT.sub("", body)
    futures = "".join(_FUTURE.findall(body))
    body = _FUTURE.sub("", body)

    lines = []
    if description:
        lines += [f"# {line}".rstrip() for line in description.splitlines()]
        lines.append("")
    lines += [
        f"TASK = {task!r}",
        f"SEED = {int(seed)!r}",
        f"CONFIG = {pprint.pformat(configuration(config), sort_dicts=False)}",
        "",
    ]
    preamble = "\n".join(lines) + "\n" + SHIM + "\n"
    return (header + futures + "\n" + preamble + body.lstrip("\n").rstrip("\n") + "\n"
            + f"\n\nMODEL_CLASS = {class_name}\n" + RUNNER)
