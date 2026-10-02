"""What kind of thing each feature column of a dataset holds.

Three kinds, because three is what a model has to treat differently:

* **numeric** — a quantity. Passed through as it is, a missing value as NaN.
* **categorical** — a label from a set: text, or a yes/no. Encoded as an
  integer code per label (`core.encoding`), which is meaningless as a number,
  so a model told the column is categorical can one-hot it or split on it
  natively instead of ordering the labels alphabetically.
* **datetime** — a point in time. Expanded into calendar features (year,
  month, day, weekday, hour), each of which *is* a quantity.

Inferred from the column's values, never from its name. A number is always
numeric here, even a whole one with three values: a column of zip codes or
ratings is a decision someone looking at the data makes, and the Data page is
where they will make it. Inference only has to be right about what cannot be a
number.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

NUMERIC = "numeric"
CATEGORICAL = "categorical"
DATETIME = "datetime"

#: Every kind, in the order the Data page lists them.
KINDS = (NUMERIC, CATEGORICAL, DATETIME)

#: What a whole date has: a four-digit year, or day, month and year as three
#: numbers. Without one, a parser completes a time ("08:30"), a month ("May")
#: or a fraction ("1/2") into a date with today's year and day — features that
#: change with the day they are computed on.
WHOLE_DATE = re.compile(r"(?<!\d)[12]\d{3}(?!\d)|(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?!\d)")

#: The share of a text column's non-missing values that must read as dates
#: for the column to be a date. Not all of them: a hand-kept spreadsheet has
#: the odd "n/a" or "unknown" in a date column, and one such cell should not
#: turn a year of timestamps into a thousand categories.
DATE_SHARE = 0.9


def kind_of(series: pd.Series) -> str:
    """The kind of one column."""
    if pd.api.types.is_bool_dtype(series):
        return CATEGORICAL
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME
    if pd.api.types.is_numeric_dtype(series):
        return NUMERIC
    values = series.dropna()
    if values.empty:
        # Nothing to go on. Numeric is the kind that costs nothing: an empty
        # column encodes as NaN either way.
        return NUMERIC
    if _reads_as_dates(values):
        return DATETIME
    return CATEGORICAL


def infer_kinds(frame: pd.DataFrame) -> dict[str, str]:
    """``{column: kind}`` for every column of *frame*, in column order."""
    return {str(name): kind_of(frame[name]) for name in frame.columns}


def _reads_as_dates(values: pd.Series) -> bool:
    """Whether a text column is dates.

    A column of plain numbers stored as text is not: `to_datetime` reads
    "2019" as a year and "3" as nothing at all, and calling either a date
    would turn a mistyped numeric column into a calendar.
    """
    text = values.astype(str)
    if pd.to_numeric(text, errors="coerce").notna().mean() >= DATE_SHARE:
        return False
    with warnings.catch_warnings():
        # "Could not infer format" is pandas falling back to parsing each value
        # on its own, which is what a column of mixed formats needs anyway.
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    whole = text.str.contains(WHOLE_DATE)
    return (parsed.notna() & whole).mean() >= DATE_SHARE
