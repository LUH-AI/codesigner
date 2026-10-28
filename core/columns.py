"""`kind_of` sorts each feature column into numeric, categorical or datetime,
the three kinds a model treats differently. Categorical labels become integer
codes (`core.encoding`), and datetimes expand into calendar features. Inference
reads values, never names, and any number stays numeric; the Data page lets a
person reclassify zip codes or ratings.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

NUMERIC = "numeric"
CATEGORICAL = "categorical"
DATETIME = "datetime"

#: Kinds in Data page order.
KINDS = (NUMERIC, CATEGORICAL, DATETIME)

#: Without a four-digit year or a three-number date, a parser fills in today's
#: year and day, and the features change daily.
WHOLE_DATE = re.compile(r"(?<!\d)[12]\d{3}(?!\d)|(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?!\d)")

#: `DATE_SHARE` stays below 1 so a stray "n/a" keeps a date column a date.
DATE_SHARE = 0.9


def kind_of(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return CATEGORICAL
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME
    if pd.api.types.is_numeric_dtype(series):
        return NUMERIC
    values = series.dropna()
    if values.empty:
        # The numeric kind costs nothing, since an empty column encodes as NaN
        # either way.
        return NUMERIC
    if _reads_as_dates(values):
        return DATETIME
    return CATEGORICAL


def infer_kinds(frame: pd.DataFrame) -> dict[str, str]:
    return {str(name): kind_of(frame[name]) for name in frame.columns}


def _reads_as_dates(values: pd.Series) -> bool:
    """`_reads_as_dates` rejects numbers stored as text, since `to_datetime`
    reads "2019" as a year.
    """
    text = values.astype(str)
    if pd.to_numeric(text, errors="coerce").notna().mean() >= DATE_SHARE:
        return False
    with warnings.catch_warnings():
        # pandas warns about format inference when it parses each value alone,
        # as mixed formats need.
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    whole = text.str.contains(WHOLE_DATE)
    return (parsed.notna() & whole).mean() >= DATE_SHARE
