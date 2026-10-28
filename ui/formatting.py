"""`ui.formatting` writes numbers for display at four significant figures, so
0.000123456 shows as 0.0001235 instead of 0.0001. Stored values, optimizer
inputs and .ihpo exports keep the full float.
"""

SIGNIFICANT_DIGITS = 4


def sigfigs(value, digits: int = SIGNIFICANT_DIGITS) -> str:
    """`sigfigs` returns a non-number as its own string, since hyperparameter
    values are often kernel names or booleans. Integers keep every digit with
    thousands separators, because rounding 128 estimators to 130 changes the
    experiment.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}g}"


def human(n: int) -> str:
    """`human` formats bytes readably."""
    for unit, size in (("TB", 1024 ** 4), ("GB", 1024 ** 3), ("MB", 1024 ** 2), ("kB", 1024)):
        if n >= size:
            return f"{n / size:.1f} {unit}".replace(".0 ", " ")
    return f"{n} B"
