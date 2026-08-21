"""How numbers are written where a person reads them.

One rule across the whole app: **four significant figures, or fewer**. Not four
decimal places — 0.8 is 0.8, not 0.8000, and 0.000123456 is 0.0001235 rather
than 0.0001. Significance is what a reader is actually after; decimal places
give a score three digits it has not earned and give a small effect none at all.

This is a display rule and only a display rule. The full float is what is stored,
what the optimizer sees, and what goes into an exported .ihpo — see
`core/io.py`. Nothing here is ever fed back into a computation.
"""

#: The default, and the only one anything currently asks for. Named so the
#: reason for the number lives in one place rather than in every call.
SIGNIFICANT_DIGITS = 4


def sigfigs(value, digits: int = SIGNIFICANT_DIGITS) -> str:
    """*value* to *digits* significant figures, as short as it can honestly be.

    Anything that is not a number comes back as its own string: this is called
    on hyperparameter values, which are as often a kernel name or a boolean as
    they are a float. Integers keep every digit and gain thousands separators —
    truncating 128 estimators to 130 would be a different experiment, and an
    integer has no rounding to hide.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}g}"
