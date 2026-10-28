"""`resolve_seed` decides each run's seed."""

from experiments.services.run import resolve_seed


def test_resolve_seed_rule():
    """`resolve_seed` keeps seeds of zero or more and randomises negatives."""
    assert resolve_seed(0) == 0
    assert resolve_seed(1234) == 1234
    assert 0 <= resolve_seed(-1) <= 2**31 - 1
