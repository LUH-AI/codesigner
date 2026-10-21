"""The seed rule a run is started with (experiments/services/run.py)."""

from experiments.services.run import resolve_seed


def test_resolve_seed_rule():
    """A non-negative seed is used verbatim; a negative one yields a valid random seed."""
    assert resolve_seed(0) == 0
    assert resolve_seed(1234) == 1234
    assert 0 <= resolve_seed(-1) <= 2**31 - 1
