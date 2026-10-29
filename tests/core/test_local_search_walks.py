"""`RecordedWalk` remembers what each candidate entry held, since SMAC's local
search keeps no record of its path.
"""

from types import SimpleNamespace

from ConfigSpace import ConfigurationSpace, Float

from core.optimizers.smac.walks import RecordedWalk, walks_to


def _configs(*values):
    """`_configs` builds one configuration for each value."""
    cs = ConfigurationSpace(seed=0)
    cs.add(Float("x", (0.0, 1.0)))
    from ConfigSpace import Configuration
    return [Configuration(cs, values={"x": v}) for v in values]


def test_every_configuration_an_entry_holds_is_kept_in_order():
    """A replacement extends the entry's path, and a plateau step to the same
    configuration adds nothing.
    """
    start_a, start_b, a, b = _configs(0.1, 0.9, 0.2, 0.3)
    sink = []
    walk = RecordedWalk([start_a, start_b], sink)

    walk[0] = a
    walk[0] = b
    walk[0] = b

    assert sink == [[start_a, a, b], [start_b]]
    assert list(walk) == [b, start_b]


def test_each_proposal_gets_the_path_that_ended_at_it_once():
    """`walks_to` gives each path to one proposal only, and a proposal no path
    reached gets None.
    """
    s1, s2, end, stranger = _configs(0.1, 0.2, 0.5, 0.8)
    paths = [[s1, end], [s2, end]]
    asked = [SimpleNamespace(config=c) for c in (end, end, stranger)]

    walks = walks_to(asked, paths)

    assert walks == [[{"x": 0.1}, {"x": 0.5}], [{"x": 0.2}, {"x": 0.5}], None]
