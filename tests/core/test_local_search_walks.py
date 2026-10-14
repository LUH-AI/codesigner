"""Contract for recording the walks SMAC's local search takes
(core/optimizers/smac_optimizer.py: `_RecordedWalk`, `_walks_to`).

SMAC climbs by replacing entries of the candidate list it is handed and keeps
no record of the path. A list that remembers what each entry held recovers it,
and each proposal is then matched to the path that ended at it.
"""

from types import SimpleNamespace

from ConfigSpace import ConfigurationSpace, Float

from core.optimizers.smac_optimizer import _RecordedWalk, _walks_to


def _configs(*values):
    """Configurations of a one-hyperparameter space, one per value of x."""
    cs = ConfigurationSpace(seed=0)
    cs.add(Float("x", (0.0, 1.0)))
    from ConfigSpace import Configuration
    return [Configuration(cs, values={"x": v}) for v in values]


def test_every_configuration_an_entry_holds_is_kept_in_order():
    """Replacing an entry extends that entry's path; a replacement with the same
    configuration (a plateau step that went nowhere) adds nothing.

    Setup: two starting points, the first stepped twice and once to itself.
    Expect: the first path start -> a -> b, the second untouched, and both
    reachable from the sink handed in.
    """
    start_a, start_b, a, b = _configs(0.1, 0.9, 0.2, 0.3)
    sink = []
    walk = _RecordedWalk([start_a, start_b], sink)

    walk[0] = a
    walk[0] = b
    walk[0] = b

    assert sink == [[start_a, a, b], [start_b]]
    assert list(walk) == [b, start_b]


def test_each_proposal_gets_the_path_that_ended_at_it_once():
    """Proposals are matched to paths by where the path ended, in order, and a
    path is given to one proposal only; one no path ended at was not climbed to.

    Setup: two paths ending at the same configuration, and three proposals —
    that configuration twice and a stranger.
    Expect: the first two take one path each, the stranger gets None.
    """
    s1, s2, end, stranger = _configs(0.1, 0.2, 0.5, 0.8)
    paths = [[s1, end], [s2, end]]
    asked = [SimpleNamespace(config=c) for c in (end, end, stranger)]

    walks = _walks_to(asked, paths)

    assert walks == [[{"x": 0.1}, {"x": 0.5}], [{"x": 0.2}, {"x": 0.5}], None]
