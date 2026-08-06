"""Everything that only matters when Codesigner is hosted for other people.

Kept in its own package because the local and private-network cases — which are
most of them — should not have to know it exists. It is always installed and
inert by default; `REQUIRE_LOGIN` is the one switch that wakes it up.
"""
