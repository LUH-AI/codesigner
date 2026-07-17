"""Step 3 TDD: contracts for the row ↔ snapshot adapter (web/services/snapshot.py).

Written before the adapter exists — imports are deferred into the tests so
the suite collects while these fail. The adapter is the single seam between
the .ihpo format and the database: import, export, and run reconstruction
all go through it, so it must preserve exactly what the snapshot-contract
tests say a store preserves.
"""

import pytest

from core import io
from core.version import VERSION

from .conftest import FIXTURES_DIR
from .test_snapshot_contract import IDENTITY_KEYS, SNAPSHOT_KEYS


def _fixture_snapshot(name: str) -> dict:
    return io.parse((FIXTURES_DIR / name).read_bytes())


@pytest.mark.django_db
def test_row_round_trip_preserves_identity_fields():
    """snapshot → Experiment row → snapshot preserves every identity field.

    This is the database analog of the .ihpo load→save contract: names,
    model/optimizer identity and params, metrics (with order), and seed all
    survive the row round-trip for both fixtures.
    """
    from web.services import snapshot as adapter

    for filename in ("test.ihpo", "test2.ihpo"):
        original = _fixture_snapshot(filename)
        row = adapter.experiment_from_snapshot(original)
        again = adapter.snapshot_from_experiment(row)

        for key in IDENTITY_KEYS:
            assert again[key] == original[key], f"{filename}: {key}"


@pytest.mark.django_db
def test_row_round_trip_preserves_result():
    """The stored result survives the row round-trip byte-for-byte.

    The result dict is opaque to the store (only optimizers interpret it),
    so the adapter must pass it through unchanged — including SMAC's
    embedded optimizer_state in the test.ihpo fixture.
    """
    from web.services import snapshot as adapter

    for filename in ("test.ihpo", "test2.ihpo"):
        original = _fixture_snapshot(filename)
        row = adapter.experiment_from_snapshot(original)
        again = adapter.snapshot_from_experiment(row)
        assert again["result"] == original["result"], filename


@pytest.mark.django_db
def test_exported_snapshot_carries_all_documented_keys_and_current_version():
    """An exported snapshot is a complete, current-version .ihpo document.

    Expect: all documented top-level keys present, version stamped with this
    app's VERSION (the writer owns the version field), and the whole thing
    accepted by io.parse — i.e. a Streamlit-loadable file.
    """
    from web.services import snapshot as adapter

    row = adapter.experiment_from_snapshot(_fixture_snapshot("test2.ihpo"))
    exported = adapter.snapshot_from_experiment(row)

    for key in SNAPSHOT_KEYS:
        assert key in exported, key
    assert exported["version"] == VERSION

    import json
    assert io.parse(json.dumps(exported).encode("utf-8"))


@pytest.mark.django_db
def test_dataset_path_reflects_stored_file_not_foreign_machine():
    """The exported dataset_path points at this store's file, or is empty.

    The fixtures carry dataset paths from another machine; importing them
    without re-supplying data must not leak that path back out. Once a
    dataset file is attached to the row, the exported path must point at it.
    """
    from web.services import snapshot as adapter

    row = adapter.experiment_from_snapshot(_fixture_snapshot("test2.ihpo"))
    exported = adapter.snapshot_from_experiment(row)
    assert exported["dataset_path"] == ""
