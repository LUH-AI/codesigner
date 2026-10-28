"""`experiments/services/snapshot.py` is the only seam between .ihpo files and
the database, so it must preserve what the snapshot-contract tests require.
"""

from importlib.metadata import version as dist_version

import pytest

from core import io

from tests.conftest import FIXTURES_DIR
from tests.core.test_snapshot_contract import IDENTITY_KEYS, SNAPSHOT_KEYS, _at


def _fixture_snapshot(name: str) -> dict:
    return io.parse((FIXTURES_DIR / name).read_bytes())


@pytest.mark.django_db
def test_row_round_trip_preserves_identity_fields():
    """`snapshot_from_experiment` keeps names, model and optimizer identity,
    ordered metrics and seed for both fixtures.
    """
    from experiments.services import snapshot as adapter

    for filename in ("test.ihpo", "test2.ihpo"):
        original = _fixture_snapshot(filename)
        row = adapter.experiment_from_snapshot(original)
        again = adapter.snapshot_from_experiment(row)

        for path in IDENTITY_KEYS:
            assert _at(again, path) == _at(original, path), f"{filename}: {path}"


@pytest.mark.django_db
def test_row_round_trip_preserves_result():
    """`snapshot_from_experiment` passes the opaque result through unchanged,
    including SMAC's `optimizer_state`.
    """
    from experiments.services import snapshot as adapter

    for filename in ("test.ihpo", "test2.ihpo"):
        original = _fixture_snapshot(filename)
        row = adapter.experiment_from_snapshot(original)
        again = adapter.snapshot_from_experiment(row)
        assert again["result"] == original["result"], filename


@pytest.mark.django_db
def test_exported_snapshot_carries_all_documented_keys_and_current_version():
    """The export carries every documented key and the package version, and
    `io.parse` accepts it.
    """
    from experiments.services import snapshot as adapter

    row = adapter.experiment_from_snapshot(_fixture_snapshot("test2.ihpo"))
    exported = adapter.snapshot_from_experiment(row)

    for key in SNAPSHOT_KEYS:
        assert key in exported, key
    assert exported["version"] == dist_version("codesigner")

    import json
    assert io.parse(json.dumps(exported).encode("utf-8"))


@pytest.mark.django_db
def test_dataset_path_reflects_stored_file_not_foreign_machine():
    """Imported fixtures carry another machine's dataset path, and the export
    must drop it.
    """
    from experiments.services import snapshot as adapter

    row = adapter.experiment_from_snapshot(_fixture_snapshot("test2.ihpo"))
    exported = adapter.snapshot_from_experiment(row)
    assert exported["dataset"]["path"] == ""
