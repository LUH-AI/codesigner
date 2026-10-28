"""Edge cases in `core.provenance` must never refuse an import or fail an
export.
"""

import hashlib

import pytest

from core import provenance
from core.optimizers import RandomOptimizer, SMACOptimizer


def test_a_file_that_is_not_there_has_no_digest(tmp_path):
    """An export must still succeed after its dataset was deleted."""
    assert provenance.sha256(tmp_path / "gone.csv") == ""


def test_hashing_a_stream_leaves_it_where_it_was_found(tmp_path):
    """The caller saves the upload after hashing."""
    import io as _io

    stream = _io.BytesIO(b"a,b\n1,2\n")
    stream.read(3)

    digest = provenance.sha256_stream(stream)

    assert digest == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert stream.tell() == 3


def test_a_record_with_no_fingerprint_never_disagrees():
    """Older `.ihpo` exports lack fingerprints and must still import."""
    assert provenance.dataset_mismatch(None, "abc") == ""
    assert provenance.dataset_mismatch({}, "abc") == ""


def test_the_same_digest_is_not_a_mismatch():
    assert provenance.dataset_mismatch({"sha256": "abc"}, "abc") == ""


def test_a_different_digest_says_which_file_and_both_digests():
    """File and both digests make the message actionable."""
    message = provenance.dataset_mismatch(
        {"sha256": "abc123def456", "filename": "iris.csv"}, "999888777666")

    assert "iris.csv" in message
    assert "abc123def456"[:12] in message
    assert "999888777666"[:12] in message


def test_a_dataset_is_described_by_shape_as_well_as_digest(tmp_path):
    """The shape describes the dataset without the file."""
    csv = tmp_path / "small.csv"
    csv.write_text("a,b,target\n1,2,x\n3,4,y\n")

    record = provenance.dataset_fingerprint(csv)

    assert record["rows"] == 2
    assert record["columns"] == 3
    assert record["column_names"] == ["a", "b", "target"]
    assert record["target_column"] == "target"


def test_a_dataset_says_how_each_feature_was_read(tmp_path):
    """Readers without the file can tell labels, numbers and dates apart."""
    csv = tmp_path / "d.csv"
    csv.write_text("city,rooms,listed,y\nBerlin,2,2024-01-05,a\nHamburg,3,2024-02-11,b\n")

    assert provenance.dataset_fingerprint(csv)["column_kinds"] == {
        "city": "categorical", "rooms": "numeric", "listed": "datetime"}


def test_a_semicolon_separated_dataset_is_read_the_same_way_the_app_reads_it(tmp_path):
    """`wine.csv` uses semicolons, as the app reads it."""
    csv = tmp_path / "euro.csv"
    csv.write_text('a;b;"quality"\n1;2;3\n')

    assert provenance.dataset_fingerprint(csv)["columns"] == 3


def test_an_unreadable_dataset_still_gets_its_digest(tmp_path):
    blob = tmp_path / "notes.csv"
    blob.write_bytes(b"\x00\x01\x02")

    record = provenance.dataset_fingerprint(blob)

    assert record["sha256"] == hashlib.sha256(b"\x00\x01\x02").hexdigest()
    assert record["rows"] is None


def test_an_optimizer_with_nothing_to_configure_resolves_to_nothing():
    assert RandomOptimizer().resolved_params() == {}


def test_a_setting_the_optimizer_no_longer_has_is_not_resolved():
    """`known_params` drops stale settings; the record must keep them out."""
    stored = {"rf_trees": 5, "withdrawn_setting": 9}

    resolved = SMACOptimizer(**SMACOptimizer.known_params(stored)).resolved_params()

    assert "withdrawn_setting" not in resolved
    assert resolved["rf_trees"] == 5


@pytest.mark.parametrize("strategy,resolved,blank", [
    ("rf", "rf_trees", "gp_restarts"),
    ("gp", "gp_restarts", "rf_trees"),
])
def test_only_the_chosen_strategys_blanks_get_answered(strategy, resolved, blank):
    """The other strategy has no such default to record."""
    record = SMACOptimizer(search_strategy=strategy).resolved_params()

    assert record[resolved] is not None
    assert record[blank] is None


def test_the_defaults_come_from_smac_rather_than_a_copy_of_them():
    """`resolved_params` reads defaults from installed SMAC signatures and
    cannot drift."""
    from smac import HyperparameterOptimizationFacade

    import inspect
    expected = inspect.signature(
        HyperparameterOptimizationFacade.get_model).parameters["n_trees"].default

    assert SMACOptimizer(search_strategy="rf").resolved_params()["rf_trees"] == expected


def test_stratification_is_left_unanswered_without_the_target():
    """Null means unchecked; False would mean unstratifiable."""
    assert provenance.evaluation(0)["stratified"] is None


@pytest.mark.parametrize("folds,scheme", [(0, "holdout"), (1, "holdout"),
                                          (2, "kfold"), (5, "kfold")])
def test_the_scheme_follows_the_fold_count(folds, scheme):
    assert provenance.evaluation(folds)["scheme"] == scheme
