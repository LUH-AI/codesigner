"""The record helpers on their own, at the edges the integration tests skip.

`tests/ui/storage/test_provenance.py` covers what an exported file says. These
cover what these functions do when there is nothing to say — a dataset that is
gone, a file written before fingerprints existed, a setting the optimizer has
since dropped. Every one of them is a case where being wrong would either refuse
a legitimate import or fail an export, and neither is worth a record.
"""

import hashlib

import pytest

from core import provenance
from core.optimizers import RandomOptimizer, SMACOptimizer


# ── digests ──────────────────────────────────────────────────────────────────

def test_a_file_that_is_not_there_has_no_digest(tmp_path):
    """Empty rather than raising: an export of an experiment whose dataset was
    deleted should still produce a file."""
    assert provenance.sha256(tmp_path / "gone.csv") == ""


def test_hashing_a_stream_leaves_it_where_it_was_found(tmp_path):
    """The upload is hashed and then saved by the caller. Reading it to the end
    and leaving it there would store an empty dataset."""
    import io as _io

    stream = _io.BytesIO(b"a,b\n1,2\n")
    stream.read(3)

    digest = provenance.sha256_stream(stream)

    assert digest == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert stream.tell() == 3


# ── what counts as a mismatch ────────────────────────────────────────────────

def test_a_record_with_no_fingerprint_never_disagrees():
    """Every `.ihpo` exported before this existed. Refusing them would have made
    the section a breaking change."""
    assert provenance.dataset_mismatch(None, "abc") == ""
    assert provenance.dataset_mismatch({}, "abc") == ""


def test_the_same_digest_is_not_a_mismatch():
    assert provenance.dataset_mismatch({"sha256": "abc"}, "abc") == ""


def test_a_different_digest_says_which_file_and_both_digests():
    """The message has to be actionable — "wrong dataset" with no names is a
    dead end when someone has several."""
    message = provenance.dataset_mismatch(
        {"sha256": "abc123def456", "filename": "iris.csv"}, "999888777666")

    assert "iris.csv" in message
    assert "abc123def456"[:12] in message
    assert "999888777666"[:12] in message


# ── the shape of a dataset ───────────────────────────────────────────────────

def test_a_dataset_is_described_by_shape_as_well_as_digest(tmp_path):
    """The digest recognises it; the shape is what makes the record readable
    without it."""
    csv = tmp_path / "small.csv"
    csv.write_text("a,b,target\n1,2,x\n3,4,y\n")

    record = provenance.dataset_fingerprint(csv)

    assert record["rows"] == 2
    assert record["columns"] == 3
    assert record["column_names"] == ["a", "b", "target"]
    assert record["target_column"] == "target"


def test_a_semicolon_separated_dataset_is_read_the_same_way_the_app_reads_it(tmp_path):
    """`wine.csv` ships with semicolons. A record that described it as one
    column would be describing a file the application never saw."""
    csv = tmp_path / "euro.csv"
    csv.write_text('a;b;"quality"\n1;2;3\n')

    assert provenance.dataset_fingerprint(csv)["columns"] == 3


def test_an_unreadable_dataset_still_gets_its_digest(tmp_path):
    """Whatever it is, it is still the file the trials came from."""
    blob = tmp_path / "notes.csv"
    blob.write_bytes(b"\x00\x01\x02")

    record = provenance.dataset_fingerprint(blob)

    assert record["sha256"] == hashlib.sha256(b"\x00\x01\x02").hexdigest()
    assert record["rows"] is None


# ── resolving the blanks ─────────────────────────────────────────────────────

def test_an_optimizer_with_nothing_to_configure_resolves_to_nothing():
    assert RandomOptimizer().resolved_params() == {}


def test_a_setting_the_optimizer_no_longer_has_is_not_resolved():
    """A stored file can name anything. `known_params` drops it on the way in,
    and the record must not reintroduce it."""
    stored = {"rf_trees": 5, "withdrawn_setting": 9}

    resolved = SMACOptimizer(**SMACOptimizer.known_params(stored)).resolved_params()

    assert "withdrawn_setting" not in resolved
    assert resolved["rf_trees"] == 5


@pytest.mark.parametrize("strategy,resolved,blank", [
    ("rf", "rf_trees", "gp_restarts"),
    ("gp", "gp_restarts", "rf_trees"),
])
def test_only_the_chosen_strategys_blanks_get_answered(strategy, resolved, blank):
    """The other strategy has no such component, so it has no default for it.
    A number there would be invented rather than recorded."""
    record = SMACOptimizer(search_strategy=strategy).resolved_params()

    assert record[resolved] is not None
    assert record[blank] is None


def test_the_defaults_come_from_smac_rather_than_a_copy_of_them():
    """Read out of the installed SMAC's signatures, so this cannot drift from
    the SMAC that will actually run."""
    from smac import HyperparameterOptimizationFacade

    import inspect
    expected = inspect.signature(
        HyperparameterOptimizationFacade.get_model).parameters["n_trees"].default

    assert SMACOptimizer(search_strategy="rf").resolved_params()["rf_trees"] == expected


# ── the evaluation scheme ────────────────────────────────────────────────────

def test_stratification_is_left_unanswered_without_the_target():
    """Null, not False. "We did not look" and "it could not be stratified" are
    different facts about the run."""
    assert provenance.evaluation(0)["stratified"] is None


@pytest.mark.parametrize("folds,scheme", [(0, "holdout"), (1, "holdout"),
                                          (2, "kfold"), (5, "kfold")])
def test_the_scheme_follows_the_fold_count(folds, scheme):
    assert provenance.evaluation(folds)["scheme"] == scheme
