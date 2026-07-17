"""Step 2.5 TDD: the Inspect page — browse any .ihpo file without a database.

Exercises the Step 2 engine (core.io parse + read-only build) through the
browser. Expectations mirror the Streamlit load dialog: a valid file yields
a browsable summary with no dataset required; an invalid file surfaces the
parse error behind the "Invalid or unreadable experiment file." message.
"""

import json

import pytest
from django.urls import reverse

from tests.conftest import FIXTURES_DIR


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_inspect_page_offers_an_upload_form(client):
    """GET on the inspect page returns an upload form.

    Expect: HTTP 200 and a file input — the page must be reachable and
    self-explanatory before any file is posted.
    """
    response = client.get(reverse("web:inspect"))
    assert response.status_code == 200
    assert b'type="file"' in response.content


def test_inspect_valid_file_shows_experiment_summary(client):
    """Posting a valid .ihpo renders its identity fields and result summary.

    Setup: the Random Search fixture (30 trials); expectations are read from
    the file itself so the test tracks fixture content, not hardcoded values.
    Expect: name, model, optimizer, seed and the trial count all appear.
    """
    snapshot = _fixture("test2.ihpo")
    with open(FIXTURES_DIR / "test2.ihpo", "rb") as f:
        response = client.post(reverse("web:inspect"), {"file": f})

    assert response.status_code == 200
    html = response.content.decode()
    assert snapshot["name"] in html
    assert snapshot["model_name"] in html
    assert snapshot["optimizer_name"] in html
    assert str(snapshot["seed"]) in html
    assert str(len(snapshot["result"]["data"])) in html


def test_inspect_needs_no_dataset_file(client):
    """A fixture whose stored dataset path is dead still renders a summary.

    The inspect page is read-only by definition (mirrors the load dialog's
    'load read-only' path), so machine-specific paths must not block it.
    """
    snapshot = _fixture("test.ihpo")   # SMAC fixture, foreign dataset path
    with open(FIXTURES_DIR / "test.ihpo", "rb") as f:
        response = client.post(reverse("web:inspect"), {"file": f})

    assert response.status_code == 200
    assert snapshot["name"] in response.content.decode()


def test_inspect_invalid_file_shows_parse_error(client):
    """Posting an unparseable file shows the load-error message and reason.

    Expect: the page renders (200, not a server error) with the
    "Invalid or unreadable experiment file." text and parse's specific
    complaint, exactly as the Streamlit load dialog surfaced it.
    """
    from django.core.files.uploadedfile import SimpleUploadedFile

    bad = SimpleUploadedFile("bad.ihpo", b"definitely not json {")
    response = client.post(reverse("web:inspect"), {"file": bad})

    assert response.status_code == 200
    html = response.content.decode()
    assert "Invalid or unreadable experiment file." in html
    assert "not valid JSON" in html
