"""Step 10: a lightweight health endpoint for the container healthcheck.

Returns 200 with a tiny body and touches nothing (no DB, no template), so a
container orchestrator can cheaply tell the web process is up.
"""

from django.urls import reverse


def test_healthz_returns_ok(client):
    resp = client.get(reverse("ui:healthz"))
    assert resp.status_code == 200
    assert resp.content == b"ok"
