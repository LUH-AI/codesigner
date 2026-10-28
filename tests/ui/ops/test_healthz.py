"""`ui:healthz` answers 200 "ok" without touching the DB or templates.
"""

from django.urls import reverse


def test_healthz_returns_ok(client):
    resp = client.get(reverse("ui:healthz"))
    assert resp.status_code == 200
    assert resp.content == b"ok"
