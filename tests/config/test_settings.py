"""SECURE_BEHIND_TLS derives a cluster of cookie/redirect/HSTS settings from
one switch, the same REQUIRE_LOGIN/ALLOW_CUSTOM_MODELS-shaped contract used
everywhere else in this settings module.

The derivation runs once, at import time (`if SECURE_BEHIND_TLS: ...` in
config/settings.py), so overriding it through pytest-django's `settings`
fixture after the fact proves nothing — that machinery only patches already-
loaded attributes, it does not re-run the block that would have set them
differently. A fresh subprocess is what actually proves settings.py wires the
switch to what it claims to.
"""

import json
import os
import subprocess
import sys

_PROBE = (
    "import django; django.setup(); "
    "from django.conf import settings; import json; "
    "print(json.dumps({"
    "'SECURE_SSL_REDIRECT': getattr(settings, 'SECURE_SSL_REDIRECT', False), "
    "'SESSION_COOKIE_SECURE': getattr(settings, 'SESSION_COOKIE_SECURE', False), "
    "'CSRF_COOKIE_SECURE': getattr(settings, 'CSRF_COOKIE_SECURE', False), "
    "'SECURE_HSTS_SECONDS': getattr(settings, 'SECURE_HSTS_SECONDS', 0), "
    "'SECURE_PROXY_SSL_HEADER': getattr(settings, 'SECURE_PROXY_SSL_HEADER', None), "
    "}))"
)


def _settings_with(**env_overrides) -> dict:
    env = {**os.environ, "DJANGO_SETTINGS_MODULE": "config.settings",
            "SECRET_KEY": "probe-only-not-a-real-secret", **env_overrides}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_off_by_default_matches_todays_behaviour():
    """Explicitly off (not just unset, so a local .env can't accidentally flip
    this test): exactly like before this setting existed."""
    values = _settings_with(SECURE_BEHIND_TLS="False")
    assert values == {
        "SECURE_SSL_REDIRECT": False, "SESSION_COOKIE_SECURE": False,
        "CSRF_COOKIE_SECURE": False, "SECURE_HSTS_SECONDS": 0,
        "SECURE_PROXY_SSL_HEADER": None,
    }


def test_on_derives_the_whole_cluster():
    """One switch turns on redirect, both secure cookies, HSTS, and the proxy
    header needed because TLS terminates in front of gunicorn, not in it."""
    values = _settings_with(SECURE_BEHIND_TLS="True")
    assert values["SECURE_SSL_REDIRECT"] is True
    assert values["SESSION_COOKIE_SECURE"] is True
    assert values["CSRF_COOKIE_SECURE"] is True
    assert values["SECURE_HSTS_SECONDS"] == 31536000
    assert values["SECURE_PROXY_SSL_HEADER"] == ["HTTP_X_FORWARDED_PROTO", "https"]
