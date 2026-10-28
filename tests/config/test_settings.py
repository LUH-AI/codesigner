"""Only a fresh subprocess proves the wiring because `SECURE_BEHIND_TLS` derives
cookie, redirect and HSTS settings at import time.
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
    """An explicit `False` keeps a local `.env` from flipping the test."""
    values = _settings_with(SECURE_BEHIND_TLS="False")
    assert values == {
        "SECURE_SSL_REDIRECT": False, "SESSION_COOKIE_SECURE": False,
        "CSRF_COOKIE_SECURE": False, "SECURE_HSTS_SECONDS": 0,
        "SECURE_PROXY_SSL_HEADER": None,
    }


def test_on_derives_the_whole_cluster():
    """TLS terminates in front of gunicorn, hence the proxy header."""
    values = _settings_with(SECURE_BEHIND_TLS="True")
    assert values["SECURE_SSL_REDIRECT"] is True
    assert values["SESSION_COOKIE_SECURE"] is True
    assert values["CSRF_COOKIE_SECURE"] is True
    assert values["SECURE_HSTS_SECONDS"] == 31536000
    assert values["SECURE_PROXY_SSL_HEADER"] == ["HTTP_X_FORWARDED_PROTO", "https"]
