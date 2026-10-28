"""WhiteNoise serves plain filenames (see STORAGES), so the `asset` tag appends
the file's mtime to bust stale browser caches.
"""

import re
from typing import cast

from django.template import Context, Template
from django.urls import reverse


def _render(path="ui/app.css"):
    return Template("{% load assets %}{% asset '" + path + "' %}").render(Context())


def test_a_static_url_carries_the_files_modification_time():
    """A restart keeps unchanged assets valid, and an edit invalidates them
    without one."""
    from pathlib import Path

    from django.contrib.staticfiles import finders

    url = _render()
    stamp = re.search(r"\?v=(\d+)$", url)

    assert stamp, url
    assert url.startswith("/static/ui/app.css?v=")
    assert int(stamp.group(1)) == int(Path(cast(str, finders.find("ui/app.css"))).stat().st_mtime)


def test_editing_the_file_changes_the_url(tmp_path, settings):
    asset = tmp_path / "probe.css"
    asset.write_text(".a {}")
    settings.STATICFILES_DIRS = [str(tmp_path)]

    before = _render("probe.css")
    import os

    os.utime(asset, (0, 1_700_000_000))
    after = _render("probe.css")

    assert before != after
    assert after.endswith("?v=1700000000")


def test_a_missing_asset_is_left_to_the_storage_to_report():
    """A typo then gives a 404 on the asset instead of a 500 on every page."""
    assert _render("ui/nope.css") == "/static/ui/nope.css"


def test_the_pages_people_load_ask_for_versioned_assets(client, settings):
    """The sign-in page carries its own stylesheet link, since that page extends
    no shell."""
    settings.REQUIRE_LOGIN = True
    signin = client.get(reverse("ui:home"), follow=True)
    shell = signin.content.decode()
    assert "ui/account/login.html" in [t.name for t in signin.templates]
    settings.REQUIRE_LOGIN = False
    home = client.get(reverse("ui:home")).content.decode()

    assert re.search(r'href="/static/ui/app\.css\?v=\d+"', home)
    assert re.search(r'src="/static/ui/poll\.js\?v=\d+"', home)
    assert re.search(r'href="/static/ui/app\.css\?v=\d+"', shell)
