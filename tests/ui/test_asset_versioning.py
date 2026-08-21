"""Static assets are asked for by a URL that changes when the file does.

The filenames WhiteNoise serves here are plain (see STORAGES), so nothing in a
static URL says which version of the file it points at. A browser holding an old
`app.css` then lays the page out by rules that are no longer the rules — which
is invisible from the server, because the server rendered the right markup.

These check the tag itself rather than a live browser: that the URL carries the
file's mtime, that editing the file changes it, and that the pages people
actually load use the tag.
"""

import re

from django.template import Context, Template
from django.urls import reverse


def _render(path="ui/app.css"):
    return Template("{% load assets %}{% asset '" + path + "' %}").render(Context())


def test_a_static_url_carries_the_files_modification_time():
    """The version has to come from the file, not from the process or the
    clock: a restart must not invalidate assets that did not change, and an
    edit must invalidate them without one."""
    from pathlib import Path

    from django.contrib.staticfiles import finders

    url = _render()
    stamp = re.search(r"\?v=(\d+)$", url)

    assert stamp, url
    assert url.startswith("/static/ui/app.css?v=")
    assert int(stamp.group(1)) == int(Path(finders.find("ui/app.css")).stat().st_mtime)


def test_editing_the_file_changes_the_url(tmp_path, settings):
    """The whole point, stated as the behaviour rather than the mechanism."""
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
    """No file, no mtime — the tag falls back rather than raising, so a typo
    surfaces as a 404 on the asset instead of a 500 on every page."""
    assert _render("ui/nope.css") == "/static/ui/nope.css"


def test_the_pages_people_load_ask_for_versioned_assets(client, settings):
    """A tag nothing uses fixes nothing. Both templates carrying a stylesheet
    link are checked — the app shell, and the sign-in page, which does not
    extend it and so had to be changed separately."""
    settings.REQUIRE_LOGIN = True
    signin = client.get(reverse("ui:home"), follow=True)
    shell = signin.content.decode()
    assert "access/login.html" in [t.name for t in signin.templates]
    settings.REQUIRE_LOGIN = False
    home = client.get(reverse("ui:home")).content.decode()

    assert re.search(r'href="/static/ui/app\.css\?v=\d+"', home)
    assert re.search(r'src="/static/ui/poll\.js\?v=\d+"', home)
    assert re.search(r'href="/static/ui/app\.css\?v=\d+"', shell)
