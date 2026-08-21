"""`{% asset %}` — a static URL that changes when the file does.

Static filenames here are plain: WhiteNoise's non-manifest storage keeps them
that way so `{% static %}` works under `runserver` and in tests without a prior
`collectstatic` (see STORAGES in config/settings.py). Plain filenames mean a
browser that has `app.css` has no way to tell a new one apart from the copy it
already holds, and a stylesheet it holds onto is a page laid out by rules that
are no longer the rules.

So the URL carries the file's modification time. Edit the file and every page
asks for a URL nobody has cached; leave it alone and the cached copy keeps being
used, which is the point of caching it.

Falls back to a bare `{% static %}` when the file cannot be found — a missing
asset is the storage's problem to report, not this tag's.
"""

from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static

register = template.Library()


@register.simple_tag
def asset(path: str) -> str:
    """The static URL for *path*, with the file's mtime as a query string."""
    found = finders.find(path)
    if not found:
        return static(path)
    from pathlib import Path

    return f"{static(path)}?v={int(Path(found).stat().st_mtime)}"
