"""`{% asset %}` appends the file's mtime to the static URL. WhiteNoise's
non-manifest storage leaves filenames plain, and the mtime stops a browser from
reusing a stale cached `app.css`. A file the finders miss falls back to a bare
`{% static %}`.
"""

from typing import cast

from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static

register = template.Library()


@register.simple_tag
def asset(path: str) -> str:
    """`asset` appends the file's mtime to *path*'s static URL."""
    found = finders.find(path)
    if not found:
        return static(path)
    from pathlib import Path

    return f"{static(path)}?v={int(Path(cast(str, found)).stat().st_mtime)}"
