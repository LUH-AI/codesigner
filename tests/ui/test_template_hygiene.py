"""Checking template sources covers templates no test renders.
"""

import re

from django.conf import settings

TEMPLATE_DIRS = ["ui/templates"]


def _templates():
    for directory in TEMPLATE_DIRS:
        yield from (settings.BASE_DIR / directory).rglob("*.html")


def test_there_are_templates_to_check():
    assert len(list(_templates())) > 10


def test_no_comment_spans_more_than_one_line():
    """A two-line `{# … #}` renders into the page; `{% comment %}` spans lines."""
    leaking = []
    for path in _templates():
        text = path.read_text()
        for match in re.finditer(r"\{#", text):
            rest = text[match.start():]
            close = rest.find("#}")
            if close == -1 or "\n" in rest[:close]:
                leaking.append(f"{path.name}:{text[:match.start()].count(chr(10)) + 1}")

    assert not leaking, f"multi-line {{# #}} comment(s) will render: {leaking}"


def test_every_template_with_text_of_its_own_marks_it_for_translation():
    """The i18n check only sees marked strings, so a template with no i18n at
    all escapes it. Exempt templates render text translated elsewhere or data.
    """
    exempt = {"_selected_config_inner.html", "_breadcrumbs.html", "_metric_select.html",
              "_optimizer_param_field.html", "_info_label.html",
              "_group_initial_points.html", "_experiment_page.html", "processing_step.html"}
    no_i18n = {p.name for p in _templates()
               if "translate" not in p.read_text()
               and p.name not in exempt
               and p.parent.name != "figures"}

    assert not no_i18n, f"template(s) with no translated text: {sorted(no_i18n)}"
