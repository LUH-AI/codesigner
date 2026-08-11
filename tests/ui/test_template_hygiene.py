"""Template mistakes that render as something rather than failing.

Checked against the template *sources* rather than rendered output, so a
template no test happens to render is covered too.
"""

import re

from django.conf import settings

TEMPLATE_DIRS = ["templates", "ui/templates", "access/templates"]


def _templates():
    for directory in TEMPLATE_DIRS:
        yield from (settings.BASE_DIR / directory).rglob("*.html")


def test_there_are_templates_to_check():
    assert len(list(_templates())) > 10


def test_no_comment_spans_more_than_one_line():
    """`{# … #}` is single-line only. Spanning two lines does not raise — the
    text is simply emitted into the page, so the note you wrote for the next
    developer ships to the browser. `{% comment %}` is the multi-line form."""
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
    """The i18n completeness check only sees strings that were *marked*, so a
    new template that forgets i18n entirely is invisible to it. This catches
    that case.

    Exempt: anything whose visible text comes from somewhere already
    translated — the figure partials render `figure.label` (translated in
    `ui/figures/catalog.py`), the breadcrumb bar renders labels from
    `ui/navigation.py`, and the metric select and selected-config panel render
    metric names, numbers and config keys, which are data. The optimizer field
    renders one setting, and every word of it — name, explanation, placeholder —
    comes from `ui/optimizer_labels.py`, which is where they are marked.
    """
    exempt = {"_selected_config_inner.html", "_breadcrumbs.html", "_metric_select.html",
              "_optimizer_param_field.html"}
    no_i18n = {p.name for p in _templates()
               if "translate" not in p.read_text()
               and p.name not in exempt
               and p.parent.name != "figures"}

    assert not no_i18n, f"template(s) with no translated text: {sorted(no_i18n)}"
