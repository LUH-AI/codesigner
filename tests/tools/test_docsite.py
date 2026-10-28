"""`tools.docsite` edits prose and refuses code changes."""

import pytest

from tools.docsite import pages
from tools.docsite import source as S

SAMPLE = '''\
"""A module.

Its second paragraph, which runs on long enough that it has been wrapped onto
a second line by hand.
"""

# A comment about X,
# over two lines.
X = 1

#: What Y is for.
Y = 2


class A:
    """One line."""

    def f(self, a,
          b):  # noqa: E501
        return a


def g():
    r"""Matches \\d, and closes on its own line.
    """
'''


def region(source, owner=None, kind=None):
    return next(r for r in S.parse(source).regions
                if (owner is None or r.owner == owner) and (kind is None or r.kind == kind))


def edit(source, r, text, flowed=True):
    return S.replace_region(source, r.start, r.raw, text, flowed)[0]


def test_finds_docstrings_comments_and_attribute_comments():
    regions = S.parse(SAMPLE).regions
    assert [(r.kind, r.owner) for r in regions] == [
        ("docstring", ""), ("comment", None), ("attr", None),
        ("docstring", "A"), ("docstring", "g"),
    ]
    assert region(SAMPLE, kind="comment").text == "A comment about X,\nover two lines."
    assert region(SAMPLE, owner="g").text == "Matches \\d, and closes on its own line."


def test_trailing_comments_and_pragmas_stay_code():
    assert not any("noqa" in r.text for r in S.parse(SAMPLE).regions)


@pytest.mark.parametrize("flowed", [False, True])
def test_unchanged_text_writes_the_same_file(flowed):
    for r in S.parse(SAMPLE).regions:
        text = S.flow(r.text) if flowed else r.text
        assert edit(SAMPLE, r, text, flowed) == SAMPLE


def test_flowed_edit_rewraps_only_the_paragraph_that_changed():
    r = region(SAMPLE, owner="")
    text = S.flow(r.text).replace("Its second paragraph,", "Its second paragraph, now longer,")
    new = edit(SAMPLE, r, text)
    assert new.startswith('"""A module.\n\nIts second paragraph, now longer, which runs on long enough that it has been\n'
                          "wrapped onto a second line by hand.\n")
    assert all(len(line) <= S.WIDTH for line in new.splitlines())
    assert new.split("# A comment")[1] == SAMPLE.split("# A comment")[1]


def test_docstring_keeps_its_prefix_and_closing_line():
    new = edit(SAMPLE, region(SAMPLE, owner="g"), "Matches \\d still.")
    assert '    r"""Matches \\d still.\n    """\n' in new


def test_comment_edit_keeps_its_marker_and_indent():
    new = edit(SAMPLE, region(SAMPLE, kind="attr"), "What Y is for, said at a good deal more length than fits on one line of the file.")
    assert "#: What Y is for, said at a good deal more length than fits on one line of the\n#: file.\nY = 2" in new


def test_emptied_comment_is_deleted():
    new = edit(SAMPLE, region(SAMPLE, kind="comment"), "")
    assert "\n\nX = 1" in new and "A comment" not in new


def test_add_docstring_under_a_multiline_signature():
    new, line = S.add_docstring(SAMPLE, "A.f", 0, "Says what f does.", True)
    lines = new.splitlines()
    assert lines[line - 2].endswith("# noqa: E501")
    assert lines[line - 1] == '        """Says what f does."""'


@pytest.mark.parametrize("text, reason", [
    ('ends """ early', "would end the docstring"),
    (r"a \d escape", "invalid escape sequence"),
])
def test_refuses_text_that_would_change_the_code(text, reason):
    with pytest.raises(S.EditError, match=reason):
        edit(SAMPLE, region(SAMPLE, owner="A"), text, flowed=False)


def test_refuses_emptying_the_only_statement_of_a_class():
    source = 'class A:\n    """Only this."""\n'
    with pytest.raises(S.EditError, match="would not parse"):
        edit(source, region(source), "")


def test_finds_text_that_moved_and_refuses_text_that_changed():
    r = region(SAMPLE, owner="A")
    moved = "# new first line\n" + SAMPLE
    assert "Two lines." in edit(moved, r, "Two lines.", flowed=False)
    with pytest.raises(S.Conflict):
        edit(SAMPLE.replace("One line.", "Changed in the IDE."), r, "Two lines.", flowed=False)


def test_every_file_in_the_repository_round_trips():
    """`edit` keeps every repository file byte for byte."""
    for path in pages.files():
        source = (pages.ROOT / path).read_text(encoding="utf-8")
        for r in S.parse(source).regions:
            assert edit(source, r, r.text, flowed=False) == source, f"{path}:{r.start}"
            assert edit(source, r, S.flow(r.text)) == source, f"{path}:{r.start} (flowed)"
