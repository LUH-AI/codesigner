"""The trade-off explainers (assistant/tradeoffs.py, read from
assistant/knowledge/explainers/tradeoffs/): each SMAC setting has exactly one,
the metric and the validation method have theirs, and each note is complete
and cites only papers in the library. The explainer tests check the files
against their template."""

import re

from assistant import sources, tradeoffs
from core.optimizers import SMACOptimizer


def test_every_smac_setting_has_one_note():
    """What: every SMAC setting is covered by exactly one note, and every note
    on `opt_` fields covers settings that exist.
    How: the notes' `applies_to` against the parameter schema."""
    covered = [name for note in tradeoffs.for_optimizer(SMACOptimizer).values()
               for name in note["applies_to"]]
    schema = {p.name for p in SMACOptimizer.params_schema}

    assert sorted(covered) == sorted(set(covered)), "a setting has two notes"
    assert set(covered) == schema


def test_every_note_is_complete_and_cites_the_library():
    """What: each note has a known verdict, says what it trades, lists when to
    stray as sentences, and cites only papers in literature.toml.
    How: every note against the library's keys."""
    keys = {paper["key"] for paper in sources.papers()}
    for name, note in tradeoffs.notes().items():
        assert note["verdict"] in tradeoffs.VERDICTS, name
        assert note["trade_off"] and note["stray_when"], name
        assert all(when.endswith(".") for when in note["stray_when"]), name
        text = note["trade_off"] + " ".join(note["stray_when"])
        cited = {key for group in re.findall(r"\[([^\]]+)\]", text)
                 for key in re.split(r",\s*", group) if re.fullmatch(r"[A-Za-z]+\+?\d{2}", key)}
        assert cited <= keys, (name, cited - keys)


def test_notes_are_found_by_the_fields_they_are_about():
    """What: asked about some fields, only the notes on them come back, a
    group's note with any one of its members; the metric and the validation
    method have notes; an optimizer without settings has none.
    How: `notes` and `for_optimizer` with a few fields."""
    from core.optimizers.random_optimizer import RandomOptimizer

    found = tradeoffs.for_optimizer(SMACOptimizer, ["trial_cap", "random_probability"])

    assert set(found) == {"initial-design-size", "random-trial-rate"}
    assert set(tradeoffs.notes(["optimize_metric", "evaluation_scheme"])) == {
        "optimized-metric", "validation-method"}
    assert tradeoffs.for_optimizer(RandomOptimizer) == {}
