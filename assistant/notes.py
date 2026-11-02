"""Each experiment's notepad (`ExperimentNotes`): one Markdown document of what
assistants and people write down for whoever works on the experiment next.

People edit it on its own page (ui:experiment_notes), the assistant with its
tools (assistant/tools/notes.py); both come through here, after checking what
the person may do with the experiment: viewing it reads the notes, editing it
writes them.

A person saves the whole document, refused if it changed since they opened
it. An assistant changes one passage, or adds one, against the document as it
is then, so it never writes over what someone saved meanwhile.
"""

from __future__ import annotations

from django.db import transaction

from .models import ExperimentNotes


class NoteError(Exception):
    """The notes cannot be written as asked. Its message says why."""


class Stale(NoteError):
    """The notes changed since the writer opened them; `notes` is how they
    are now."""

    def __init__(self, notes) -> None:
        super().__init__("the notes changed since they were opened")
        self.notes = notes


def notes_of(exp):
    """*exp*'s notepad, unsaved and empty if nothing was written yet."""
    return (ExperimentNotes.objects.filter(experiment=exp).select_related("updated_by").first()
            or ExperimentNotes(experiment=exp))


def as_dict(notes) -> dict:
    """*notes* as the page and the model are given them."""
    who = notes.updated_by.get_username() if notes.updated_by is not None else ""
    if notes.by_assistant:
        who = f"{who}'s assistant" if who else "the assistant"
    return {"text": notes.text, "by": who or ("someone" if notes.pk else ""),
            "by_assistant": notes.by_assistant,
            "updated_at": notes.updated_at.isoformat() if notes.pk else ""}


def _checked(text: str) -> str:
    if len(text) > ExperimentNotes.MAX_LENGTH:
        raise NoteError(f"the notes can be at most {ExperimentNotes.MAX_LENGTH} characters; "
                        f"these are {len(text)}")
    return text


def _author(user):
    return user if getattr(user, "is_authenticated", False) else None


def save(exp, text: str, *, user, seen: str):
    """A person saves *text* as *exp*'s notes. *seen* is the `updated_at` of
    the notes they opened ("" for none yet), and the save is refused with
    `Stale` if they have changed since. Returns the notes."""
    text = _checked((text or "").replace("\r\n", "\n").rstrip())
    with transaction.atomic():
        notes, created = ExperimentNotes.objects.select_for_update().get_or_create(experiment=exp)
        if not created and as_dict(notes)["updated_at"] != seen and notes.text != text:
            raise Stale(notes)
        notes.text, notes.updated_by = text, _author(user)
        notes.by_assistant, notes.conversation = False, None
        notes.save()
    return notes


def edit(exp, text: str, *, user, conversation, replace: str = ""):
    """An assistant, in *conversation*, changes *exp*'s notes: the passage
    *replace*, which must be in them exactly once, becomes *text*; without
    *replace*, *text* is added at the end as a paragraph of its own. Returns
    the notes."""
    text = (text or "").strip()
    with transaction.atomic():
        notes, _ = ExperimentNotes.objects.select_for_update().get_or_create(experiment=exp)
        if replace:
            found = notes.text.count(replace)
            if found != 1:
                raise NoteError("the passage to replace is not in the notes" if not found else
                                f"the passage to replace is in the notes {found} times; "
                                "quote more of it")
            changed = notes.text.replace(replace, text)
        elif text:
            changed = (notes.text.rstrip() + "\n\n" + text).strip()
        else:
            raise NoteError("give the text to add, or the passage to replace")
        notes.text, notes.updated_by = _checked(changed), _author(user)
        notes.by_assistant, notes.conversation = True, conversation
        notes.save()
    return notes


#: How much of the notepad the system prompt carries, in characters.
PROMPT_CHARACTERS = 4000


def for_prompt(exp) -> str:
    """The start of *exp*'s notes, up to `PROMPT_CHARACTERS`, saying how much
    more there is."""
    text = notes_of(exp).text
    if len(text) <= PROMPT_CHARACTERS:
        return text
    return (text[:PROMPT_CHARACTERS]
            + f"\n[… {len(text) - PROMPT_CHARACTERS} more characters: read_notes]")
