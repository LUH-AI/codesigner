"""`assistant.notes` stores each experiment's notepad. `save` raises `Stale` on
a conflict, and `edit` never overwrites a concurrent save.
"""

from __future__ import annotations

from django.db import transaction

from .models import ExperimentNotes


class NoteError(Exception):
    """`NoteError` messages explain failures."""


class Stale(NoteError):

    def __init__(self, notes) -> None:
        super().__init__("the notes changed since they were opened")
        self.notes = notes


def notes_of(exp):
    return (ExperimentNotes.objects.filter(experiment=exp).select_related("updated_by").first()
            or ExperimentNotes(experiment=exp))


def as_dict(notes) -> dict:
    from access.models import shown_name

    who = shown_name(notes.updated_by) if notes.updated_by is not None else ""
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
    """`save` raises `Stale` when other text was saved since *seen*."""
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
    """`edit` raises `NoteError` unless *replace* occurs once; without
    *replace*, `edit` appends."""
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


#: `PROMPT_CHARACTERS` caps prompt notes.
PROMPT_CHARACTERS = 4000


def for_prompt(exp) -> str:
    """Truncated notes end with `read_notes`."""
    text = notes_of(exp).text
    if len(text) <= PROMPT_CHARACTERS:
        return text
    return (text[:PROMPT_CHARACTERS]
            + f"\n[… {len(text) - PROMPT_CHARACTERS} more characters: read_notes]")
