"""An experiment's notepad as a page of its own, beside its dashboard: one
Markdown document that people who may edit the experiment write in an editor,
and everyone who may view it reads. Assistants write it too, with their tools
(assistant/tools/notes.py); both go through assistant/notes.py.
"""

from django.core.exceptions import PermissionDenied
from django.shortcuts import redirect, render

from access.policy import EDIT, VIEW, policy

from ..permissions import BOTH, experiment_view


@experiment_view(VIEW, drafts=BOTH)
def experiment_notes(request, exp):
    """`experiment_notes` shows *exp*'s notes, in the editor for a person who
    may edit it. A save made over a newer version is refused: the page comes
    back with the person's text still in the editor and the newer notes
    beside it, to merge and save again."""
    from assistant import notes as notepad

    may_edit = policy().may(request, exp, EDIT)
    context = {"may_edit": may_edit, "draft": exp.draft}
    if request.method == "POST":
        if not may_edit:
            raise PermissionDenied
        text = request.POST.get("text", "")
        try:
            notepad.save(exp, text, user=request.user, seen=request.POST.get("seen", ""))
        except notepad.Stale as stale:
            context.update(stale=notepad.as_dict(stale.notes), text=text)
        except notepad.NoteError as error:
            context.update(error=str(error), text=text)
        else:
            return redirect("ui:experiment_notes", pk=exp.pk)
    current = notepad.notes_of(exp)
    notes = notepad.as_dict(current)
    context.setdefault("text", notes["text"])
    context.update(notes=notes, updated_at=current.updated_at if current.pk else None,
                   max_length=notepad.ExperimentNotes.MAX_LENGTH)
    if exp.draft:
        context.update(experiment=exp, summary={"name": exp.title})
    else:
        from .dashboard import experiment_header

        context.update(experiment_header(request, exp))
    return render(request, "ui/experiment_notes.html", context)
