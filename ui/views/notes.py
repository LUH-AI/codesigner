"""`ui.views.notes` shows an experiment's Markdown notepad beside its dashboard.
Editors and assistants both save through `assistant.notes`.
"""

from django.core.exceptions import PermissionDenied
from django.shortcuts import redirect, render

from access.policy import EDIT, VIEW, policy

from ..permissions import BOTH, experiment_view


@experiment_view(VIEW, drafts=BOTH)
def experiment_notes(request, exp):
    """`experiment_notes` shows *exp*'s notes, editable for a person with EDIT.
    A stale save raises `.Stale`, and the page shows the person's text beside
    the newer notes."""
    from assistant import notes as notepad

    may_edit = policy().may(request, exp, EDIT)
    # `editing` opens the page on Edit, where a refused save's text waits.
    context = {"may_edit": may_edit, "draft": exp.draft, "editing": False,
               "stale": None, "error": ""}
    if request.method == "POST":
        if not may_edit:
            raise PermissionDenied
        text = request.POST.get("text", "")
        try:
            notepad.save(exp, text, user=request.user, seen=request.POST.get("seen", ""))
        except notepad.Stale as stale:
            context.update(stale=notepad.as_dict(stale.notes), text=text, editing=True)
        except notepad.NoteError as error:
            context.update(error=str(error), text=text, editing=True)
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
