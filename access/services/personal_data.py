"""`export_zip` gathers what codesigner stores about one person into a zip they
download themselves (DSGVO Art. 15 and 20): `account.json` with their account,
memberships, accepted terms, shares, bin, history entries, runs, notes and
assistant chats, and an `.ihpo` per experiment they own.
"""

import io
import json
import zipfile

from django.utils import timezone


def _when(value):
    return value.isoformat() if value else None


def collect(user) -> dict:
    """`collect` returns everything in `account.json`, oldest first within each
    list."""
    from access.models import account_of
    from assistant.models import AgentAction, ExperimentNotes
    from experiments.models import BinEntry, ExperimentEvent, ExperimentShare, Run

    account = account_of(user)

    def named(experiment):
        return {"identifier": experiment.identifier, "name": experiment.title}

    return {
        "exported_at": _when(timezone.now()),
        "account": {
            "email": user.email,
            "display_name": account.display_name,
            "joined": _when(user.date_joined),
            "last_sign_in": _when(user.last_login),
            "active": user.is_active,
            "password_changed": _when(account.password_changed_at),
        },
        "memberships": [
            {"group": m.group.name, "role": m.role, "joined": _when(m.joined_at)}
            for m in user.memberships.select_related("group").order_by("joined_at")],
        "accepted_terms": [
            {"version": a.version, "accepted": _when(a.accepted_at)}
            for a in user.agreements.order_by("accepted_at")],
        "shared_with_you": [
            {**named(s.experiment), "level": s.level, "since": _when(s.granted_at)}
            for s in ExperimentShare.objects.filter(user=user)
            .select_related("experiment__data").order_by("granted_at")],
        "you_shared": [
            {**named(s.experiment), "with": s.user.email, "level": s.level,
             "since": _when(s.granted_at)}
            for s in ExperimentShare.objects.filter(granted_by=user)
            .select_related("experiment__data", "user").order_by("granted_at")],
        "in_your_bin": [
            {**named(b.experiment), "since": _when(b.binned_at)}
            for b in BinEntry.objects.filter(user=user)
            .select_related("experiment__data").order_by("binned_at")],
        "history_entries": [
            {"experiment": e.data.experiment.identifier if hasattr(e.data, "experiment") else None,
             "kind": e.kind, "at": _when(e.at), "payload": e.payload}
            for e in ExperimentEvent.objects.filter(actor=user)
            .select_related("data__experiment").order_by("at", "id")],
        "runs_started": [
            {**named(r.experiment), "status": r.status, "queued": _when(r.created_at),
             "started": _when(r.started_at), "finished": _when(r.finished_at),
             "trials": r.trial_count}
            for r in Run.objects.filter(started_by=user)
            .select_related("experiment__data").order_by("id")],
        "notes_last_edited": [
            {**named(n.experiment), "text": n.text, "edited": _when(n.updated_at)}
            for n in ExperimentNotes.objects.filter(updated_by=user)
            .select_related("experiment__data")],
        "assistant_conversations": [
            {"started": _when(c.created_at), "last_active": _when(c.updated_at),
             "messages": [{"role": m.role, "text": m.content, "at": _when(m.created_at)}
                          for m in c.messages.order_by("id")]}
            for c in user.conversations.order_by("created_at")],
        "assistant_actions": [
            {"tool": a.tool, "arguments": a.arguments, "outcome": a.outcome,
             "at": _when(a.created_at)}
            for a in AgentAction.objects.filter(user=user).order_by("created_at")],
    }


def export_zip(user) -> bytes:
    """`export_zip` adds an `.ihpo` per experiment *user* owns, in use or in a
    bin, under `experiments/`, with real times and no stack traces."""
    from experiments.services.snapshot import export

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("account.json", json.dumps(collect(user), indent=2,
                                                   ensure_ascii=False, default=str))
        for experiment in user.experiments.select_related("data").order_by("id"):
            bundle.writestr(f"experiments/{experiment.identifier}.ihpo",
                            export(experiment, "absolute", False))
    return buffer.getvalue()
