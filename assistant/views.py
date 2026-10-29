"""What the chat on a page talks to: one POST to say something, and a stream of
what happens next.

The stream is server-sent events, the browser's own `EventSource`: one
connection, the events in order, each with its id, and — when the connection
drops, or the page is left and another opened — a reconnect that sends the last
id it saw (`Last-Event-ID`) and carries on from there. Each connection is
closed after `ASSISTANT_STREAM_SECONDS` and the browser reconnects by itself,
so none is held open indefinitely.

Under ASGI the stream is asynchronous and holds nothing while it waits; under
WSGI (the development server) it holds a thread. The same view serves both.
`?after=<id>` without the stream's Accept header answers once, as JSON, for
tests and anything that cannot hold a stream.
"""

from __future__ import annotations

import json
import time

from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.http import Http404, HttpResponseBadRequest, JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_GET, require_POST

from access.policy import VIEW

from ui.permissions import BOTH, resolve_experiment

from . import turns
from .models import AssistantEvent, Conversation

#: The longest message a person may send in one go.
MAX_MESSAGE = 4000
#: How often the stream looks for new events, in seconds.
POLL_SECONDS = 0.3
#: How often an idle stream says it is still there, in seconds.
KEEP_ALIVE_SECONDS = 15.0
#: How many events a page replays when it opens: the end of a long conversation.
REPLAY = 300


def _enabled():
    if not settings.ASSISTANT_ENABLED:
        raise Http404


def _owner(request):
    user = request.user
    return user if user.is_authenticated else None


def conversations_of(request):
    """The conversations *request*'s person may open: their own, or — without
    accounts — the instance's."""
    return Conversation.objects.filter(owner=_owner(request))


def conversation_for_page(request, experiment_id=None):
    """The conversation a page shows: its experiment's on an experiment's page,
    else the one the person is planning in. None when there is none yet."""
    found = conversations_of(request)
    if experiment_id is not None:
        found = found.filter(experiment_id=experiment_id)
    else:
        found = found.filter(phase=Conversation.PLANNING)
    return found.order_by("-updated_at").first()


def chat_context(request, experiment_id=None, url_name="") -> dict:
    """What the chat (assistant/_chat.html) needs on a page about
    *experiment_id*, or about none: the conversation it continues, where its
    replay starts, whether anything has been said in it yet, and — while
    planning — the dataset and what can be chosen instead."""
    conversation = _moved_on(conversation_for_page(request, experiment_id))
    planning = experiment_id is None and (conversation is None
                                          or conversation.phase == Conversation.PLANNING)
    out = {
        "conversation": conversation.pk if conversation else "",
        "experiment": experiment_id or "",
        "after": replay_from(conversation),
        "live_after": (conversation.events.order_by("-id").values_list("id", flat=True).first()
                       if conversation else None) or 0,
        "busy": bool(conversation and turns.busy(conversation)),
        "started": bool(conversation and conversation.events.filter(kind="user").exists()),
        "planning": planning,
        "fill": _fill(conversation, url_name) if request.method == "GET" else [],
    }
    if planning:
        from access.policy import groups_to_choose_from
        from core.datasets import demo_datasets

        out["dataset"] = conversation.dataset_name if conversation else ""
        out["dataset_is_demo"] = bool(conversation and conversation.demo_dataset)
        out["dataset_upload"] = conversation.dataset_name if conversation and conversation.dataset else ""
        out["demos"] = sorted(demo_datasets())
        out["groups"] = [(g.pk, g.name) for g in groups_to_choose_from(request.user)]
    return out


def _fill(conversation, url_name) -> list:
    """What the page at *url_name* is filled with from *conversation*'s
    approved plan: a setup step until it has been saved, when the values saved
    are the person's; the run form until the experiment's first run."""
    from ui import setup_steps as steps

    from .plan import RUN, fill_for
    from .tools.client import FILLABLE

    if conversation is None or not conversation.plan or conversation.experiment is None:
        return []
    exp = conversation.experiment
    step = FILLABLE.get(url_name)
    if step is not None:
        if conversation.phase != Conversation.SETUP or not exp.draft or steps.saved(exp, step):
            return []
        return fill_for(conversation.plan, step)
    if url_name == "ui:experiment_detail" and not exp.draft and not exp.runs.exists():
        return fill_for(conversation.plan, RUN)
    return []


def _moved_on(conversation):
    """*conversation*, in the dashboard phase once its draft has been created."""
    if (conversation is not None and conversation.phase == Conversation.SETUP
            and conversation.experiment is not None and not conversation.experiment.draft):
        conversation.phase = Conversation.DASHBOARD
        conversation.save(update_fields=["phase"])
    return conversation


def replay_from(conversation) -> int:
    """The event id a page opening *conversation* starts after."""
    if conversation is None:
        return 0
    ids = list(conversation.events.order_by("-id").values_list("id", flat=True)[:REPLAY])
    return ids[-1] - 1 if ids else 0


@require_POST
def message(request):
    """The person says *text*, in *conversation* or a new one (about
    *experiment*, when the page is one's). Starts the turn and answers with
    where to follow it."""
    _enabled()
    text = (request.POST.get("text") or "").strip()
    if not text:
        return HttpResponseBadRequest("nothing said")
    if len(text) > MAX_MESSAGE:
        return HttpResponseBadRequest("too long")

    conversation = _conversation(request)
    if conversation is None:
        experiment = None
        if request.POST.get("experiment"):
            try:
                experiment = resolve_experiment(request, _int(request.POST["experiment"]),
                                                VIEW, BOTH)
            except PermissionDenied:
                raise Http404
        conversation = _new_conversation(request, experiment)

    note_browser(request, conversation)
    after = conversation.events.order_by("-id").values_list("id", flat=True).first() or 0
    if not turns.start(conversation, text):
        return JsonResponse({"conversation": conversation.pk, "busy": True}, status=409)
    return JsonResponse({"conversation": conversation.pk, "after": after})


@require_POST
def dataset(request):
    """The person chooses the dataset to plan with: a demo by name, or an upload.

    Kept with the conversation, checked as setup step 1 would check it
    (`drafts.check_dataset`). No draft is made for it yet: that happens when the
    plan is approved. With `clear`, the dataset is let go. Only while the
    conversation is planning."""
    from ui import drafts

    from .loop import emit

    _enabled()
    conversation = _conversation(request) or _new_conversation(request)
    if conversation.phase != Conversation.PLANNING:
        return JsonResponse({"error": "This conversation's plan is approved."}, status=400)
    if request.POST.get("clear"):
        conversation.forget_dataset()
        conversation.save()
        emit(conversation, "dataset", name="", demo=False)
        return JsonResponse({"conversation": conversation.pk, "dataset": ""})
    upload = request.FILES.get("dataset_file")
    stem, refusal = drafts.check_dataset(request.POST.get("demo") or "", upload)
    if not refusal:
        group = request.POST.get("group")
        chosen, refusal = drafts.check_group(request, _int(group) if group else None)
    if refusal:
        return JsonResponse({"conversation": conversation.pk, "error": refusal}, status=400)
    conversation.forget_dataset()
    conversation.demo_dataset = stem
    if not stem:
        conversation.dataset_filename = upload.name.rsplit("/", 1)[-1]
        conversation.dataset.save(conversation.dataset_filename, upload, save=False)
    conversation.group = chosen
    conversation.save()
    emit(conversation, "dataset", name=conversation.dataset_name, demo=bool(stem))
    return JsonResponse({"conversation": conversation.pk, "dataset": conversation.dataset_name})


@require_POST
def approve(request):
    """The person approves the plan: checked once more against the setup's own
    forms, then — and only now — a draft is made, owned by them, and the
    dataset put on it as setup step 1 would. The conversation moves on to
    filling the setup in. Only the person can: no tool does this."""
    from django.core.files import File

    from ui import drafts

    from experiments.services import storage
    from ui import setup_steps as steps

    from . import plan as plans
    from .loop import emit

    _enabled()
    conversation = _conversation(request)
    if conversation is None:
        raise Http404
    if conversation.phase != Conversation.PLANNING:
        return JsonResponse({"problems": ["There is no plan to approve."]}, status=400)
    if turns.busy(conversation):
        return JsonResponse({"problems": ["The assistant is still answering."]}, status=409)
    found = plans.problems(conversation.plan, request, conversation.dataset_path(),
                           conversation.group_id)
    if found:
        return JsonResponse({"problems": found}, status=400)

    exp = drafts.new(request)
    if conversation.dataset:
        with conversation.dataset.open("rb") as handle:
            outcome = drafts.stage_dataset(request, exp, upload=File(handle, name=conversation.dataset_name),
                                           group=conversation.group_id)
    else:
        outcome = drafts.stage_dataset(request, exp, demo=conversation.demo_dataset,
                                       group=conversation.group_id)
    if not outcome.saved:
        if not outcome.gone:
            storage.delete_experiment_files(exp)
        return JsonResponse({"problems": [outcome.refusal]
                             + [text for _, text in outcome.notices if text]}, status=400)
    conversation.forget_dataset()
    conversation.experiment = exp
    conversation.phase = Conversation.SETUP
    conversation.browser = browser_of(request)
    conversation.save()
    url = steps.url(exp, steps.SETUP)
    emit(conversation, "approved", url=url)
    return JsonResponse({"url": url})


@require_POST
def reset(request):
    """The person throws the conversation away and starts again: the chat is
    deleted, with the dataset it was planning for, so the home page is as it
    was before anything was said. Only while planning — nothing else was made
    yet. What the assistant did stays in the audit (`AgentAction`)."""
    _enabled()
    conversation = _conversation(request)
    if conversation is None:
        return JsonResponse({"reset": True})
    if conversation.phase != Conversation.PLANNING:
        return JsonResponse({"error": "Only a conversation still planning can be reset."},
                            status=400)
    if turns.busy(conversation):
        return JsonResponse({"error": "The assistant is still answering."}, status=409)
    conversation.forget_dataset()
    conversation.delete()
    return JsonResponse({"reset": True})


@require_GET
def window(request):
    """The chat in a window of its own, popped out of a page: the conversation
    about *experiment*, or the one being planned."""
    from django.shortcuts import render

    _enabled()
    experiment_id = None
    if request.GET.get("experiment"):
        try:
            experiment_id = resolve_experiment(request, _int(request.GET["experiment"]),
                                               VIEW, BOTH).pk
        except PermissionDenied:
            raise Http404
    return render(request, "assistant/window.html",
                  {"assistant": chat_context(request, experiment_id)})


def browser_of(request) -> str:
    """*request*'s browser session, started if it has none yet."""
    if not request.session.session_key:
        request.session.save()
    return request.session.session_key


def note_browser(request, conversation) -> None:
    """Make *request*'s browser the one *conversation*'s actions go to."""
    browser = browser_of(request)
    if conversation.browser != browser:
        conversation.browser = browser
        conversation.save(update_fields=["browser"])


def _conversation(request):
    """The person's conversation named in the POST, or None if none is; 404 for
    one that is not theirs."""
    if not request.POST.get("conversation"):
        return None
    found = conversations_of(request).filter(pk=_int(request.POST["conversation"])).first()
    if found is None:
        raise Http404
    return found


def _new_conversation(request, experiment=None):
    """A conversation for *request*'s person: planning one, or — started from an
    experiment's page — about setting up that draft, or about that experiment."""
    if settings.REQUIRE_LOGIN and not request.session.session_key:
        request.session.save()
    phase = Conversation.PLANNING
    if experiment is not None:
        phase = Conversation.SETUP if experiment.draft else Conversation.DASHBOARD
    return Conversation.objects.create(
        owner=_owner(request), experiment=experiment, phase=phase,
        session_key=(request.session.session_key or "") if settings.REQUIRE_LOGIN else "")


@require_GET
def events(request, pk):
    """What has happened in conversation *pk* since the last event the page saw."""
    _enabled()
    conversation = conversations_of(request).filter(pk=pk).first()
    if conversation is None:
        raise Http404
    after = _int(request.headers.get("Last-Event-ID") or request.GET.get("after") or 0)
    if "text/event-stream" not in request.headers.get("Accept", ""):
        found = AssistantEvent.objects.filter(conversation_id=pk, id__gt=after)[:REPLAY]
        return JsonResponse({"events": [_as_dict(e) for e in found],
                             "busy": turns.busy(conversation)})

    from django.core.handlers.asgi import ASGIRequest

    stream = _async_stream if isinstance(request, ASGIRequest) else _sync_stream
    response = StreamingHttpResponse(stream(pk, after), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def _sync_stream(pk, after):
    from django.db import close_old_connections

    yield "retry: 1000\n\n"
    last, until, quiet = after, time.monotonic() + settings.ASSISTANT_STREAM_SECONDS, time.monotonic()
    while time.monotonic() < until:
        found = list(AssistantEvent.objects.filter(conversation_id=pk, id__gt=last)[:REPLAY])
        for event in found:
            last = event.id
            yield _frame(event)
        if found:
            quiet = time.monotonic()
        elif time.monotonic() - quiet > KEEP_ALIVE_SECONDS:
            quiet = time.monotonic()
            yield ": still here\n\n"
        close_old_connections()
        time.sleep(POLL_SECONDS)


async def _async_stream(pk, after):
    import asyncio

    from asgiref.sync import sync_to_async

    def fetch(last):
        return list(AssistantEvent.objects.filter(conversation_id=pk, id__gt=last)[:REPLAY])

    yield "retry: 1000\n\n"
    last, until, quiet = after, time.monotonic() + settings.ASSISTANT_STREAM_SECONDS, time.monotonic()
    while time.monotonic() < until:
        found = await sync_to_async(fetch)(last)
        for event in found:
            last = event.id
            yield _frame(event)
        if found:
            quiet = time.monotonic()
        elif time.monotonic() - quiet > KEEP_ALIVE_SECONDS:
            quiet = time.monotonic()
            yield ": still here\n\n"
        await asyncio.sleep(POLL_SECONDS)


def _frame(event) -> str:
    """One event in the stream's format."""
    return f"id: {event.id}\nevent: {event.kind}\ndata: {json.dumps(event.payload)}\n\n"


def _as_dict(event) -> dict:
    return {"id": event.id, "kind": event.kind, "payload": event.payload}


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise Http404


#: The most a page may report of its fields, and of the elements it names.
PAGE_ITEMS = 300
#: The longest value of a field kept from a page's report.
PAGE_VALUE = 500


@require_POST
def page(request):
    """The page open in this browser says what it is (assistant/acts.py): its
    path and title, its form's fields and values, and the elements it names.
    Sent as JSON when a page opens and when it is used again."""
    import json as _json

    from django.urls import Resolver404, resolve
    from django.utils import timezone

    from .models import BrowserPage

    _enabled()
    try:
        report = _json.loads(request.body or b"{}")
    except ValueError:
        return HttpResponseBadRequest("not JSON")
    if not isinstance(report, dict):
        return HttpResponseBadRequest("not an object")
    path = str(report.get("path") or "")[:500]
    url_name, experiment_id = "", None
    try:
        match = resolve(path)
        url_name = match.view_name
        if getattr(match.func, "experiment_action", None):
            experiment_id = match.kwargs.get("pk")
    except Resolver404:
        pass
    BrowserPage.objects.update_or_create(
        browser=browser_of(request),
        defaults={"path": path, "url_name": url_name, "experiment_id": experiment_id,
                  "title": str(report.get("title") or "")[:300],
                  "fields": _clipped(report.get("fields")),
                  "targets": _clipped(report.get("targets")),
                  "seen_at": timezone.now()})
    return JsonResponse({"ok": True})


def _clipped(items) -> list:
    """A page's list of fields or elements, kept to plain, bounded values."""
    out = []
    for item in (items if isinstance(items, list) else [])[:PAGE_ITEMS]:
        if not isinstance(item, dict):
            continue
        kept = {}
        for key, value in item.items():
            if key == "options" and isinstance(value, list):
                kept[key] = [[str(v)[:PAGE_VALUE] for v in (o if isinstance(o, list) else [o])][:2]
                             for o in value[:PAGE_ITEMS]]
            elif isinstance(value, (bool, int, float)) or value is None:
                kept[str(key)[:50]] = value
            else:
                kept[str(key)[:50]] = str(value)[:PAGE_VALUE]
        out.append(kept)
    return out


@require_GET
def acts(request):
    """The actions for this browser's pages to carry out: those still pending,
    then each new one, as server-sent events (assistant/acts.py)."""
    _enabled()
    browser = browser_of(request)
    from django.core.handlers.asgi import ASGIRequest

    stream = _async_acts if isinstance(request, ASGIRequest) else _sync_acts
    response = StreamingHttpResponse(stream(browser), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


#: How often a page following its actions is noted as open, in seconds.
SEEN_SECONDS = 10.0


def _act_frame(act) -> str:
    data = {"id": act.id, "kind": act.kind, "args": act.args, "path": act.path}
    return f"id: {act.id}\nevent: act\ndata: {json.dumps(data)}\n\n"


def _touch(browser) -> None:
    from django.utils import timezone

    from .models import BrowserPage

    BrowserPage.objects.filter(browser=browser).update(seen_at=timezone.now())


def _sync_acts(browser):
    from django.db import close_old_connections

    from . import acts as client

    yield "retry: 1000\n\n"
    sent, until = set(), time.monotonic() + settings.ASSISTANT_STREAM_SECONDS
    quiet = seen = 0.0
    while time.monotonic() < until:
        for act in client.pending(browser).exclude(id__in=sent):
            sent.add(act.id)
            yield _act_frame(act)
        now = time.monotonic()
        if now - seen > SEEN_SECONDS:
            seen = now
            _touch(browser)
        if now - quiet > KEEP_ALIVE_SECONDS:
            quiet = now
            yield ": still here\n\n"
        close_old_connections()
        time.sleep(POLL_SECONDS)


async def _async_acts(browser):
    import asyncio

    from asgiref.sync import sync_to_async

    from . import acts as client

    def fetch(sent):
        return list(client.pending(browser).exclude(id__in=sent))

    yield "retry: 1000\n\n"
    sent, until = set(), time.monotonic() + settings.ASSISTANT_STREAM_SECONDS
    quiet = seen = 0.0
    while time.monotonic() < until:
        for act in await sync_to_async(fetch)(set(sent)):
            sent.add(act.id)
            yield _act_frame(act)
        now = time.monotonic()
        if now - seen > SEEN_SECONDS:
            seen = now
            await sync_to_async(_touch)(browser)
        if now - quiet > KEEP_ALIVE_SECONDS:
            quiet = now
            yield ": still here\n\n"
        await asyncio.sleep(POLL_SECONDS)


@require_POST
def acted(request):
    """A page says how an action went: *act*, *ok*, and a word on it."""
    from .models import ClientAct

    _enabled()
    status = ClientAct.DONE if request.POST.get("ok") == "1" else ClientAct.FAILED
    ClientAct.objects.filter(pk=_int(request.POST.get("act")), browser=browser_of(request),
                             status=ClientAct.PENDING).update(
        status=status, detail=(request.POST.get("detail") or "")[:1000])
    return JsonResponse({"ok": True})
