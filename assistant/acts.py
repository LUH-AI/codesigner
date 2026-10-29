"""The assistant acting on the person's page, in their browser.

A tool does not change anything on the server for these: it records a
`ClientAct` addressed to the browser session the person last wrote from, and
the page carries it out (assistant/static/assistant/act.js). Every Codesigner
page of that session follows the session's actions (`assistant:acts`); the one
used last carries each out and says how it went (`assistant:acted`), and the
tool waits a little for that. An action not carried out yet goes to whichever
page of the session opens next, so the person navigating stops nothing.

The page also says what it is (`assistant:page`): which page, its fields and
their values, what it names for the assistant, and which tab of figures it
shows. It says so whenever any of that changes — opening, coming back into
use, a field changed, a tab switched — so the record (`BrowserPage`) stands
for the page until the next report and is read without asking again. Only
without a record does `current` ask the page (`ClientAct.REPORT`). That is what
the assistant sees of it. Nothing a page reports is taken as permission: what may be done is
checked on the server, as the person, before an action is sent.

There is no action that clicks or submits. Saving, creating, running and
deleting stay the person's.
"""

from __future__ import annotations

import time
from datetime import timedelta

from django.conf import settings
from django.utils import timezone

from .models import BrowserPage, ClientAct

#: How long an action waits for a page before it is no longer carried out.
EXPIRES = timedelta(minutes=2)
#: How often a tool waiting on the page looks again, in seconds.
POLL_SECONDS = 0.2


def page_of(browser: str):
    """`page_of` returns the page *browser* has open, or None when none has been seen lately."""
    if not browser:
        return None
    since = timezone.now() - timedelta(seconds=settings.ASSISTANT_PAGE_SECONDS)
    return BrowserPage.objects.filter(browser=browser, seen_at__gte=since).first()


#: How long `current` waits at most for a page asked to say what it shows, in
#: seconds; never longer than an action waits.
REPORT_SECONDS = 3.0


def current(conversation):
    """`current` returns the page *conversation*'s person has open: the record
    of what it last reported, or, without one, what it reports when asked now.
    None when no page answers."""
    page = page_of(conversation.browser)
    if page is not None or not conversation.browser:
        return page
    act = ClientAct.objects.create(browser=conversation.browser, conversation=conversation,
                                   kind=ClientAct.REPORT, args={})
    wait(act, min(REPORT_SECONDS, settings.ASSISTANT_ACT_SECONDS))
    if act.status == ClientAct.PENDING:
        ClientAct.objects.filter(pk=act.pk).update(status=ClientAct.FAILED,
                                                   detail="No page answered.")
    return page_of(conversation.browser)


def pending(browser: str):
    """`pending` returns *browser*'s actions still to be carried out, oldest first."""
    return ClientAct.objects.filter(browser=browser, status=ClientAct.PENDING,
                                    created_at__gte=timezone.now() - EXPIRES)


def send(conversation, kind: str, args: dict, path: str = "") -> ClientAct:
    """`send` asks the person's page to do *kind* with *args* (only on the page
    at *path*, when given) and waits for it to say how it went, up to
    `ASSISTANT_ACT_SECONDS`. `send` returns the action as it then stands."""
    act = ClientAct.objects.create(browser=conversation.browser, conversation=conversation,
                                   kind=kind, args=args, path=path)
    return wait(act, settings.ASSISTANT_ACT_SECONDS)


def wait(act: ClientAct, seconds: float) -> ClientAct:
    """`wait` returns *act* once its page has reported, or as it is after *seconds*."""
    until = time.monotonic() + seconds
    while True:
        act.refresh_from_db()
        if act.status != ClientAct.PENDING or time.monotonic() >= until:
            return act
        time.sleep(POLL_SECONDS)


def outcome(act: ClientAct) -> str:
    """`outcome` returns what the model is told of *act*."""
    if act.status == ClientAct.DONE:
        return "Done on the person's page." + (f" {act.detail}" if act.detail else "")
    if act.status == ClientAct.FAILED:
        return f"The page could not do it: {act.detail}"
    return ("Sent; the page has not confirmed it yet. It is done when the page next "
            "has the chance, unless the person has left that page.")
