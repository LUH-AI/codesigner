"""`assistant.acts` lets the assistant act on the person's page.

A tool records a `ClientAct` for the person's last browser session, whose page
(assistant/static/assistant/act.js) runs it and reports to `assistant:acted`.
Tools check permissions as the person before `send`, and no action clicks or
submits.
"""

from __future__ import annotations

import time
from datetime import timedelta

from django.conf import settings
from django.utils import timezone

from .models import BrowserPage, ClientAct

#: Unanswered actions expire.
EXPIRES = timedelta(minutes=2)
POLL_SECONDS = 0.2


def page_of(browser: str):
    if not browser:
        return None
    since = timezone.now() - timedelta(seconds=settings.ASSISTANT_PAGE_SECONDS)
    return BrowserPage.objects.filter(browser=browser, seen_at__gte=since).first()


#: Seconds `current` waits for a page report.
REPORT_SECONDS = 3.0


def current(conversation):
    """`current` returns the open page, or None."""
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
    return ClientAct.objects.filter(browser=browser, status=ClientAct.PENDING,
                                    created_at__gte=timezone.now() - EXPIRES)


def send(conversation, kind: str, args: dict, path: str = "") -> ClientAct:
    """`send` waits up to `ASSISTANT_ACT_SECONDS` for the page to do *kind*."""
    act = ClientAct.objects.create(browser=conversation.browser, conversation=conversation,
                                   kind=kind, args=args, path=path)
    return wait(act, settings.ASSISTANT_ACT_SECONDS)


def wait(act: ClientAct, seconds: float) -> ClientAct:
    until = time.monotonic() + seconds
    while True:
        act.refresh_from_db()
        if act.status != ClientAct.PENDING or time.monotonic() >= until:
            return act
        time.sleep(POLL_SECONDS)


def outcome(act: ClientAct) -> str:
    if act.status == ClientAct.DONE:
        return "Done on the person's page." + (f" {act.detail}" if act.detail else "")
    if act.status == ClientAct.FAILED:
        return f"The page could not do it: {act.detail}"
    return ("Sent; the page has not confirmed it yet. It is done when the page next "
            "has the chance, unless the person has left that page.")
