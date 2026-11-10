"""`sweep` deletes personal data past its retention, as the privacy policy
states it: expired sessions, failed sign-ins (with their IP address) after
`PRIVACY_FAILED_SIGN_IN_DAYS`, and assistant chats, actions and page reports
nobody touched for `ASSISTANT_RETENTION_DAYS`."""

from datetime import timedelta

from django.conf import settings
from django.utils import timezone


def sweep(dry_run: bool = False) -> dict:
    """`sweep` returns how many rows of each kind it deleted, or would delete
    with *dry_run*."""
    from django.contrib.sessions.models import Session

    from assistant.models import AgentAction, BrowserPage, Conversation

    now = timezone.now()
    sign_ins = now - timedelta(days=settings.PRIVACY_FAILED_SIGN_IN_DAYS)
    chats = now - timedelta(days=settings.ASSISTANT_RETENTION_DAYS)
    due = {
        "sessions": Session.objects.filter(expire_date__lt=now),
        "conversations": Conversation.objects.filter(updated_at__lt=chats),
        "assistant_actions": AgentAction.objects.filter(created_at__lt=chats),
        "page_reports": BrowserPage.objects.filter(seen_at__lt=chats),
    }
    try:
        from axes.models import AccessAttempt, AccessFailureLog, AccessLog
    except ImportError:
        pass
    else:
        due["failed_sign_ins"] = AccessFailureLog.objects.filter(attempt_time__lt=sign_ins)
        due["lockouts"] = AccessAttempt.objects.filter(attempt_time__lt=sign_ins)
        due["sign_in_log"] = AccessLog.objects.filter(attempt_time__lt=sign_ins)
    counts = {name: rows.count() for name, rows in due.items()}
    if not dry_run:
        # A conversation's uploaded dataset is a file the row deletion leaves.
        for conversation in due["conversations"].exclude(dataset=""):
            conversation.dataset.delete(save=False)
        for rows in due.values():
            rows.delete()
    return counts
