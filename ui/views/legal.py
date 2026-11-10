"""`imprint`, `privacy` and `terms` show the deployment's legal texts
(`LEGAL_PAGES`, `ACCOUNT_AGREEMENT`) to everybody, signed in or signed out,
in the active language. Each page answers 404 while its hook is unset."""

from django.contrib.auth.decorators import login_not_required
from django.http import Http404
from django.shortcuts import render
from django.utils.safestring import mark_safe
from django.utils.translation import gettext as _

from access.onboarding import agreement, legal_pages


def _page(request, title, html):
    if not html:
        raise Http404
    return render(request, "ui/legal.html", {"heading": title, "body": mark_safe(html)})


@login_not_required
def imprint(request):
    return _page(request, _("Imprint"), (legal_pages() or {}).get("imprint"))


@login_not_required
def privacy(request):
    return _page(request, _("Privacy policy"), (legal_pages() or {}).get("privacy"))


@login_not_required
def terms(request):
    found = agreement()
    return _page(request, _("Terms of use"), found[1] if found else None)
