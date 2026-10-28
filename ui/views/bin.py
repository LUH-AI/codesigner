from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import DELETE, policy
from experiments.services import bin as bin_service

from ..permissions import experiment_view
from .export import _export


@experiment_view(DELETE)
def experiment_delete(request, exp):
    """GET confirms; POST moves *exp* to its people's bins.
    """
    if request.method == "POST":
        bin_service.delete(exp, by=request.user)
        messages.success(request, _("Moved %(name)s to the bin.") % {"name": exp.title})
        return redirect("ui:home")
    return render(request, "ui/delete_confirm.html", {
        "experiment": exp,
        "people": _bin_holders(exp),
    })


def _bin_holders(exp):
    from django.contrib.auth import get_user_model
    if not settings.REQUIRE_LOGIN:
        return []
    return list(get_user_model().objects.filter(pk__in=bin_service.its_people(exp))
                .order_by("username"))


def _binned(request, pk):
    from django.shortcuts import get_object_or_404
    return get_object_or_404(policy().bin(request), pk=pk)


def experiment_bin(request):
    binned = (policy().bin(request)
              .select_related("owner", "deleted_by").order_by("-deleted_at"))
    return render(request, "ui/bin.html", {
        "binned": [{"experiment": exp,
                    # No site bin without accounts, so no expiry.
                    "expires_at": (bin_service.expires_at(exp)
                                   if settings.REQUIRE_LOGIN else None)}
                   for exp in binned],
        "retention_days": settings.BIN_RETENTION_DAYS,
    })


@require_POST
def bin_restore(request, pk):
    exp = _binned(request, pk)
    bin_service.restore(exp, by=request.user)
    messages.success(request, _("Restored %(name)s.") % {"name": exp.title})
    return redirect("ui:experiment_detail", pk=exp.pk)


def bin_download(request, pk):
    """Binning leaves export rules unchanged."""
    return _export(request, _binned(request, pk),
                   cancel_url=reverse("ui:experiment_bin"))


def bin_dismiss(request, pk):
    """`bin_dismiss` confirms on GET. On POST it drops the experiment from the
    requester's bin only, and the last dismissal leaves the experiment to the
    site admins' bin. Without accounts, POST purges the experiment.
    """
    exp = _binned(request, pk)
    if request.method == "POST":
        if settings.REQUIRE_LOGIN:
            bin_service.dismiss(exp, request.user)
        else:
            bin_service.purge(exp)
        return redirect("ui:experiment_bin")
    # Bin entries decide the move to the site bin.
    others = [entry.user for entry in exp.bin_entries.select_related("user")
              .exclude(user_id=request.user.pk).order_by("user__username")]
    return render(request, "ui/bin_dismiss_confirm.html",
                  {"experiment": exp, "others": others})
