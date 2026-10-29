from django.conf import settings
from django.contrib import messages
from django.http import Http404, HttpResponseBadRequest
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import SHARE
from experiments.models import ExperimentShare
from experiments.services import sharing

from ..permissions import experiment_view


def _back_to_sharing(request, exp):
    """`_back_to_sharing` returns to the sharing controls' page, else the
    experiment."""
    from ..navigation import safe_next
    return redirect(safe_next(request) or reverse("ui:experiment_detail", args=[exp.pk]))


def _colleague(request, exp):
    """`_colleague` only searches the experiment's group, so an outside id gets
    the same 404 as a missing one.
    """
    from django.contrib.auth import get_user_model
    if not settings.REQUIRE_LOGIN or exp.group_id is None:
        raise Http404
    try:
        pk = int(request.POST.get("user", ""))
    except ValueError:
        raise Http404
    colleague = (get_user_model().objects
                 .filter(pk=pk, memberships__group_id=exp.group_id).first())
    if colleague is None or colleague.pk == exp.owner_id:
        raise Http404
    return colleague


@require_POST
@experiment_view(SHARE)
def experiment_share(request, exp):
    """`experiment_share` sets one group member's access to none, viewer or
    contributor.
    """
    colleague = _colleague(request, exp)
    level = request.POST.get("level", "")
    if level not in dict(ExperimentShare.LEVELS) and level != "none":
        return HttpResponseBadRequest(_("Unknown sharing level."))
    sharing.set_access(exp, colleague, level, granted_by=request.user)
    return _back_to_sharing(request, exp)


@require_POST
@experiment_view(SHARE)
def experiment_transfer(request, exp):
    """`experiment_transfer` hands the experiment to a contributor, who only
    gains `SHARE` and `RENAME`. The previous owner stays a contributor.
    """
    colleague = _colleague(request, exp)
    if not sharing.transfer(exp, colleague):
        raise Http404
    messages.success(request, _("%(name)s now belongs to %(user)s. You are a "
                                "contributor.")
                     % {"name": exp.title, "user": colleague.get_username()})
    return _back_to_sharing(request, exp)
