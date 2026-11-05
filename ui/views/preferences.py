from django.conf import settings
from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from access.policy import EDIT, policy
from experiments.models import GlobalSettings
from experiments.services import timestamps
from experiments.services.settings import coerce_settings, set_global_defaults

from .. import times
from ..experiment_settings import SETTING_DEFAULTS, global_defaults, resolve_settings
from ..forms import ExperimentSettingsFields, ExperimentSettingsForm
from ..permissions import experiment_view


def _posted_settings(request):
    """`_posted_settings` reads the settings form through `.coerce_settings`."""
    return coerce_settings(request.POST, SETTING_DEFAULTS)


@experiment_view(EDIT)
def experiment_settings(request, exp):
    if request.method == "POST":
        if "reset" in request.POST or request.POST.get("use_default_settings"):
            exp.use_default_settings = True
            exp.settings = {}
            exp.save(update_fields=["use_default_settings", "settings"])
            return redirect("ui:experiment_settings", pk=exp.pk)

        posted = _posted_settings(request)

        if ("save_as_default" in request.POST
                or "confirm_save_as_default" in request.POST):
            if not policy().may_change_defaults(request):
                raise PermissionDenied

        if "save_as_default" in request.POST:
            return render(request, "ui/save_as_default_confirm.html", {
                "experiment": exp,
                "pending": [key for key, on in posted.items() if on],
            })

        if "confirm_save_as_default" in request.POST:
            set_global_defaults(posted)
            return redirect("ui:experiment_settings", pk=exp.pk)

        exp.use_default_settings = False
        exp.settings = posted
        exp.save(update_fields=["use_default_settings", "settings"])
        return redirect("ui:experiment_settings", pk=exp.pk)

    form = ExperimentSettingsForm(initial={
        **resolve_settings(exp),
        "use_default_settings": exp.use_default_settings,
    })
    return render(request, "ui/experiment_settings.html",
                  {"experiment": exp, "form": form})


def appearance(request):
    """`appearance` stores the theme in a year-long cookie and the assistant
    switch on the account.
    """
    from ..context_processors import THEME_COOKIE, THEMES
    if request.method == "POST":
        response = redirect("ui:appearance")
        chosen = request.POST.get("theme")
        if chosen in THEMES:
            response.set_cookie(THEME_COOKIE, chosen, max_age=365 * 24 * 3600,
                                samesite="Lax", secure=request.is_secure())
        if request.POST.get("assistant") in ("on", "off"):
            times.set_assistant_on(request.user, request.POST["assistant"] == "on")
        return response
    return render(request, "ui/appearance.html", {
        "themes": [("light", _("Light")), ("dark", _("Dark")),
                   ("system", _("Same as my system"))],
        "assistant_offered": settings.ASSISTANT_ENABLED,
        "assistant_on": times.assistant_on(request.user),
    })


def export_preferences(request):
    """`export_preferences` stores the time format with the account, for every
    browser."""
    if request.method == "POST":
        times.set_reader_preference(request.user, request.POST.get("export_timestamps"))
        return redirect("ui:export_preferences")
    return render(request, "ui/export_preferences.html", {
        "modes": [(mode, *times.LABELS[mode]) for mode in timestamps.MODES],
        "chosen": times.reader_preference(request.user),
    })


def account(request):
    """`account` shows the signed-in user's account and their one grant,
    custom models.
    """
    if not settings.REQUIRE_LOGIN:
        raise Http404
    from access.models import account_of
    from access.services.users import set_display_name

    from .accounts import DisplayNameForm

    form = DisplayNameForm(request.POST or None,
                           initial={"display_name": account_of(request.user).display_name})
    if request.method == "POST" and form.is_valid():
        set_display_name(request.user, form.cleaned_data["display_name"])
        messages.success(request, _("Saved your display name."))
        return redirect("ui:account")
    current = policy()
    return render(request, "ui/account.html", {
        "name_form": form,
        "powers": [
            (_("Upload and run custom models"),
             current.may_upload_models(request)),
        ],
        # `.ALLOW_CUSTOM_MODELS` gates every account's grant.
        "custom_models_off": not settings.ALLOW_CUSTOM_MODELS,
    })


def default_experiment_settings(request):
    if not policy().may_change_defaults(request):
        raise PermissionDenied
    gs = GlobalSettings.get_solo()
    if request.method == "POST":
        set_global_defaults(_posted_settings(request))
        return redirect("ui:default_experiment_settings")

    form = ExperimentSettingsFields(initial=global_defaults())
    return render(request, "ui/default_experiment_settings.html", {"form": form})
