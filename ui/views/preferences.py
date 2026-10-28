from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from access.policy import EDIT, policy
from experiments.models import GlobalSettings
from experiments.services import timestamps
from experiments.services.settings import SETTING_BOUNDS, SETTING_CHOICES

from .. import times
from ..experiment_settings import SETTING_DEFAULTS, global_defaults, resolve_settings
from ..forms import ExperimentSettingsFields, ExperimentSettingsForm
from ..permissions import experiment_view


def _posted_settings(request):
    """`_posted_settings` coerces each field to the type of its
    `SETTING_DEFAULTS` entry. An absent checkbox is off, numbers clamp to
    `SETTING_BOUNDS`, and unreadable or off-list values take the default, since
    zero would mean "no limit" or "refuse every prior".
    """
    posted = {}
    for key, default in SETTING_DEFAULTS.items():
        if isinstance(default, bool):
            posted[key] = bool(request.POST.get(key))
            continue
        try:
            value = type(default)(request.POST.get(key, ""))
        except (TypeError, ValueError):
            value = default
        low, high = SETTING_BOUNDS.get(key, (None, None))
        if low is not None:
            value = max(low, min(high, value))
        if key in SETTING_CHOICES and value not in SETTING_CHOICES[key]:
            value = default
        posted[key] = value
    return posted


@experiment_view(EDIT)
def experiment_settings(request, exp):
    """`experiment_settings` inherits, overrides, resets or promotes one
    experiment's settings; optimizer settings stay fixed from creation.
    """
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
            gs = GlobalSettings.get_solo()
            gs.default_experiment_settings = posted
            gs.save(update_fields=["default_experiment_settings"])
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
    """`appearance` saves the theme (light, dark or system) in a year-long
    cookie and ignores any other choice.
    """
    from ..context_processors import THEME_COOKIE, THEMES
    if request.method == "POST":
        response = redirect("ui:appearance")
        chosen = request.POST.get("theme")
        if chosen in THEMES:
            response.set_cookie(THEME_COOKIE, chosen, max_age=365 * 24 * 3600,
                                samesite="Lax", secure=request.is_secure())
        return response
    return render(request, "ui/appearance.html", {
        "themes": [("light", _("Light")), ("dark", _("Dark")),
                   ("system", _("Same as my system"))],
    })


def export_preferences(request):
    """`export_preferences` keeps the reader's export time format on the
    account, so the format follows the reader to any browser."""
    if request.method == "POST":
        times.set_reader_preference(request.user, request.POST.get("export_timestamps"))
        return redirect("ui:export_preferences")
    return render(request, "ui/export_preferences.html", {
        "modes": [(mode, *times.LABELS[mode]) for mode in timestamps.MODES],
        "chosen": times.reader_preference(request.user),
    })


def account(request):
    """`account` lists what the signed-in user may do, assembled from four
    grants held directly or through a group. With `REQUIRE_LOGIN` off, the page
    is a 404.
    """
    if not settings.REQUIRE_LOGIN:
        raise Http404
    current = policy()
    return render(request, "ui/account.html", {
        "powers": [
            (_("See every experiment on this instance"),
             request.user.has_perm("access.view_all_experiments")),
            (_("Run, edit and delete other people's experiments"),
             request.user.has_perm("access.manage_experiments")),
            (_("Change the settings every experiment inherits"),
             current.may_change_defaults(request)),
            (_("Upload and run custom models"),
             current.may_upload_models(request)),
        ],
        # An instance-wide floor, separate from this account's grants.
        "custom_models_off": not settings.ALLOW_CUSTOM_MODELS,
        "is_administrator": request.user.is_superuser,
    })


def default_experiment_settings(request):
    """`default_experiment_settings` edits the defaults inheriting experiments
    use.
    """
    if not policy().may_change_defaults(request):
        raise PermissionDenied
    gs = GlobalSettings.get_solo()
    if request.method == "POST":
        gs.default_experiment_settings = _posted_settings(request)
        gs.save(update_fields=["default_experiment_settings"])
        return redirect("ui:default_experiment_settings")

    form = ExperimentSettingsFields(initial=global_defaults())
    return render(request, "ui/default_experiment_settings.html", {"form": form})
