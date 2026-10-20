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
from ..forms import DefaultExperimentSettingsForm, ExperimentSettingsForm
from ..permissions import experiment_view


def _posted_settings(request):
    """The settings as submitted, coerced to the type of each default.

    Almost every setting is a checkbox, where absent means off. The exceptions
    are the numeric ones — counts like `ice_max_curves`, and
    `prior_acceptance_tolerance`, which is a fraction of an objective's own
    scale — clamped to `SETTING_BOUNDS`. A display preference typed into a box
    is not worth failing a form over, and anything unreadable falls back to the
    built-in default rather than to zero, which would mean something specific
    and wrong ("no limit", or "refuse every prior").

    Keyed off `type(default)` rather than a second list of which settings are
    numbers, so declaring one in `SETTING_DEFAULTS` is all it takes — and that
    type is what each is read as, so a count stays a count and a tolerance
    keeps its decimals.
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
    """One experiment's settings: inherit the defaults, override, reset, or
    promote its own settings to be the defaults (which asks first).

    How the search runs is not here. An optimizer's settings are chosen when the
    experiment is created and fixed for its life, like `cv_folds` — see
    `optimizer_form._posted_params`.
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
            # Ask before changing what every inheriting experiment shows.
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
    """How the pages look in this browser: light, dark, or as the system is.

    Kept in a cookie for a year — see `context_processors.theme` on why there
    and not on the account. A choice that is not one of the three is ignored.
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
    """How this reader's exported files carry times, unless an experiment says
    otherwise (ui/times.py). Kept for the reader rather than for
    the browser: it is a decision about what they send, wherever they send it
    from."""
    if request.method == "POST":
        times.set_reader_preference(request.user, request.POST.get("export_timestamps"))
        return redirect("ui:export_preferences")
    return render(request, "ui/export_preferences.html", {
        "modes": [(mode, *times.LABELS[mode]) for mode in timestamps.MODES],
        "chosen": times.reader_preference(request.user),
    })


def account(request):
    """Who you are signed in as, and what you may do here.

    Assembled rather than looked up: the answer now comes from four separate
    grants, held directly or through a group, and before this there was no page
    anywhere that could say what they added up to. An operator reading a bug
    report needs it as much as the person does.

    Absent without accounts — `REQUIRE_LOGIN` off means there is no account to
    describe and every answer would be an unconditional yes.
    """
    if not settings.REQUIRE_LOGIN:
        raise Http404
    current = policy()
    return render(request, "ui/account.html", {
        # Each paired with what it lets you do rather than its codename, which
        # names the grant and not the consequence.
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
        # Its own line because it is an instance-wide floor rather than
        # something about this account: off means off, for everyone.
        "custom_models_off": not settings.ALLOW_CUSTOM_MODELS,
        "is_administrator": request.user.is_superuser,
    })


def default_experiment_settings(request):
    """Edit the default experiment settings that inheriting experiments use.

    Every key in the schema is a checkbox, so an absent key means unchecked.
    """
    if not policy().may_change_defaults(request):
        raise PermissionDenied
    gs = GlobalSettings.get_solo()
    if request.method == "POST":
        gs.default_experiment_settings = _posted_settings(request)
        gs.save(update_fields=["default_experiment_settings"])
        return redirect("ui:default_experiment_settings")

    form = DefaultExperimentSettingsForm(initial=global_defaults())
    return render(request, "ui/default_experiment_settings.html", {"form": form})
