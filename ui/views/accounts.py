"""`accounts` serves sign-in and the password and agreement pages.
"""

from urllib.parse import urlencode

from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model, views as auth_views
from django.contrib.auth.decorators import login_not_required  # pyright: ignore[reportAttributeAccessIssue]
from django.contrib.auth.forms import AuthenticationForm, SetPasswordForm
from django.contrib.auth.password_validation import password_validators_help_texts
from django.http import Http404
from django.shortcuts import redirect, render
from django.urls import reverse, reverse_lazy
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy

from access.backends import account_named, operator
from access.models import DISPLAY_NAME_MIN, Account, account_of
from access.onboarding import onboarding
from access.services import setup_links
from access.services.passwords import accept_agreement, agreement_due, password_set
from access.services.users import set_display_name

from ..navigation import safe_next

#: The key stores the first-sign-in account's pk.
FIRST_SIGN_IN = "first_sign_in"


def _first_sign_in_account(identifier: str):
    user = account_named(identifier)
    if (user is None or not user.is_active or operator(user) or user.has_usable_password()
            or account_of(user).password_pending != Account.FIRST_SIGN_IN):
        return None
    return user


def _first_sign_in_account_by_pk(pk):
    user = get_user_model().objects.filter(pk=pk).first() if pk else None
    return _first_sign_in_account(user.get_username()) if user else None


class SignInForm(AuthenticationForm):
    """`SignInForm` lets a first sign-in submit a username alone; `clean` still
    fails an empty password."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["password"].required = False

    def clean(self):
        if not self.cleaned_data.get("password"):
            raise self.get_invalid_login_error()
        return super().clean()


class SignInView(auth_views.LoginView):
    template_name = "ui/account/login.html"
    form_class = SignInForm

    def post(self, request, *args, **kwargs):
        user = _first_sign_in_account(request.POST.get("username", ""))
        if user is not None:
            request.session[FIRST_SIGN_IN] = user.pk
            target = reverse("ui:first_password")
            following = safe_next(request)
            return redirect(f"{target}?{urlencode({'next': following})}" if following else target)
        return super().post(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        return {**super().get_context_data(**kwargs), "onboarding": onboarding()}


def display_name_field():
    return forms.CharField(
        required=False, min_length=DISPLAY_NAME_MIN, max_length=150,
        label=gettext_lazy("Display name"),
        help_text=gettext_lazy("Shown to others instead of your email address."))


class DisplayNameForm(forms.Form):
    display_name = display_name_field()


class DeleteAccountForm(forms.Form):
    """`DeleteAccountForm` asks for the password and the address typed out, so
    the deletion is deliberate."""

    password = forms.CharField(label=gettext_lazy("Your password"), widget=forms.PasswordInput(
        attrs={"autocomplete": "current-password"}))
    email = forms.CharField(label=gettext_lazy("Your email address, typed out"))

    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user

    def clean_password(self):
        if not self.user.check_password(self.cleaned_data["password"]):
            raise forms.ValidationError(_("That is not your password."))
        return self.cleaned_data["password"]

    def clean_email(self):
        if self.cleaned_data["email"].strip().lower() != (self.user.email or "").lower():
            raise forms.ValidationError(_("That is not the address you sign in with."))
        return self.cleaned_data["email"]


class PasswordSetupForm(SetPasswordForm):
    """`PasswordSetupForm` also takes an optional display name, and asks for
    the agreement when one is due, with a confirmation that the privacy
    policy was read when `LEGAL_PAGES` supplies one. `password_rules` lists
    the validators' requirements once, under both password fields."""

    display_name = display_name_field()
    accept = forms.BooleanField(required=False, label=gettext_lazy("I accept the terms of use."))
    privacy_read = forms.BooleanField(required=False,
                                      label=gettext_lazy("I have read the privacy policy."))
    field_order = ["new_password1", "new_password2", "display_name", "accept", "privacy_read"]

    def __init__(self, user, *args, terms=None, **kwargs):
        super().__init__(user, *args, **kwargs)
        self.fields["new_password1"].label = _("Password")
        self.fields["new_password2"].label = _("Confirm password")
        name = self.fields["display_name"]
        name.label = _("Display name (optional)")
        name.initial = account_of(user).display_name
        name.widget.attrs["placeholder"] = user.email or ""
        for field in ("new_password1", "new_password2", "display_name"):
            self.fields[field].help_text = ""
        self.password_rules = password_validators_help_texts()
        self.terms = terms
        if terms is None:
            del self.fields["accept"]
        else:
            self.fields["accept"].required = True
        if terms is None or not settings.LEGAL_PAGES:
            del self.fields["privacy_read"]
        else:
            self.fields["privacy_read"].required = True

    def save(self, commit=True):
        user = super().save(commit)
        set_display_name(user, self.cleaned_data.get("display_name", ""))
        if self.terms is not None:
            accept_agreement(user, self.terms[0])
        password_set(user)
        return user


def _set_password_page(request, user, *, title, done):
    """`_set_password_page` calls *done* before saving a valid form and renders
    `link_dead.html` when *done* returns False. A saved form leads to the
    sign-in page."""
    terms = agreement_due(user)
    form = PasswordSetupForm(user, request.POST or None, terms=terms)
    if request.method == "POST" and form.is_valid():
        if not done():
            return render(request, "ui/account/link_dead.html", status=410)
        form.save()
        messages.success(request, _("Your account is ready. Sign in with your new password."))
        target = reverse("ui:login")
        if following := safe_next(request):
            target += "?" + urlencode({"next": following})
        return redirect(target)
    return render(request, "ui/account/set_password.html", {
        "form": form, "terms": terms, "title": title})


@login_not_required
def first_password(request):
    user = _first_sign_in_account_by_pk(request.session.get(FIRST_SIGN_IN))
    if user is None:
        return redirect("ui:login")

    def done():
        request.session.pop(FIRST_SIGN_IN, None)
        return _first_sign_in_account_by_pk(user.pk) is not None

    return _set_password_page(request, user, title=_("Choose your password"), done=done)


@login_not_required
def account_setup(request, token):
    """`account_setup` only uses the `SetupLink` on a valid POST, since mail
    scanners open links."""
    link = setup_links.live(token)
    if link is None or not link.user.is_active or operator(link.user):
        return render(request, "ui/account/link_dead.html", status=410)
    return _set_password_page(request, link.user, title=_("Set your password"),
                              done=lambda: setup_links.use(link))


def agreement_page(request):
    """`.AccountGateMiddleware` redirects to `agreement_page` while an agreement
    is due."""
    terms = agreement_due(request.user)
    if terms is None:
        return redirect(safe_next(request) or "ui:home")
    if (request.method == "POST" and request.POST.get("accept")
            and (request.POST.get("privacy_read") or not settings.LEGAL_PAGES)):
        accept_agreement(request.user, terms[0])
        return redirect(safe_next(request) or "ui:home")
    return render(request, "ui/account/agreement.html", {"terms": terms})


@login_not_required
def password_forgot(request):
    """`password_forgot` passes a matched account to `reset` and shows one
    message either way."""
    hook = onboarding()
    if not hook.offers_reset:
        raise Http404
    if request.method == "POST":
        user = hook.find(request.POST.get("identifier", ""))
        if user is not None and user.is_active and not operator(user):
            hook.reset(user)
        messages.info(request, _("If an account matches, a message is on its way."))
        return redirect("ui:login")
    return render(request, "ui/account/password_forgot.html", {"onboarding": hook})


class PasswordChangeView(auth_views.PasswordChangeView):
    template_name = "ui/account/password_change.html"
    success_url = reverse_lazy("ui:password_changed")

    def form_valid(self, form):
        response = super().form_valid(form)
        password_set(self.request.user)
        return response
