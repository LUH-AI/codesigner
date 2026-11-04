"""Signing in, and the pages that set a password without the old one: the
first sign-in of a `LocalOnboarding` account, a one-time `SetupLink`, and the
agreement `ACCOUNT_AGREEMENT` supplies. `ui:password_forgot` appears only for
an onboarding with `offers_reset`.
"""

from urllib.parse import urlencode

from django import forms
from django.contrib import messages
from django.contrib.auth import get_user_model, login, views as auth_views
from django.contrib.auth.decorators import login_not_required  # pyright: ignore[reportAttributeAccessIssue]
from django.contrib.auth.forms import AuthenticationForm, SetPasswordForm
from django.http import Http404
from django.shortcuts import redirect, render
from django.urls import reverse, reverse_lazy
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy

from access.backends import account_named, operator
from access.models import Account, account_of
from access.onboarding import onboarding
from access.services import setup_links
from access.services.passwords import accept_agreement, agreement_due, password_set

from ..navigation import safe_next

#: The session key naming the account whose first password is being chosen.
FIRST_SIGN_IN = "first_sign_in"
BACKEND = "access.backends.SiteBackend"


def _first_sign_in_account(identifier: str):
    """`_first_sign_in_account` returns the account of *identifier* when it is
    waiting for its first sign-in to choose a password, else None."""
    user = account_named(identifier)
    if (user is None or not user.is_active or operator(user) or user.has_usable_password()
            or account_of(user).password_pending != Account.FIRST_SIGN_IN):
        return None
    return user


def _first_sign_in_account_by_pk(pk):
    user = get_user_model().objects.filter(pk=pk).first() if pk else None
    return _first_sign_in_account(user.get_username()) if user else None


class SignInForm(AuthenticationForm):
    """`SignInForm` leaves the password optional in the browser, so a
    first sign-in can go by username alone; an empty password still fails
    `AuthenticationForm.clean`."""

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


class PasswordSetupForm(SetPasswordForm):
    """`PasswordSetupForm` sets a password without the old one, and asks for
    the agreement when one is due."""

    accept = forms.BooleanField(required=False, label=gettext_lazy("I accept these terms."))

    def __init__(self, user, *args, terms=None, **kwargs):
        super().__init__(user, *args, **kwargs)
        self.terms = terms
        if terms is None:
            del self.fields["accept"]
        else:
            self.fields["accept"].required = True

    def save(self, commit=True):
        user = super().save(commit)
        if self.terms is not None:
            accept_agreement(user, self.terms[0])
        password_set(user)
        return user


def _set_password_page(request, user, *, title, done):
    """`_set_password_page` shows and handles `PasswordSetupForm` for *user*;
    *done* runs once, after a valid form and before the password is saved, and
    returns False when the page is no longer good for anything."""
    terms = agreement_due(user)
    form = PasswordSetupForm(user, request.POST or None, terms=terms)
    if request.method == "POST" and form.is_valid():
        if not done():
            return render(request, "ui/account/link_dead.html", status=410)
        form.save()
        login(request, user, backend=BACKEND)
        return redirect(safe_next(request) or "ui:home")
    return render(request, "ui/account/set_password.html", {
        "form": form, "terms": terms, "title": title, "who": user.get_username()})


@login_not_required
def first_password(request):
    """`first_password` lets a `LocalOnboarding` account named on the sign-in
    page choose its password."""
    user = _first_sign_in_account_by_pk(request.session.get(FIRST_SIGN_IN))
    if user is None:
        return redirect("ui:login")

    def done():
        request.session.pop(FIRST_SIGN_IN, None)
        return _first_sign_in_account_by_pk(user.pk) is not None

    return _set_password_page(request, user, title=_("Choose your password"), done=done)


@login_not_required
def account_setup(request, token):
    """`account_setup` opens a `SetupLink`. Opening it uses nothing up, since
    mail scanners open links too; only setting the password does."""
    link = setup_links.live(token)
    if link is None or not link.user.is_active or operator(link.user):
        return render(request, "ui/account/link_dead.html", status=410)
    return _set_password_page(request, link.user, title=_("Set your password"),
                              done=lambda: setup_links.use(link))


def agreement_page(request):
    """`agreement_page` asks a signed-in person to accept the current
    agreement, which `.AccountGateMiddleware` sends them to."""
    terms = agreement_due(request.user)
    if terms is None:
        return redirect(safe_next(request) or "ui:home")
    if request.method == "POST" and request.POST.get("accept"):
        accept_agreement(request.user, terms[0])
        return redirect(safe_next(request) or "ui:home")
    return render(request, "ui/account/agreement.html", {"terms": terms})


@login_not_required
def password_forgot(request):
    """`password_forgot` hands the identifier typed to `.onboarding`'s `reset`.
    The answer is the same whether or not an account matched."""
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
