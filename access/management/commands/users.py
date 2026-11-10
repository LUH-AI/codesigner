"""`manage.py users` lists and manages accounts. WHO is an email address.
"""

from django.contrib.auth import get_user_model
from django.core.management.base import CommandError

from access.backends import operator
from access.onboarding import onboarding
from access.services import users as user_service
from access.services.passwords import require_new_password
from experiments.management.jsoncommand import JsonCommand


def user_row(user) -> dict:
    return {"email": user.email, "display_name": getattr(getattr(user, "account", None), "display_name", ""),
            "status": getattr(user, "status", None), "active": user.is_active,
            "groups": ", ".join(sorted(m.group.name for m in user.memberships.all())),
            "experiments": getattr(user, "experiment_count", None),
            "last_login": user.last_login, "joined": user.date_joined}


class Command(JsonCommand):
    help = "List and manage accounts."

    def add_subcommands(self, sub):
        sub.add_parser("list")
        for name in ("deactivate", "activate", "delete", "reset-password"):
            sub.add_parser(name).add_argument("who")
        email = sub.add_parser("set-email", help="Change the address somebody signs in with.")
        email.add_argument("who")
        email.add_argument("address")
        custom = sub.add_parser("custom-models")
        custom.add_argument("who")
        custom.add_argument("state", choices=["on", "off"])
        renew = sub.add_parser("require-new-password",
                               help="Make people set a new password at their next request.")
        renew.add_argument("who", nargs="*")
        renew.add_argument("--all", action="store_true",
                           help="Everybody who can sign in to the site.")

    def do_list(self, options):
        return [user_row(u) for u in user_service.list_users()]

    def do_deactivate(self, options):
        user = user_service.named(options["who"])
        user_service.set_active(user, False)
        return {"email": user.email, "active": False}

    def do_activate(self, options):
        user = user_service.named(options["who"])
        user_service.set_active(user, True)
        return {"email": user.email, "active": True}

    def do_delete(self, options):
        user = user_service.named(options["who"])
        user_service.delete_user(user)
        return {"deleted": options["who"]}

    def do_set_email(self, options):
        user = user_service.named(options["who"])
        user_service.set_email(user, options["address"])
        return {"email": user.email}

    def do_custom_models(self, options):
        user = user_service.named(options["who"])
        user_service.set_custom_models(user, options["state"] == "on")
        return {"email": user.email, "custom_models": options["state"] == "on"}

    def do_reset_password(self, options):
        return onboarding().reset(user_service.named(options["who"]))

    def do_require_new_password(self, options):
        if options["all"] == bool(options["who"]):
            raise CommandError("Give either names or --all.")
        if options["all"]:
            users = [u for u in get_user_model().objects.filter(is_active=True)
                     if not operator(u) and u.has_usable_password()]
        else:
            users = [user_service.named(who) for who in options["who"]]
        return {"required": require_new_password(users)}
