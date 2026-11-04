"""`manage.py setup_links` issues one-time password links and sweeps accounts
whose links expired.
"""

from access.services import setup_links
from access.services import users as user_service
from experiments.management.jsoncommand import JsonCommand


class Command(JsonCommand):
    help = "Issue one-time password links, and sweep accounts whose links expired."

    def add_subcommands(self, sub):
        issue = sub.add_parser("issue")
        issue.add_argument("who")
        issue.add_argument("--hours", type=float, default=24.0)
        issue.add_argument("--base-url", default="",
                           help="Prefix for the printed link, such as https://example.org.")
        sub.add_parser("sweep")

    def do_issue(self, options):
        user = user_service.named(options["who"])
        path = setup_links.issue(user, options["hours"])
        return {"username": user.get_username(),
                "link": options["base_url"].rstrip("/") + path}

    def do_sweep(self, options):
        return {"removed": setup_links.sweep()}
