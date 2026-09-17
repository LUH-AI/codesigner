"""An instance with people in it, to sign in to and look at.

Setting up groups, roles and sharing by hand to check one page is tedious enough
that it does not get done. This builds the whole shape in one go:

    manage.py seed_demo            # build it, leaving anything already there
    manage.py seed_demo --reset    # tear the demo down first and rebuild

Every account's password is its username, which is exactly as bad as it sounds
and exactly why this refuses to run unless `DEBUG` is on or `--force` is given.

What it makes, and why each piece is there:

* a **site admin** in a group of one, so the Site tab has an owner and the
  "a site admin sees no experiments" claim has somebody to be true of;
* **two groups at different limits**, so one of them is full and the seat
  arithmetic is visible rather than theoretical;
* a **lead and two members** in each, so the lead's two views have something to
  be different about;
* experiments at **all three sharing levels** — group, named, and neither —
  because the interesting cases are the boundaries between them;
* **finished runs** with real trial counts and durations, so Usage totals
  something; and one run left **running with a job id**, so Jobs has a row and
  Stop has a target.
"""

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from access.models import Group, Membership

from ...models import Experiment, Run

#: Groups this command owns. `--reset` removes these and everyone in them, and
#: nothing else — a demo that deleted an operator's real group would be a worse
#: problem than the one it solves.
DEMO_GROUPS = ("site", "vision-lab", "nlp-group")


class Command(BaseCommand):
    help = "Create a worked example of groups, roles and sharing, to debug against."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true",
                            help="delete the demo groups and their people first")
        parser.add_argument("--force", action="store_true",
                            help="run even with DEBUG off. The passwords here are "
                                 "the usernames; do not do this on a real instance.")

    @transaction.atomic
    def handle(self, *args, **options):
        if not settings.DEBUG and not options["force"]:
            raise CommandError(
                "DEBUG is off. Every account this creates has its username as "
                "its password, so it refuses to run on what might be a real "
                "instance. Pass --force if this is not one.")

        if options["reset"]:
            self._reset()

        site = self._site_admin()
        vision = self._group("vision-lab", limit=5)
        nlp = self._group("nlp-group", limit=3)

        lead_v = self._person("vera", vision, Membership.LEAD)
        ana = self._person("ana", vision)
        ben = self._person("ben", vision)
        lead_n = self._person("nils", nlp, Membership.LEAD)
        cleo = self._person("cleo", nlp)

        # All three sharing levels, on one person's work, so a colleague signing
        # in sees exactly two of the three and can be asked why.
        self._experiment(ana, "Wine — shared with the group", shared=True,
                         runs=[(24, 41.2), (12, 18.9)])
        self._experiment(ana, "Wine — shared with Ben", shared_with=[ben],
                         runs=[(8, 12.4)])
        self._experiment(ana, "Wine — not shared", runs=[(30, 55.0)])
        self._experiment(ben, "Iris — shared with the group", shared=True,
                         runs=[(15, 9.8)])
        self._experiment(lead_v, "Vera's own", runs=[(6, 7.1)])
        # Another group's, so the boundary has something on the other side.
        self._experiment(cleo, "Sentiment — shared with the group", shared=True,
                         runs=[(18, 27.3)])
        self._experiment(lead_n, "Nils's own")

        # One left running, with a job id, so the Jobs panel is not empty and
        # Stop has something to stop.
        running = self._experiment(ben, "Iris — running now")
        Run.objects.get_or_create(
            experiment=running, status="running",
            defaults={"primary_metric": "accuracy",
                      "stopping": {"max_trials": 60}, "started_by": ben,
                      "backend": "slurm", "job_id": "2519001",
                      "started_at": timezone.now(), "trial_offset": 0})

        self._report(site, [(vision, lead_v, [ana, ben]), (nlp, lead_n, [cleo])])

    # ── the pieces ───────────────────────────────────────────────────────────

    def _reset(self) -> None:
        User = get_user_model()
        groups = Group.objects.filter(name__in=DEMO_GROUPS)
        users = User.objects.filter(membership__group__in=groups)
        # Experiments first: `Membership.group` is PROTECT and `owner` is
        # SET_NULL, so deleting people first would leave rows nobody can reach,
        # which is the exact state `access/admin.py` refuses to create by hand.
        Experiment.objects.filter(owner__in=users).delete()
        Membership.objects.filter(group__in=groups).delete()
        users.delete()
        groups.delete()
        self.stdout.write("Removed the demo groups and their people.")

    def _site_admin(self):
        User = get_user_model()
        user, made = User.objects.get_or_create(
            username="site", defaults={"email": "site@example.org"})
        if made:
            user.set_password("site")
            user.save(update_fields=["password"])
        user.user_permissions.add(Permission.objects.get(
            content_type__app_label="access", codename="manage_site"))
        group = self._group("site", limit=1)
        Membership.objects.get_or_create(
            user=user, defaults={"group": group, "role": Membership.LEAD})
        return user

    def _group(self, name, *, limit):
        group, _ = Group.objects.get_or_create(
            name=name, defaults={"user_limit": limit})
        return group

    def _person(self, username, group, role=Membership.MEMBER):
        User = get_user_model()
        user, made = User.objects.get_or_create(
            username=username, defaults={"email": f"{username}@example.org"})
        if made:
            user.set_password(username)
            user.save(update_fields=["password"])
        Membership.objects.get_or_create(
            user=user, defaults={"group": group, "role": role})
        return user

    def _experiment(self, owner, name, *, shared=False, shared_with=(), runs=()):
        exp, _ = Experiment.objects.get_or_create(
            name=name, owner=owner,
            defaults={"model_name": "Random Forest",
                      "optimizer_name": "SMAC",
                      "metric_names": ["accuracy"], "current_metric": "accuracy",
                      "seed": 0, "shared": shared})
        exp.shared_with.set(shared_with)
        for trials, seconds in runs:
            Run.objects.get_or_create(
                experiment=exp, trial_count=trials,
                defaults={"primary_metric": "accuracy", "status": "done",
                          "stopping": {"max_trials": trials},
                          "started_by": owner, "trial_seconds": seconds,
                          "trial_offset": 0, "stopped_by": "max_trials",
                          "started_at": timezone.now(),
                          "finished_at": timezone.now()})
        return exp

    def _report(self, site, groups) -> None:
        out, ok = self.stdout, self.style.SUCCESS
        out.write(ok("\nSeeded. Every password is the username.\n"))
        out.write(f"  {'site':10s} site admin — Site tab, and no experiments anywhere")
        for group, lead, members in groups:
            out.write(f"\n  {group.name} ({group.member_count}/{group.user_limit} seats)")
            out.write(f"  {'  ' + lead.get_username():10s} group lead — Group tab, "
                      "sees everyone here")
            for member in members:
                out.write(f"  {'  ' + member.get_username():10s} member")
        out.write("\nTry: sign in as ben and look for ana's three experiments — "
                  "he should see two.\n")
