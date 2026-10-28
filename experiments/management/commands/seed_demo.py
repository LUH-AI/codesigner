"""`seed_demo` builds a demo instance with groups, roles and sharing to sign in
to.

Every password equals its username, so the command refuses to run unless `DEBUG`
is on or `--force` is given. The demo includes two groups at different seat
limits, experiments at all three sharing levels, finished runs, and one running
job.
"""

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from access.models import Group, Membership
from core.metrics import metrics_for
from experiments.models import Experiment, ExperimentShare, Run

VIEWER, CONTRIBUTOR = ExperimentShare.VIEWER, ExperimentShare.CONTRIBUTOR

#: `--reset` removes only these groups and their people.
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

        lead_v = self._person("vera", vision, Membership.PRIMARY_LEAD)
        ana = self._person("ana", vision)
        ben = self._person("ben", vision)
        lead_n = self._person("nils", nlp, Membership.PRIMARY_LEAD)
        cleo = self._person("cleo", nlp)

        # Ana's three levels; a colleague sees two of them.
        self._experiment(ana, "Wine — Ben contributes", shares={ben: CONTRIBUTOR},
                         runs=[(24, 41.2), (12, 18.9)])
        self._experiment(ana, "Wine — Ben may look", shares={ben: VIEWER},
                         runs=[(8, 12.4)])
        self._experiment(ana, "Wine — not shared", runs=[(30, 55.0)])
        self._experiment(ana, "Diabetes — regression", runs=[(20, 14.6)], task="regression")
        self._experiment(ben, "Iris — Ana may look", shares={ana: VIEWER},
                         runs=[(15, 9.8)])
        self._experiment(lead_v, "Vera's own", runs=[(6, 7.1)])
        # Experiments in nlp-group test the group boundary.
        self._experiment(cleo, "Sentiment", runs=[(18, 27.3)])
        self._experiment(lead_n, "Nils's own")

        # Gives the Jobs panel a row and Stop a target.
        running = self._experiment(ben, "Iris — running now")
        Run.objects.get_or_create(
            experiment=running, status="running",
            defaults={"primary_metric": "accuracy",
                      "stopping": {"max_trials": 60}, "started_by": ben,
                      "backend": "slurm", "job_id": "2519001",
                      "started_at": timezone.now(), "trial_offset": 0})

        self._report(site, [(vision, lead_v, [ana, ben]), (nlp, lead_n, [cleo])])


    def _reset(self) -> None:
        User = get_user_model()
        groups = Group.objects.filter(name__in=DEMO_GROUPS)
        # Distinct, so a user in two demo groups appears once.
        users = User.objects.filter(memberships__group__in=groups).distinct()
        # Experiments go first, since deleting people would leave them ownerless
        # (`owner` is SET_NULL).
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
            user=user, group=group, defaults={"role": Membership.PRIMARY_LEAD})
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
        # Pair-keyed: a person may belong to several groups.
        Membership.objects.get_or_create(
            user=user, group=group, defaults={"role": role})
        return user

    def _experiment(self, owner, name, *, shares=None, runs=(), task="classification"):
        # `get_or_create` would look up a missing column and write only the
        # instance half, so the manager creates both halves.
        exp = Experiment.objects.filter(data__name=name, owner=owner).first()
        if exp is None:
            metric = metrics_for(task)[0]
            exp = Experiment.objects.create(
                name=name, owner=owner, model_name="Random Forest",
                optimizer_name="SMAC", metric_names=metrics_for(task),
                current_metric=metric, task=task, seed=0)
        for user, level in (shares or {}).items():
            ExperimentShare.objects.update_or_create(
                experiment=exp, user=user,
                defaults={"level": level, "granted_by": owner})
        for trials, seconds in runs:
            Run.objects.get_or_create(
                experiment=exp, trial_count=trials,
                defaults={"primary_metric": exp.data.current_metric or "accuracy",
                          "status": "done",
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
            out.write(f"  {'  ' + lead.get_username():10s} group lead — "
                      "sees and runs everyone's here, deletes only their own")
            for member in members:
                out.write(f"  {'  ' + member.get_username():10s} member")
        out.write("\nTry: sign in as ben and look for ana's three experiments — "
                  "he should see two, and be able to run only one of them.\n")
