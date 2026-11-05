"""`manage.py jobs` lists and stops in-progress runs of every group.
"""

from django.utils.translation import gettext as _

from access.services.errors import ServiceError
from experiments.management.jsoncommand import JsonCommand
from experiments.models import Run
from experiments.services import site as site_service


class Command(JsonCommand):
    help = "List and stop the runs in progress."

    def add_subcommands(self, sub):
        sub.add_parser("list")
        stop = sub.add_parser("stop")
        stop.add_argument("run", type=int, help="The run's number, from `jobs list`.")
        stop.add_argument("--reason", default="")

    def do_list(self, options):
        return [site_service.job_row(run) for run in site_service.active_runs()]

    def do_stop(self, options):
        run = Run.objects.filter(pk=options["run"], status__in=Run.ACTIVE_STATUSES).first()
        if run is None:
            raise ServiceError(_("Run %(pk)s is not in progress.") % {"pk": options["run"]})
        site_service.stop_job(run, options["reason"] or _("Stopped by a site administrator."))
        job = run.job
        return {"stopped": run.pk, "job_id": job.scheduler_id if job else ""}
