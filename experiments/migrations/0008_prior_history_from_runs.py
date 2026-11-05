"""The history records a prior when a run takes it up
(`experiments.services.history.record_run_priors`), no longer at every edit of
the figure. This migration drops the prior entries recorded per edit, including
`prior_reset`, and writes them again from each run's own priors. A `created`
entry loses its copy of the set-up, which the timeline rebuilds from the
experiment and its later changes.
"""

from django.db import migrations

OLD_PRIOR_KINDS = ("prior_stated", "prior_edited", "prior_withdrawn", "prior_reset")


def priors_from_runs(apps, schema_editor):
    Experiment = apps.get_model("experiments", "Experiment")
    ExperimentEvent = apps.get_model("experiments", "ExperimentEvent")
    Run = apps.get_model("experiments", "Run")

    ExperimentEvent.objects.filter(kind__in=OLD_PRIOR_KINDS).delete()
    ExperimentEvent.objects.filter(kind="created").update(payload={})
    for exp in Experiment.objects.filter(draft=False).only("pk", "data_id"):
        before = {}
        for run in Run.objects.filter(experiment_id=exp.pk).order_by("pk"):
            now = run.priors or {}
            for hp in sorted(set(before) | set(now)):
                old, new = before.get(hp), now.get(hp)
                if old == new:
                    continue
                kind = ("prior_stated" if old is None
                        else "prior_withdrawn" if new is None else "prior_edited")
                ExperimentEvent.objects.create(
                    data_id=exp.data_id, at=run.created_at, at_trial=run.trial_offset or 0,
                    kind=kind, payload={"hyperparameter": hp, "old": old, "new": new},
                    run_id=run.pk, actor_id=run.started_by_id)
            before = now


class Migration(migrations.Migration):

    dependencies = [
        ("experiments", "0007_experiment_group_share"),
    ]

    operations = [
        migrations.RunPython(priors_from_runs, migrations.RunPython.noop),
    ]
