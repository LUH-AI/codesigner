from django.urls import path

from . import panels, views

app_name = "ui"

urlpatterns = [
    path("", views.home, name="home"),
    path("experiments/", views.experiment_list, name="experiment_list"),
    path("healthz/", views.healthz, name="healthz"),
    path("settings/account/", views.account, name="account"),
    # The two management surfaces. Neither goes through `@experiment_view`: a
    # group lead reaches their colleagues' work through the policy's queryset,
    # and a site admin reaches no experiment at all. See ui/panels.py.
    path("group/", panels.group_people, name="group_people"),
    path("group/add/", panels.group_add_person, name="group_add_person"),
    path("group/<int:pk>/remove/", panels.group_remove_person, name="group_remove_person"),
    path("group/experiments/", panels.group_work, name="group_work"),
    path("site/", panels.site_groups, name="site_groups"),
    path("site/groups/save/", panels.site_group_save, name="site_group_save"),
    path("site/usage/", panels.site_usage, name="site_usage"),
    path("site/jobs/", panels.site_jobs, name="site_jobs"),
    path("site/jobs/<int:pk>/stop/", panels.site_job_stop, name="site_job_stop"),
    path("settings/appearance/", views.appearance, name="appearance"),
    path("settings/experiment-defaults/", views.default_experiment_settings, name="default_experiment_settings"),
    path("experiments/new/", views.new_experiment, name="new_experiment"),
    path("experiments/import/", views.import_experiment, name="import_experiment"),
    path("experiments/<int:pk>/", views.experiment_detail, name="experiment_detail"),
    path("experiments/<int:pk>/run/", views.experiment_run, name="experiment_run"),
    path("experiments/<int:pk>/status/", views.run_status, name="run_status"),
    path("experiments/<int:pk>/env-status/", views.env_status, name="env_status"),
    path("experiments/<int:pk>/prepare-env/", views.prepare_env, name="prepare_env"),
    path("experiments/<int:pk>/cancel/", views.run_cancel, name="run_cancel"),
    path("experiments/<int:pk>/force-stop/", views.run_force_stop, name="run_force_stop"),
    path("experiments/<int:pk>/share/", views.experiment_share, name="experiment_share"),
    path("experiments/<int:pk>/trial-panel/", views.trial_panel, name="trial_panel"),
    path("experiments/<int:pk>/trial-ablation/", views.trial_ablation, name="trial_ablation"),
    path("experiments/<int:pk>/trial-traceback/", views.trial_traceback, name="trial_traceback"),
    path("experiments/<int:pk>/partial-dependence/", views.partial_dependence, name="partial_dependence"),
    path("experiments/<int:pk>/local-effects/", views.local_effects, name="local_effects"),
    path("experiments/<int:pk>/surrogate-uncertainty/", views.surrogate_uncertainty,
         name="surrogate_uncertainty"),
    path("experiments/<int:pk>/figures/", views.metric_figures, name="metric_figures"),
    path("experiments/<int:pk>/compute-analytics/", views.experiment_compute_analytics,
         name="experiment_compute_analytics"),
    path("experiments/<int:pk>/export/", views.experiment_export, name="experiment_export"),
    path("experiments/<int:pk>/settings/", views.experiment_settings, name="experiment_settings"),
    path("experiments/<int:pk>/delete/", views.experiment_delete, name="experiment_delete"),
]
