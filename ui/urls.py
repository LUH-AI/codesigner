from django.urls import path

from . import views

app_name = "ui"

urlpatterns = [
    path("", views.home, name="home"),
    path("experiments/", views.experiment_list, name="experiment_list"),
    path("healthz/", views.healthz, name="healthz"),
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
    path("experiments/<int:pk>/share/", views.experiment_share, name="experiment_share"),
    path("experiments/<int:pk>/trial-panel/", views.trial_panel, name="trial_panel"),
    path("experiments/<int:pk>/export/", views.experiment_export, name="experiment_export"),
    path("experiments/<int:pk>/settings/", views.experiment_settings, name="experiment_settings"),
    path("experiments/<int:pk>/delete/", views.experiment_delete, name="experiment_delete"),
]
