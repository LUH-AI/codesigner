from django.urls import path

from . import views

app_name = "web"

urlpatterns = [
    path("", views.home, name="home"),
    path("inspect/", views.inspect, name="inspect"),
    path("experiments/new/", views.new_experiment, name="new_experiment"),
    path("experiments/import/", views.import_experiment, name="import_experiment"),
    path("experiments/<int:pk>/", views.experiment_detail, name="experiment_detail"),
    path("experiments/<int:pk>/run/", views.experiment_run, name="experiment_run"),
    path("experiments/<int:pk>/status/", views.run_status, name="run_status"),
    path("experiments/<int:pk>/cancel/", views.run_cancel, name="run_cancel"),
    path("experiments/<int:pk>/export/", views.experiment_export, name="experiment_export"),
    path("experiments/<int:pk>/delete/", views.experiment_delete, name="experiment_delete"),
]
