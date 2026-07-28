from django.contrib import admin

from .models import Experiment, GlobalSettings, Run


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ("name", "identifier", "model_name", "optimizer_name", "primary_metric", "seed", "created_at")
    search_fields = ("name", "identifier")


@admin.register(GlobalSettings)
class GlobalSettingsAdmin(admin.ModelAdmin):
    list_display = ("__str__",)


@admin.register(Run)
class RunAdmin(admin.ModelAdmin):
    list_display = ("experiment", "status", "n_trials", "primary_metric", "started_at", "finished_at")
    list_filter = ("status",)
