from django.contrib import admin

from .models import Experiment, Run


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ("name", "model_name", "optimizer_name", "primary_metric", "seed", "created_at")
    search_fields = ("name",)


@admin.register(Run)
class RunAdmin(admin.ModelAdmin):
    list_display = ("experiment", "status", "n_trials", "primary_metric", "started_at", "finished_at")
    list_filter = ("status",)
