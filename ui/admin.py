from django.contrib import admin

from .models import Experiment, GlobalSettings, Run


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    # owner and shared are here because the admin is the only place to change
    # them for someone else: reassigning an ownerless experiment after an
    # instance gains accounts, or handing one over when a person leaves.
    list_display = ("name", "identifier", "owner", "shared", "model_name",
                    "optimizer_name", "primary_metric", "seed", "created_at")
    list_filter = ("shared",)
    search_fields = ("name", "identifier", "owner__username")


@admin.register(GlobalSettings)
class GlobalSettingsAdmin(admin.ModelAdmin):
    list_display = ("__str__",)


@admin.register(Run)
class RunAdmin(admin.ModelAdmin):
    list_display = ("experiment", "status", "n_trials", "primary_metric", "started_at", "finished_at")
    list_filter = ("status",)
