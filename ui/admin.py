from django.contrib import admin

from .models import Experiment, ExperimentData, ExperimentShare, GlobalSettings, Run


class ExperimentShareInline(admin.TabularInline):
    # Only members of the experiment's group; `ExperimentShare.clean` refuses
    # anybody else, and the admin runs it.
    model = ExperimentShare
    extra = 0
    fields = ("user", "level", "granted_by", "granted_at")
    readonly_fields = ("granted_at",)
    raw_id_fields = ("user", "granted_by")


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    inlines = [ExperimentShareInline]
    # owner and shared are here because the admin is the only place to change
    # them for someone else: reassigning an ownerless experiment after an
    # instance gains accounts, or handing one over when a person leaves.
    #
    # The columns describing the work itself come through `data`, which is the
    # other half of the experiment — see `ui/models.py`. `list_display` cannot
    # follow a relation, so each is a small accessor; `search_fields` and
    # `list_filter` can, and do.
    list_display = ("name", "identifier", "owner", "group", "model_name",
                    "optimizer_name", "current_metric", "seed", "created_at")
    list_filter = ("group",)
    search_fields = ("data__name", "identifier", "owner__username")
    list_select_related = ("data", "owner")

    @admin.display(ordering="data__name", description="name")
    def name(self, obj):
        return obj.data.name

    @admin.display(ordering="data__model_name", description="model name")
    def model_name(self, obj):
        return obj.data.model_name

    @admin.display(ordering="data__optimizer_name", description="optimizer name")
    def optimizer_name(self, obj):
        return obj.data.optimizer_name

    @admin.display(ordering="data__current_metric", description="current metric")
    def current_metric(self, obj):
        return obj.data.current_metric

    @admin.display(ordering="data__seed", description="seed")
    def seed(self, obj):
        return obj.data.seed


@admin.register(ExperimentData)
class ExperimentDataAdmin(admin.ModelAdmin):
    """The portable half, registered so an operator can reach it directly.

    Mostly they will not: the instance-side row is the one with an owner and a
    group on it, and that is the page somebody arrives at. This exists so the
    work is not unreachable when its `Experiment` is missing — which the
    OneToOne makes impossible today, and which a later bug could not hide.
    """

    list_display = ("name", "model_name", "optimizer_name", "current_metric", "seed")
    search_fields = ("name",)


@admin.register(GlobalSettings)
class GlobalSettingsAdmin(admin.ModelAdmin):
    list_display = ("__str__",)


@admin.register(Run)
class RunAdmin(admin.ModelAdmin):
    list_display = ("experiment", "status", "stopped_by", "primary_metric", "started_at", "finished_at")
    list_filter = ("status",)
