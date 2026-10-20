from django.contrib.auth import views as auth_views
from django.urls import path, reverse_lazy

from .views import (
    analytics, bin, dashboard, data_handling, export, groups, imports, pages, preferences,
    priors, runs, setup, sharing, site,
)

app_name = "ui"

urlpatterns = [
    path("", pages.home, name="home"),
    path("experiments/", pages.experiment_list, name="experiment_list"),
    path("healthz/", pages.healthz, name="healthz"),
    path("settings/account/", preferences.account, name="account"),
    # Signing in and out.
    #
    # Mounted unconditionally, like the middleware, so the two modes differ by one
    # boolean and not by which routes exist. With `REQUIRE_LOGIN` off these pages are
    # reachable but pointless — nothing sends you to them.
    #
    # Signing in, signing out, and changing the password you are signed in with.
    #
    # Not password *reset*, which is the other thing: it mails a link to somebody who
    # cannot sign in, so it needs a configured mail server — an operator decision and a
    # whole surface of its own. Changing a password you already know needs none, which
    # is why one is here and the other is not. Until reset is asked for, a forgotten
    # password is reset by an administrator in the Django admin.
    #
    # There is no registration either, by design. Nothing here accepts an
    # unauthenticated POST that creates an account: on an instance where uploading a
    # model is arbitrary code execution, an open door is the thing most worth not
    # having. Accounts are created by an administrator.
    path("accounts/login/", auth_views.LoginView.as_view(template_name="ui/account/login.html"),
         name="login"),
    # Django's LogoutView is POST-only, which is right: a GET logout can be
    # triggered by any page that can make the browser fetch a URL.
    path("accounts/logout/", auth_views.LogoutView.as_view(), name="logout"),
    # Django's own views and its own form, which already enforces the password
    # validators in settings and asks for the current password before changing
    # it — so a borrowed, still-signed-in browser cannot lock its owner out.
    path("accounts/password/", auth_views.PasswordChangeView.as_view(
        template_name="ui/account/password_change.html",
        success_url=reverse_lazy("ui:password_changed")), name="password_change"),
    path("accounts/password/done/", auth_views.PasswordChangeDoneView.as_view(
        template_name="ui/account/password_changed.html"), name="password_changed"),
    # The two management surfaces. Neither goes through `@experiment_view`: a
    # group lead reaches their colleagues' work through the policy's queryset,
    # and a site admin reaches no experiment at all. See ui/views/groups.py.
    # A group is named in the URL, because a person may be in several. The
    # membership routes carry their group as well as their own pk: the view
    # then looks the membership up *within* that group, which makes the
    # boundary a lookup rather than a check somebody has to remember.
    path("group/", groups.group_index, name="group_index"),
    path("group/<int:group_pk>/", groups.group_detail, name="group_detail"),
    path("group/<int:group_pk>/add/", groups.group_add_person, name="group_add_person"),
    path("group/<int:group_pk>/people/<int:pk>/remove/", groups.group_remove_person,
         name="group_remove_person"),
    path("group/<int:group_pk>/people/<int:pk>/role/", groups.group_set_role,
         name="group_set_role"),
    path("group/<int:group_pk>/people/<int:pk>/primary/", groups.group_transfer_primary,
         name="group_transfer_primary"),
    path("site/", site.site_groups, name="site_groups"),
    path("site/groups/save/", site.site_group_save, name="site_group_save"),
    path("site/groups/delete/", site.site_group_delete, name="site_group_delete"),
    path("site/trash/", site.site_trash, name="site_trash"),
    path("site/trash/<int:pk>/rehome/", site.site_trash_rehome, name="site_trash_rehome"),
    path("site/trash/<int:pk>/delete/", site.site_trash_delete, name="site_trash_delete"),
    path("site/usage/", site.site_usage, name="site_usage"),
    path("site/jobs/", site.site_jobs, name="site_jobs"),
    path("site/jobs/<int:pk>/stop/", site.site_job_stop, name="site_job_stop"),
    path("settings/appearance/", preferences.appearance, name="appearance"),
    path("settings/export/", preferences.export_preferences, name="export_preferences"),
    path("settings/experiment-defaults/", preferences.default_experiment_settings, name="default_experiment_settings"),
    path("experiments/new/", pages.new_experiment, name="new_experiment"),
    path("experiments/<int:pk>/data/", data_handling.data_overview, name="data_overview"),
    path("experiments/<int:pk>/data/<slug:section>/", data_handling.data_section, name="data_section"),
    # A draft's setup (ui/setup_steps.py): its three steps, creating it, and
    # throwing it away. Drafts only — see `experiment_view`'s `drafts`.
    path("experiments/<int:pk>/setup/", setup.setup, name="setup"),
    path("experiments/<int:pk>/setup/optimizer/", setup.setup_optimizer,
         name="setup_optimizer"),
    path("experiments/<int:pk>/setup/data/", setup.setup_data, name="setup_data"),
    path("experiments/<int:pk>/setup/priors/", setup.setup_priors, name="setup_priors"),
    path("experiments/<int:pk>/setup/resume/", setup.setup_resume, name="setup_resume"),
    path("experiments/<int:pk>/setup/create/", setup.setup_create, name="setup_create"),
    path("experiments/<int:pk>/setup/discard/", setup.setup_discard,
         name="setup_discard"),
    path("experiments/import/", imports.import_experiment, name="import_experiment"),
    path("experiments/<int:pk>/", dashboard.experiment_detail, name="experiment_detail"),
    path("experiments/<int:pk>/run/", runs.experiment_run, name="experiment_run"),
    path("experiments/<int:pk>/optimizer/", dashboard.experiment_optimizer,
         name="experiment_optimizer"),
    path("experiments/<int:pk>/rename/", dashboard.experiment_rename, name="experiment_rename"),
    path("experiments/<int:pk>/status/", runs.run_status, name="run_status"),
    path("experiments/<int:pk>/env-status/", runs.env_status, name="env_status"),
    path("experiments/<int:pk>/prepare-env/", runs.prepare_env, name="prepare_env"),
    path("experiments/<int:pk>/cancel/", runs.run_cancel, name="run_cancel"),
    path("experiments/<int:pk>/force-stop/", runs.run_force_stop, name="run_force_stop"),
    path("experiments/<int:pk>/share/", sharing.experiment_share, name="experiment_share"),
    path("experiments/<int:pk>/transfer/", sharing.experiment_transfer, name="experiment_transfer"),
    path("bin/", bin.experiment_bin, name="experiment_bin"),
    path("bin/<int:pk>/restore/", bin.bin_restore, name="bin_restore"),
    path("bin/<int:pk>/download/", bin.bin_download, name="bin_download"),
    path("bin/<int:pk>/dismiss/", bin.bin_dismiss, name="bin_dismiss"),
    path("experiments/<int:pk>/trial-panel/", dashboard.trial_panel, name="trial_panel"),
    path("experiments/<int:pk>/proposal-panel/", priors.proposal_panel, name="proposal_panel"),
    path("experiments/<int:pk>/run-proposal/", priors.run_proposal, name="run_proposal"),
    path("experiments/<int:pk>/projected-prior/preview/", priors.projected_prior_preview,
         name="projected_prior_preview"),
    path("experiments/<int:pk>/projected-prior/state/", priors.state_projected_prior,
         name="state_projected_prior"),
    path("experiments/<int:pk>/projected-prior/withdraw/", priors.withdraw_projected_prior,
         name="withdraw_projected_prior"),
    path("experiments/<int:pk>/trial-ablation/", analytics.trial_ablation, name="trial_ablation"),
    path("experiments/<int:pk>/trial-traceback/", analytics.trial_traceback, name="trial_traceback"),
    path("experiments/<int:pk>/partial-dependence/", analytics.partial_dependence, name="partial_dependence"),
    path("experiments/<int:pk>/forecast/", analytics.forecast_figure, name="forecast"),
    path("experiments/<int:pk>/acquisition-slice/", priors.acquisition_slice,
         name="acquisition_slice"),
    path("experiments/<int:pk>/local-effects/", analytics.local_effects, name="local_effects"),
    path("experiments/<int:pk>/prior/", priors.save_prior, name="save_prior"),
    path("experiments/<int:pk>/layout/", dashboard.save_layout, name="save_layout"),
    path("experiments/<int:pk>/prior/evaluate/", priors.evaluate_prior,
         name="evaluate_prior"),
    path("experiments/<int:pk>/prior/reset/", priors.reset_prior,
         name="reset_prior"),
    path("experiments/<int:pk>/surrogate-uncertainty/", analytics.surrogate_uncertainty,
         name="surrogate_uncertainty"),
    path("experiments/<int:pk>/figures/", analytics.metric_figures, name="metric_figures"),
    path("experiments/<int:pk>/hypershap/", analytics.hypershap_figures, name="hypershap_figures"),
    path("experiments/<int:pk>/surrogate-prediction/", analytics.surrogate_prediction,
         name="surrogate_prediction"),
    path("experiments/<int:pk>/next-trials/", priors.next_trials, name="next_trials"),
    path("experiments/<int:pk>/compute-analytics/", analytics.experiment_compute_analytics,
         name="experiment_compute_analytics"),
    path("experiments/<int:pk>/export/", export.experiment_export, name="experiment_export"),
    path("experiments/<int:pk>/export-model/", export.experiment_export_model,
         name="experiment_export_model"),
    path("experiments/<int:pk>/export-trial-parameters/", export.experiment_export_trial_model,
         name="experiment_export_trial_model"),
    path("experiments/<int:pk>/settings/", preferences.experiment_settings, name="experiment_settings"),
    path("experiments/<int:pk>/timeline/", export.experiment_timeline, name="experiment_timeline"),
    path("experiments/<int:pk>/delete/", bin.experiment_delete, name="experiment_delete"),
]
