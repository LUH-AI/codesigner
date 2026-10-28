"""Every view here reaches only drafts (`drafts=ONLY`); `ui/setup_steps.py`
defines the four steps, and `ui/drafts.py` does the saving.
"""

from pathlib import Path

from django.contrib import messages
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import DELETE, EDIT
from experiments.services import storage
from experiments.services import snapshot as snapshot_adapter

from .. import drafts, optimizer_form
from .. import setup_steps as steps
from ..figures.data_handling import SECTIONS
from ..permissions import ONLY, experiment_view
from .common import tell
from .dashboard import _prior_only_context, experiment_header
from .data_handling import tabs_for
from .pages import _demo_columns, _demo_time_columns, _evaluation_schemes, _model_tasks


def _page(request, exp, step, template, **context):
    return render(request, template, {
        **experiment_header(request, exp, None),
        "steps": steps.bar(exp, step),
        "step": step,
        **context,
    })


@experiment_view(EDIT, drafts=ONLY)
def setup_resume(request, exp):
    """`setup_resume` opens the first unsaved step."""
    return redirect(steps.resume(exp))


@experiment_view(EDIT, drafts=ONLY)
def setup(request, exp):
    """`setup` shows step 1, what the experiment is."""
    form = None
    if request.method == "POST":
        outcome = drafts.save_step(request, exp, steps.SETUP, request.POST, request.FILES)
        tell(request, outcome.notices)
        if outcome.gone:
            return redirect("ui:home")
        if outcome.saved:
            return redirect(steps.url(exp, steps.OPTIMIZER))
        form = outcome.form
    return _page(request, exp, steps.SETUP, "ui/setup_step_setup.html",
                 **setup_form_context(request, exp, form))


@experiment_view(EDIT, drafts=ONLY)
def setup_optimizer(request, exp):
    """`setup_optimizer` shows step 2, the optimizer and its settings."""
    if not steps.reachable(exp, steps.OPTIMIZER):
        return redirect(steps.url(exp, steps.SETUP))
    if request.method == "POST":
        outcome = drafts.save_step(request, exp, steps.OPTIMIZER, request.POST)
        if not outcome.saved:
            messages.error(request, outcome.refusal)
            return redirect(steps.url(exp, steps.OPTIMIZER))
        return redirect(steps.url(exp, steps.DATA))
    return _page(request, exp, steps.OPTIMIZER, "ui/setup_step_optimizer.html",
                 **optimizer_form.context(exp))


@experiment_view(EDIT, drafts=ONLY)
def setup_data(request, exp):
    """`setup_data` shows step 3, the Data Handling panels settable on Overview."""
    if not steps.reachable(exp, steps.DATA):
        return redirect(steps.url(exp, steps.SETUP))
    if request.method == "POST":
        outcome = drafts.save_step(request, exp, steps.DATA, request.POST)
        if not outcome.saved:
            messages.error(request, outcome.refusal)
            return redirect(steps.url(exp, steps.DATA))
        return redirect(steps.url(exp, steps.PRIORS))
    here = steps.url(exp, steps.DATA)
    return _page(request, exp, steps.DATA, "ui/setup_step_data.html",
                 tabs=tabs_for(exp, lambda s: here + ("" if s is SECTIONS[0] else f"#{s.slug}"),
                               setup=True),
                 section=SECTIONS[0], locked=False, save_url=here,
                 save_label=_("Save and continue"),
                 uploaded_model=bool(exp.data.model_file))


@experiment_view(EDIT, drafts=ONLY)
def setup_priors(request, exp):
    """`setup_priors` shows step 4, the prior figure and the Create button."""
    if not steps.reachable(exp, steps.PRIORS):
        return redirect(steps.url(exp, steps.SETUP))
    built = snapshot_adapter.rebuild_experiment(exp)
    return _page(request, exp, steps.PRIORS, "ui/setup_step_priors.html",
                 not_ready=steps.not_ready(exp),
                 run_default_metric=(exp.data.metric_names or [None])[0],
                 **_prior_only_context(exp, built))


@require_POST
@experiment_view(EDIT, drafts=ONLY)
def setup_create(request, exp):
    """`setup_create` turns the draft into an experiment once steps 1 to 3 are
    saved."""
    refusal = drafts.create(request, exp)
    if refusal:
        messages.error(request, refusal)
        return redirect(steps.resume(exp))
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(DELETE, drafts=ONLY)
def setup_discard(request, exp):
    """`setup_discard` deletes the draft and its files, skipping the bin."""
    name = exp.title
    storage.delete_experiment_files(exp)
    messages.success(request, _("Discarded the draft “%(name)s”.") % {"name": name})
    return redirect("ui:home")


# Step 1

def setup_form_context(request, exp, form=None) -> dict:
    """`setup_form_context` builds step 1's template context with the draft's
    current files."""
    data = exp.data
    if form is None:
        form = drafts.setup_form(request, exp)
        form.initial.update(drafts.setup_initial(exp))
    return {
        "form": form,
        "evaluation_schemes": _evaluation_schemes(),
        "model_tasks": _model_tasks(),
        "demo_time_columns": _demo_time_columns(),
        "demo_columns": _demo_columns(),
        "current_dataset": (data.demo_dataset or (Path(data.dataset.name).name if data.dataset else "")),
        "current_dataset_upload": Path(data.dataset.name).name if data.dataset else "",
        "current_model": Path(data.model_file.name).name if data.model_file else "",
    }
