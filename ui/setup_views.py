"""A draft's setup: its four steps, creating it, and throwing it away.

See ui/services/setup.py for what the steps are and what saving each means.
Every view here reaches only drafts (`experiment_view(..., drafts=ONLY)`); an
experiment that has been created is sent to its dashboard instead.
"""

from pathlib import Path

from django.contrib import messages
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST


from .permissions import DELETE, EDIT, ONLY, experiment_view
from .services import drafts, optimizer_form
from .services import setup as steps


def _page(request, exp, step, template, **context):
    from .views import experiment_header

    return render(request, template, {
        **experiment_header(request, exp, None),
        "steps": steps.bar(exp, step),
        "step": step,
        **context,
    })


@experiment_view(EDIT, drafts=ONLY)
def setup_resume(request, exp):
    """A draft opened from the sidebar: the first step not yet saved."""
    return redirect(steps.resume(exp))


@experiment_view(EDIT, drafts=ONLY)
def setup(request, exp):
    """Step 1: what the experiment is, filled from the draft; Save keeps it
    and moves on."""
    form = None
    if request.method == "POST":
        from .views import tell

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
    """Step 2: the optimizer and its settings; Save keeps them and moves on."""
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
    """Step 3: the Data Handling page, its settable panels on Overview; Save
    keeps their choices and moves on."""
    from datahandling.sections import SECTIONS
    from datahandling.views import tabs_for

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
    """Step 4: what is believed about where the optimum is — the prior figure,
    as an experiment with no trials shows it — and Create."""
    from .views import _prior_only_context, _rebuild_experiment

    if not steps.reachable(exp, steps.PRIORS):
        return redirect(steps.url(exp, steps.SETUP))
    built = _rebuild_experiment(exp)
    return _page(request, exp, steps.PRIORS, "ui/setup_step_priors.html",
                 not_ready=steps.not_ready(exp),
                 run_default_metric=(exp.data.metric_names or [None])[0],
                 **_prior_only_context(exp, built))


@require_POST
@experiment_view(EDIT, drafts=ONLY)
def setup_create(request, exp):
    """Make the draft an experiment: refused until steps 1 to 3 are saved.
    Its history begins here, with how it was set up — Data Handling's choices
    among it, which the timeline lists as an entry of their own — and each
    prior its steps stated."""
    refusal = drafts.create(request, exp)
    if refusal:
        messages.error(request, refusal)
        return redirect(steps.resume(exp))
    return redirect("ui:experiment_detail", pk=exp.pk)


@require_POST
@experiment_view(DELETE, drafts=ONLY)
def setup_discard(request, exp):
    """Throw the draft away, files and all. Nothing was run, so there is
    nothing for a bin to keep."""
    from .services import storage

    name = exp.title
    storage.delete_experiment_files(exp)
    messages.success(request, _("Discarded the draft “%(name)s”.") % {"name": name})
    return redirect("ui:home")


# ── step 1 ───────────────────────────────────────────────────────────────────

def setup_form_context(request, exp, form=None) -> dict:
    """What step 1's template needs: the form (bound after a failed Save, else
    filled from the draft), and what the draft already has, so keeping it is
    visible."""
    from .views import _demo_columns, _demo_time_columns, _evaluation_schemes, _model_tasks

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
