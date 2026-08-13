"""Pure decision logic for the run / metric-change flow.

An experiment carries a *current* metric (the one the optimizer optimizes right
now) and an *original* metric (the one its first run used, kept only so the
page can say the two have diverged). Running again may change the current
metric, and every change is confirmed first: the trials already accumulated
were *chosen* under the old objective, so the sample carries its bias forward
even though the optimizer re-reads their scores under the new one.

Viewing a metric (the results switcher) is separate and client-side; here
*chosen_metric* is strictly the metric the Run form asks to optimize.
"""


def decide_run(original_metric, current_metric, chosen_metric):
    """Decide what a Run press should do.

    Returns (action, optimize_metric):
      * ("first", chosen)  — nothing committed yet; run and commit the metric.
      * ("warn", chosen)   — would change the metric being optimized; confirm
                             before running.
      * ("run", current)   — the chosen metric is already the one being
                             optimized; just run.

    Every change is confirmed, not only the first. It used to warn once, on the
    divergence from *original*, and afterwards run with the current metric no
    matter what the form asked for — so a second change was silently discarded
    and the experiment could never be steered again. Changing the metric is a
    real choice each time it is made, and the caveat that prompts the
    confirmation applies just as much on the third change as on the first.
    """
    if original_metric is None:
        return "first", chosen_metric
    if chosen_metric != current_metric:
        return "warn", chosen_metric
    return "run", current_metric


def resolve_metric_change(decision, current_metric, chosen_metric):
    """Resolve the metric-change confirmation to the metric to optimize.

    "new" optimizes the chosen candidate; "old" stays on what the experiment is
    optimizing *now* — not the original, which after an earlier change is a
    metric the user did not ask to return to. Anything else (cancel) means
    don't run.
    """
    if decision == "new":
        return chosen_metric
    if decision == "old":
        return current_metric
    return None


def apply_metrics(original_metric, optimize_metric):
    """The (current, original) an experiment should carry after a run.

    Current always becomes the optimized metric; original is pinned once, on
    the first run, and immutable thereafter.
    """
    new_original = optimize_metric if original_metric is None else original_metric
    return optimize_metric, new_original
