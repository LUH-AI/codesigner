"""Pure decision logic for the run / metric-change flow.

An experiment carries a *primary* metric (the one the optimizer optimizes) and
an *original* metric (the one its first run used). Running again may change the
primary; the first time it diverges from the original the user is warned, since
it makes the accumulated trials optimize a different objective.

Viewing a metric (the results switcher) is separate and client-side; here
*chosen_metric* is strictly the metric the Run form asks to optimize.
"""


def decide_run(original_metric, primary_metric, chosen_metric):
    """Decide what a Run press should do.

    Returns (action, optimize_metric):
      * ("first", chosen)  — nothing committed yet; run and commit the metric.
      * ("warn", chosen)   — would change the optimized metric while it still
                             matches the original; confirm before running.
      * ("run", primary)   — run with the current primary metric.
    """
    if original_metric is None:
        return "first", chosen_metric
    if chosen_metric != primary_metric and primary_metric == original_metric:
        return "warn", chosen_metric
    return "run", primary_metric


def resolve_metric_change(decision, original_metric, chosen_metric):
    """Resolve the metric-change confirmation to the metric to optimize.

    "new" optimizes the chosen candidate; "old" keeps the original; anything
    else (cancel) means don't run.
    """
    if decision == "new":
        return chosen_metric
    if decision == "old":
        return original_metric
    return None


def apply_metrics(original_metric, optimize_metric):
    """The (primary, original) an experiment should carry after a run.

    Primary always becomes the optimized metric; original is pinned once, on
    the first run, and immutable thereafter.
    """
    new_original = optimize_metric if original_metric is None else original_metric
    return optimize_metric, new_original
