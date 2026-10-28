"""`decide_run` confirms every change of the *current* metric, because the
existing trials were chosen under the old objective and carry its bias. The
*original* metric only lets the page flag divergence.
"""


def decide_run(original_metric, current_metric, chosen_metric):
    """`decide_run` returns "first" with the chosen metric on a first run,
    "warn" with it when the optimized metric would change, and "run" with the
    current metric otherwise. Each change asks for confirmation, including the
    third.
    """
    if original_metric is None:
        return "first", chosen_metric
    if chosen_metric != current_metric:
        return "warn", chosen_metric
    return "run", current_metric


def resolve_metric_change(decision, current_metric, chosen_metric):
    """"new" picks the chosen metric and "old" keeps the current one, which may
    differ from the original after earlier changes. Any other decision yields
    None, and nothing runs.
    """
    if decision == "new":
        return chosen_metric
    if decision == "old":
        return current_metric
    return None


def apply_metrics(original_metric, optimize_metric):
    """The original metric is set on the first run and never changes.
    """
    new_original = optimize_metric if original_metric is None else original_metric
    return optimize_metric, new_original
