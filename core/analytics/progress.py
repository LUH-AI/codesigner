from __future__ import annotations


#: Below `SETTLED` of the achievable gain, the remainder is rounding.
SETTLED = 0.02


def tuning_progress(result, metric, ablation):
    """Tunability decomposes best achievable minus default and the ablation
    decomposes the ablated trial minus default, both as order-1 FSII terms from
    the same zeroed baseline, so subtracting is valid. Achievable is a lower
    bound from 10,000 random draws, so `beyond` marks a search that beat the
    estimate, and order-1 terms miss the order-2 share. `tuning_progress`
    returns nothing for a result without `hyperparameter_tunability_total`. The
    result is `(rows, settled)`, with *settled* True when the remainder is
    rounding.
    """
    shares = result.hyperparameter_importance.get(metric, {})
    total = result.hyperparameter_tunability_total.get(metric, 0.0)
    if not shares or not total or not ablation:
        return [], False

    rows = []
    for name, share in shares.items():
        achievable = share * total
        # A negative value means the trial drifted worse than the default, the
        # one finding actionable alone.
        banked = ablation.get(name, 0.0)
        rows.append({
            "name": name,
            "achievable": achievable,
            "banked": banked,
            # A negative remainder is no remainder.
            "remaining": max(0.0, achievable - banked),
            "banked_share": (banked / achievable) if achievable else 0.0,
            # The search beat 10,000 random draws on this axis.
            "beyond": banked > achievable > 0,
        })
    rows.sort(key=lambda row: row["remaining"], reverse=True)
    left = sum(row["remaining"] for row in rows)
    return rows, left <= SETTLED * (total or 1.0)
