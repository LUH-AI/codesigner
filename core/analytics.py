"""What a finished search's stored analytics say when read together."""

#: Below this share of the achievable gain, "what is left" is rounding rather
#: than headroom, and a pie of it would be a picture of noise.
SETTLED = 0.02


def tuning_progress(result, metric, ablation):
    """What each hyperparameter was worth, and how much of it has been banked.

    Tunability decomposes (best achievable − default); the ablation decomposes
    (this trial − default). Both are FSII order-1 terms of an order-2 fit,
    against the same baseline, and both games are zeroed at that baseline —
    measured, `v(none) = 0` for each — so the two are in the same units from the
    same origin and the subtraction is well posed.

    **Two things it is not**, both measured rather than assumed, and both
    reasons to read the ordering rather than the digits:

    - *Achievable is a lower bound.* The tunability game's maximum comes from
      10,000 random draws of the configuration space, and a real optimizer beats
      random search — that is what it is for. So a well-tuned hyperparameter can
      show more banked than achievable, and it is not an error: it means the
      search found something the estimate did not. `beyond` marks those, and
      they are the strongest evidence an axis is finished, not a contradiction.
    - *Order-1 is not the whole game.* Measured on a four-hyperparameter run, the
      order-1 terms account for about 70-100% of each game's own endpoint, the
      rest living in order-2 interactions that neither number carries. Both
      sides are deflated by the same mechanism, so the comparison survives it
      better than either figure alone does.

    Both halves are needed in the metric's own units, and only one of them is
    stored that way: `hyperparameter_importance` is a share of 100%, and
    `hyperparameter_tunability_total` is what it is a share *of*. A result
    written before that field existed has no scale to multiply by, and gets
    nothing rather than a ratio computed against the wrong denominator.

    Returns `(rows, settled)`. Each row is one hyperparameter's achievable,
    banked and remaining value; *settled* is True when the remaining total is
    rounding, which is a finding — this trial has taken essentially everything
    there was — and not a figure.
    """
    shares = result.hyperparameter_importance.get(metric, {})
    total = result.hyperparameter_tunability_total.get(metric, 0.0)
    if not shares or not total or not ablation:
        return [], False

    rows = []
    for name, share in shares.items():
        achievable = share * total
        # Signed on purpose. A negative one says this trial's value for that
        # hyperparameter is *worse* than the default — drift picked up while
        # chasing whichever one mattered — and flooring it here would hide the
        # one finding that is actionable on its own.
        banked = ablation.get(name, 0.0)
        rows.append({
            "name": name,
            "achievable": achievable,
            "banked": banked,
            # Floored, because a negative amount left is not an amount left.
            "remaining": max(0.0, achievable - banked),
            "banked_share": (banked / achievable) if achievable else 0.0,
            # Past the estimated ceiling: the search found something 10,000
            # random draws did not. Marked rather than quietly floored, because
            # it says something — this axis is not merely finished, it is
            # finished past where the estimate could see.
            "beyond": banked > achievable > 0,
        })
    rows.sort(key=lambda row: row["remaining"], reverse=True)
    left = sum(row["remaining"] for row in rows)
    return rows, left <= SETTLED * (total or 1.0)
