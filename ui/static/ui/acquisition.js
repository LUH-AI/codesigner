/* The acquisition-and-beliefs figure.
 *
 * The one figure on the page that draws itself. Every other figure is a Plotly
 * payload built in `ui/figures/plots.py` and handed to `Plotly.react`; this one
 * gets a styled skeleton from there (`acquisition_slice_plot`) with its belief
 * and acquisition traces empty, and fills them here.
 *
 * That split is deliberate. The belief is a thing the reader states and drags,
 * so the traces depending on it change faster than any round trip; and once
 * they are computed here, computing them in Python as well would mean two
 * implementations of expected improvement that have to keep agreeing forever.
 * So Python owns the palette, the axes and the panel structure, and this file
 * owns the arithmetic. Everything it needs travels in the figure's own
 * `layout.meta.acquisition` — the same contract `layout.meta.selection` uses
 * for the selection bus, so neither script needs to know about the other.
 *
 * Loaded with `defer` from the figure's own partial, which is what guarantees
 * it runs after `plotly.min.js` (a classic in-body script much further down the
 * page). `document.currentScript` is therefore null here; find nodes by id.
 */
(function () {
    "use strict";

    var SQ2PI = Math.sqrt(2 * Math.PI);

    /* Abramowitz & Stegun 7.1.26. Accurate to ~1.5e-7, which is several orders
     * better than anything here needs: the acquisition function is only ever
     * ranked, and it is drawn 400 pixels wide. */
    function erf(x) {
        var s = x < 0 ? -1 : 1, t;
        x = Math.abs(x);
        t = 1 / (1 + 0.3275911 * x);
        return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
            - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
    }
    function cdf(z) { return 0.5 * (1 + erf(z / Math.SQRT2)); }
    function pdf(z) { return Math.exp(-0.5 * z * z) / SQ2PI; }

    /* Expected improvement over `eta`, written for whichever direction the
     * metric runs. `gain` is how much better than the incumbent this point
     * would be if the surrogate is right; the formula is the same either way
     * once the sign is settled, which is why the orientation is handled here
     * and not by pre-flipping the data. */
    function expectedImprovement(mu, sigma, eta, higherIsBetter) {
        var gain = higherIsBetter ? mu - eta : eta - mu, z;
        if (!(sigma > 1e-12)) return Math.max(gain, 0);
        z = gain / sigma;
        return gain * cdf(z) + sigma * pdf(z);
    }

    /* The fiction: the mean that would have produced `target` on its own.
     *
     * Expected improvement falls monotonically as a point gets worse, so the
     * inverse exists and bisection finds it. This is the only sense in which a
     * belief moves a performance curve — it is not a surrogate anyone fitted,
     * and the caller draws it dashed for that reason.
     *
     * Returns null where the answer is off the bracket: a belief sharp enough
     * claims more improvement than any finite mean could justify, or writes a
     * region off entirely. The caller leaves a gap rather than pinning the
     * curve to the panel edge, which would read as a real plateau. */
    function impliedMean(target, sigma, eta, higherIsBetter, lo, hi) {
        var best = higherIsBetter ? hi : lo, worst = higherIsBetter ? lo : hi, mid, i;
        if (target >= expectedImprovement(best, sigma, eta, higherIsBetter)) return null;
        if (target <= expectedImprovement(worst, sigma, eta, higherIsBetter)) return null;
        for (i = 0; i < 44; i++) {
            mid = 0.5 * (best + worst);
            if (expectedImprovement(mid, sigma, eta, higherIsBetter) > target) best = mid;
            else worst = mid;
        }
        return 0.5 * (best + worst);
    }

    /* A belief's density at each position. Normal, on the normalized [0, 1]
     * representation the positions are already in — which is the space
     * ConfigSpace and SMAC evaluate a prior in, so what is drawn here is what
     * the optimizer would read. Unnormalized: only ratios matter, since this
     * multiplies a function that is itself only ranked. */
    function density(positions, centre, width) {
        var out = [], i, z;
        for (i = 0; i < positions.length; i++) {
            z = (positions[i] - centre) / width;
            out.push(Math.exp(-0.5 * z * z) / (width * SQ2PI));
        }
        return out;
    }

    function maxOf(values) {
        var m = 0, i;
        for (i = 0; i < values.length; i++) if (values[i] > m) m = values[i];
        return m;
    }

    /* Everything the three panels show, for one belief. `belief` is null when
     * none has been stated, in which case the weight is 1 everywhere and the
     * weighted acquisition is the plain one — the figure then reads as "what
     * the run would do next, left alone". */
    function compute(meta, belief) {
        var n = meta.positions.length,
            acq = [], weighted = [], prior = [], fiction = [],
            lo = Infinity, hi = -Infinity,
            i, w, raw, aMax, wMax, pad;

        for (i = 0; i < n; i++) {
            if (meta.mu[i] < lo) lo = meta.mu[i];
            if (meta.mu[i] > hi) hi = meta.mu[i];
        }
        /* A generous bracket for the inversion: the fiction routinely runs well
         * past anything the surrogate predicts, which is the point of it. The
         * padding is measured once, before either end moves. */
        pad = 3 * (hi - lo) + 1;
        lo -= pad;
        hi += pad;

        for (i = 0; i < n; i++) {
            acq.push(expectedImprovement(meta.mu[i], meta.sigma[i], meta.eta, meta.higherIsBetter));
        }
        if (belief) {
            raw = density(meta.positions, belief.centre, belief.width);
            for (i = 0; i < n; i++) {
                /* The floor goes on before the exponent, not after: zero raised
                 * to any positive power is still zero, so flooring afterwards
                 * would leave a written-off region at exactly zero and lose the
                 * ordering the floor exists to keep. Same reason SMAC's
                 * AbstractAcquisitionWeight adds it where it does. */
                w = Math.pow(raw[i] + 1e-12, belief.exponent);
                prior.push(w);
                weighted.push(acq[i] * w);
            }
        } else {
            for (i = 0; i < n; i++) { prior.push(1); weighted.push(acq[i]); }
        }

        for (i = 0; i < n; i++) {
            fiction.push(belief
                ? impliedMean(weighted[i], meta.sigma[i], meta.eta, meta.higherIsBetter, lo, hi)
                : null);
        }

        /* Each curve to its own maximum. Only the ranking of an acquisition
         * function means anything, and a belief can scale it by many orders of
         * magnitude — on a shared axis the unweighted curve would vanish. */
        aMax = maxOf(acq) || 1;
        wMax = maxOf(weighted) || 1;
        for (i = 0; i < n; i++) { acq[i] /= aMax; weighted[i] /= wMax; }

        return {acquisition: acq, weighted: weighted, belief: prior, fiction: fiction};
    }

    /* ── wiring ─────────────────────────────────────────────────────────── */

    var section = document.querySelector('[data-figure="acquisition_slice"]');
    if (!section) return;

    var url = section.getAttribute("data-acquisition-url"),
        plotEl = document.getElementById("figure-acquisition_slice"),
        emptyEl = document.getElementById("figure-acquisition_slice-empty"),
        warnEl = document.getElementById("acq-warning"),
        promptEl = document.getElementById("acq-compute"),
        promptBtn = document.getElementById("acq-compute-btn"),
        hpSelect = document.getElementById("acq-hp-select"),
        metricSelect = document.getElementById("metric-select"),
        cache = {},
        inflight = {},
        state = null,
        autocompute = true;

    try {
        var flags = JSON.parse(document.getElementById("autocompute-data").textContent);
        if (flags && flags.acquisition_slice === false) autocompute = false;
    } catch (e) { /* no settings shipped: compute, which is the default */ }

    function metric() { return metricSelect ? metricSelect.value : ""; }

    function setWarning(text) {
        if (!warnEl) return;
        warnEl.textContent = text || "";
        warnEl.hidden = !text;
    }
    function setPrompt(pending) { if (promptEl) promptEl.hidden = !pending; }

    /* Drawn from the cached payload rather than from whatever is on screen, so
     * a redraw after the page purges this plot (see the `figures:redrawn`
     * listener) rebuilds it from the same source as the first draw. */
    function render() {
        if (!state || !plotEl) return;
        var meta = state.figure.layout.meta.acquisition,
            values = compute(meta, state.belief),
            t = meta.traces;
        /* One call, not a react plus a restyle: the traces are filled in place
         * and handed over together. */
        state.figure.data[t.fiction].y = values.fiction;
        state.figure.data[t.belief].y = values.belief;
        state.figure.data[t.acquisition].y = values.acquisition;
        state.figure.data[t.weighted].y = values.weighted;

        /* Un-hidden *before* the draw, and this is not optional twice over.
         *
         * The page's own `draw(key, null)` hides a figure whose payload is
         * null, and this figure's payload is always null — it ships no
         * server-rendered plot. So by the time anything here runs, the div is
         * already `hidden`, and every other figure gets un-hidden by that same
         * `draw` on its way past. This one bypasses it, so it has to do the job
         * itself or it draws into a div nobody can see.
         *
         * And before, not after: a hidden element has no width, so Plotly would
         * size the plot to nothing and keep that size once it was revealed. */
        plotEl.hidden = false;
        if (emptyEl) emptyEl.hidden = true;
        Plotly.react(plotEl, state.figure.data, state.figure.layout,
                     {displaylogo: false, responsive: true});
    }

    /* The mirror of `render`, and of the page's own `draw(key, null)`: empty the
     * plot and hide the div again, so a figure with nothing to show leaves no
     * blank panel behind. `showEmpty` separates "we looked and there is nothing"
     * — which says so — from "nothing has been asked for yet", which the Compute
     * prompt above it is already explaining. */
    function clear(showEmpty) {
        if (plotEl) {
            if (plotEl.data) Plotly.purge(plotEl);
            plotEl.hidden = true;
        }
        if (emptyEl) emptyEl.hidden = !showEmpty;
    }

    function show(data) {
        if (!data.figure) { state = null; clear(true); setWarning(data.warning); return; }
        state = {figure: data.figure, belief: null};
        setWarning(data.warning);
        render();
    }

    function refresh(asked) {
        if (!hpSelect || !plotEl) return;
        var m = metric(), hp = hpSelect.value, key;
        if (!hp) return;
        key = m + ":" + hp;

        function apply(data) {
            /* Stale by the time it resolved — the reader moved on while it was
             * in flight. */
            if (metric() !== m || hpSelect.value !== hp) return;
            setPrompt(false);
            show(data);
        }

        if (cache[key]) { apply(cache[key]); return; }
        if (!autocompute && !asked) {
            setPrompt(true);
            state = null;
            clear(false);
            setWarning(null);
            return;
        }
        /* Already asked for and still in the air. Without this a metric switch
         * would fetch the same slice twice: once because the page announced it
         * had redrawn, and once because this script had not yet cached it. */
        if (inflight[key]) return;
        inflight[key] = true;
        setPrompt(false);
        fetch(url + "?metric=" + encodeURIComponent(m) + "&hp=" + encodeURIComponent(hp))
            .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
            .then(function (data) { cache[key] = data; apply(data); })
            .catch(function () { setWarning("Could not compute the acquisition slice."); })
            .then(function () { delete inflight[key]; });
    }

    if (hpSelect) hpSelect.addEventListener("change", function () { refresh(); });
    if (promptBtn) promptBtn.addEventListener("click", function () { refresh(true); });

    /* The page rebuilds every figure when the metric changes, and purges any
     * whose payload is null on the way through — including this one, which
     * ships no server-rendered plot at all. That purge lands at a moment this
     * script cannot predict, so the page announces itself instead: it
     * dispatches `figures:redrawn` once it has finished, and we put ours back.
     * The same idiom `poll:swapped` and `trials:sorted` already use.
     *
     * This is also why there is no listener of our own on the metric select.
     * There would be two then, and ours would fire second — after the page had
     * already purged us — so between the two we would redraw the *previous*
     * metric's slice from stale state and then immediately replace it. Waiting
     * to be told the page has settled is both simpler and flicker-free. */
    document.addEventListener("figures:redrawn", function () { refresh(); });

    refresh();
}());
