/* The acquisition-and-priors figure.
 *
 * The one figure on the page that draws itself. Every other figure is a Plotly
 * payload built in `ui/figures/plots.py` and handed to `Plotly.react`; this one
 * gets a styled skeleton from there (`acquisition_slice_plot`) with its prior
 * and acquisition traces empty, and fills them here.
 *
 * That split is deliberate. The prior is a thing the reader states and drags,
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

    //: Every figure here draws the same way: no modebar, and responsive so a
    //: page-width slot is actually used.
    var PLOT_CONFIG = {displaylogo: false, responsive: true, displayModeBar: false};

    /* The top of the prior panel's y axis, and the same number
     * `plots.PRIOR_AXIS_TOP` gives the layout — three places have to agree
     * about it: the server's range, the range this file reasserts when the
     * shape changes, and the pointer arithmetic that converts between a
     * position on screen and a height in [0, 1].
     *
     * Well above 1, not just clear of it. A uniform prior is a flat line at 1,
     * and with barely any headroom it sat against the top edge reading as the
     * frame — so the panel looked empty until a shape was chosen, which is the
     * opposite of what the resting state should say. */
    var PRIOR_AXIS_TOP = 1.25;

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
     * prior moves a performance curve — it is not a surrogate anyone fitted,
     * and the caller draws it dashed for that reason.
     *
     * Returns null where the answer is off the bracket: a prior sharp enough
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

    /* The distributions a prior can take, and what each is parameterised by.
     *
     * One table rather than a branch per control: the panel renders whatever a
     * kind declares, so adding a shape is an entry here and a density below,
     * not a new widget. `label` is the parameter's own symbol, which reads the
     * same in every locale and so belongs here. `string` names a translated
     * *description* in the template's `acq-prior-strings` block, used as the
     * control's tooltip — that is prose, and the message catalogs are built
     * from templates and Python, so it could not live in this file.
     *
     * `drag` names which parameters the pointer gesture moves. Normal has a
     * location and a scale, so pressing and pulling map onto it directly. Beta
     * is parameterised by shape rather than position, so it has none and is set
     * through its fields — which is the honest reason the two differ, rather
     * than an omission. */
    var KINDS = {
        uniform: {params: []},
        normal: {
            params: [{key: "mu", label: "μ", string: "mu",
                      min: 0, max: 1, step: 0.005, value: 0.5},
                     /* Up to the width of the axis itself. Beyond that a
                      * Normal on the unit interval is indistinguishable from
                      * uniform, and below 0.004 it is a spike between two grid
                      * points that nothing can resolve. */
                     {key: "sigma", label: "σ", string: "sigma",
                      min: 0.004, max: 1.0, step: 0.002, value: 0.07}],
        },
        beta: {
            params: [{key: "alpha", label: "α", string: "alpha",
                      min: 0.1, max: 20, step: 0.1, value: 2},
                     {key: "beta", label: "β", string: "beta",
                      min: 0.1, max: 20, step: 0.1, value: 2}]
        },
        /* A density given by its values rather than by parameters: points a
         * reader placed, with a smooth curve through them.
         *
         * A categorical hyperparameter is the same object with x pinned to the
         * choice indices — "one weight per choice" and "a curve through points"
         * differ only in whether a point may slide sideways, which is a
         * constraint on the editor and not a second editor. */
        tabulated: {params: [], points: true}
    };


    function currentMeta() {
        return (state && state.meta) || {};
    }

    function defaultsFor(kind) {
        var out = {}, list = (KINDS[kind] || KINDS.uniform).params, i;
        for (i = 0; i < list.length; i++) out[list[i].key] = list[i].value;
        if (KINDS[kind] && KINDS[kind].points) out.points = startingPoints(currentMeta());
        return out;
    }

    /* Where a density used to be computed, and deliberately is not.
     *
     * It comes from the server now — `ui/views._prior_density` over
     * `core.priors` — and arrives with the payload. There is no second
     * implementation of a Gaussian, a Beta or a hand-placed curve to keep in
     * step with the first, which is the whole point: a disagreement between
     * them would not have raised, it would have weighted the search by one
     * shape while this figure drew another.
     *
     * Nothing needs it locally any more. Only the freeform prior takes a
     * pointer, and every shape with parameters is set through fields that
     * already round-trip, so the density comes back in the same response that
     * saves them. For a log-scaled hyperparameter that matters twice over: the
     * one implementation states the density on the vectorized axis, where a
     * Gaussian is the log-normal such a hyperparameter deserves, and no part of
     * this file has to know a transform was involved.
     */
    function densityFor(which) {
        return (state && state.density && state.density[which]) || null;
    }

    /* Where a tabulated prior's points start: evenly spaced and flat, so the
     * first drag states something rather than starting from a shape nobody
     * chose. A categorical gets one per choice, pinned to its index. */
    function startingPoints(meta) {
        var out = [], n, i;
        if (meta && meta.kind === "categorical" && meta.positions) {
            for (i = 0; i < meta.positions.length; i++) out.push([toUnitOf(meta, meta.positions[i]), 0.5]);
            return out;
        }
        n = 5;
        for (i = 0; i < n; i++) out.push([i / (n - 1), 0.5]);
        return out;
    }

    function toUnitOf(meta, x) {
        var lo = meta.span[0], width = (meta.span[1] - meta.span[0]) || 1;
        return (x - lo) / width;
    }

    /* Which grid slot a sampled configuration's position falls in. Binary
     * search rather than a scan: the slice grid is sorted by construction, and
     * this runs once per sampled point. */
    function nearestIndex(positions, x) {
        var lo = 0, hi = positions.length - 1, mid;
        if (x <= positions[0]) return 0;
        if (x >= positions[hi]) return hi;
        while (hi - lo > 1) {
            mid = (lo + hi) >> 1;
            if (positions[mid] <= x) lo = mid; else hi = mid;
        }
        return (x - positions[lo]) <= (positions[hi] - x) ? lo : hi;
    }

    /* Which slot each sampled configuration belongs to. Depends only on the
     * payload, not on the prior, so it is worked out once and kept — otherwise
     * every drag frame would repeat two thousand binary searches to reach the
     * same answer. */
    function cloudSlots(meta) {
        var out = [], i;
        if (meta._slots) return meta._slots;
        for (i = 0; i < meta.cloud.positions.length; i++) {
            out.push(nearestIndex(meta.positions, meta.cloud.positions[i]));
        }
        meta._slots = out;
        return out;
    }

    function maxOf(values) {
        var m = 0, i;
        for (i = 0; i < values.length; i++) if (values[i] > m) m = values[i];
        return m;
    }

    /* Everything the three panels show, for one prior. `prior` is null when
     * none has been stated, in which case the weight is 1 everywhere and the
     * weighted acquisition is the plain one — the figure then reads as "what
     * the run would do next, left alone". */
    function compute(meta, prior) {
        var n = meta.positions.length,
            acq = [], weighted = [], weights = [], fiction = [],
            lo = Infinity, hi = -Infinity,
            band, k, stated, pts, envelope, slots, cloudWeight, j, slot, value,
            i, w, raw, scale, peak, pad, bare;

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
        /* A uniform prior is the same statement as no prior: it multiplies the
         * acquisition by a constant, which cannot change a ranking. So it
         * weights nothing, and casts no ghost — under it the implied curve
         * would sit exactly on the predicted one by construction, which is
         * clutter rather than information. */
        stated = !!(prior && prior.kind && prior.kind !== "uniform");
        if (stated) {
            raw = densityFor("grid");
            if (!raw || raw.length !== n) stated = false;
        }
        if (stated) {
            for (i = 0; i < n; i++) {
                /* The floor goes on before the exponent, not after: zero raised
                 * to any positive power is still zero, so flooring afterwards
                 * would leave a written-off region at exactly zero and lose the
                 * ordering the floor exists to keep. Same reason SMAC's
                 * AbstractAcquisitionWeight adds it where it does. */
                w = Math.pow(raw[i] + 1e-12, prior.exponent);
                weights.push(w);
                weighted.push(acq[i] * w);
            }
        } else {
            for (i = 0; i < n; i++) { weights.push(1); weighted.push(acq[i]); }
        }

        for (i = 0; i < n; i++) {
            fiction.push(stated
                ? impliedMean(weighted[i], meta.sigma[i], meta.eta, meta.higherIsBetter, lo, hi)
                : null);
        }

        /* Both curves through one divisor, so they can be read against each
         * other.
         *
         * Each to its own maximum — what this did before — makes them
         * incomparable: a prior that lifted the acquisition everywhere and one
         * that flattened it draw the same picture, because each is rescaled to
         * fill the panel. The silhouette survives and the effect does not.
         *
         * But a common divisor alone is not enough either, and that is why the
         * old code existed: a prior sharp enough has a weight peaking in the
         * hundreds, so `weighted` would set the divisor and squash the
         * unweighted curve to invisibility.
         *
         * So the weight is normalized to peak at 1 first. That is free rather
         * than a distortion — an acquisition function is only ever ranked, so
         * scaling it by a constant cannot change which configuration wins, and
         * SMAC's own argmax is untouched. What it buys is a picture that reads:
         * the weighted curve lies at or below the unweighted one, meeting it
         * where the prior is strongest, and the gap elsewhere is what the
         * prior has written off. */
        peak = maxOf(weights) || 1;
        for (i = 0; i < n; i++) {
            weighted[i] = weighted[i] / peak;
        }
        /* The panel curve takes the same normalization, for every shape whose
         * height has no units of its own: a Gaussian's density depends on its
         * σ, so undivided it would move the axis every time a parameter did.
         *
         * A freeform prior is the exception, and `applyPriorAxis` says why —
         * its heights *are* the panel's units, put there by hand on the same
         * [0, 1] the knots are stored in. Normalizing it draws the curve
         * somewhere the knots are not, which for the flat starting shape meant
         * a row of knots at half height under a curve pinned at one. The
         * weighting above is untouched: that division is what keeps the
         * weighted acquisition comparable to the unweighted one, and it is
         * ranking-invariant either way. */
        if (!(stated && prior.kind === "tabulated")) {
            for (i = 0; i < n; i++) weights[i] = weights[i] / peak;
        }

        /* The fiction's own spread, from the same σ the surrogate reports: the
         * fiction is a mean, and a prior does not claim to have narrowed the
         * model's uncertainty — only to have moved where the interest is.
         *
         * One band, and a faint one. This is an inference about a surrogate
         * nobody fitted, so stepping it like the measured model's spread would
         * lend it the same standing. Null wherever the inversion returned
         * nothing, so the band breaks exactly where the line it surrounds
         * does. */
        k = meta.fictionSigma || 1;
        band = [[], []];
        for (i = 0; i < n; i++) {
            if (fiction[i] === null || fiction[i] === undefined) {
                band[0].push(null);
                band[1].push(null);
            } else {
                band[0].push(fiction[i] - k * meta.sigma[i]);
                band[1].push(fiction[i] + k * meta.sigma[i]);
            }
        }

        /* The control points, back in axis coordinates so the marker trace can
         * carry them. Empty for every shape that is not tabulated, which is how
         * the trace disappears without being removed.
         *
         * Raised to the decay exponent, which is what the curve beside them is:
         * the freeform curve is left un-normalized above so that the two share
         * one scale, and the exponent is the only thing between a stated height
         * and a drawn one. `_through` is monotone (Fritsch-Carlson) and does
         * not overshoot its control points, so the density at a knot's x is
         * that knot's own height and the marker lands on the line. */
        pts = {x: [], y: []};
        if (stated && prior.kind === "tabulated" && prior.params.points) {
            for (i = 0; i < prior.params.points.length; i++) {
                pts.x.push(meta.span[0] + prior.params.points[i][0] *
                           ((meta.span[1] - meta.span[0]) || 1));
                pts.y.push(decayed(prior.params.points[i][1], prior.exponent));
            }
        }

        /* The shadow: the best acquisition reachable at each value of this
         * hyperparameter when the others are *free*, estimated from
         * configurations drawn across the whole space.
         *
         * The curve above it is a slice — every other hyperparameter pinned at
         * the incumbent — and SMAC's maximizer is under no such constraint. So
         * the gap between the two is how much interest the search can reach by
         * moving something this panel is holding still, which is most of the
         * answer to why the next configuration is not the curve's peak.
         *
         * Expected improvement is computed here, with the same function the
         * curve uses, from the mean and spread the server sent. A maximum per
         * slot rather than a mean: what matters is what the search *could*
         * find there, not how a region scores on average. Empty slots stay null
         * so the line breaks instead of interpolating across a gap in the
         * sample. */
        envelope = null;
        bare = null;
        if (meta.cloud && meta.cloud.mu && meta.cloud.mu.length) {
            slots = cloudSlots(meta);
            cloudWeight = stated ? densityFor("cloud") : null;
            if (stated && !cloudWeight) cloudWeight = null;
            envelope = [];
            /* The same envelope with the prior left out, kept alongside it.
             *
             * This is the curve the prior is actually acting on. The pair at
             * the incumbent answers a narrower question — one line through the
             * space with everything else frozen — while this one is taken over
             * the whole sampled space, which is where the search really looks.
             * Drawn without its unweighted twin, a weighted envelope says how
             * interest is distributed but not what the reader changed.
             *
             * Its own maximum, not the weighted one's: the two are separate
             * maxima over the same points, and a prior can move which sample
             * wins a slot. */
            bare = stated && cloudWeight ? [] : null;
            for (i = 0; i < n; i++) {
                envelope.push(null);
                if (bare) bare.push(null);
            }
            for (j = 0; j < meta.cloud.mu.length; j++) {
                value = expectedImprovement(meta.cloud.mu[j], meta.cloud.sigma[j],
                                            meta.eta, meta.higherIsBetter);
                slot = slots[j];
                if (bare && (bare[slot] === null || value > bare[slot])) {
                    bare[slot] = value;
                }
                if (stated && cloudWeight) {
                    value *= Math.pow(cloudWeight[j] + 1e-12, prior.exponent) / peak;
                }
                if (envelope[slot] === null || value > envelope[slot]) envelope[slot] = value;
            }
        }

        /* One divisor for everything the bottom panel draws, taken after all
         * four are known.
         *
         * It cannot be the curve's own maximum, which is what this did before.
         * The candidates the search would actually run routinely outscore
         * anything on the slice — measured at 9x on a Gaussian-process run —
         * and the panel's axis is pinned to [0, 1.08], so scaling to the curve
         * put them off the top of the figure instead of beside it. Dividing by
         * the largest of the four puts the tallest thing at 1 and leaves every
         * other one in true proportion to it.
         *
         * When the candidates dominate, the slice curve comes out small. That
         * is not a display fault — it is the finding: the line this panel draws
         * is a poor region compared with what the search has found elsewhere. */
        scale = maxOf(acq);
        if (maxOf(weighted) > scale) scale = maxOf(weighted);
        if (envelope && maxOf(envelope) > scale) scale = maxOf(envelope);
        /* The ghost is in the divisor too. Weighting can only lower a sample's
         * score relative to the peak, so the unweighted envelope is usually the
         * taller of the two — left out, it would be the one curve on the panel
         * drawn off the top of a pinned axis. */
        if (bare && maxOf(bare) > scale) scale = maxOf(bare);
        scale = scale || 1;

        for (i = 0; i < n; i++) {
            acq[i] /= scale;
            weighted[i] /= scale;
            if (envelope && envelope[i] !== null) envelope[i] /= scale;
            if (bare && bare[i] !== null) bare[i] /= scale;
        }

        return {acquisition: acq, weighted: weighted, prior: weights,
                fiction: fiction, fictionBand: band, points: pts, cloud: envelope,
                cloudBare: bare,
                /* The bare density, before the floor and the exponent. That is
                 * what a prior *is*; SMAC's own PriorWeight applies the
                 * exponent and the decay, so sending the weight would apply
                 * them twice. */
                density: raw || []};
    }

    /* Every trace a prior touches, paired with what it becomes: the fiction
     * and its bands on the top panel, the prior on the middle one, both curves
     * on the bottom. Gathered in one place so a redraw is a single call — trace
     * `y` is `editType: "calc"`, so one call per trace is one full recalc each,
     * and there are now ten of them rather than four.
     *
     * Indices come from `meta.traces`, never from counting: the server adds
     * bands and a marker conditionally, so the numbers move. */
    /* The incumbent's place on the acquisition curve: the grid point nearest
     * its own value, at that point's acquisition.
     *
     * Nearest rather than interpolated. The curve is drawn as straight segments
     * between grid points, so a marker at an interpolated height sits off the
     * line wherever the grid is coarse — which is exactly where it would be
     * noticed.
     *
     * Empty when the slice is switched off, when the server sent no incumbent
     * (a prior-only payload has none), or before there is a curve to sit on.
     * An empty marker trace draws nothing and keeps its name out of the legend
     * the same way the hidden curves do. */
    function incumbentAt(meta, values, showing) {
        var pos = meta.positions || [], best = -1, gap, closest = Infinity, i;
        if (!showing || !meta.incumbent || !values.acquisition
                || !values.acquisition.length) {
            return {x: [], y: []};
        }
        for (i = 0; i < pos.length; i++) {
            gap = Math.abs(pos[i] - meta.incumbent[0]);
            if (gap < closest) { closest = gap; best = i; }
        }
        if (best < 0) return {x: [], y: []};
        return {x: [pos[best]], y: [values.acquisition[best]]};
    }


    /* What a stated prior changes, grouped by the figure that owns it — one
     * entry per figure, so each is redrawn in a single call.
     *
     * Indices come from `meta.traces`, never from counting: the server adds a
     * marker and a cloud trace conditionally, so the numbers move. Splitting
     * the figure up made that contract carry its weight rather than test it.
     *
     * `priorPoints` is the one trace whose x moves — a control point is dragged
     * along the axis as well as up it — so x travels with y. Everything else
     * re-sends the grid it already had, which costs a copy and keeps one path. */
    function priorTraces(meta, values) {
        var t = meta.traces, grid = meta.positions, showing, at,
            out = {acquisition: {indices: [], xs: [], ys: []},
                   prior: {indices: [], xs: [], ys: []},
                   surrogate: {indices: [], xs: [], ys: []}},
            pair = t.surrogate.fictionBand;

        function add(where, index, x, y) {
            out[where].indices.push(index);
            out[where].xs.push(x);
            out[where].ys.push(y);
        }

        /* Hidden rather than emptied: an empty trace leaves its name in the
         * legend claiming a curve that is not drawn. `visible` takes it out of
         * both at once. */
        showing = !sliceToggle || sliceToggle.checked;
        add("acquisition", t.acquisition.acquisition,
            grid, showing ? values.acquisition : []);
        add("acquisition", t.acquisition.weighted,
            grid, showing ? values.weighted : []);
        /* With the slice curves, because it is a point on them. */
        if (typeof t.acquisition.incumbent === "number") {
            at = incumbentAt(meta, values, showing);
            add("acquisition", t.acquisition.incumbent, at.x, at.y);
        }
        if (typeof t.acquisition.cloud === "number" && values.cloud) {
            add("acquisition", t.acquisition.cloud, grid, values.cloud);
        }
        /* Empty unless a prior is stated, where it would sit exactly under the
         * weighted one and say nothing. */
        if (typeof t.acquisition.cloudBare === "number") {
            add("acquisition", t.acquisition.cloudBare, grid,
                values.cloudBare || []);
        }
        add("prior", t.prior.prior, grid, values.prior);
        add("prior", t.prior.priorPoints, values.points.x, values.points.y);
        add("surrogate", t.surrogate.fiction, grid, values.fiction);
        if (pair) {
            add("surrogate", pair[0], grid, values.fictionBand[0]);
            add("surrogate", pair[1], grid, values.fictionBand[1]);
        }
        return out;
    }

    /* ── wiring ─────────────────────────────────────────────────────────── */

    var section = document.querySelector('[data-figure="acquisition_slice"]');
    if (!section) return;

    var url = section.getAttribute("data-acquisition-url"),
        priorUrl = section.getAttribute("data-prior-url"),
        rewalkBtn = document.getElementById("acq-rewalk"),
        rewalkLabel = "",
        csrf = section.getAttribute("data-csrf"),
        saveTimer = 0,
        plotEl = document.getElementById("figure-acquisition_slice"),
        /* The three figures, in reading order. Separate divs so each can carry
         * its own controls underneath; they share an x axis by construction,
         * since the server gives all three the same span, ticks and positions. */
        panels = {
            acquisition: document.getElementById("acq-plot-acquisition"),
            prior: document.getElementById("acq-plot-prior"),
            surrogate: document.getElementById("acq-plot-surrogate")
        },
        emptyEl = document.getElementById("figure-acquisition_slice-empty"),
        warnEl = document.getElementById("acq-warning"),
        promptEl = document.getElementById("acq-compute"),
        promptBtn = document.getElementById("acq-compute-btn"),
        hpSelect = document.getElementById("acq-hp-select"),
        initialEl = document.getElementById("acq-initial"),
        sliceToggle = document.getElementById("acq-show-slice"),
        betaField = document.getElementById("acq-prior-beta"),
        strengthEl = document.getElementById("acq-prior-strength"),
        delayToggle = document.getElementById("acq-prior-delay"),
        delayWrap = document.getElementById("acq-delay-wrap"),
        evaluateBtn = document.getElementById("acq-evaluate"),
        resetBtn = document.getElementById("acq-reset"),
        resetUrl = section.getAttribute("data-reset-url"),
        verdictEl = document.getElementById("acq-verdict"),
        verdictDialog = document.getElementById("acq-verdict-dialog"),
        verdictDetail = document.getElementById("acq-verdict-detail"),
        evaluateUrl = section.getAttribute("data-evaluate-url"),
        /* No run behind this page: the prior panel is all there is, and the
         * only request worth making is the one that needs no model. */
        priorOnlyPage = section.getAttribute("data-prior-only") === "1",
        metricSelect = document.getElementById("metric-select"),
        cache = {},
        defaultBeta = null,
        inflight = {},
        state = null,
        autocompute = true;

    try {
        var flags = JSON.parse(document.getElementById("autocompute-data").textContent);
        if (flags && flags.acquisition_slice === false) autocompute = false;
    } catch (e) { /* no settings shipped: compute, which is the default */ }

    /* The page's selector where there is one. Without a run there is no metric
     * switcher on the page at all, so the card carries the experiment's own —
     * a prior is stated against a hyperparameter, and the metric only decides
     * which surrogate would have been fitted. */
    function metric() {
        return metricSelect ? metricSelect.value
                            : (section.getAttribute("data-metric") || "");
    }

    function setWarning(text) {
        if (!warnEl) return;
        warnEl.textContent = text || "";
        warnEl.hidden = !text;
    }
    function setPrompt(pending) { if (promptEl) promptEl.hidden = !pending; }

    /* The initial design runs before there is a model to weight, so a prior
     * stated now cannot touch the trials still to come out of it. Said plainly
     * and only while it is true — once the design is exhausted the banner goes,
     * because from then on a prior does reach every trial. */
    function setInitialDesign(info) {
        if (!initialEl) return;
        if (!info || info.done) {
            /* Cleared as well as hidden: an element that keeps its text is one
             * style rule away from showing it again. */
            initialEl.textContent = "";
            initialEl.hidden = true;
            return;
        }
        initialEl.textContent = text("initial-design", {
            trials: info.trials, size: info.size, name: info.name || "?"});
        initialEl.hidden = false;
    }

    /* Drawn from the cached payload rather than from whatever is on screen, so
     * a redraw after the page purges this plot (see the `figures:redrawn`
     * listener) rebuilds it from the same source as the first draw. */
    function render() {
        if (!state || !plotEl) return;
        var values = compute(state.meta, state.prior),
            filled = priorTraces(state.meta, values), name, group, i;

        /* Filled in place, then handed over — one `Plotly.react` per figure,
         * never a react plus a restyle. */
        for (name in filled) {
            if (!Object.prototype.hasOwnProperty.call(filled, name)) continue;
            /* A prior-only payload carries one figure. The other two are what
             * the surrogate is for, and arrive when it has been fitted. */
            if (!state.figures[name]) continue;
            group = filled[name];
            for (i = 0; i < group.indices.length; i++) {
                state.figures[name].data[group.indices[i]].y = group.ys[i];
                state.figures[name].data[group.indices[i]].x = group.xs[i];
            }
        }

        /* Un-hidden *before* the draw, and this is not optional twice over.
         *
         * The page's own `draw(key, null)` hides a figure whose payload is
         * null, and this figure's payload is always null — it ships no
         * server-rendered plot. So by the time anything here runs, the wrapper
         * is already `hidden`, and every other figure gets un-hidden by that
         * same `draw` on its way past. This one bypasses it, so it has to do
         * the job itself or it draws into a div nobody can see.
         *
         * And before, not after: a hidden element has no width, so Plotly would
         * size each plot to nothing and keep that size once it was revealed. */
        plotEl.hidden = false;
        if (emptyEl) emptyEl.hidden = true;

        /* No modebar. Its zoom and pan were the only working interaction these
         * had and an accidental one — every axis is pinned, so the buttons
         * would be dead controls sitting over the figures. The one real
         * interaction is the prior drag, bound after the draw because Plotly
         * builds the drag layer as part of drawing and rebuilds it each time. */
        if (state.figures.acquisition) {
            Plotly.react(panels.acquisition, state.figures.acquisition.data,
                         state.figures.acquisition.layout, PLOT_CONFIG);
        }
        if (state.figures.surrogate) {
            Plotly.react(panels.surrogate, state.figures.surrogate.data,
                         state.figures.surrogate.layout, PLOT_CONFIG);
        }
        Plotly.react(panels.prior, state.figures.prior.data,
                     state.figures.prior.layout, PLOT_CONFIG).then(bindPrior);
    }

    /* The mirror of `render`, and of the page's own `draw(key, null)`: empty the
     * plot and hide the div again, so a figure with nothing to show leaves no
     * blank panel behind. `showEmpty` separates "we looked and there is nothing"
     * — which says so — from "nothing has been asked for yet", which the Compute
     * prompt above it is already explaining. */
    function clear(showEmpty) {
        var name;
        for (name in panels) {
            if (!Object.prototype.hasOwnProperty.call(panels, name)) continue;
            if (panels[name] && panels[name].data) Plotly.purge(panels[name]);
        }
        if (plotEl) plotEl.hidden = true;
        if (emptyEl) emptyEl.hidden = !showEmpty;
    }

    /* ── stating a prior ───────────────────────────────────────────────────
     *
     * The middle panel is the input: press to say where the good values are,
     * drag an edge to say how sure you are, double-click to withdraw it.
     *
     * Plotly's own shape editing cannot drive this. Of the places the bundle
     * emits `plotly_relayouting`, none is in shape code — a shape drag repaints
     * the SVG path directly and reports only on mouse-up, which is far too late
     * to recompute a curve against. So the gesture is hand-rolled on the
     * panel's own drag layer, which the server leaves free by setting
     * `dragmode=False`.
     *
     * Pointer Events rather than mouse events: touch and pen come for free, and
     * `setPointerCapture` keeps the gesture alive once the pointer leaves the
     * panel — which it does constantly, because the interesting widths are
     * narrow and the panel is short. */

    var POINT_PX = 11,          /* how close counts as grabbing a control point */
        MIN_POINTS = 2,         /* below this there is no curve to speak of */
        held = -1,              /* which control point the gesture has hold of */
        drag = null,
        frame = 0,
        bound = null;

    /* Row 2's drag layer. Matched on the y half of the subplot id because
     * `shared_xaxes` leaves the x half ambiguous ("xy2" or "x2y2"), while y2 is
     * the middle panel either way. */
    /* The prior figure's drag layer. It is its own figure now with a single
     * subplot, so there is no ambiguity left to match around — the split paid
     * for itself here. */
    function priorLayer() {
        return panels.prior ? panels.prior.querySelector(".nsewdrag") : null;
    }

    function axisSpan() { return state.meta.span; }

    /* Axis units per pixel, from the bounding box alone — no Plotly internals,
     * no axis objects. The x axis is pinned server-side with an explicit range
     * and `fixedrange` precisely so this stays pure geometry. */
    function perPixel(box) {
        var span = axisSpan();
        return (span[1] - span[0]) / box.width;
    }

    /* Which shapes the panel itself can edit, which is only the freeform one.
     *
     * A Normal and a Beta are fully stated by two numbers that already have
     * fields and sliders beneath the figure, and those say what they are: μ, σ,
     * α, β, with their ranges visible. Dragging the curve was a second way to
     * set the same two numbers, less precisely and without naming them — and it
     * made every panel look editable, which invited dragging shapes that have
     * nothing to give. The freeform prior is different in kind: its points
     * *are* the statement, and no field could carry them. */
    function editable() {
        var kind = state && state.prior && state.prior.kind;
        return !!(kind && KINDS[kind] && KINDS[kind].points);
    }

    /* One `Plotly.update` per figure — three rather than the one this cost
     * before the split. That is the price of the controls sitting under the
     * graph they belong to, and it is paid per frame, so each figure still
     * gets exactly one call and never two. */
    function redrawPrior() {
        var meta, values, filled, name, group, i;
        if (!state) return;
        meta = state.meta;
        values = compute(meta, state.prior);
        filled = priorTraces(meta, values);
        for (name in filled) {
            if (!Object.prototype.hasOwnProperty.call(filled, name)) continue;
            group = filled[name];
            if (!state.figures[name]) continue;
            if (!group.indices.length || !panels[name] || !panels[name].data) continue;
            /* Written back to the payload as well as to the screen: the page
             * purges these on a metric switch and `render` rebuilds from there,
             * so a dragged prior has to survive in the payload too. */
            for (i = 0; i < group.indices.length; i++) {
                state.figures[name].data[group.indices[i]].y = group.ys[i];
                state.figures[name].data[group.indices[i]].x = group.xs[i];
            }
            Plotly.update(panels[name], {x: group.xs, y: group.ys}, {}, group.indices);
        }
    }

    /* One repaint per frame. A pointer reports far faster than three panels can
     * redraw, and without this every move would queue a recalc that is already
     * stale by the time it runs. */
    function scheduleRedraw() {
        if (frame) return;
        frame = requestAnimationFrame(function () { frame = 0; redrawPrior(); });
    }

    /* Only the freeform prior reaches a gesture. Everything else on this panel
     * is stated in the fields below it, so a press here has nothing to mean —
     * and the cursor says so before it is pressed. */
    function onPointerDown(ev) {
        if (!state || ev.button) return;
        if (state.prior && state.prior.kind === "tabulated") {
            onTabulatedDown(ev, ev.currentTarget.getBoundingClientRect());
        }
    }

    function onPointerMove(ev) {
        if (!drag || !state || !state.prior) return;
        if (state.prior.kind === "tabulated") {
            onTabulatedMove(ev, ev.currentTarget.getBoundingClientRect());
        }
    }

    /* ── tabulated priors ───────────────────────────────────────────────────
     *
     * Editing a density by its values rather than its parameters. The panel's y
     * axis is switched to a plain linear [0, 1] for this — see `applyPriorAxis`
     * — so a point's position is pure geometry in both directions, the same
     * property the x axis already had.
     */

    /* A freeform prior's knots are stated heights; the curve through them is
     * what the search currently weights by, which is that density raised to the
     * decay exponent. The knots are drawn decayed too, so the picture is one
     * statement rather than two: where the curve runs now, and the points that
     * put it there, on the same scale.
     *
     * Which makes the drag a conversion in both directions — the pointer is a
     * position on the decayed curve, and what gets stored is the stated height
     * underneath it. Below `DECAY_FLOOR` that inversion stops meaning anything:
     * the exponent is near zero, every height draws at nearly 1, and the root
     * that would undo it sends the smallest difference in pointer position to
     * the whole of [0, 1]. A prior that faded that far states nothing, and the
     * flat curve says so; editing it moves the knots as though it had not
     * faded, which is the one reading that stays usable. */
    var DECAY_FLOOR = 0.02;

    function decayExponent() {
        var e = state && state.prior ? state.prior.exponent : 1;
        return typeof e === "number" && e >= DECAY_FLOOR ? e : 1;
    }

    /* A stated height, as it is drawn. */
    function decayed(h, exponent) {
        return Math.pow(h + 1e-12, exponent);
    }

    /* And back: a height read off the panel, as it would have to be stated. */
    function stated(v, exponent) {
        return Math.min(Math.max(Math.pow(v, 1 / exponent), 0), 1);
    }


    /* The panel's y axis runs to PRIOR_AXIS_TOP, not to 1: a knot at full
     * height needs headroom above it or its marker is drawn half outside the
     * plotting area. Both directions of the conversion have to know that, or
     * the pointer lands a few per cent away from where Plotly put the knot —
     * near enough to look like a jitter, far enough to miss a grab. */
    function pointerUnit(box, ev) {
        return {
            u: Math.min(Math.max((ev.clientX - box.left) / box.width, 0), 1),
            /* Where the pointer is on the drawn curve, converted to the
             * height that would have to be stated to put it there. */
            v: stated(Math.min(Math.max(
                ((box.bottom - ev.clientY) / box.height) * PRIOR_AXIS_TOP, 0), 1),
                decayExponent())
        };
    }

    function nearestPoint(box, ev) {
        var points = state.prior.params.points || [],
            best = -1, bestDistance = POINT_PX, i, dx, dy, d;
        for (i = 0; i < points.length; i++) {
            dx = (box.left + points[i][0] * box.width) - ev.clientX;
            dy = (box.bottom - (decayed(points[i][1], decayExponent())
                                / PRIOR_AXIS_TOP) * box.height) - ev.clientY;
            d = Math.sqrt(dx * dx + dy * dy);
            if (d < bestDistance) { bestDistance = d; best = i; }
        }
        return best;
    }

    function onTabulatedDown(ev, box) {
        var at = pointerUnit(box, ev), points = state.prior.params.points, i;
        held = nearestPoint(box, ev);

        /* Missing every point adds one, which is how a curve gains detail —
         * except on a categorical, where the points are the choices and there
         * is no "between" to add one at. */
        if (held < 0) {
            if (currentMeta().kind === "categorical") return;
            for (i = 0; i < points.length && points[i][0] < at.u; i++) { /* find place */ }
            points.splice(i, 0, [at.u, at.v]);
            held = i;
        }
        drag = "point";
        ev.currentTarget.setPointerCapture(ev.pointerId);
        ev.preventDefault();
        scheduleRedraw();
    }

    function onTabulatedMove(ev, box) {
        var at = pointerUnit(box, ev), points = state.prior.params.points, p;
        if (held < 0 || held >= points.length) return;
        p = points[held];
        p[1] = at.v;
        /* x is pinned on a categorical — a choice does not move — and elsewhere
         * it is held between its neighbours rather than re-sorted, so the point
         * being dragged keeps its identity through the whole gesture. */
        if (currentMeta().kind !== "categorical") {
            p[0] = Math.min(Math.max(at.u,
                            held > 0 ? points[held - 1][0] + 1e-4 : 0),
                            held < points.length - 1 ? points[held + 1][0] - 1e-4 : 1);
        }
        ev.preventDefault();
        scheduleRedraw();
    }

    /* Double-clicking a point removes it; double-clicking anywhere else is the
     * withdraw gesture the other shapes use. */
    function onTabulatedDoubleClick(ev, box) {
        var points = state.prior.params.points, i = nearestPoint(box, ev);
        if (i < 0) return false;
        if (points.length > MIN_POINTS) points.splice(i, 1);
        held = -1;
        scheduleRedraw();
        return true;
    }

    /* A hand-drawn height has no units, and a log axis cannot reach zero, so a
     * tabulated prior gets a linear [0, 1] panel: relative preference, which is
     * all an unnormalized density means anyway. Every other shape keeps the log
     * axis, where a sharp peak is legible instead of a spike on a flat floor. */
    function applyPriorAxis(kind) {
        /* One axis for every shape, and it never moves.
         *
         * A hand-drawn height has no units and a log axis cannot reach zero, so
         * the freeform prior always needed a plain [0, 1] panel. Every other
         * shape had a log axis that auto-ranged, which meant the scale changed
         * under the reader as they adjusted a parameter — the curve would look
         * identical while the axis labels told a different story. Since the
         * curve is now peak-normalized (see `compute`), they can all share the
         * freeform one.
         *
         * The cost is that a very sharp prior draws as a spike rather than
         * being opened out by the logarithm. That is what a very sharp prior
         * is, and a moving axis was a worse way to say it. */
        var layout = state && state.figures ? state.figures.prior.layout : null,
            wanted = {type: "linear", range: [0, PRIOR_AXIS_TOP],
                      fixedrange: true, autorange: false};
        if (!layout) return;
        layout.yaxis = layout.yaxis || {};
        layout.yaxis.type = wanted.type;
        layout.yaxis.fixedrange = wanted.fixedrange;
        layout.yaxis.autorange = wanted.autorange;
        if (wanted.range) layout.yaxis.range = wanted.range;
        else delete layout.yaxis.range;
        if (panels.prior && panels.prior.data) {
            Plotly.relayout(panels.prior, {
                "yaxis.type": wanted.type,
                "yaxis.fixedrange": wanted.fixedrange,
                "yaxis.autorange": wanted.autorange,
                "yaxis.range": wanted.range || null
            });
        }
    }

    function onPointerUp(ev) {
        if (!drag) return;
        drag = null;
        held = -1;
        savePrior();
        if (ev.currentTarget.hasPointerCapture(ev.pointerId)) {
            ev.currentTarget.releasePointerCapture(ev.pointerId);
        }
    }

    /* Withdrawing the prior. Double-click is already Plotly's own "put it
     * back how it was" everywhere else on the page. */
    function onDoubleClick(ev) {
        if (!state) return;
        /* On a tabulated prior this first means "remove this point", and only
         * falls through to withdrawing when the pointer is not on one. */
        if (state.prior && state.prior.kind === "tabulated"
                && onTabulatedDoubleClick(ev, ev.currentTarget.getBoundingClientRect())) {
            ev.preventDefault();
            return;
        }
        if (!editable()) return;
        state.prior = {kind: "uniform", params: {}, exponent: 1, decay: "none",
                       betaRatio: defaultBeta};
        drag = null;
        ev.preventDefault();
        syncControls();
        scheduleRedraw();
        savePrior();
    }

    /* Plotly builds a fresh drag layer on every draw, so this runs after each
     * one. Comparing element identity is what stops it binding twice to the
     * same node; listeners on a replaced node go with it. */
    function bindPrior() {
        var hit = priorLayer();
        if (!hit || hit === bound) return;
        bound = hit;
        hit.style.touchAction = "none";   /* or the browser scrolls instead */
        hit.style.cursor = editable() ? "crosshair" : "default";
        hit.addEventListener("pointerdown", onPointerDown);
        hit.addEventListener("pointermove", onPointerMove);
        hit.addEventListener("pointerup", onPointerUp);
        hit.addEventListener("pointercancel", onPointerUp);
        hit.addEventListener("dblclick", onDoubleClick);
    }

    /* ── the controls ───────────────────────────────────────────────────────
     *
     * A number field and a slider per parameter of whichever shape is chosen.
     * All three ways of stating a prior — the picker, the fields, the drag —
     * write to the same object, and the drag writes back into the fields, so
     * they cannot disagree about what is currently stated.
     */

    var kindSelect = document.getElementById("acq-prior-kind"),
        decaySelect = document.getElementById("acq-prior-decay"),
        paramsBox = document.getElementById("acq-prior-params"),
        stringsEl = document.getElementById("acq-prior-strings"),
        fields = {};

    /* The strings this script writes onto the page, read from the template that
     * can translate them. `text("key", {name: value})` fills `{name}` holes,
     * which is the form a translator can reorder — "Trial 3 of 12" does not put
     * its numbers in that order in every language. Braces rather than
     * `%(name)s`: extracting a template doubles every `%`, so that form could
     * never match at runtime. */
    function text(key, values) {
        var box = document.getElementById("acq-strings"),
            out = box ? (box.getAttribute("data-" + key) || "") : "";
        if (!values) return out;
        return out.replace(/\{(\w+)\}/g, function (whole, name) {
            return Object.prototype.hasOwnProperty.call(values, name)
                ? values[name] : whole;
        });
    }

    function labelFor(name) {
        return (stringsEl && stringsEl.dataset[name]) || name;
    }

    /* A Normal or a Beta over choice indices is a curve drawn through things
     * that have no order between them, so a categorical hyperparameter is
     * offered Uniform and the point editor and nothing else. That editor is the
     * freeform one, with its points pinned to the choices. */
    function offerKinds() {
        var categorical = currentMeta().kind === "categorical", i, option;
        if (!kindSelect) return;
        for (i = 0; i < kindSelect.options.length; i++) {
            option = kindSelect.options[i];
            option.hidden = categorical
                && (option.value === "normal" || option.value === "beta");
        }
    }

    /* Rebuilt when the shape changes, and only then: each shape has its own
     * parameters, and nothing else moves them. The panel above is inert for
     * every shape that has fields, so these inputs are the single way in and
     * there is no second writer to keep them in step with. */
    function renderParams() {
        var kind = (state && state.prior && state.prior.kind) || "uniform",
            spec = (KINDS[kind] || KINDS.uniform).params, i, param, row, number, slider;
        fields = {};
        if (!paramsBox) return;
        paramsBox.textContent = "";
        if (!state || !state.prior) return;

        for (i = 0; i < spec.length; i++) {
            param = spec[i];
            row = document.createElement("label");
            row.className = "prior-param";
            row.appendChild(document.createTextNode(param.label + " "));
            row.title = labelFor(param.string);

            number = document.createElement("input");
            slider = document.createElement("input");
            number.type = "number";
            slider.type = "range";
            [number, slider].forEach(function (el) {
                el.min = param.min;
                el.max = param.max;
                el.step = param.step;
                el.value = state.prior.params[param.key];
            });
            row.appendChild(number);
            row.appendChild(slider);
            paramsBox.appendChild(row);

            fields[param.key] = {number: number, slider: slider, spec: param};
            bindField(param.key);
        }
    }

    function bindField(key) {
        var pair = fields[key];
        function take() {
            var v = parseFloat(this.value);
            if (!isFinite(v)) return;          /* mid-typing, "-" or "" */
            v = Math.min(Math.max(v, pair.spec.min), pair.spec.max);
            state.prior.params[key] = v;
            pair.number.value = v;
            pair.slider.value = v;
            scheduleRedraw();
            savePrior();
        }
        pair.number.addEventListener("input", take);
        pair.slider.addEventListener("input", take);
    }

    /* ── keeping it ─────────────────────────────────────────────────────────
     *
     * A prior is not a way of looking at a run, it is a statement about the next
     * one, so it outlives the page. Saved a beat after the reader stops rather
     * than on every keystroke or every dragged point: both produce edits far
     * faster than a round trip, and what matters is where they left it.
     */
    /* What the optimizer said it would run next, drawn onto the acquisition
     * figure as a rug along the top.
     *
     * Held on `state.figures` rather than pushed straight at Plotly, because
     * `render` redraws from that cached payload on every pointer move — a
     * restyle would be wiped by the next drag. It only rewrites the indices
     * `priorTraces` names, so these two survive it untouched.
     *
     * Position and rank only. Both come from the search itself; a height would
     * have to be an acquisition value computed here, and the two disagree by
     * fractions of a percent — enough to put the "runs next" marker on a
     * visibly shorter point than one ranked below it. */
    function showCandidates(list) {
        if (setCandidates(list)) render();
    }

    /* How far apart two candidates must be to be drawn apart, in axis units.
     *
     * A pixel question, so it is answered in pixels. The panel is measured
     * rather than assumed because the figure is full-width and responsive, and
     * the same slate is legible at 1200px and a single blob at 400.
     *
     * Falls back to a hundredth of the axis when the plot area cannot be
     * measured — before the first draw, or while the card is hidden, where
     * every box is zero and every candidate would otherwise merge into one. */
    function mergeTolerance() {
        var layer = panels.acquisition
                    ? panels.acquisition.querySelector(".nsewdrag") : null,
            box = layer ? layer.getBoundingClientRect() : null,
            span = state.meta.span,
            width = Math.abs(span[1] - span[0]) || 1;
        if (!box || !box.width) return width / 100;
        return (state.meta.markerPx || 13) * perPixel(box);
    }

    /* Candidates the axis cannot separate, gathered into one marker each.
     *
     * They collide for two different reasons and both are common. An integer or
     * categorical hyperparameter has few reachable positions, so exact ties are
     * routine — three of twelve landing on the same `max_depth`. And local
     * search converges: a whole slate can span two percent of the axis, which
     * is narrower than a single marker.
     *
     * Merged rather than spread. The alternative is a vertical offset, and this
     * figure gave up height deliberately — the markers say *where*, never *how
     * good*, because the rank comes from the search and any height here would
     * be the browser's own arithmetic disagreeing with it. A jitter would hand
     * that claim straight back, in a form the reader cannot even question.
     *
     * And the pile-up is the finding. A slate crowded into a sliver of the axis
     * says the search has stopped discriminating on this hyperparameter, which
     * is worth seeing; spreading it out would draw exactly the opposite. */
    function cluster(list, tolerance) {
        var out = [], i, c, last;
        for (i = 0; i < list.length; i++) {
            c = list[i];
            last = out.length ? out[out.length - 1] : null;
            if (last && c.position - last.at <= tolerance) {
                last.members.push(c);
                /* The marker sits on the group, not on its first arrival. */
                last.at = (last.at * (last.members.length - 1) + c.position)
                          / last.members.length;
            } else {
                out.push({at: c.position, members: [c]});
            }
        }
        return out;
    }

    /* One marker's hover.
     *
     * Every configuration in a pile spelled out was a wall of text taller than
     * the figure, which Plotly then pushed off the edge — and three near-
     * identical blocks of hyperparameters are hard to read even when they fit.
     * A cluster names its ranks on one line and spells out only the best of
     * them, which is the one that would actually run; the others differ from it
     * in dimensions this axis is not showing anyway.
     */
    function hoverFor(group, total) {
        var members = group.members, best = members[0], ranks = [], i;
        for (i = 0; i < members.length; i++) ranks.push(members[i].rank);

        if (members.length === 1) {
            return "<b>" + text("rank", {rank: best.rank, total: total})
                   + (best.rank === 1 ? " — " + text("runs-next") : "") + "</b>"
                   + "<br>" + best.origin + "<br><br>" + best.label;
        }
        return "<b>" + text("pile", {count: members.length}) + "</b><br>"
               + text("pile-ranks", {ranks: ranks.join(", "), total: total})
               + "<br><br><b>" + text("pile-best", {rank: best.rank}) + "</b>"
               + "<br>" + best.label;
    }

    function setCandidates(list) {
        var idx = state && state.meta && state.meta.traces
                  && state.meta.traces.acquisition,
            y = (state && state.meta && state.meta.rugY) || 1.03,
            trace, sorted, groups, i, g;
        if (!state || !idx || idx.candidates === undefined) return false;

        state.candidates = (list && list.length) ? list : null;
        trace = state.figures.acquisition.data[idx.candidates];
        trace.x = []; trace.y = []; trace.text = []; trace.customdata = [];
        if (!state.candidates) return true;

        /* By position to cluster, having arrived ranked. Each group keeps its
         * own members in rank order so the hover reads best-first. */
        sorted = list.slice().sort(function (a, b) { return a.position - b.position; });
        groups = cluster(sorted, mergeTolerance());

        for (i = 0; i < groups.length; i++) {
            g = groups[i];
            g.members.sort(function (a, b) { return a.rank - b.rank; });
            trace.x.push(g.at);
            trace.y.push(y);
            /* A bare count, and only where there is something to count. */
            trace.text.push(g.members.length > 1 ? String(g.members.length) : "");
            trace.customdata.push([hoverFor(g, list.length)]);
        }
        return true;
    }

    function savePrior() {
        if (saveTimer) clearTimeout(saveTimer);
        /* Shorter than it was. This is no longer only a save — the density
         * comes back in the same response, so this delay is how long the curve
         * lags the fields. Long enough to coalesce a slider drag, short enough
         * to feel like the figure is following. */
        saveTimer = setTimeout(function () { saveTimer = 0; postPrior(); }, 180);
    }

    function postPrior(rewalk) {
        if (!priorUrl || !state || !state.hp) return;
        /* The verdict was about the belief as it stood. Any save means it has
         * moved, so the old answer goes rather than sitting under a curve it no
         * longer describes — a stale approval is the one outcome here that
         * could mislead somebody into not asking again. */
        setVerdict(null);
        var meta = currentMeta(),
            stated = !!(state.prior && state.prior.kind !== "uniform"),
            body = {hp: state.hp, prior: null, rewalk: !!rewalk};

        if (stated) {
            body.prior = {kind: state.prior.kind, params: state.prior.params,
                          decay: {shape: state.prior.decay,
                                  beta_ratio: state.prior.betaRatio},
                          /* The server refuses it once the design is over, so
                           * this is what was asked for rather than what will
                           * hold — see `_inside_initial_design`. */
                          delay_decay: !!(delayToggle && delayToggle.checked)};
        }
        /* Where the density is wanted back. The server evaluates it at these
         * positions and returns it in this same response, so a save and a
         * redraw are one round trip rather than two — and the curve the reader
         * ends up looking at came from the same function the optimizer will
         * weight by. */
        body.meta = {positions: meta.positions, span: meta.span,
                     cloud: {positions: (meta.cloud || {}).positions || []}};

        fetch(priorUrl, {
            method: "POST",
            headers: {"Content-Type": "application/json", "X-CSRFToken": csrf},
            body: JSON.stringify(body)
        }).then(function (r) {
            if (!r.ok) { setWarning(text("save-failed")); return null; }
            return r.json().catch(function () { return null; });
        }).then(function (data) {
            /* Only ever applied to the slice that asked for it. The reader can
             * change hyperparameter while a walk is in flight, and these
             * positions are coordinates on the axis they were computed for. */
            if (!data || !state || state.hp !== body.hp) return;
            /* The density for what was just stated. Held on `state` so
             * `render` draws from it, the same way every other server-supplied
             * number on this figure is held. */
            if (data.density) { state.density = data.density; render(); }
            if (!rewalk) return;
            if (data.candidates) showCandidates(data.candidates);
            else setWarning(text("no-optimizer"));
        }).catch(function () { setWarning(text("save-failed")); })
          .then(function () { if (rewalk) setBusy(false); });
    }

    function setBusy(on) {
        if (!rewalkBtn) return;
        rewalkBtn.disabled = !!on;
        rewalkBtn.textContent = on ? text("asking") : rewalkLabel;
    }

    /* ── judging a belief ──────────────────────────────────────────────────
     *
     * DynaBO's safeguard, asked for rather than continuous. It costs a model
     * fit and two hundred acquisition evaluations, so it is a button for the
     * same reason the re-walk is one.
     *
     * The verdict is SMAC's: the server offers the prior to `add_prior` behind
     * an `IncumbentComparisonPolicy` and reports whether it was registered.
     * Nothing here re-decides it, and nothing here is written — a refused
     * belief stays exactly where the reader left it until they say otherwise.
     */

    function setVerdict(message, kind) {
        if (!verdictEl) return;
        if (!message) {
            verdictEl.textContent = "";
            verdictEl.hidden = true;
            return;
        }
        /* One element, three tones. `alert` carries the shape; the modifier
         * says whether this is a finding, a refusal or neither. */
        verdictEl.className = "alert" + (kind ? " " + kind : "");
        verdictEl.textContent = message;
        verdictEl.hidden = false;
    }

    /* Withdrawing, through the control that already does it: setting the shape
     * to uniform is what "no prior" means everywhere else in this file, and
     * dispatching the event runs the same handler a reader would. One path, so
     * a discard cannot drift from a withdrawal. */
    function discardPrior() {
        if (!kindSelect) return;
        kindSelect.value = "uniform";
        kindSelect.dispatchEvent(new Event("change"));
    }

    function judgePrior() {
        if (!evaluateUrl || !state || !state.hp || !evaluateBtn) return;
        var body = new FormData();
        body.append("hp", state.hp);
        evaluateBtn.disabled = true;
        setVerdict(text("judging"), "");
        fetch(evaluateUrl, {method: "POST", headers: {"X-CSRFToken": csrf},
                            body: body})
            .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
            .then(function (data) {
                setVerdict(data.message,
                           data.verdict === "rejected" ? "warning" : "");
                /* Only a refusal asks a question. Approval and "could not be
                 * judged" are both things to read, not things to answer. */
                if (data.verdict === "rejected" && verdictDialog) {
                    if (verdictDetail) verdictDetail.textContent = data.message;
                    if (verdictDialog.showModal) verdictDialog.showModal();
                }
            })
            .catch(function () { setVerdict(text("judge-failed"), "warning"); })
            .then(function () { evaluateBtn.disabled = false; });
    }

    if (delayToggle) {
        delayToggle.addEventListener("change", function () {
            if (!state || !state.prior) return;
            state.prior.delayDecay = delayToggle.checked;
            /* A save, because it changes what the search will do — and the
             * response brings back the strength the belief now reports, which
             * is the whole visible difference. */
            savePrior();
        });
    }
    /* Back to what the last run searched under, or to uniform when there was
     * nothing. The server decides which — it is the side that knows what each
     * run recorded — and hands back the restored belief, so the panel is redrawn
     * from the same answer every other request gives it rather than from a
     * second reconstruction here. */
    function resetPrior() {
        if (!resetUrl || !state || !state.hp || !resetBtn) return;
        var body = new FormData();
        body.append("hp", state.hp);
        resetBtn.disabled = true;
        fetch(resetUrl, {method: "POST", headers: {"X-CSRFToken": csrf},
                         body: body})
            .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
            .then(function (data) {
                var restored = data.prior && data.prior.kind;
                state.prior = restored
                    ? {kind: data.prior.kind,
                       params: data.prior.params || {},
                       exponent: data.prior.exponent || 1,
                       decay: (data.prior.decay || {}).shape || "none",
                       delayDecay: data.prior.delay_decay !== false,
                       betaRatio: (data.prior.decay || {}).beta_ratio != null
                                  ? (data.prior.decay || {}).beta_ratio : defaultBeta}
                    : {kind: "uniform", params: defaultsFor("uniform"),
                       exponent: 1, decay: "none", delayDecay: true,
                       betaRatio: defaultBeta};
                /* Said out loud, because the two outcomes look the same on an
                 * untouched panel and mean different things: put back to
                 * something, or put back to nothing because there was nothing. */
                setVerdict(text(data.restored ? "reset-to-last" : "reset-to-uniform"), "");
                syncControls();
                scheduleRedraw();
                /* No save. The server has already written it — this is what it
                 * wrote — and posting it back would re-anchor a belief that was
                 * restored precisely to keep its anchor. */
                showCandidates(null);
            })
            .catch(function () { setVerdict(text("reset-failed"), "warning"); })
            .then(function () { resetBtn.disabled = false; });
    }

    if (resetBtn) {
        resetBtn.addEventListener("click", resetPrior);
    }
    if (evaluateBtn) {
        evaluateBtn.addEventListener("click", judgePrior);
    }
    if (verdictDialog) {
        var keepBtn = document.getElementById("acq-verdict-keep"),
            discardBtn = document.getElementById("acq-verdict-discard");
        if (keepBtn) {
            keepBtn.addEventListener("click", function () { verdictDialog.close(); });
        }
        if (discardBtn) {
            discardBtn.addEventListener("click", function () {
                verdictDialog.close();
                discardPrior();
                /* The banner goes with the belief it was about. */
                setVerdict(null);
            });
        }
    }


    /* What the stated belief currently counts for.
     *
     * The exponent has always been here — it is what the curve is drawn from,
     * and what the knots are now drawn from too — and the panel never said what
     * it was, so a decayed prior looked like a differently-shaped one. The
     * number comes from the server with the payload (`_decayed_prior`), because
     * the schedules are SMAC's; nothing here recomputes it.
     *
     * Only while it is doing something. Without a decay the exponent is 1 for
     * the whole run, and a line saying so on every panel is noise; a uniform
     * prior states nothing to fade in the first place. */
    function showStrength(kind) {
        var e = state && state.prior ? state.prior.exponent : null,
            shape = (state && state.prior && state.prior.decay) || "none";
        if (!strengthEl) return;
        if (kind === "uniform" || shape === "none" || typeof e !== "number") {
            strengthEl.textContent = "";
            strengthEl.hidden = true;
            return;
        }
        /* Two decimals: the exponent moves slowly and the reader is watching
         * for "most of it" against "hardly any of it", not for a fourth digit. */
        strengthEl.textContent = text("strength", {strength: e.toFixed(2)});
        strengthEl.hidden = false;
    }


    /* The whole row, after anything that changes the shape. */
    function syncControls() {
        var kind = (state && state.prior && state.prior.kind) || "uniform";
        if (kindSelect) kindSelect.value = kind;
        if (decaySelect) {
            /* Nothing to fade. Decay shrinks a prior's exponent toward one as
             * the run proceeds, and a uniform prior is already the constant it
             * would decay to. */
            decaySelect.disabled = kind === "uniform";
            decaySelect.value = (state && state.prior && state.prior.decay) || "none";
        }
        if (betaField) {
            /* β scales a schedule, so it is only live once there is one to
             * scale. `none` returns the exponent 1 whatever β is. */
            var shape = (state && state.prior && state.prior.decay) || "none";
            betaField.disabled = kind === "uniform" || shape === "none";
            var ratio = (state && state.prior && state.prior.betaRatio != null)
                        ? state.prior.betaRatio : defaultBeta;
            if (ratio != null) betaField.value = ratio;
        }
        showStrength(kind);
        if (delayWrap) {
            /* Nothing to wait for once the design is over, and nothing to fade
             * when no prior is stated. */
            delayWrap.hidden = !(state && state.insideDesign) || kind === "uniform";
        }
        if (delayToggle && state && state.prior) {
            delayToggle.checked = state.prior.delayDecay !== false;
        }
        if (evaluateBtn) evaluateBtn.disabled = kind === "uniform";
        offerKinds();
        applyPriorAxis(kind);
        renderParams();
        /* The panel only invites a drag when there is something to drag. */
        if (bound) bound.style.cursor = editable() ? "crosshair" : "default";
    }

    if (sliceToggle) {
        sliceToggle.addEventListener("change", function () { scheduleRedraw(); });
    }

    if (betaField) {
        betaField.addEventListener("input", function () {
            var v = parseFloat(betaField.value);
            if (!state || !state.prior || !isFinite(v)) return;   /* mid-typing */
            state.prior.betaRatio = Math.min(Math.max(v, parseFloat(betaField.min)),
                                             parseFloat(betaField.max));
            /* Not redrawn here. β only enters through the exponent, which the
             * server computes from the schedule — so the curve moves on the
             * next fetch rather than on a second implementation of the decay
             * living in this file. */
            savePrior();
        });
    }

    if (rewalkBtn) {
        rewalkLabel = rewalkBtn.textContent;
        /* Straight past the debounce. The reader has just said "now", and the
         * saved prior has to be on the server before the optimizer is built
         * from it — a pending timer would have this walk run against the
         * previous prior and quietly return the wrong answer. */
        rewalkBtn.addEventListener("click", requery);
    }

    /* Save whatever is stated, then ask the optimizer what it would run next.
     * Straight past the debounce: the reader has just said "now", and a pending
     * timer would have the walk run against the previous prior and quietly
     * return the wrong answer. */
    function requery() {
        /* Checked here rather than left to `postPrior`, which returns without a
         * round trip when there is no slice to ask about — and would leave the
         * button disabled with nothing coming back to re-enable it. */
        if (!state || !state.hp) return;
        if (saveTimer) { clearTimeout(saveTimer); saveTimer = 0; }
        setBusy(true);
        postPrior(true);
    }

    if (kindSelect) {
        kindSelect.addEventListener("change", function () {
            if (!state) return;
            var kind = kindSelect.value;
            state.prior = {
                kind: kind,
                params: defaultsFor(kind),
                exponent: 1,
                /* Carried across a change of shape: how a prior fades is a
                 * separate statement from what shape it has. */
                decay: (state.prior && state.prior.decay) || "none",
                betaRatio: (state.prior && state.prior.betaRatio) || defaultBeta
            };
            syncControls();
            scheduleRedraw();
            savePrior();
        });
    }
    if (decaySelect) {
        /* Recorded, not drawn. Decay is an exponent over the trial count, so it
         * says nothing about a run that has already finished — it is what would
         * be handed to `add_prior` when a prior steers a live one. No redraw,
         * because there is honestly nothing to redraw. */
        decaySelect.addEventListener("change", function () {
            if (state && state.prior) state.prior.decay = decaySelect.value;
            /* The exponent changes shape, so the drawn weight does too — the
             * server sends back the new one on the next fetch, and until then
             * the field at least stops lying about whether β applies. */
            syncControls();
            savePrior();
        });
    }

    function show(data, hp) {
        if (!data.figures || !data.meta) {
            state = null; clear(true); setWarning(data.warning);
            setInitialDesign(data.initialDesign);
            return;
        }
        /* A prior is a statement about one hyperparameter's axis. It survives
         * a metric switch, which redraws this figure from cache underneath the
         * reader, and is dropped on a move to a different hyperparameter, where
         * the same centre and width would mean something else entirely. */
        var kept = state && state.hp === hp ? state.prior : null;
        /* Nothing held over from this session, but something was stated in an
         * earlier one: put the controls back where they were left. Only the
         * shape and its parameters are restored — the knots are recomputed from
         * them, so there is one source for what the curve is. */
        if (!kept && data.prior && data.prior.kind) {
            kept = {kind: data.prior.kind, params: data.prior.params || {},
                    exponent: data.prior.exponent || 1,
                    decay: (data.prior.decay || {}).shape || "none",
                    /* Absent means the belief predates the choice, and the
                     * choice is on by default — but a stored `false` is an
                     * answer and has to survive the round trip. */
                    delayDecay: data.prior.delay_decay !== false,
                    betaRatio: (data.prior.decay || {}).beta_ratio};
        }
        /* β opens on DynaBO's initialisation — the trial budget over ten — for
         * a prior that has not stated one. The server computes it, because it
         * is the only side that knows the budget. */
        if (kept && kept.betaRatio == null) kept.betaRatio = data.betaRatioDefault;
        defaultBeta = data.betaRatioDefault;
        /* Decay is a function of how many trials have run, not of anything the
         * reader did, so the server's exponent wins even over a prior held
         * across a metric switch. Without this the figure would keep showing an
         * exponent from whenever the panel was first opened, and a decaying
         * prior would look like it never faded. */
        if (kept && data.prior && typeof data.prior.exponent === "number") {
            kept.exponent = data.prior.exponent;
        }
        state = {figures: data.figures, meta: data.meta, prior: kept, hp: hp,
                 candidates: null, density: data.density || null,
                 /* Whether there is still a design to wait for, which decides
                  * whether the wait is worth offering. Said by the server
                  * rather than read off `initialDesign`, which is also null for
                  * a run whose optimizer records no design at all. */
                 insideDesign: !!data.insideDesign};
        /* Deliberately not carried over. A candidate's position is a coordinate
         * on one hyperparameter's axis, and what is drawn below is now a
         * different axis or a different metric's surrogate. Re-ask. */
        setCandidates(null);
        setWarning(data.warning);
        setInitialDesign(data.initialDesign);
        syncControls();
        render();
    }

    function refresh(asked, andRequery) {
        if (!hpSelect || !plotEl) return;
        var m = metric(), hp = hpSelect.value, key;
        if (!hp) return;
        key = m + ":" + hp;

        function apply(data) {
            /* Stale by the time it resolved — the reader moved on while it was
             * in flight. */
            if (metric() !== m || hpSelect.value !== hp) return;
            setPrompt(false);
            show(data, hp);
            if (andRequery && rewalkBtn && !rewalkBtn.disabled) requery();
        }

        if (cache[key]) { apply(cache[key]); return; }
        /* Nothing to fit and nothing to wait for. Asking for the full slice
         * would be asking the server for a surrogate over no trials, which it
         * refuses — correctly, and the reader would see the refusal rather
         * than the panel they came for. */
        if (priorOnlyPage) { drawPriorOnly(m, hp); return; }
        if (!autocompute && !asked) {
            setPrompt(true);
            setWarning(null);
            /* The prompt is about the surrogate, which is the expensive half.
             * A prior needs no model, so it is drawn rather than withheld
             * behind a button that is not about it. */
            drawPriorOnly(m, hp);
            return;
        }
        /* Drawn first and replaced when the fit lands, so a prior is editable
         * from the moment the page opens rather than after a model. */
        if (!state || state.hp !== hp) drawPriorOnly(m, hp);
        /* Already asked for and still in the air. Without this a metric switch
         * would fetch the same slice twice: once because the page announced it
         * had redrawn, and once because this script had not yet cached it. */
        if (inflight[key]) return;
        inflight[key] = true;
        setPrompt(false);
        fetch(url + "?metric=" + encodeURIComponent(m) + "&hp=" + encodeURIComponent(hp))
            .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
            .then(function (data) { cache[key] = data; apply(data); })
            .catch(function () { setWarning(text("slice-failed")); })
            .then(function () { delete inflight[key]; });
    }

    /* The prior on its own: no surrogate, no wait. Its payload carries one
     * figure, and `show` takes it like any other — the acquisition and
     * surrogate panels are simply not in it yet. */
    function drawPriorOnly(m, hp) {
        var key = "prior:" + m + ":" + hp;
        if (inflight[key]) return;
        inflight[key] = true;
        fetch(url + "?prior_only=1&metric=" + encodeURIComponent(m)
              + "&hp=" + encodeURIComponent(hp))
            .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
            .then(function (data) {
                /* Dropped if the reader moved on: this is the cheaper half of
                 * a race it must not win. */
                if (metric() !== m || hpSelect.value !== hp) return;
                /* And dropped if the full slice for *this* hyperparameter has
                 * already landed, which is the race. Not if one landed for any
                 * hyperparameter at all: after pressing Compute once, every
                 * hyperparameter switched to afterwards found a stale
                 * `figures.acquisition` here and drew no prior, so the panel
                 * stayed behind the Compute prompt for the rest of the session
                 * — which is exactly what the prompt is not supposed to gate. */
                if (state && state.hp === hp
                        && state.figures && state.figures.acquisition) return;
                if (data && data.figures) show(data, hp);
            })
            .catch(function () { /* the full fetch reports for both */ })
            .then(function () { delete inflight[key]; });
    }

    if (hpSelect) hpSelect.addEventListener("change", function () { refresh(); });
    if (promptBtn) {
        /* Compute is the reader asking for everything this panel can tell them,
         * and what the optimizer would run next is part of that. Asking twice —
         * once for the figure, once for the candidates — is a distinction only
         * this code knows about. */
        promptBtn.addEventListener("click", function () { refresh(true, true); });
    }

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

    /* New trials landed. Two things this panel shows are functions of the trial
     * count and of nothing else: the surrogate, and the prior's exponent, which
     * decay shrinks toward one as the run proceeds. Both are computed server
     * side, so the cache is what would freeze them — it is keyed on metric and
     * hyperparameter, neither of which a new trial changes.
     *
     * So the cache is emptied rather than the key extended. A stale entry here
     * is not a slightly old picture; it is a prior claiming a strength it no
     * longer has, which is the one thing this figure exists to report
     * honestly.
     *
     * The candidates go with it, and are not re-asked. They were what the
     * optimizer would do given the trials it had, and it no longer has those
     * trials. Asking again costs a model fit and a maximization, which is not
     * something to spend on every poll without being asked — the button is
     * right there. */
    /* The merge tolerance is a pixel width, so a resize can turn one marker
     * into three or three into one. Plotly reflows the panels itself; this
     * re-decides what they are allowed to show apart. Debounced, because a
     * drag-resize fires this continuously and each pass re-sorts the slate. */
    var resizeTimer = 0;
    window.addEventListener("resize", function () {
        if (!state || !state.candidates) return;
        if (resizeTimer) clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function () {
            resizeTimer = 0;
            if (state && state.candidates) showCandidates(state.candidates);
        }, 150);
    });

    document.addEventListener("poll:swapped", function () {
        /* Carried into `refresh` as though the reader had just pressed Compute,
         * and only when this panel is actually showing something. Emptying the
         * cache puts an autocompute-off figure back behind its prompt
         * otherwise: the reader asked once, got a picture, and would watch it
         * vanish on the next poll. Having computed it once is the asking. */
        var showing = !!state;
        cache = {};
        if (showing) showCandidates(null);
        refresh(showing);
    });

    refresh();
}());
