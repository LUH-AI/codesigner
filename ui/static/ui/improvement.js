/* The acquisition functions, the one implementation the page has: expected
 * improvement and probability of improvement, as SMAC computes them.
 *
 * Its own file because two figures need it and neither should need the other:
 * the acquisition figure draws it along one hyperparameter, and the
 * configuration cube draws it across the whole space it is showing. Loaded
 * before both, as a classic script, so it is there whichever arrives first. */
(function () {
    "use strict";

    if (window.codesignerImprovement) return;

    var SQ2PI = Math.sqrt(2 * Math.PI);

    /* Abramowitz & Stegun 7.1.26. Accurate to ~1.5e-7, which is several orders
     * better than anything here needs: the acquisition function is only ever
     * ranked, and it is drawn a few hundred pixels wide. */
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
    function expectedImprovement(mu, sigma, eta, higherIsBetter, xi) {
        var gain = (higherIsBetter ? mu - eta : eta - mu) - (xi || 0), z;
        if (!(sigma > 1e-12)) return Math.max(gain, 0);
        z = gain / sigma;
        return gain * cdf(z) + sigma * pdf(z);
    }

    /* The chance this point beats the incumbent by at least `xi`. */
    function probabilityOfImprovement(mu, sigma, eta, higherIsBetter, xi) {
        var gain = (higherIsBetter ? mu - eta : eta - mu) - (xi || 0);
        if (!(sigma > 1e-12)) return gain > 0 ? 1 : 0;
        return cdf(gain / sigma);
    }

    var KINDS = {ei: expectedImprovement, pi: probabilityOfImprovement};

    /* The run's own acquisition function, as the server names it
     * ({kind, xi}), with the margin bound in: called as
     * `(mu, sigma, eta, higherIsBetter)`. Expected improvement when the run
     * names none. */
    function acquisition(spec) {
        var f = KINDS[spec && spec.kind] || expectedImprovement,
            xi = (spec && spec.xi) || 0;
        return function (mu, sigma, eta, higherIsBetter) {
            return f(mu, sigma, eta, higherIsBetter, xi);
        };
    }

    /* Its short name, for a scale's title. */
    function abbreviation(spec) {
        return KINDS[spec && spec.kind] ? spec.kind.toUpperCase() : "EI";
    }

    window.codesignerImprovement = {expectedImprovement: expectedImprovement,
                                    probabilityOfImprovement: probabilityOfImprovement,
                                    acquisition: acquisition,
                                    abbreviation: abbreviation};
})();
