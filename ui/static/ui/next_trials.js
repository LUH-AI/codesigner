/* What the optimizer would run next: one question, asked from either figure
 * that draws the answer.
 *
 * The acquisition figure and the configuration cube each carry a "Next Trials"
 * checkbox with a Requery button beneath it, and the two pairs are one pair of
 * controls. Ticking either shows the proposals in both, asking the optimizer
 * first if it has not been asked since the trials last moved; Requery asks
 * again. While an answer is being worked out both buttons are disabled, and
 * while a run is adding trials both pairs are, since every new trial would
 * make the answer stale.
 *
 * The answer is announced as `candidates:updated` with `candidates` null when
 * the proposals are hidden or there are none; each figure places them on its
 * own axes. A figure may set `before` to a function returning a promise, which
 * is waited on before asking — the acquisition figure uses it to land a prior
 * still being saved, so the optimizer is built from what the reader stated —
 * and `anchor` to a function naming the hyperparameter the trial budget is
 * anchored at.
 *
 * Loaded after computed.js, whose `running` it reads.
 */
(function () {
    "use strict";

    if (window.codesignerNextTrials) return;

    var computed = window.codesignerComputed || null,
        candidates = null,
        queried = false,
        busy = false,
        shown = false;

    function boxes() { return Array.prototype.slice.call(document.querySelectorAll("[data-next-trials]")); }
    function buttons() { return Array.prototype.slice.call(document.querySelectorAll("[data-requery]")); }
    function locked() { return !!(computed && computed.running); }

    function sync() {
        var off = locked();
        boxes().forEach(function (box) { box.checked = shown; box.disabled = off; });
        buttons().forEach(function (button) { button.disabled = off || busy || !shown; });
    }

    function announce(unavailable) {
        document.dispatchEvent(new CustomEvent("candidates:updated", {detail: {
            candidates: shown ? candidates : null,
            unavailable: !!unavailable
        }}));
    }

    function requery() {
        var box = boxes()[0], url = box && box.getAttribute("data-url");
        if (busy || locked() || !url) return;
        busy = true;
        sync();
        var anchor = api.anchor ? api.anchor() : null,
            ready = api.before ? api.before() : null;
        Promise.resolve(ready)
            .then(function () {
                return fetch(url + (anchor ? "?hp=" + encodeURIComponent(anchor) : ""));
            })
            .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
            .then(function (data) {
                candidates = data.candidates || null;
                queried = true;
                busy = false;
                sync();
                announce(!data.candidates);
            })
            .catch(function () {
                busy = false;
                sync();
            });
    }

    document.addEventListener("change", function (event) {
        if (!event.target.matches || !event.target.matches("[data-next-trials]")) return;
        shown = event.target.checked;
        sync();
        if (shown && !queried) requery();
        else announce();
    });
    document.addEventListener("click", function (event) {
        var button = event.target.closest && event.target.closest("[data-requery]");
        if (button && !button.disabled) requery();
    });
    /* New trials: the proposals were for the trials there were, and are
     * dropped. Not asked again on their own — a run adds trials every few
     * seconds — and unticked, so a tick is what asks. */
    document.addEventListener("computed:forgotten", function () {
        candidates = null;
        queried = false;
        shown = false;
        sync();
        announce();
    });

    var api = window.codesignerNextTrials = {
        before: null,
        anchor: null,
        /* What a figure drawn after the announcement should show. */
        get candidates() { return shown ? candidates : null; },
        sync: sync
    };

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", sync);
    else sync();
}());
