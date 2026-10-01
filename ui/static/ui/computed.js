/* What the figures behind a Compute button have computed, kept until the
 * results they were computed from change.
 *
 * Partial dependence, local explanation, local effects, the projection's
 * surrogate uncertainty and the acquisition function each fit a model on
 * request and draw what comes back. That answer depends only on the
 * experiment's results and settings, so it is kept here and drawn again
 * whenever it is asked for — on another hyperparameter and back, another tab
 * and back, another page and back — without fitting anything twice.
 *
 * Kept in sessionStorage, so it outlives the page but not the browser tab, and
 * labelled with the results it came from (`#computed-version`, which the server
 * builds from the trials, the last run and the settings). Anything labelled
 * with other results is not this experiment's any more and is dropped. So is
 * everything when the page is reloaded (F5): that is the reader asking for it
 * all again. And during a run, when new trials arrive, it is dropped as they
 * do — see `forget` — and nothing is fetched on its own until the run ends,
 * since every poll would otherwise refit every figure.
 *
 * Loaded before the page script and acquisition.js, which both read it as
 * `window.codesignerComputed`.
 */
(function () {
    "use strict";

    var versionEl = document.getElementById("computed-version");
    var info = versionEl ? JSON.parse(versionEl.textContent) : null;
    var storeKey = info ? "codesigner:computed:" + info.experiment : null;
    var memory = {};
    var running = false;

    function reloaded() {
        try {
            var nav = performance.getEntriesByType("navigation")[0];
            return !!nav && nav.type === "reload";
        } catch (e) {
            return false;
        }
    }

    function read() {
        if (!storeKey) return;
        try {
            if (reloaded()) { sessionStorage.removeItem(storeKey); return; }
            var stored = JSON.parse(sessionStorage.getItem(storeKey) || "null");
            if (stored && stored.version === info.version) memory = stored.entries || {};
            else sessionStorage.removeItem(storeKey);
        } catch (e) {
            memory = {};
        }
    }

    /* Everything, every time: the entries are few. If the tab's storage is
     * full, it is given up on rather than half-written — the answers are still
     * kept for as long as this page is open. */
    function write() {
        if (!storeKey) return;
        try {
            sessionStorage.setItem(storeKey, JSON.stringify({version: info.version, entries: memory}));
        } catch (e) {
            try { sessionStorage.removeItem(storeKey); } catch (e2) { /* nothing to do */ }
        }
    }

    function slot(name) {
        return memory[name] || (memory[name] = {});
    }

    var api = {
        get: function (name, key) { return slot(name)[key]; },
        set: function (name, key, data) { slot(name)[key] = data; write(); },
        /* Every answer, from memory and from the tab: the results they came
         * from have moved on. The figures are told, so they can put their
         * Compute buttons back rather than go on showing what is now stale. */
        forget: function () {
            memory = {};
            write();
            document.dispatchEvent(new CustomEvent("computed:forgotten"));
        },
        /* Set while a run is adding trials. A figure then only computes when
         * its button is pressed, whatever its settings say. */
        get running() { return running; },
        set running(value) { running = !!value; },
        /* An object that reads and writes one figure's answers, for code that
         * indexes a cache as `cache[key]`. */
        cache: function (name) {
            return new Proxy({}, {
                get: function (target, key) { return typeof key === "string" ? api.get(name, key) : undefined; },
                set: function (target, key, value) { api.set(name, key, value); return true; },
                has: function (target, key) { return api.get(name, key) !== undefined; },
            });
        },
    };

    read();
    window.codesignerComputed = api;
})();
