// Minimal polling shim (no external dependency). Any element with
//   hx-get="URL" hx-trigger="every Ns"
// is polled every N seconds: the response HTML replaces the element (and the
// replacement is re-wired), unless the response carries `HX-Refresh: true`, in
// which case the whole page reloads. Enough for live run-status updates.
(function () {
    // A hidden tab is nobody looking. Browsers throttle background timers but
    // they do not stop the requests, so without this a forgotten tab kept
    // asking a server for figures nobody could see — for as long as the run
    // lasted. Held rather than skipped: the poll fires the moment the tab comes
    // back, instead of waiting out another interval first.
    //
    // Navigating away needs nothing: the timer lives on the page it was started
    // from, so leaving the experiment ends its polling by taking the page with
    // it.
    function whenVisible(run) {
        if (!document.hidden) { run(); return; }
        document.addEventListener("visibilitychange", function resume() {
            if (document.hidden) return;
            document.removeEventListener("visibilitychange", resume);
            run();
        });
    }

    function wire(el) {
        const url = el.getAttribute("hx-get");
        const match = (el.getAttribute("hx-trigger") || "").match(/every\s+(\d+)s/);
        if (!url || !match) return;
        const ms = parseInt(match[1], 10) * 1000;
        setTimeout(function () { whenVisible(function () { poll(el, url); }); }, ms);
    }

    function poll(el, url) {
        fetch(url, {headers: {"HX-Request": "true"}})
            .then(function (r) {
                if (r.headers.get("HX-Refresh") === "true") {
                    window.location.reload();
                    return null;
                }
                return r.text();
            })
            .then(function (html) {
                if (html === null) return;
                const tmp = document.createElement("div");
                tmp.innerHTML = html.trim();
                const fresh = tmp.firstElementChild;
                if (!fresh) { wire(el); return; }
                el.replaceWith(fresh);
                wire(fresh);
                // The page has no other way to notice a swap happened, and a
                // swapped-in fragment can carry more than the status it
                // replaced — the live figure payloads and the trials table's
                // new rows ride inside the run-status poll rather than in a
                // second loop of their own. Dispatched on the document so a
                // listener does not have to find the element first.
                document.dispatchEvent(new CustomEvent("poll:swapped", {
                    detail: {element: fresh},
                }));
            })
            .catch(function () { wire(el); });  // retry next cycle
    }

    document.querySelectorAll("[hx-get][hx-trigger]").forEach(wire);
})();
