// Minimal polling shim (no external dependency). Any element with
//   hx-get="URL" hx-trigger="every Ns"
// is polled every N seconds: the response HTML replaces the element (and the
// replacement is re-wired), unless the response carries `HX-Refresh: true`, in
// which case the whole page reloads. Enough for live run-status updates.
(function () {
    function wire(el) {
        const url = el.getAttribute("hx-get");
        const match = (el.getAttribute("hx-trigger") || "").match(/every\s+(\d+)s/);
        if (!url || !match) return;
        const ms = parseInt(match[1], 10) * 1000;
        setTimeout(function () {
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
                    if (fresh) { el.replaceWith(fresh); wire(fresh); }
                    else { wire(el); }
                })
                .catch(function () { wire(el); });  // retry next cycle
        }, ms);
    }
    document.querySelectorAll("[hx-get][hx-trigger]").forEach(wire);
})();
