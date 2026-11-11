/* `cardBusy(keys, promise)` shows a spinner beside the title of each card
 * named in *keys* (a figure key or a list of them) until *promise* settles,
 * and returns *promise*. Requests overlapping on one card keep the spinner
 * until the last of them settles. Only the slow requests use it: a card whose
 * figure is computed on request, not one redrawn from what the page holds. */
(function () {
    "use strict";
    var label = (document.currentScript && document.currentScript.dataset.label) || "";
    var counts = {};

    function mark(key, on) {
        document.querySelectorAll('[data-figure="' + key + '"]').forEach(function (card) {
            var title = card.querySelector(".card-head .subheader");
            if (!title) return;
            var spin = title.querySelector(":scope > .card-throbber");
            if (on && !spin) {
                spin = document.createElement("span");
                spin.className = "card-throbber";
                spin.setAttribute("role", "status");
                spin.setAttribute("aria-label", label);
                spin.title = label;
                title.appendChild(spin);
            } else if (!on && spin) {
                spin.remove();
            }
        });
    }

    window.cardBusy = function (keys, promise) {
        keys = [].concat(keys).filter(Boolean);
        keys.forEach(function (key) {
            counts[key] = (counts[key] || 0) + 1;
            if (counts[key] === 1) mark(key, true);
        });
        function done() {
            keys.forEach(function (key) {
                counts[key] -= 1;
                if (counts[key] <= 0) { delete counts[key]; mark(key, false); }
            });
        }
        Promise.resolve(promise).then(done, done);
        return promise;
    };
})();
