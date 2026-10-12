// What lets the assistant act on this page (assistant/acts.py).
//
// The page says what it is: its path, its form's fields and values, and the
// elements it names for the assistant with data-assist. It follows the
// actions for this browser and carries out the ones meant for it — going to a
// page, filling fields, pointing something out — then says how each went.
// When several pages of the browser are open, the one used last acts.
//
// Nothing here clicks or submits anything: saving, creating, running and
// deleting stay the person's.
(function () {
    "use strict";
    var script = document.currentScript;
    var cfg = script.dataset;
    var ACTOR = "assistant-actor";
    var me = Math.random().toString(36).slice(2);
    // The whole page, run form and sidebars included; the chat is left out.
    var main = document.body;

    function store(set) {
        try {
            if (set === undefined) return localStorage.getItem(ACTOR);
            if (set === null) localStorage.removeItem(ACTOR);
            else localStorage.setItem(ACTOR, set);
        } catch (e) { return set === undefined ? me : undefined; }
    }
    function claim() { store(me); }
    function acting() {
        var who = store();
        if (!who) { claim(); return true; }
        return who === me;
    }

    function post(url, body, keepalive) {
        return fetch(url, {
            method: "POST", body: body, credentials: "same-origin", keepalive: !!keepalive,
            headers: {"X-CSRFToken": cfg.csrf}
        });
    }

    // ── What the page is ───────────────────────────────────────────────────
    var SKIP = {hidden: 1, file: 1, password: 1, submit: 1, button: 1, reset: 1, image: 1};

    function labelOf(el) {
        if (el.labels && el.labels.length) return el.labels[0].textContent.trim();
        var field = el.closest(".field");
        var label = field && field.querySelector("label");
        return (el.getAttribute("aria-label") || (label && label.textContent) || "").trim();
    }

    function controls() {
        return Array.prototype.filter.call(
            main.querySelectorAll("input[name], select[name], textarea[name]"),
            function (el) { return !SKIP[el.type] && !el.closest("[data-assistant]"); });
    }

    function fields() {
        var seen = {};
        var out = [];
        controls().forEach(function (el) {
            if (el.type === "radio") {
                if (seen[el.name]) return;
                seen[el.name] = 1;
                var group = main.querySelectorAll('input[type=radio][name="' + CSS.escape(el.name) + '"]');
                var chosen = main.querySelector('input[type=radio][name="' + CSS.escape(el.name) + '"]:checked');
                out.push({name: el.name, label: labelOf(el), type: "radio", fillable: true,
                          value: chosen ? chosen.value : "", visible: el.offsetParent !== null,
                          options: Array.prototype.map.call(group, function (r) { return [r.value, labelOf(r)]; })});
                return;
            }
            var item = {name: el.name, label: labelOf(el), type: el.type || el.tagName.toLowerCase(),
                        fillable: true, visible: el.offsetParent !== null};
            if (el.type === "checkbox") item.value = el.checked;
            else item.value = el.value;
            if (el.tagName === "SELECT") {
                item.options = Array.prototype.map.call(el.options, function (o) {
                    return [o.value, o.textContent.trim()];
                });
            }
            out.push(item);
        });
        return out;
    }

    function targets() {
        return Array.prototype.map.call(document.querySelectorAll("[data-assist]"), function (el) {
            var text = el.getAttribute("aria-label") || el.textContent || "";
            return {name: el.dataset.assist, label: text.replace(/\s+/g, " ").trim().slice(0, 80)};
        });
    }

    var reportTimer = null;
    function report() {
        clearTimeout(reportTimer);
        reportTimer = setTimeout(function () {
            post(cfg.pageUrl, JSON.stringify({
                path: location.pathname, title: document.title, fields: fields(), targets: targets()
            })).catch(function () {});
        }, 150);
    }

    // ── Doing things ───────────────────────────────────────────────────────
    function wrapperOf(el) { return el.closest(".field") || el; }

    function setOne(name, value, reason) {
        var els = main.querySelectorAll('[name="' + CSS.escape(name) + '"]');
        els = Array.prototype.filter.call(els, function (el) {
            return !SKIP[el.type] && !el.closest("[data-assistant]");
        });
        if (!els.length) return false;
        var el = els[0];
        if (el.type === "radio") {
            var match = els.filter(function (r) { return r.value === String(value); })[0];
            if (!match) return false;
            match.checked = true;
            el = match;
        } else if (el.type === "checkbox") {
            el.checked = value === true || value === "true" || value === "on" || value === 1;
        } else if (el.tagName === "SELECT") {
            var has = Array.prototype.some.call(el.options, function (o) { return o.value === String(value); });
            if (!has) return false;
            el.value = String(value);
        } else {
            el.value = value === null || value === undefined ? "" : String(value);
        }
        el.dispatchEvent(new Event("input", {bubbles: true}));
        el.dispatchEvent(new Event("change", {bubbles: true}));
        var box = wrapperOf(el);
        box.classList.add("assistant-filled");
        box.title = reason || cfg.textFilled;
        return true;
    }

    // Fields in order, each tried again a moment later when it is not there
    // yet: choosing one field can bring the next onto the page.
    function fill(items, done) {
        var missed = [];
        var i = 0;
        function next(tries) {
            if (i >= items.length) return done(missed);
            var item = items[i];
            if (setOne(item.name, item.value, item.reason)) {
                i += 1;
                setTimeout(function () { next(0); }, 60);
            } else if (tries < 4) {
                setTimeout(function () { next(tries + 1); }, 250);
            } else {
                missed.push(item.name);
                i += 1;
                next(0);
            }
        }
        next(0);
    }

    function highlight(name) {
        var el = document.querySelector('[data-assist="' + CSS.escape(name) + '"]');
        if (!el) {
            var field = main.querySelector('[name="' + CSS.escape(name) + '"]');
            el = field && wrapperOf(field);
        }
        if (!el) return false;
        el.scrollIntoView({behavior: "smooth", block: "center"});
        el.classList.remove("assistant-highlight");
        void el.offsetWidth;
        el.classList.add("assistant-highlight");
        setTimeout(function () { el.classList.remove("assistant-highlight"); }, 4000);
        return true;
    }

    function acted(id, ok, detail, keepalive) {
        var body = new FormData();
        body.append("act", id);
        body.append("ok", ok ? "1" : "0");
        body.append("detail", detail || "");
        return post(cfg.actedUrl, body, keepalive).catch(function () {});
    }

    var done = {};
    function carryOut(act) {
        if (done[act.id] || !acting()) return;
        done[act.id] = 1;
        if (act.path && act.path !== location.pathname) {
            acted(act.id, false, "The person is on " + location.pathname + " now, not " + act.path + ".");
            return;
        }
        var args = act.args || {};
        if (act.kind === "navigate") {
            acted(act.id, true, "Opened " + args.url + ".", true).then(function () {
                location.assign(args.url);
            });
        } else if (act.kind === "fill") {
            var values = args.values || {};
            var items = Object.keys(values).map(function (name) {
                return {name: name, value: values[name], reason: args.reason};
            });
            fill(items, function (missed) {
                report();
                if (missed.length) acted(act.id, false, "Not on the page: " + missed.join(", ") + ".");
                else acted(act.id, true, "Filled " + Object.keys(values).join(", ") + ".");
            });
        } else if (act.kind === "highlight") {
            if (highlight(args.target)) acted(act.id, true, "");
            else acted(act.id, false, "Nothing called " + args.target + " is on the page.");
        } else {
            acted(act.id, false, "Not something this page does.");
        }
    }

    // ── Starting ───────────────────────────────────────────────────────────
    claim();
    window.addEventListener("focus", function () { claim(); report(); });
    document.addEventListener("visibilitychange", function () {
        if (document.visibilityState === "visible") { claim(); report(); }
    });
    window.addEventListener("pagehide", function () { if (store() === me) store(null); });
    main.addEventListener("change", function (e) {
        if (e.isTrusted) report();
    });

    var planned = document.getElementById("assistant-fill");
    var items = planned ? JSON.parse(planned.textContent || "[]") : [];
    if (items && items.length) fill(items, function () { report(); });
    else report();

    var source = new EventSource(cfg.actsUrl);
    source.addEventListener("act", function (e) { carryOut(JSON.parse(e.data)); });
})();
