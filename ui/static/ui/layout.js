/* The experiment page's tabs, and the size of every figure on them.
 *
 * The server renders each figure once, into its slot on its home tab, and an
 * empty slot on Overview for each one pinned there (see ui/layout.py). This
 * script does three things with that and nothing else:
 *
 *  - shows one tab at a time, and moves each figure into the slot on the tab
 *    being shown. A figure is one element with one id wherever it appears, so
 *    the page script that draws into `#figure-<key>` never has to know which
 *    tab it is on;
 *  - sizes each slot on the grid from its shape and its width in columns;
 *  - redraws a plot whenever its slot changes size, which Plotly's own
 *    `responsive` does not do — it only follows the window;
 *  - gives every slot its controls — a grip to drag it by, a pin to put it on
 *    Overview or take it off, narrower and wider — and saves the arrangement
 *    they leave behind, as this reader's for this experiment.
 *
 * Loaded after the page script, so every figure has been drawn once already.
 */
(function () {
    "use strict";

    var tabs = document.querySelectorAll(".fig-tab");
    var panels = document.querySelectorAll(".fig-panel");
    if (!tabs.length) return;

    /* ── sizes, and where each figure goes ─────────────────────────────── */

    /* The stylesheet gives every slot a size from its width and shape, so the
     * page script's first draw lands at the right size. This then places them:
     * each figure goes as high on its tab as it can, in the order the tab
     * lists them, straight under whatever is already in the columns it needs —
     * not under the bottom of the tallest thing beside it, which is what the
     * grid's own rows would do. A figure two columns wide sits on columns 1–2
     * or 3–4, never 2–3.
     *
     * Heights, from a column's width:
     *   squat   3:2 — two thirds as tall as it is wide
     *   long    3:1
     *   table   as many one-column-squat heights as its rows need, whole ones
     *   stack, or any figure with nothing to draw yet: as tall as it is */

    var RATIO = {squat: 2 / 3, long: 1 / 3};

    function shapeOf(slot) {
        var m = /\bshape-(\w+)/.exec(slot.className);
        return m ? m[1] : "squat";
    }

    /* A figure is one card, or one per metric with only the viewed metric's
     * showing (the best configuration). Every one of them moves with the
     * figure; the one showing is what the slot is sized to. */
    function cardsIn(slot) {
        return slot.querySelectorAll(":scope > [data-figure]");
    }

    function cardIn(slot) {
        var cards = cardsIn(slot), i;
        for (i = 0; i < cards.length; i++) if (!cards[i].hidden) return cards[i];
        return cards[0] || null;
    }

    /* Nothing to draw yet: the figure's own plot is empty, and it is saying so
     * — its "no data" note or its Compute button is showing. Both conditions,
     * because a figure that has drawn something can still offer to compute
     * more (the projection's surrogate uncertainty), and one that has drawn
     * nothing may simply be showing a table instead. */
    function isEmpty(slot) {
        var key = slot.dataset.key, card = cardIn(slot), plot, note, compute;
        if (!card) return false;
        plot = document.getElementById("figure-" + key);
        if (!plot || plot.querySelector(".main-svg")) return false;
        note = document.getElementById("figure-" + key + "-empty");
        if (note && !note.hidden) return true;
        compute = card.querySelector('button[id$="compute-btn"]');
        return !!(compute && !compute.closest("[hidden]"));
    }

    function size(slot) {
        var empty = isEmpty(slot);
        slot.classList.toggle("is-empty", empty);
        slot.classList.toggle("is-content", empty || shapeOf(slot) === "stack");
    }

    /* How many columns, and how wide each is, for *grid* as wide as it is now.
     *
     * A column's width is a length in the same unit as the text — the
     * stylesheet's --col-min / --col-max on `.fig-grid`, in rem — not a share
     * of the window. A figure's text is a fixed size, so a figure sized as a
     * share of the window is the same text in a smaller box on a smaller
     * screen, until its legend sits on its title; and browser zoom, which
     * grows the text, would leave the box where it was. Sized in rem, the two
     * keep their proportion on any screen and at any zoom, and what changes is
     * how many fit in a row.
     *
     * So: the most columns of 4, 2 and 1 that are each at least --col-min
     * wide, filling the row up to --col-max each. Never 3: a stored
     * arrangement is in quarters and halves of the row (see ui/layout.py),
     * and folds in half onto two (see `arrange`) but not onto three. Whatever
     * the columns leave of the row stays empty, to the right. */
    function geometryOf(grid) {
        var style = getComputedStyle(grid),
            gap = parseFloat(style.columnGap) || 0,
            rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16,
            least = remToPx(style.getPropertyValue("--col-min"), rem, 0),
            most = remToPx(style.getPropertyValue("--col-max"), rem, Infinity),
            width = grid.clientWidth, cols, fill;
        for (cols = 4; cols > 1; cols /= 2) {
            fill = (width - (cols - 1) * gap) / cols;
            if (fill >= least) break;
        }
        fill = (width - (cols - 1) * gap) / cols;
        return {cols: cols, gap: gap, colW: Math.max(0, Math.min(fill, most))};
    }

    function remToPx(value, rem, fallback) {
        var n = parseFloat(value);
        if (!(n > 0)) return fallback;
        return /rem\s*$/.test(value) ? n * rem : n;
    }

    function columnsOf(grid) {
        return geometryOf(grid).cols;
    }

    /* A card's own height at *width*, whatever its slot is now. */
    function natural(slot, width) {
        var card = cardIn(slot), h;
        if (!card) return 0;
        slot.style.width = width + "px";
        slot.style.height = "auto";
        card.style.flex = "none";
        card.classList.add("is-measured");
        h = card.offsetHeight;
        card.classList.remove("is-measured");
        card.style.flex = "";
        return h;
    }

    /* Where each slot goes, and how big it is: one pass over a tab. */
    var placed = new WeakMap();
    /* Each grid's columns as last arranged, for finding the one under the
     * pointer. */
    var laidOut = new WeakMap();

    /* A figure's column, if it has one that fits it: the first column of the
     * pair it sits in when it is two wide, the first of the row when four. */
    function fits(c, w) {
        return c >= 0 && c + w <= 4 && c % w === 0;
    }

    /* The highest place a figure *w* columns wide could go on *tops*, the
     * leftmost of equals. */
    function lowest(tops, w) {
        var best = 0, top = Infinity, start, t, i;
        for (start = 0; start + w <= tops.length; start += w) {
            for (t = 0, i = start; i < start + w; i++) t = Math.max(t, tops[i]);
            if (t < top - 0.5) { top = t; best = start; }
        }
        return best;
    }

    /* On the four-column grid every figure has a column of its own and stays in
     * it: a figure above growing or shrinking moves it up or down, never across.
     * One with no column yet — a tab nobody has rearranged — is given the
     * highest place it would have at its shape's own size, so that whether a
     * figure above has been computed yet does not decide where this one goes.
     * The column is kept on the slot from then on, and saved with the
     * arrangement the first time the reader changes anything on the tab.
     *
     * Four columns are two halves of two side by side, and two columns show
     * the left half and then the right: each half keeps its own arrangement
     * exactly — a figure in the first or third of four goes in the first of
     * two, one in the second or fourth in the second, one two wide spans both
     * — so what sits side by side on a larger screen still does on a smaller.
     * A figure with no column yet, and every figure on one column, takes the
     * highest place there is. Nothing here changes a stored column. */
    function arrange(grid) {
        var geometry = geometryOf(grid), cols = geometry.cols,
            gap = geometry.gap, colW = geometry.colW,
            unit = colW * RATIO.squat, four = cols === 4,
            tops = [], nominal = [], c, bottom = 0;
        if (!grid.clientWidth) return;
        for (c = 0; c < cols; c++) { tops.push(0); nominal.push(0); }
        grid.classList.add("is-arranged");
        grid.dataset.cols = String(cols);
        grid.style.setProperty("--cols", String(cols));
        laidOut.set(grid, geometry);

        inOrder(grid, cols).forEach(function (slot) {
            var shape = shapeOf(slot), w = Math.min(parseInt(slot.dataset.w, 10) || 1, cols),
                width = w * colW + (w - 1) * gap, ratio = RATIO[shape] || RATIO.squat,
                height, normal, start, top = 0, ntop = 0, k, i;
            size(slot);
            if (cols === 1 && shape === "long") ratio = RATIO.squat;
            if (shape === "table") {
                k = Math.max(1, Math.ceil((natural(slot, width) + gap) / (unit + gap)));
                height = normal = k * unit + (k - 1) * gap;
            } else if (slot.classList.contains("is-content")) {
                height = natural(slot, width);
                normal = width * (shape === "stack" ? RATIO.long : ratio);
            } else {
                height = normal = width * ratio;
            }
            start = parseInt(slot.dataset.c, 10);
            if (cols === 2 && fits(start, parseInt(slot.dataset.w, 10) || 1)) {
                start = w === 2 ? 0 : start % 2;
            } else if (!four) {
                start = lowest(tops, w);
            } else if (!fits(start, w)) {
                start = lowest(nominal, w);
                slot.dataset.c = String(start);
            }
            for (i = start; i < start + w; i++) {
                top = Math.max(top, tops[i]);
                ntop = Math.max(ntop, nominal[i]);
            }
            for (i = start; i < start + w; i++) {
                tops[i] = top + height + gap;
                nominal[i] = ntop + normal + gap;
            }
            bottom = Math.max(bottom, top + height);
            placed.set(slot, {start: start, top: top, w: w, height: height});
            slot.style.left = (start * (colW + gap)) + "px";
            slot.style.top = top + "px";
            slot.style.width = width + "px";
            slot.style.height = height + "px";
        });
        grid.style.height = grid.querySelector(":scope > .fig-slot") ? bottom + "px" : "";
    }

    /* *grid*'s slots in the order they are placed: as listed, except that on
     * two columns the left half of the four comes before the right. */
    function inOrder(grid, cols) {
        var slots = Array.prototype.slice.call(grid.querySelectorAll(":scope > .fig-slot"));
        if (cols !== 2) return slots;
        function half(slot) {
            var c = parseInt(slot.dataset.c, 10);
            return fits(c, parseInt(slot.dataset.w, 10) || 1) && c >= 2 ? 1 : 0;
        }
        return slots.map(function (slot, i) { return {slot: slot, i: i, half: half(slot)}; })
            .sort(function (a, b) { return a.half - b.half || a.i - b.i; })
            .map(function (entry) { return entry.slot; });
    }

    /* Which column of *grid* is under *x*, for a figure *w* wide: the first
     * column of the pair under it when it is two wide, the first when four. */
    function columnAt(grid, x, w) {
        var rect = grid.getBoundingClientRect(), geometry = laidOut.get(grid) || geometryOf(grid),
            gap = geometry.gap, colW = geometry.colW,
            raw = Math.floor((x - rect.left + gap / 2) / (colW + gap));
        raw = Math.min(Math.max(raw, 0), 3);
        return Math.min(Math.floor(raw / w) * w, 4 - w);
    }

    function sizeAll(root) {
        (root || document).querySelectorAll(".fig-grid").forEach(arrange);
    }

    /* ── redrawing plots at their new size ───────────────────────────────── */

    /* Batched to one pass per frame: a drag or a resize changes several slots
     * at once, and each would otherwise ask Plotly for its own reflow. */
    var dirty = new Set(), dirtyGrids = new Set(), frame = 0;

    function flush() {
        frame = 0;
        dirtyGrids.forEach(arrange);
        dirtyGrids.clear();
        dirty.forEach(function (slot) {
            if (!slot.offsetWidth) return;     // emptied: its figure is elsewhere
            slot.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
                if (gd.offsetWidth && window.Plotly) window.Plotly.Plots.resize(gd);
            });
        });
        dirty.clear();
    }

    function schedule() {
        if (!frame) frame = requestAnimationFrame(flush);
    }

    function touch(slot) {
        dirty.add(slot);
        schedule();
    }

    function rearrange(grid) {
        dirtyGrids.add(grid);
        grid.querySelectorAll(":scope > .fig-slot").forEach(function (slot) { dirty.add(slot); });
        schedule();
    }

    /* A slot resized redraws its plots. A card resized — a figure sized by
     * its content that grew, a table given another page of rows — re-places
     * its tab, since everything under it moves. */
    var resized = new ResizeObserver(function (entries) {
        entries.forEach(function (entry) {
            var slot = entry.target.closest(".fig-slot");
            if (!slot) return;
            if (entry.target === slot) touch(slot);
            else if (slot.classList.contains("is-content") || shapeOf(slot) === "table") {
                rearrange(slot.parentNode);
            }
        });
    });

    /* A figure fills, empties, or changes what it shows without its slot
     * changing size — a Compute button replaced by a plot, a note revealed. */
    var changed = new MutationObserver(function (records) {
        var grids = new Set();
        records.forEach(function (r) {
            var slot = r.target.closest && r.target.closest(".fig-slot");
            if (slot && slot.parentNode) grids.add(slot.parentNode);
        });
        grids.forEach(rearrange);
    });

    function watch(slot) {
        resized.observe(slot);
        cardsIn(slot).forEach(function (card) {
            resized.observe(card);
            changed.observe(card, {attributes: true, attributeFilter: ["hidden"],
                                   childList: true, subtree: true});
        });
    }

    /* The grid's width decides its columns and every height on it. */
    var gridResized = new ResizeObserver(function (entries) {
        entries.forEach(function (entry) {
            rearrange(entry.target);
            entry.target.querySelectorAll(".fig-slot").forEach(function (slot) {
                if (slot.querySelector(":scope > .fig-grip")) refresh(slot);
            });
        });
    });

    /* ── tabs ────────────────────────────────────────────────────────────── */

    function cardsFor(key) {
        return document.querySelectorAll('.fig-slot > [data-figure="' + key + '"]');
    }

    /* Each figure into the slot on *panel*, wherever it was before. */
    function gather(panel) {
        panel.querySelectorAll(".fig-slot").forEach(function (slot) {
            var moved = false;
            cardsFor(slot.dataset.key).forEach(function (card) {
                if (card.parentNode !== slot) {
                    slot.appendChild(card);
                    moved = true;
                }
            });
            if (moved) watch(slot);
        });
    }

    function show(name) {
        var found = false;
        panels.forEach(function (panel) {
            if (panel.dataset.tab === name) found = true;
        });
        if (!found) name = tabs[0].dataset.tab;
        tabs.forEach(function (tab) {
            var on = tab.dataset.tab === name;
            tab.setAttribute("aria-selected", on ? "true" : "false");
            tab.tabIndex = on ? 0 : -1;
        });
        panels.forEach(function (panel) {
            var on = panel.dataset.tab === name;
            panel.classList.toggle("is-inactive", !on);
            panel.inert = !on;
            if (on) gather(panel);
        });
        document.querySelectorAll(".fig-grid").forEach(rearrange);
        return name;
    }

    function fromHash() {
        return (window.location.hash || "").replace(/^#/, "");
    }

    tabs.forEach(function (tab) {
        tab.addEventListener("click", function () {
            var name = show(tab.dataset.tab);
            /* In the address bar, so a reload or a shared link opens the same
             * tab — replaced rather than pushed, so Back still leaves the page. */
            if (window.history && window.history.replaceState) {
                window.history.replaceState(null, "", "#" + name);
            }
        });
        /* Arrow keys move between tabs, as a tablist's do. */
        tab.addEventListener("keydown", function (ev) {
            var list = Array.prototype.slice.call(tabs), i = list.indexOf(tab);
            if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
                var next = list[(i + (ev.key === "ArrowRight" ? 1 : list.length - 1)) % list.length];
                next.focus();
                next.click();
                ev.preventDefault();
            }
        });
    });

    /* ── the arrangement, and keeping it ─────────────────────────────────── */

    var area = document.querySelector(".fig-area");
    var layoutEl = document.getElementById("figure-layout-data");
    var layout = layoutEl ? JSON.parse(layoutEl.textContent) : null;
    var OVERVIEW = "overview";

    function say(key) { return (area && area.dataset[key]) || ""; }

    var saveTimer = 0;

    /* Shortly after the last change rather than on each one: a drag or a run
     * of clicks on Wider is one decision, and one request. */
    /* Every slot's column, written into the arrangement: once the reader has
     * changed anything on a tab, the columns the figures were in stay theirs. */
    function keepColumns() {
        document.querySelectorAll(".fig-grid > .fig-slot").forEach(function (slot) {
            var item = entry(tabOf(slot), slot.dataset.key), c = parseInt(slot.dataset.c, 10);
            if (item && !isNaN(c)) item.c = c;
        });
    }

    function save() {
        if (!layout || !area || !area.dataset.saveUrl) return;
        keepColumns();
        if (saveTimer) clearTimeout(saveTimer);
        saveTimer = setTimeout(function () {
            saveTimer = 0;
            post({layout: layout});
        }, 400);
    }

    function post(body) {
        return fetch(area.dataset.saveUrl, {
            method: "POST",
            headers: {"Content-Type": "application/json", "X-CSRFToken": area.dataset.csrf},
            body: JSON.stringify(body)
        }).then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); });
    }

    function entries(tab) { return layout.tabs[tab] || (layout.tabs[tab] = []); }

    function entry(tab, key) {
        var list = entries(tab), i;
        for (i = 0; i < list.length; i++) if (list[i].key === key) return list[i];
        return null;
    }

    function tabOf(slot) { return slot.closest(".fig-panel").dataset.tab; }

    function gridOf(tab) {
        return document.querySelector('.fig-panel[data-tab="' + tab + '"] .fig-grid');
    }

    function slotOn(tab, key) {
        var grid = gridOf(tab);
        return grid ? grid.querySelector('.fig-slot[data-key="' + key + '"]') : null;
    }

    function isPinned(key) { return !!entry(OVERVIEW, key); }

    /* The order of a tab's slots after a drag, written back into the
     * arrangement. Figures switched off for this experiment have no slot, so
     * they keep the places they had and the ones on screen fill around them. */
    function readOrder(tab) {
        var grid = gridOf(tab), list = entries(tab), shown = {}, order, merged = [], i, j;
        order = Array.prototype.map.call(grid.querySelectorAll(":scope > .fig-slot"),
                                         function (slot) { return slot.dataset.key; });
        order.forEach(function (key) { shown[key] = true; });
        for (i = 0, j = 0; i < list.length; i++) {
            if (shown[list[i].key]) merged.push(entry(tab, order[j++]));
            else merged.push(list[i]);
        }
        layout.tabs[tab] = merged;
    }

    /* ── controls ──────────────────────────────────────────────────────────── */

    function button(cls, label, text, onClick) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = cls;
        b.title = label;
        b.setAttribute("aria-label", label);
        b.textContent = text;
        b.addEventListener("click", function (ev) { ev.stopPropagation(); onClick(ev); });
        return b;
    }

    function widthOf(slot) { return parseInt(slot.dataset.w, 10) || 1; }

    /* The widths a slot can be made on this window: the ones its shape allows
     * (1, 2 or the whole row of 4), no wider than the grid is — so a button is
     * only offered where pressing it changes what is on screen — and where in
     * that list it is now. */
    function steps(slot) {
        var cols = columnsOf(slot.parentNode),
            widths = (slot.dataset.widths || "1").split(" ").map(Number)
                .filter(function (w) { return w <= cols; }),
            shown = Math.min(widthOf(slot), cols), at = 0, i;
        if (!widths.length) widths = [Number((slot.dataset.widths || "1").split(" ")[0])];
        for (i = 0; i < widths.length; i++) if (widths[i] <= shown) at = i;
        return {widths: widths, at: at};
    }

    function refresh(slot) {
        var step = steps(slot), pin = slot.querySelector(":scope > .fig-pin"),
            narrower = slot.querySelector(".fig-narrower"),
            wider = slot.querySelector(".fig-wider"),
            onOverview = tabOf(slot) === OVERVIEW, pinned = onOverview || isPinned(slot.dataset.key);
        if (narrower) narrower.hidden = step.at <= 0;
        if (wider) wider.hidden = step.at >= step.widths.length - 1;
        if (pin) {
            pin.setAttribute("aria-pressed", pinned ? "true" : "false");
            pin.title = pinned ? say("unpin") : say("pin");
            pin.setAttribute("aria-label", pin.title);
        }
    }

    /* Wider grows to the left when there is a column there it can take: a
     * figure in column 2 becomes columns 1–2, one in column 4 becomes 3–4, and
     * one going the whole row starts at column 1. At the height it was: the
     * figure beside it in the column it grows into comes after it in the order
     * instead, and so moves down beneath it, keeping its own column. */
    function growLeft(slot, w) {
        var at = placed.get(slot), c = parseInt(slot.dataset.c, 10), want, sibling, there;
        if (isNaN(c)) return;
        want = Math.floor(c / w) * w;
        slot.dataset.c = String(want);
        if (!at || want >= c) return;
        for (sibling = slot.parentNode.firstElementChild; sibling && sibling !== slot;
             sibling = sibling.nextElementSibling) {
            there = placed.get(sibling);
            if (!there || !sibling.classList.contains("fig-slot")) continue;
            if (there.start >= want && there.start < c
                    && there.top < at.top + at.height && there.top + there.height > at.top) {
                slot.parentNode.insertBefore(slot, sibling);
                readOrder(tabOf(slot));
                return;
            }
        }
    }

    function resize(slot, by) {
        var step = steps(slot), i = Math.min(Math.max(step.at + by, 0), step.widths.length - 1),
            w = step.widths[i], item = entry(tabOf(slot), slot.dataset.key);
        if (i === step.at) return;
        if (by > 0) growLeft(slot, w);
        slot.dataset.w = String(w);
        if (item) item.w = w;
        arrange(slot.parentNode);
        slot.parentNode.querySelectorAll(":scope > .fig-slot").forEach(function (s) {
            refresh(s);
            touch(s);
        });
        save();
    }

    function decorate(slot) {
        if (slot.querySelector(":scope > .fig-grip")) { refresh(slot); return; }
        var sizes = document.createElement("div");
        sizes.className = "fig-size";
        sizes.appendChild(button("fig-narrower", say("narrower"), "−",
                                 function () { resize(slot, -1); }));
        sizes.appendChild(button("fig-wider", say("wider"), "+",
                                 function () { resize(slot, 1); }));
        slot.appendChild(button("fig-grip", say("drag"), "⠿", function () {}));
        slot.appendChild(button("fig-pin", say("pin"), "📌",
                                function () { togglePin(slot); }));
        slot.appendChild(sizes);
        refresh(slot);
    }

    function emptyNote(tab, on) {
        var grid = gridOf(tab), note = grid.querySelector(".fig-empty-tab");
        if (on && !note) {
            note = document.createElement("p");
            note.className = "caption fig-empty-tab";
            note.textContent = say("nothingPinned");
            grid.appendChild(note);
        } else if (!on && note) {
            note.remove();
        }
    }

    /* On Overview: the figure leaves it and goes home. On its home tab: it
     * goes on Overview as well, at the end, as wide as it is here. */
    function togglePin(slot) {
        var key = slot.dataset.key, home = slot.dataset.home, there, homeSlot;
        if (tabOf(slot) === OVERVIEW || isPinned(key)) {
            layout.tabs[OVERVIEW] = entries(OVERVIEW).filter(function (e) { return e.key !== key; });
            there = slotOn(OVERVIEW, key);
            homeSlot = slotOn(home, key);
            if (there) {
                if (homeSlot) cardsIn(there).forEach(function (card) { homeSlot.appendChild(card); });
                there.remove();
                if (homeSlot) { watch(homeSlot); rearrange(homeSlot.parentNode); }
            }
            if (!gridOf(OVERVIEW).querySelector(".fig-slot")) emptyNote(OVERVIEW, true);
            rearrange(gridOf(OVERVIEW));
        } else {
            entries(OVERVIEW).push({key: key, w: widthOf(slot)});
            there = slot.cloneNode(false);
            there.removeAttribute("style");
            /* Overview has columns of its own; it finds one there. */
            there.removeAttribute("data-c");
            emptyNote(OVERVIEW, false);
            gridOf(OVERVIEW).appendChild(there);
            decorate(there);
            resized.observe(there);
            rearrange(gridOf(OVERVIEW));
        }
        document.querySelectorAll('.fig-slot[data-key="' + key + '"]').forEach(refresh);
        save();
    }

    /* ── dragging ──────────────────────────────────────────────────────────── */

    /* Where the figure being dragged would go across: the column under the
     * pointer. Its place in the order is Sortable's; its column is this, so a
     * figure can be dropped into any column rather than only into the place
     * the order happens to put it. */
    var dragging = null, pointerX = null, followFrame = 0;

    function follow() {
        if (!dragging || pointerX === null || columnsOf(dragging.grid) !== 4) return false;
        var slot = dragging.slot, w = Math.min(parseInt(slot.dataset.w, 10) || 1, 4),
            c = String(columnAt(dragging.grid, pointerX, w));
        if (slot.dataset.c === c) return false;
        slot.dataset.c = c;
        return true;
    }

    ["pointermove", "mousemove", "touchmove"].forEach(function (type) {
        document.addEventListener(type, function (ev) {
            if (!dragging) return;
            pointerX = ev.touches ? ev.touches[0].clientX : ev.clientX;
            if (followFrame) return;
            followFrame = requestAnimationFrame(function () {
                followFrame = 0;
                if (follow()) arrange(dragging.grid);
            });
        }, {capture: true, passive: true});
    });

    /* One list per tab, so nothing is dragged off the tab it lives on. By the
     * grip only: everything else on a figure is something to click. */
    if (window.Sortable && layout) {
        document.querySelectorAll(".fig-grid").forEach(function (grid) {
            var tab = grid.closest(".fig-panel").dataset.tab;
            window.Sortable.create(grid, {
                group: "fig-" + tab,
                handle: ".fig-grip",
                draggable: ".fig-slot",
                animation: 180,
                /* Pointer events rather than the browser's native drag: the
                 * figure follows the pointer as itself, the way a widget does
                 * on a phone, instead of as a faded screenshot — and it works
                 * the same on a touch screen. */
                forceFallback: true,
                fallbackOnBody: true,
                ghostClass: "fig-ghost",
                chosenClass: "fig-chosen",
                /* Placed again as the figure passes each place, so the others
                 * make room while it is still being dragged. Synchronously,
                 * so Sortable animates them from where they were to there. */
                onStart: function (ev) { dragging = {grid: grid, slot: ev.item}; },
                onChange: function () { follow(); arrange(grid); },
                onEnd: function (ev) {
                    follow();
                    dragging = null;
                    arrange(grid);
                    grid.querySelectorAll(":scope > .fig-slot").forEach(touch);
                    readOrder(tab);
                    save();
                }
            });
        });
    }

    document.querySelectorAll(".fig-slot").forEach(function (slot) {
        if (cardsIn(slot).length) watch(slot);
    });
    if (layout) document.querySelectorAll(".fig-slot").forEach(decorate);
    document.querySelectorAll(".fig-grid").forEach(function (grid) {
        gridResized.observe(grid);
    });
    window.addEventListener("hashchange", function () { show(fromHash()); });
    show(fromHash());
})();
