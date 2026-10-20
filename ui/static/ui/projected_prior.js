/* The Projected prior card: shaping a belief drawn in the configuration cube,
 * stating it, and changing or withdrawing one already stated.
 *
 * The cube picks the points and draws; this card holds the parameters and
 * talks to the server, which works out everything a belief is (see
 * ui/projected.py) — nothing here computes a density. They talk by
 * events on the document:
 *
 *   prior-draft:points       cube → card: the points picked, and the view
 *   prior-draft:points-edited card → cube: a point's coordinates typed in
 *   prior-draft:preview      card → cube: the draft's weight and its shapes
 *   prior-draft:cleared      either way: no draft any more
 *   prior-draft:refused      cube → card: a point it would not put down
 *   prior-draft:outside      card → cube: a point the server refused
 *   projected-priors:changed card → everyone: the stated ones and their weight
 *
 * The card draws the prior itself as well: its shape in its own frame, which
 * the server evaluates with the same density the search weighs by
 * (`core.projected_prior.profile`).
 */
(function () {
    "use strict";

    var section = document.querySelector('[data-figure="projected_prior"]');
    if (!section) return;

    var urls = {preview: section.dataset.previewUrl, state: section.dataset.stateUrl,
                withdraw: section.dataset.withdrawUrl},
        csrf = section.dataset.csrf,
        canEdit = section.dataset.canEdit === "1",
        strings = section.querySelector("[data-pp-strings]").dataset,
        el = {
            empty: document.getElementById("pp-empty"),
            error: document.getElementById("pp-error"),
            draft: document.getElementById("pp-draft"),
            summary: document.getElementById("pp-summary"),
            points: document.getElementById("pp-points"),
            params: document.getElementById("pp-params"),
            state: document.getElementById("pp-state"),
            discard: document.getElementById("pp-discard"),
            stated: document.getElementById("pp-stated"),
            decay: document.getElementById("pp-decay"),
            beta: document.getElementById("pp-beta"),
            plot: document.getElementById("pp-plot")
        },
        colours = {},
        /* Why the last point was not put down, kept on show until another
         * point is. */
        refusal = null,
        stated = [],
        /* The prior being drawn — `{view, dims, hyperparameters, points,
         * params, label}` — or null. */
        draft = null,
        /* The id of the stated prior being changed, or null. */
        editing = null,
        timer = 0,
        asked = 0;

    try {
        stated = JSON.parse(document.getElementById("projected-priors-data").textContent) || [];
    } catch (e) { stated = []; }
    try {
        colours = JSON.parse(document.getElementById("selection-colors-data").textContent) || {};
    } catch (e) { colours = {}; }

    function text(name, values) {
        var out = strings[name.replace(/-([a-z])/g, function (_m, c) { return c.toUpperCase(); })] || name;
        Object.keys(values || {}).forEach(function (key) { out = out.split("{" + key + "}").join(values[key]); });
        return out;
    }
    function metric() {
        var select = document.getElementById("metric-select");
        return select ? select.value : "";
    }
    function tell(name, detail) {
        document.dispatchEvent(new CustomEvent(name, {detail: detail || {}}));
    }
    function setError(message) {
        el.error.textContent = message || "";
        el.error.hidden = !message;
    }
    function round(v) { return Number(Number(v).toPrecision(4)); }

    function post(url, body) {
        return fetch(url, {method: "POST", body: JSON.stringify(body),
                           headers: {"Content-Type": "application/json", "X-CSRFToken": csrf}})
            .then(function (r) {
                return r.json().catch(function () { return {}; }).then(function (data) {
                    if (!r.ok) return Promise.reject(data.error ? data : {error: text("failed")});
                    return data;
                });
            }, function () { return Promise.reject({error: text("failed")}); });
    }
    function reason(failure) { return (failure && failure.error) || String(failure); }

    /* ── the prior, drawn ───────────────────────────────────────────────── */

    /* Its shape in its own frame: across a bump, along and across a ridge,
     * over a slab's plane — with the points it was drawn through and its
     * peak. A picture of the belief rather than of the space, which the cube
     * already is. */
    function drawProfile(profile) {
        if (!el.plot || !window.Plotly) return;
        if (!profile) { Plotly.purge(el.plot); el.plot.hidden = true; return; }
        el.plot.hidden = false;
        var colour = colours.prior || "#B26315",
            scale = [[0, "rgba(255,255,255,0)"], [1, colour]],
            traces = [], layout = {
                margin: {t: 8, r: 12, b: 40, l: 48}, height: 220, showlegend: false,
                paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                xaxis: {zeroline: false}, yaxis: {zeroline: false}};
        if (profile.kind === "radial") {
            traces.push({type: "scatter", mode: "lines", x: profile.x, y: profile.y,
                         fill: "tozeroy", line: {color: colour, width: 2}, hoverinfo: "skip"});
            traces.push({type: "scatter", mode: "lines", hoverinfo: "skip",
                         x: [-profile.width, -profile.width, null, profile.width, profile.width],
                         y: [0, 1, null, 0, 1], line: {color: colour, width: 1, dash: "dot"}});
            layout.xaxis.title = {text: text("across")};
            layout.yaxis.range = [0, 1.05];
            layout.yaxis.title = {text: text("weight")};
        } else {
            traces.push({type: "heatmap", x: profile.x, y: profile.y, z: profile.z,
                         colorscale: scale, zmin: 0, zmax: 1, showscale: false, hoverinfo: "skip"});
            var points = profile.points;
            if (profile.kind === "slab") {
                traces.push({type: "scatter", mode: "lines", hoverinfo: "skip",
                             x: points.map(function (p) { return p[0]; }).concat([points[0][0]]),
                             y: points.map(function (p) { return p[1]; }).concat([points[0][1]]),
                             line: {color: colour, width: 1, dash: "dot"}});
            } else {
                traces.push({type: "scatter", mode: "lines", hoverinfo: "skip",
                             x: [profile.x[0], profile.x[profile.x.length - 1], null,
                                 profile.x[0], profile.x[profile.x.length - 1]],
                             y: [-profile.width, -profile.width, null, profile.width, profile.width],
                             line: {color: colour, width: 1, dash: "dot"}});
            }
            traces.push({type: "scatter", mode: "markers+text", hoverinfo: "skip",
                         x: points.map(function (p) { return p[0]; }),
                         y: points.map(function (p) { return p[1]; }),
                         text: points.map(function (_p, i) { return String(i + 1); }),
                         textposition: "top center", textfont: {color: colour},
                         marker: {size: 9, color: colour, line: {color: "#FFFFFF", width: 1.5}}});
            traces.push({type: "scatter", mode: "markers", hoverinfo: "skip",
                         x: [profile.mode[0]], y: [profile.mode[1]],
                         marker: {symbol: "diamond", size: 11, color: colour}});
            layout.xaxis.title = {text: profile.kind === "slab" ? text("along-1") : text("along")};
            layout.yaxis.title = {text: profile.kind === "slab" ? text("along-2") : text("across")};
        }
        Plotly.react(el.plot, traces, layout, {displayModeBar: false, responsive: true});
    }

    function decay() {
        return {shape: el.decay.value, beta_ratio: parseFloat(el.beta.value) || null};
    }

    /* ── the parameters ─────────────────────────────────────────────────── */

    /* One control per parameter, as the prior card draws them: its name, an
     * exact number, a coarse slider. The ranges come from the record: a
     * position runs a little past the points either side, a spread up to
     * twice their span, a width up to ten times its default. */
    function paramControls(params, along, onChange) {
        el.params.innerHTML = "";
        var defs = [];
        for (var j = 0; j < along; j++) {
            defs.push({key: "mean", index: j, label: along > 1 ? text("position-n", {n: j + 1}) : text("position"),
                       help: text("position-help"), min: -0.5, max: 1.5, step: 0.01});
        }
        for (j = 0; j < along; j++) {
            defs.push({key: "spread", index: j, label: along > 1 ? text("spread-n", {n: j + 1}) : text("spread"),
                       help: text("spread-help"), min: 0.01, max: 2, step: 0.01});
        }
        var width = params.width || 0.1;
        defs.push({key: "width", index: null, label: text("width"), help: text("width-help"),
                   min: round(width / 20), max: round(width * 10), step: round(width / 20)});
        defs.forEach(function (def) {
            var value = def.index === null ? params.width : params[def.key][def.index];
            var label = document.createElement("label");
            label.className = "prior-param";
            label.title = def.help;
            label.appendChild(document.createTextNode(def.label));
            var number = document.createElement("input");
            number.type = "number";
            var range = document.createElement("input");
            range.type = "range";
            [number, range].forEach(function (input) {
                input.min = def.min; input.max = def.max; input.step = def.step;
                input.value = round(value);
                input.disabled = !canEdit && !!editing;
                input.addEventListener("input", function () {
                    var v = parseFloat(input.value);
                    if (!isFinite(v)) return;
                    (input === number ? range : number).value = v;
                    if (def.index === null) params.width = v; else params[def.key][def.index] = v;
                    onChange();
                });
            });
            label.appendChild(number);
            label.appendChild(range);
            el.params.appendChild(label);
        });
    }

    /* ── the draft ──────────────────────────────────────────────────────── */

    function showDraft() {
        el.empty.hidden = !!(draft || editing) || stated.length > 0;
        el.draft.hidden = !(draft || editing);
        el.discard.hidden = !!editing;
        el.state.textContent = editing ? text("done") : el.state.dataset.label;
        el.state.disabled = !editing && (!canEdit || !draft || !draft.record);
        el.state.title = canEdit ? "" : text("not-edit");
    }

    function renderPoints() {
        el.points.innerHTML = "";
        if (!draft) return;
        draft.points.forEach(function (point, i) {
            var row = document.createElement("div");
            row.className = "pp-point";
            row.appendChild(document.createTextNode(text("point", {n: i + 1})));
            point.forEach(function (value, k) {
                var input = document.createElement("input");
                input.type = "number";
                input.step = "any";
                input.value = round(value);
                input.setAttribute("aria-label", text("point", {n: i + 1}) + " " + (k + 1));
                input.addEventListener("change", function () {
                    var v = parseFloat(input.value);
                    if (!isFinite(v)) return;
                    draft.points[i][k] = v;
                    tell("prior-draft:points-edited", {points: draft.points});
                    preview();
                });
                row.appendChild(input);
            });
            el.points.appendChild(row);
        });
    }

    function preview() {
        if (timer) clearTimeout(timer);
        timer = setTimeout(function () {
            timer = 0;
            if (!draft) return;
            var mine = ++asked;
            post(urls.preview, {metric: draft.metric, draft: {
                view: draft.view, hyperparameters: draft.hyperparameters,
                points: draft.points, params: draft.params}})
                .then(function (data) {
                    if (mine !== asked || !draft) return;
                    setError(refusal);
                    draft.record = data.record;
                    drawProfile(data.profile);
                    if (!draft.params) {
                        draft.params = JSON.parse(JSON.stringify(data.record.params));
                        paramControls(draft.params, draft.points.length - 1, preview);
                    }
                    el.summary.textContent = data.record.label;
                    showDraft();
                    tell("prior-draft:preview", {metric: draft.metric, view: draft.view,
                                                 weights: data.weights, geometry: data.geometry});
                })
                .catch(function (failure) {
                    if (mine !== asked) return;
                    setError(reason(failure));
                    if (draft) draft.record = null;
                    showDraft();
                    tell("prior-draft:preview", {metric: draft && draft.metric, weights: null,
                                                 geometry: null});
                    /* One point outside the search space: it comes off in
                     * the cube, and the rest are asked about again. */
                    if (failure && typeof failure.point === "number") {
                        refusal = reason(failure);
                        tell("prior-draft:outside", {point: failure.point});
                    }
                });
        }, 150);
    }

    function clearDraft(announce) {
        draft = null;
        asked++;
        el.points.innerHTML = "";
        el.params.innerHTML = "";
        refusal = null;
        setError(null);
        drawProfile(null);
        showDraft();
        if (announce) tell("prior-draft:cleared");
    }

    document.addEventListener("prior-draft:points", function (event) {
        var d = event.detail || {};
        if (!d.points || !d.points.length) { clearDraft(false); return; }
        if (!d.refused) refusal = null;
        if (editing) stopEditing();
        /* The parameters carry over while the shape stays the same kind —
         * the same view, as many points — and start afresh otherwise. */
        var keep = draft && draft.view === d.view && draft.points.length === d.points.length;
        draft = {view: d.view, dims: d.dims, hyperparameters: d.hyperparameters || [],
                 points: d.points.map(function (p) { return p.slice(); }), metric: d.metric,
                 params: keep ? draft.params : null, record: null};
        if (!keep) el.params.innerHTML = "";
        el.summary.textContent = "";
        renderPoints();
        showDraft();
        preview();
    });
    document.addEventListener("prior-draft:cleared", function (event) {
        /* Lifted because its last point was refused: why stays on show. */
        var kept = event.detail && event.detail.refused ? refusal : null;
        clearDraft(false);
        if (kept) { refusal = kept; setError(kept); }
    });
    document.addEventListener("prior-draft:refused", function (event) {
        refusal = (event.detail && event.detail.message) || text("outside");
        setError(refusal);
    });

    /* ── stated ─────────────────────────────────────────────────────────── */

    function changed(data) {
        stated = data.stated || [];
        renderStated();
        tell("projected-priors:changed", {stated: stated, weights: data.weights, metric: metric()});
    }

    function renderStated() {
        el.stated.innerHTML = "";
        stated.forEach(function (record) {
            var item = document.createElement("li");
            item.className = "pp-item" + (record.id === editing ? " is-editing" : "");
            var name = document.createElement("span");
            name.textContent = record.label + " · " + text("from-trial", {trial: record.at_trial});
            item.appendChild(name);
            var edit = document.createElement("button");
            edit.type = "button";
            edit.textContent = record.id === editing ? text("done") : text("edit");
            edit.disabled = !canEdit;
            edit.addEventListener("click", function () {
                if (editing === record.id) stopEditing(); else startEditing(record);
            });
            var withdraw = document.createElement("button");
            withdraw.type = "button";
            withdraw.textContent = text("withdraw");
            withdraw.disabled = !canEdit;
            withdraw.addEventListener("click", function () {
                if (editing === record.id) stopEditing();
                post(urls.withdraw, {metric: metric(), id: record.id}).then(changed)
                    .catch(function (m) { setError(reason(m)); });
            });
            item.appendChild(edit);
            item.appendChild(withdraw);
            el.stated.appendChild(item);
        });
        showDraft();
    }

    var saveTimer = 0;
    function saveEdit() {
        if (saveTimer) clearTimeout(saveTimer);
        saveTimer = setTimeout(function () {
            saveTimer = 0;
            var record = stated.filter(function (r) { return r.id === editing; })[0];
            if (!record) return;
            post(urls.state, {metric: metric(), id: record.id, params: record.params,
                              decay: decay()})
                .then(function (data) {
                    setError(null);
                    changed(data);
                    var now = stated.filter(function (r) { return r.id === editing; })[0];
                    if (now) drawProfile(now.profile);
                })
                .catch(function (m) { setError(reason(m)); });
        }, 180);
    }

    function startEditing(record) {
        if (draft) clearDraft(true);
        editing = record.id;
        el.summary.textContent = text("editing", {label: record.label});
        el.points.innerHTML = "";
        var decayRecord = record.decay || {};
        el.decay.value = decayRecord.shape || "none";
        if (decayRecord.beta_ratio) el.beta.value = decayRecord.beta_ratio;
        el.beta.disabled = el.decay.value === "none";
        paramControls(record.params, record.points.length - 1, saveEdit);
        drawProfile(record.profile);
        renderStated();
    }

    function stopEditing() {
        editing = null;
        el.params.innerHTML = "";
        el.summary.textContent = "";
        drawProfile(null);
        renderStated();
    }

    el.state.dataset.label = el.state.textContent;
    el.state.addEventListener("click", function () {
        if (editing) { stopEditing(); return; }
        if (!draft || !draft.record) return;
        el.state.disabled = true;
        post(urls.state, {metric: draft.metric, decay: decay(), draft: {
            view: draft.view, hyperparameters: draft.hyperparameters,
            points: draft.points, params: draft.params}})
            .then(function (data) { clearDraft(true); changed(data); })
            .catch(function (m) { setError(reason(m)); showDraft(); });
    });
    el.discard.addEventListener("click", function () { clearDraft(true); });
    el.decay.addEventListener("change", function () {
        el.beta.disabled = el.decay.value === "none";
        if (editing) saveEdit();
    });
    el.beta.addEventListener("change", function () { if (editing) saveEdit(); });

    renderStated();
    showDraft();
}());
