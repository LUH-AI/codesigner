/* The chat beside every page (assistant/_chat.html).
 *
 * Saying something is a POST; what happens next arrives through the
 * conversation's event stream, an EventSource, which reconnects by itself —
 * also after the page is left and the next one opened — and carries on from
 * the last event it saw. The events are written by the turn as it runs
 * (assistant/loop.py), so this script only draws them.
 *
 * Model text is drawn with textContent and never as markup: the model's words
 * are data on this page, like everything else the server sends.
 *
 * The handling of each kind of event follows A3S's chat
 * (https://github.com/LUH-AI/A3S, BSD-3-Clause, Copyright (c) Marcel Wever):
 * a status line while the model works, a card per tool, buttons for a choice. */
(function () {
    "use strict";
    var root = document.querySelector("[data-assistant]");
    if (!root || root.dataset.ready) return;
    root.dataset.ready = "1";

    var log = root.querySelector(".assistant-log");
    var status = root.querySelector(".assistant-status");
    var form = root.querySelector(".assistant-form");
    var box = form.querySelector("textarea");
    var send = form.querySelector("button[type=submit]");
    var text = function (key, values) {
        var s = root.dataset["text" + key.charAt(0).toUpperCase() + key.slice(1)] || "";
        Object.keys(values || {}).forEach(function (k) { s = s.replace("{" + k + "}", values[k]); });
        return s;
    };

    var conversation = root.dataset.conversation || "";
    var after = parseInt(root.dataset.after || "0", 10);
    // Events up to this one had happened when the page was drawn: they are
    // shown at once, and only what arrives later is written out as it comes.
    var liveAfter = parseInt(root.dataset.liveAfter || "0", 10);
    var source = null;
    var reply = null;      // the assistant's message being written
    var card = null;       // the last tool's card, waiting for its result
    var planCard = null;   // the plan as it stands, redrawn as it changes
    var datasetBox = root.querySelector("[data-assistant-dataset]");
    var resetButton = root.querySelector("[data-assistant-reset]");

    function eventsUrl(id) {
        return root.dataset.eventsUrl.replace("/0/", "/" + id + "/") + "?after=" + after;
    }

    // The log follows what is written while it is scrolled to its end; scrolled
    // up to read, it stays where it is until the person sends something.
    var following = true;
    log.addEventListener("scroll", function () {
        following = log.scrollHeight - log.scrollTop - log.clientHeight < 48;
    });
    function scroll(force) {
        if (force) following = true;
        if (following) log.scrollTop = log.scrollHeight;
    }

    // ── The model's formatting ────────────────────────────────────────────
    // A reply's Markdown, as far as a chat needs it: paragraphs, headings,
    // lists, quotes, tables, code, bold, italics and links. Built from elements
    // and text nodes, never from markup, so nothing the model writes can add
    // to the page beyond these.
    var INLINE = [
        ["code", /`([^`\n]+)`/],
        ["link", /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/],
        ["strong", /\*\*(?=\S)([\s\S]*?\S)\*\*|__(?=\S)([\s\S]*?\S)__(?!\w)/],
        ["em", /\*(?=[^\s*])([^*\n]*?[^\s*])\*|(?<![\w])_(?=\S)([^_\n]*?\S)_(?![\w])/],
        ["del", /~~(?=\S)([^~\n]*?\S)~~/]
    ];

    function inline(text, into) {
        while (text) {
            var best = null;
            INLINE.forEach(function (rule) {
                var m = rule[1].exec(text);
                if (m && (!best || m.index < best.m.index)) best = {kind: rule[0], m: m};
            });
            if (!best) { into.appendChild(document.createTextNode(text)); return; }
            if (best.m.index) into.appendChild(document.createTextNode(text.slice(0, best.m.index)));
            var el;
            if (best.kind === "code") {
                el = document.createElement("code");
                el.textContent = best.m[1];
            } else if (best.kind === "link") {
                el = document.createElement("a");
                el.href = best.m[2];
                el.target = "_blank";
                el.rel = "noopener noreferrer";
                inline(best.m[1], el);
            } else {
                el = document.createElement(best.kind);
                inline(best.m[1] !== undefined ? best.m[1] : best.m[2], el);
            }
            into.appendChild(el);
            text = text.slice(best.m.index + best.m[0].length);
        }
    }

    function cells(line) {
        return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(function (c) { return c.trim(); });
    }

    function markdown(source) {
        var out = document.createDocumentFragment();
        var lines = source.replace(/\r/g, "").replace(/^\s*\n/, "").split("\n");
        var i = 0, paragraph = [];
        function flush() {
            if (!paragraph.length) return;
            var p = document.createElement("p");
            paragraph.forEach(function (line, n) {
                if (n) p.appendChild(document.createElement("br"));
                inline(line, p);
            });
            out.appendChild(p);
            paragraph = [];
        }
        while (i < lines.length) {
            var line = lines[i];
            var fence = /^\s*(```|~~~)/.exec(line);
            if (fence) {
                flush();
                var code = [];
                i += 1;
                while (i < lines.length && lines[i].trim().indexOf(fence[1]) !== 0) code.push(lines[i++]);
                i += 1;
                var pre = document.createElement("pre"), c = document.createElement("code");
                c.textContent = code.join("\n");
                pre.appendChild(c);
                out.appendChild(pre);
                continue;
            }
            if (!line.trim()) { flush(); i += 1; continue; }
            var heading = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
            if (heading) {
                flush();
                var h = document.createElement("p");
                h.className = "assistant-heading";
                inline(heading[2], h);
                out.appendChild(h);
                i += 1;
                continue;
            }
            if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) {
                flush();
                out.appendChild(document.createElement("hr"));
                i += 1;
                continue;
            }
            if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
                flush();
                var table = document.createElement("table");
                var head = document.createElement("tr");
                cells(line).forEach(function (cell) {
                    var th = document.createElement("th");
                    inline(cell, th);
                    head.appendChild(th);
                });
                var thead = document.createElement("thead");
                thead.appendChild(head);
                table.appendChild(thead);
                var tbody = document.createElement("tbody");
                i += 2;
                while (i < lines.length && /^\s*\|/.test(lines[i])) {
                    var row = document.createElement("tr");
                    cells(lines[i]).forEach(function (cell) {
                        var td = document.createElement("td");
                        inline(cell, td);
                        row.appendChild(td);
                    });
                    tbody.appendChild(row);
                    i += 1;
                }
                table.appendChild(tbody);
                var wrap = document.createElement("div");
                wrap.className = "assistant-table";
                wrap.appendChild(table);
                out.appendChild(wrap);
                continue;
            }
            var item = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(line);
            if (item) {
                flush();
                var ordered = /\d/.test(item[1]);
                var list = document.createElement(ordered ? "ol" : "ul");
                if (ordered && parseInt(item[1], 10) !== 1) list.start = parseInt(item[1], 10);
                while (i < lines.length) {
                    var next = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(lines[i]);
                    if (!next || /\d/.test(next[1]) !== ordered) break;
                    var li = document.createElement("li");
                    inline(next[2], li);
                    i += 1;
                    // A line indented under an item continues it.
                    while (i < lines.length && /^\s{2,}\S/.test(lines[i])
                           && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
                        li.appendChild(document.createElement("br"));
                        inline(lines[i].trim(), li);
                        i += 1;
                    }
                    list.appendChild(li);
                }
                out.appendChild(list);
                continue;
            }
            if (/^\s*>/.test(line)) {
                flush();
                var quote = document.createElement("blockquote");
                var quoted = [];
                while (i < lines.length && /^\s*>/.test(lines[i])) quoted.push(lines[i++].replace(/^\s*>\s?/, ""));
                quote.appendChild(markdown(quoted.join("\n")));
                out.appendChild(quote);
                continue;
            }
            paragraph.push(line);
            i += 1;
        }
        flush();
        return out;
    }

    // ── Writing a reply out ───────────────────────────────────────────────
    // Text arrives in pieces of any size; a reply shows it a few characters a
    // frame, faster the further behind it is, so it reads as written rather
    // than dropped in blocks and never trails by more than a moment.
    var writing = [];
    var frame = null;
    function writer(el) {
        return {el: el, text: "", shown: 0};
    }
    function draw_(w) {
        w.el.replaceChildren(markdown(w.text.slice(0, w.shown)));
    }
    function write(w, piece, live) {
        w.text += piece;
        if (!live) {
            w.shown = w.text.length;
            draw_(w);
            scroll();
            return;
        }
        if (writing.indexOf(w) < 0) writing.push(w);
        if (frame === null) frame = requestAnimationFrame(tick);
    }
    function tick() {
        frame = null;
        writing = writing.filter(function (w) {
            var behind = w.text.length - w.shown;
            w.shown += Math.max(3, Math.ceil(behind / 8));
            if (w.shown < w.text.length && /[\ud800-\udbff]/.test(w.text.charAt(w.shown - 1))) w.shown += 1;
            w.shown = Math.min(w.shown, w.text.length);
            draw_(w);
            return w.shown < w.text.length;
        });
        scroll();
        if (writing.length) frame = requestAnimationFrame(tick);
    }

    function bubble(role, content) {
        var div = document.createElement("div");
        div.className = "assistant-msg " + role;
        div.textContent = content || "";
        log.appendChild(div);
        scroll();
        return div;
    }

    function setStatus(message) {
        status.textContent = message || "";
        status.hidden = !message;
    }

    function waiting(on) {
        send.disabled = on;
        box.disabled = on;
        if (!on) box.focus();
    }

    function choices(question, options) {
        var group = document.createElement("div");
        group.className = "assistant-choices";
        if (question) {
            var asked = document.createElement("p");
            asked.textContent = question;
            group.appendChild(asked);
        }
        options.forEach(function (option) {
            var b = document.createElement("button");
            b.type = "button";
            b.className = "btn";
            b.textContent = option;
            b.addEventListener("click", function () {
                group.querySelectorAll("button").forEach(function (x) { x.disabled = true; });
                say(option);
            });
            group.appendChild(b);
        });
        log.appendChild(group);
        scroll();
    }

    function label(key) {
        return root.dataset["label" + key.charAt(0).toUpperCase() + key.slice(1)] || key;
    }

    function shown(value) {
        if (value === null || value === undefined || value === "") return "—";
        if (typeof value !== "object") return String(value);
        return Object.keys(value).map(function (k) {
            var inner = value[k];
            return k + ": " + (inner && typeof inner === "object" ? JSON.stringify(inner) : inner);
        }).join(", ");
    }

    // A chart the assistant drew from the data, under its reply: Plotly, from
    // the page when it has it, else loaded once. Colours follow the page.
    var plotly = null;
    function withPlotly(then) {
        if (window.Plotly) return then(window.Plotly);
        if (!plotly) {
            plotly = new Promise(function (resolve, reject) {
                var script = document.createElement("script");
                script.src = root.dataset.plotlyUrl;
                script.onload = function () { resolve(window.Plotly); };
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }
        plotly.then(then).catch(function () {});
    }

    function drawChart(title, spec) {
        if (!spec || !spec.data) return;
        var box = document.createElement("div");
        box.className = "assistant-chart";
        box.setAttribute("role", "img");
        box.setAttribute("aria-label", title || "");
        log.appendChild(box);
        scroll();
        var colour = getComputedStyle(root).color;
        var layout = Object.assign({}, spec.layout || {}, {
            paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
            font: {color: colour, size: 11}
        });
        withPlotly(function (Plotly) {
            Plotly.newPlot(box, spec.data, layout, {displayModeBar: false, responsive: true});
        });
    }

    // The papers an answer cites, under it, each linking to where it can be
    // read. Only web links are drawn as links.
    function drawSources(papers) {
        if (!papers.length) return;
        var list = document.createElement("ol");
        list.className = "assistant-sources";
        papers.forEach(function (paper) {
            var item = document.createElement("li");
            var label = "[" + paper.key + "] " + paper.authors + " (" + paper.year + "). " + paper.title;
            if (/^https:\/\//.test(paper.link || "")) {
                var link = document.createElement("a");
                link.href = paper.link;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.textContent = label;
                item.appendChild(link);
            } else {
                item.textContent = label;
            }
            list.appendChild(item);
        });
        log.appendChild(list);
        scroll();
    }

    function drawPlan(plan, problems) {
        if (planCard) planCard.remove();
        planCard = document.createElement("div");
        planCard.className = "assistant-plan";
        var title = document.createElement("h3");
        title.textContent = text("plan");
        planCard.appendChild(title);
        var rows = document.createElement("dl");
        ["goal", "name", "task", "target_column", "model_name", "evaluation", "optimizer",
         "processing", "metric", "stopping", "seed"].forEach(function (key) {
            if (!(key in plan)) return;
            var dt = document.createElement("dt");
            dt.textContent = label(key);
            var dd = document.createElement("dd");
            dd.textContent = shown(plan[key]);
            rows.appendChild(dt);
            rows.appendChild(dd);
        });
        planCard.appendChild(rows);
        if (problems && problems.length) {
            var notReady = document.createElement("p");
            notReady.className = "caption";
            notReady.textContent = text("notReady");
            var list = document.createElement("ul");
            problems.forEach(function (problem) {
                var li = document.createElement("li");
                li.textContent = problem;
                list.appendChild(li);
            });
            planCard.appendChild(notReady);
            planCard.appendChild(list);
        }
        var approve = document.createElement("button");
        approve.type = "button";
        approve.className = "btn-primary";
        approve.textContent = text("approve");
        approve.disabled = !!(problems && problems.length);
        approve.addEventListener("click", function () { approvePlan(approve); });
        planCard.appendChild(approve);
        log.appendChild(planCard);
        scroll();
    }

    function post(url, body) {
        if (conversation) body.append("conversation", conversation);
        return fetch(url, {
            method: "POST", body: body, credentials: "same-origin",
            headers: {"X-CSRFToken": root.dataset.csrf}
        }).then(function (response) {
            return response.json().catch(function () { return {}; }).then(function (data) {
                return {ok: response.ok, status: response.status, data: data};
            });
        });
    }

    function approvePlan(button) {
        button.disabled = true;
        post(root.dataset.approveUrl, new FormData()).then(function (r) {
            if (r.ok && r.data.url) {
                window.location.href = r.data.url;
                return;
            }
            bubble("notice", (r.data.problems || [text("error")]).join(" "));
            button.disabled = false;
        }).catch(function () {
            bubble("notice", text("error"));
            button.disabled = false;
        });
    }

    function chooseDataset(fields) {
        var body = new FormData();
        Object.keys(fields).forEach(function (k) { body.append(k, fields[k]); });
        var group = root.querySelector("#assistant-group");
        if (group) body.append("group", group.value);
        post(root.dataset.datasetUrl, body).then(function (r) {
            if (r.data.conversation && String(r.data.conversation) !== conversation) {
                after = 0;
                follow(r.data.conversation);
            }
            if (!r.ok) bubble("notice", r.data.error || text("error"));
        }).catch(function () { bubble("notice", text("error")); });
    }

    // The dataset beside the box: an upload, or a demo when the checkbox above
    // it says so. Choosing one, or removing it, tells the conversation at once.
    if (datasetBox) {
        var useDemo = datasetBox.querySelector("#assistant-use-demo");
        var demo = datasetBox.querySelector("#assistant-demo");
        var upload = datasetBox.querySelector("#assistant-upload");
        var uploadPane = datasetBox.querySelector('[data-dataset-pane="upload"]');
        var panes = function () {
            demo.hidden = !useDemo.checked;
            uploadPane.hidden = useDemo.checked;
        };
        useDemo.addEventListener("change", panes);
        panes();
        demo.addEventListener("change", function () {
            if (demo.value) chooseDataset({demo: demo.value});
        });
        upload.addEventListener("change", function () {
            if (upload.files && upload.files[0]) chooseDataset({dataset_file: upload.files[0]});
        });
        upload.addEventListener("file-pick-clear", function () {
            if (conversation) chooseDataset({clear: "1"});
        });
    }

    // On the home page the box sits in the middle of the page until the
    // assistant starts answering; then the conversation takes the space above.
    function answering() { root.classList.remove("is-empty"); }

    var ended = {
        unavailable: "unavailable", budget: "budget", steps: "stopped",
        tool_calls: "stopped", repeating: "stopped", failed: "failed"
    };

    function draw(kind, payload, live) {
        if (kind === "user") {
            if (resetButton) resetButton.hidden = false;
            reply = null;
            bubble("user", payload.text);
            scroll(true);
            waiting(true);
        } else if (kind === "status") {
            answering();
            setStatus(text(payload.state === "thinking" ? "thinking" : "working"));
        } else if (kind === "text") {
            answering();
            if (!reply) reply = writer(bubble("assistant", ""));
            write(reply, payload.text, live);
            setStatus("");
        } else if (kind === "tool_call") {
            reply = null;
            card = document.createElement("details");
            card.className = "assistant-tool";
            var summary = document.createElement("summary");
            summary.textContent = text("tool", {tool: payload.tool});
            var args = document.createElement("pre");
            args.textContent = JSON.stringify(payload.arguments, null, 2);
            card.appendChild(summary);
            card.appendChild(args);
            log.appendChild(card);
            scroll();
        } else if (kind === "tool_result") {
            if (card) {
                var out = document.createElement("pre");
                out.textContent = payload.output;
                card.appendChild(out);
            }
        } else if (kind === "choices") {
            choices(payload.question, payload.options || []);
        } else if (kind === "plan") {
            drawPlan(payload.plan || {}, payload.problems || []);
        } else if (kind === "dataset") {
            // On the home page the picker names it; elsewhere the log does.
            if (datasetBox) {
                hasDataset = !!payload.name;
                unmark();
                var picked = datasetBox.querySelector("[data-file-pick-name]");
                if (!payload.demo) {
                    picked.textContent = payload.name || picked.dataset.empty;
                    picked.parentElement.classList.toggle("has-file", !!payload.name);
                }
            } else {
                bubble("notice dataset", text("dataset", {name: payload.name}));
            }
        } else if (kind === "sources") {
            drawSources(payload.papers || []);
        } else if (kind === "chart") {
            drawChart(payload.title, payload.spec);
        } else if (kind === "approved") {
            if (planCard) planCard.querySelectorAll("button").forEach(function (b) { b.disabled = true; });
            bubble("notice dataset", text("approved"));
        } else if (kind === "turn_end") {
            answering();
            setStatus("");
            reply = null;
            if (ended[payload.reason]) bubble("notice", text(ended[payload.reason]));
            waiting(false);
        }
    }

    function follow(id) {
        if (source) source.close();
        conversation = String(id);
        root.dataset.conversation = conversation;
        source = new EventSource(eventsUrl(conversation));
        ["user", "status", "text", "tool_call", "tool_result", "choices", "turn_end",
         "plan", "dataset", "approved", "chart", "sources"]
            .forEach(function (kind) {
                source.addEventListener(kind, function (e) {
                    var id = parseInt(e.lastEventId || "0", 10);
                    after = Math.max(after, id);
                    draw(kind, JSON.parse(e.data), id > liveAfter);
                });
            });
    }

    function say(message) {
        var body = new FormData();
        body.append("text", message);
        if (conversation) body.append("conversation", conversation);
        else if (root.dataset.experiment) body.append("experiment", root.dataset.experiment);
        waiting(true);
        fetch(root.dataset.messageUrl, {
            method: "POST", body: body, credentials: "same-origin",
            headers: {"X-CSRFToken": root.dataset.csrf}
        }).then(function (response) {
            return response.json().catch(function () { return {}; }).then(function (data) {
                if (response.status === 409) {
                    bubble("notice", text("busy"));
                    return;
                }
                if (!response.ok) throw new Error(response.status);
                if (String(data.conversation) !== conversation) {
                    after = data.after || 0;
                    follow(data.conversation);
                }
            });
        }).catch(function () {
            bubble("notice", text("error"));
            waiting(false);
        });
    }

    // Sending needs something to say, and on the home page a dataset to plan
    // with: what is missing is marked, and said, until it is given.
    var missing = form.querySelector("[data-assistant-missing]");
    var hasDataset = !!(datasetBox && datasetBox.hasAttribute("data-has-dataset"));
    function mark(prompt, dataset) {
        box.classList.toggle("is-missing", prompt);
        if (datasetBox) datasetBox.classList.toggle("is-missing", dataset);
        var said = [];
        if (dataset) said.push(text("needDataset"));
        if (prompt) said.push(text("needPrompt"));
        missing.textContent = said.join(" ");
        missing.hidden = !said.length;
    }
    function unmark() {
        mark(box.classList.contains("is-missing") && !box.value.trim(),
             !!datasetBox && datasetBox.classList.contains("is-missing") && !hasDataset);
    }
    form.addEventListener("submit", function (e) {
        e.preventDefault();
        var message = box.value.trim();
        var noDataset = !!datasetBox && !hasDataset;
        if (!message || noDataset) return mark(!message, noDataset);
        mark(false, false);
        box.value = "";
        say(message);
    });
    box.addEventListener("input", unmark);
    box.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            form.requestSubmit();
        }
    });
    // Reset (the home page), shown once something has been said: after
    // asking, the chat is deleted, and the draft it made with it, and the page
    // starts again as it was.
    var resetDialog = root.querySelector("[data-assistant-reset-dialog]");
    if (resetButton && resetDialog) {
        resetButton.addEventListener("click", function () { resetDialog.showModal(); });
        resetDialog.querySelector("[data-reset-cancel]").addEventListener("click", function () {
            resetDialog.close();
        });
        resetDialog.querySelector("[data-reset-confirm]").addEventListener("click", function () {
            post(root.dataset.resetUrl, new FormData()).then(function (r) {
                resetDialog.close();
                if (r.ok) window.location.reload();
                else bubble("notice", r.data.error || text("error"));
            }).catch(function () {
                resetDialog.close();
                bubble("notice", text("error"));
            });
        });
    }

    // Where the chat is, shared by every page of this browser and the window
    // it pops out into: "open" or "closed" in the dock, or "popped" out. Kept in
    // localStorage, whose change every other page and the window hear, so each
    // follows: a page's dock opens, closes or says the chat is elsewhere, and
    // the window closes itself when the chat comes back.
    var KEY = "assistant-dock", WINDOW = "codesigner-assistant";
    function where() {
        try { return localStorage.getItem(KEY) || "closed"; } catch (e) { return "closed"; }
    }
    function put(state) {
        try { localStorage.setItem(KEY, state); } catch (e) { /* this page alone */ }
    }

    var dock = root.closest("[data-assistant-dock]");
    if (dock) {
        var bar = root.querySelector("[data-assistant-toggle]");
        var show = function (state) {
            dock.classList.toggle("collapsed", state !== "open");
            dock.classList.toggle("popped", state === "popped");
            bar.setAttribute("aria-expanded", state === "open" ? "true" : "false");
            if (state === "open") scroll();
        };
        var go = function (state) {
            put(state);
            show(state);
            if (state === "open") box.focus();
        };
        bar.addEventListener("click", function () {
            var state = where();
            if (state !== "popped") return go(state === "open" ? "closed" : "open");
            // Brought to the front if it is still there. Asking for a window by
            // its name finds it without reloading it; if there is none, what
            // opens is blank, and the chat comes back here instead.
            var w = window.open("", WINDOW);
            var blank = true;
            try { blank = !w || w.location.href === "about:blank"; } catch (e) { blank = false; }
            if (blank) {
                if (w) w.close();
                go("open");
            } else {
                w.focus();
            }
        });
        root.querySelector("[data-assistant-collapse]").addEventListener("click", function () { go("closed"); });
        root.querySelector("[data-assistant-return]").addEventListener("click", function () { go("open"); });
        root.querySelector("[data-assistant-popout]").addEventListener("click", function () {
            var w = window.open(root.dataset.windowUrl, WINDOW, "popup,width=460,height=720");
            if (w) go("popped");
        });
        window.addEventListener("storage", function (e) { if (e.key === KEY) show(where()); });
        show(where());

        // Its size, set by dragging the corner grip and kept for every page of
        // this browser; held inside the window when the window is smaller.
        var SIZE = "assistant-dock-size";
        var grip = root.querySelector("[data-assistant-grip]");
        var size = null;
        try { size = JSON.parse(localStorage.getItem(SIZE) || "null"); } catch (e) { size = null; }
        var fit = function (w, h) {
            w = Math.round(Math.min(Math.max(w, 288), window.innerWidth - 40));
            h = Math.round(Math.min(Math.max(h, 240), window.innerHeight - 32));
            dock.style.setProperty("--dock-width", w + "px");
            dock.style.setProperty("--dock-height", h + "px");
            return {w: w, h: h};
        };
        if (size && size.w && size.h) fit(size.w, size.h);
        window.addEventListener("resize", function () { if (size) fit(size.w, size.h); });
        var start = null;
        grip.addEventListener("pointerdown", function (e) {
            e.preventDefault();
            grip.setPointerCapture(e.pointerId);
            start = {x: e.clientX, y: e.clientY, w: dock.offsetWidth, h: root.offsetHeight};
        });
        grip.addEventListener("pointermove", function (e) {
            if (start) fit(start.w + start.x - e.clientX, start.h + start.y - e.clientY);
        });
        var stop = function () {
            if (!start) return;
            start = null;
            size = {w: dock.offsetWidth, h: root.offsetHeight};
            try { localStorage.setItem(SIZE, JSON.stringify(size)); } catch (e) { /* this page alone */ }
            scroll();
        };
        grip.addEventListener("pointerup", stop);
        grip.addEventListener("pointercancel", stop);
    }

    // The window of its own: it says so while it is open, and goes when the
    // chat is brought back — by its own button or from a page.
    if (root.dataset.mode === "window") {
        put("popped");
        root.querySelector("[data-assistant-return]").addEventListener("click", function () {
            put("open");
            window.close();
        });
        window.addEventListener("storage", function (e) {
            if (e.key === KEY && where() !== "popped") window.close();
        });
        window.addEventListener("pagehide", function () {
            if (where() === "popped") put("closed");
        });
    }

    if (conversation) follow(conversation);
    if (root.dataset.busy) waiting(true);
})();
