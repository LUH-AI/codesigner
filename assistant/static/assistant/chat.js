/* The chats beside every page (assistant/_dock.html, assistant/_chat.html).
 *
 * Each chat on the page is a panel, `[data-assistant]`, run by `chat`: in the
 * dock a tab each (the general chat, the experiment's, a visitor's), on the
 * home page and in a window of its own just one.
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
    // The home page with a visitor has the script twice: its own chat's, and
    // the dock's.
    if (document.documentElement.dataset.assistantChats) return;
    document.documentElement.dataset.assistantChats = "1";

    // A chart the assistant drew from the data, under its reply: Plotly, from
    // the page when it has it, else loaded once for every chat on the page.
    var plotly = null;
    function withPlotly(url, then) {
        if (window.Plotly) return then(window.Plotly);
        if (!plotly) {
            plotly = new Promise(function (resolve, reject) {
                var script = document.createElement("script");
                script.src = url;
                script.onload = function () { resolve(window.Plotly); };
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }
        plotly.then(then).catch(function () {});
    }

    // One chat, in the panel *root*; what the dock needs of it is returned.
    function chat(root) {
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

    // A reply's Markdown is drawn by ui/markdown.js.
    var markdown = window.codesignerMarkdown;

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

    // Colours follow the page.
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
        withPlotly(root.dataset.plotlyUrl, function (Plotly) {
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
        var actions = document.createElement("div");
        actions.className = "assistant-plan-actions";
        var blocked = !!(problems && problems.length);
        [["draft", "", false], ["create", "btn-primary", true]].forEach(function (spec) {
            var button = document.createElement("button");
            button.type = "button";
            button.className = spec[1];
            button.textContent = text(spec[0] === "draft" ? "reviewDraft" : "approve");
            button.disabled = blocked;
            button.addEventListener("click", function () { approvePlan(spec[2]); });
            actions.appendChild(button);
        });
        var reject = document.createElement("button");
        reject.type = "button";
        reject.textContent = text("reject");
        reject.addEventListener("click", rejectPlan);
        actions.appendChild(reject);
        planCard.appendChild(actions);
        log.appendChild(planCard);
        scroll();
    }

    function planButtons(disabled) {
        if (planCard) planCard.querySelectorAll("button").forEach(function (b) { b.disabled = disabled; });
    }

    function planRejected() {
        if (!planCard) return;
        planCard.classList.add("is-rejected");
        var actions = planCard.querySelector(".assistant-plan-actions");
        if (actions) actions.remove();
        var note = document.createElement("p");
        note.className = "caption";
        note.textContent = text("rejected");
        planCard.appendChild(note);
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

    function approvePlan(create) {
        planButtons(true);
        var body = new FormData();
        if (create) body.append("create", "1");
        post(root.dataset.approveUrl, body).then(function (r) {
            if (r.ok && r.data.url) {
                window.location.href = r.data.url;
                return;
            }
            bubble("notice", (r.data.problems || [text("error")]).join(" "));
            planButtons(false);
        }).catch(function () {
            bubble("notice", text("error"));
            planButtons(false);
        });
    }

    function rejectPlan() {
        planButtons(true);
        post(root.dataset.rejectUrl, new FormData()).then(function (r) {
            if (r.ok) return;
            bubble("notice", (r.data.problems || [text("error")]).join(" "));
            planButtons(false);
        }).catch(function () {
            bubble("notice", text("error"));
            planButtons(false);
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
        } else if (kind === "withdrawn") {
            // The answer cited a paper the library lacks; the model answers
            // again, and this one stays visible but marked.
            var answers = log.querySelectorAll(".assistant-msg.assistant");
            if (answers.length) answers[answers.length - 1].classList.add("is-withdrawn");
            reply = null;
            bubble("notice", text("withdrawn", {keys: (payload.keys || []).map(function (k) {
                return "[" + k + "]"; }).join(", ")}));
        } else if (kind === "chart") {
            drawChart(payload.title, payload.spec);
        } else if (kind === "approved") {
            planButtons(true);
            bubble("notice dataset", text("approved"));
        } else if (kind === "rejected") {
            planRejected();
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
         "plan", "dataset", "approved", "rejected", "chart", "sources", "withdrawn"]
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
    // Reset, shown once something has been said: after asking, the chat is
    // deleted (assistant/views.py `reset` says what stays), and the page
    // starts again with it empty.
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

    // A visiting chat's way back to its own experiment, where the visit ends:
    // pressed on the chat itself, or on its tab in the dock.
    function goBack(button) {
        button.disabled = true;
        post(root.dataset.backUrl, new FormData()).then(function (r) {
            window.location.assign(r.data.url || window.location.href);
        }).catch(function () {
            button.disabled = false;
            bubble("notice", text("error"));
        });
    }
    var back = root.querySelector("[data-assistant-back]");
    if (back) back.addEventListener("click", function () { goBack(back); });

    if (conversation) follow(conversation);
    if (root.dataset.busy) waiting(true);
    return {
        root: root,
        scroll: scroll,
        focus: function () { if (!box.disabled) box.focus(); },
        back: goBack,
        windowUrl: function (base) {
            if (conversation) return base + "?conversation=" + encodeURIComponent(conversation);
            if (root.dataset.experiment) return base + "?experiment=" + encodeURIComponent(root.dataset.experiment);
            return base;
        }
    };
    }

    var chats = Array.prototype.map.call(document.querySelectorAll("[data-assistant]"), chat);

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

    var dock = document.querySelector("[data-assistant-dock]");
    var shell = dock && dock.querySelector("[data-assistant-shell]");
    if (shell) {
        // The tab showing; the others keep following their conversations.
        var docked = chats.filter(function (c) { return shell.contains(c.root); });
        var tabs = shell.querySelectorAll("[data-assistant-tab]");
        var current = docked.filter(function (c) { return !c.root.hidden; })[0] || docked[0];
        var pick = function (key) {
            docked.forEach(function (c) {
                c.root.hidden = c.root.dataset.tab !== key;
                if (!c.root.hidden) current = c;
            });
            tabs.forEach(function (t) {
                t.setAttribute("aria-selected", t.dataset.assistantTab === key ? "true" : "false");
            });
            current.scroll(true);
            current.focus();
        };
        tabs.forEach(function (t) {
            t.addEventListener("click", function () { pick(t.dataset.assistantTab); });
        });
        var visitor = docked.filter(function (c) { return c.root.dataset.tab === "visitor"; })[0];
        var tabBack = shell.querySelector(".assistant-tabs [data-assistant-back]");
        if (visitor && tabBack) {
            tabBack.addEventListener("click", function () { visitor.back(tabBack); });
        }

        var bar = shell.querySelector("[data-assistant-toggle]");
        var show = function (state) {
            dock.classList.toggle("collapsed", state !== "open");
            dock.classList.toggle("popped", state === "popped");
            bar.setAttribute("aria-expanded", state === "open" ? "true" : "false");
            if (state === "open") current.scroll();
        };
        var go = function (state) {
            put(state);
            show(state);
            if (state === "open") current.focus();
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
        shell.querySelector("[data-assistant-collapse]").addEventListener("click", function () { go("closed"); });
        shell.querySelector("[data-assistant-return]").addEventListener("click", function () { go("open"); });
        // The window takes the tab showing.
        shell.querySelector("[data-assistant-popout]").addEventListener("click", function () {
            var w = window.open(current.windowUrl(shell.dataset.windowUrl), WINDOW,
                                "popup,width=460,height=720");
            if (w) go("popped");
        });
        window.addEventListener("storage", function (e) { if (e.key === KEY) show(where()); });
        // A visitor, or a chat that is answering, has just brought the person
        // here: the dock opens on it.
        var arriving = current.root.dataset.tab === "visitor" || current.root.dataset.busy;
        show(arriving && where() === "closed" ? "open" : where());

        // Its size, set by dragging the corner grip and kept for every page of
        // this browser; held inside the window when the window is smaller.
        var SIZE = "assistant-dock-size";
        var grip = shell.querySelector("[data-assistant-grip]");
        var size = null;
        try { size = JSON.parse(localStorage.getItem(SIZE) || "null"); } catch (e) { size = null; }
        var fit = function (w, h) {
            // 26rem at the least: narrower, a line of an answer is a few words.
            var least = 26 * parseFloat(getComputedStyle(document.documentElement).fontSize);
            w = Math.round(Math.min(Math.max(w, least), window.innerWidth - 40));
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
            start = {x: e.clientX, y: e.clientY, w: dock.offsetWidth, h: shell.offsetHeight};
        });
        grip.addEventListener("pointermove", function (e) {
            if (start) fit(start.w + start.x - e.clientX, start.h + start.y - e.clientY);
        });
        var stop = function () {
            if (!start) return;
            start = null;
            size = {w: dock.offsetWidth, h: shell.offsetHeight};
            try { localStorage.setItem(SIZE, JSON.stringify(size)); } catch (e) { /* this page alone */ }
            current.scroll();
        };
        grip.addEventListener("pointerup", stop);
        grip.addEventListener("pointercancel", stop);
    }

    // The window of its own: it says so while it is open, and goes when the
    // chat is brought back — by its own button or from a page.
    var own = document.querySelector('[data-assistant][data-mode="window"]');
    if (own) {
        put("popped");
        own.querySelector("[data-assistant-return]").addEventListener("click", function () {
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
})();
