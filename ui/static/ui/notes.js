/* An experiment's notes page (ui/experiment_notes.html): the notes drawn from
 * their Markdown (ui/markdown.js), the View and Edit tabs, and the editor's
 * preview, redrawn as the person types. The page asks before it is left with
 * changes unsaved. */
(function () {
    "use strict";
    var draw = window.codesignerMarkdown;
    document.querySelectorAll("[data-markdown]").forEach(function (el) {
        el.replaceChildren(draw(el.textContent));
        el.classList.add("is-drawn");
    });

    var page = document.querySelector("[data-notes-page]");
    if (!page) return;
    var tabs = page.querySelectorAll(".fig-tab");
    var panels = page.querySelectorAll(".fig-panel");

    /* As the dashboard's tabs (ui/layout.js): the tab in the address bar's
     * hash, replaced rather than pushed, so a reload keeps it and Back still
     * leaves the page. */
    function show(name, remember) {
        var known = Array.prototype.some.call(tabs, function (t) { return t.dataset.tab === name; });
        if (!known) return;
        tabs.forEach(function (tab) {
            var on = tab.dataset.tab === name;
            tab.setAttribute("aria-selected", on ? "true" : "false");
            tab.tabIndex = on ? 0 : -1;
        });
        panels.forEach(function (panel) {
            var on = panel.dataset.tab === name;
            panel.classList.toggle("is-inactive", !on);
            panel.inert = !on;
        });
        if (remember && window.history && window.history.replaceState) {
            window.history.replaceState(null, "", name === "view" ? window.location.pathname : "#" + name);
        }
    }
    tabs.forEach(function (tab) {
        tab.addEventListener("click", function () { show(tab.dataset.tab, true); });
    });
    var opened = page.querySelector('.fig-tab[aria-selected="true"]');
    show(window.location.hash === "#edit" ? "edit" : opened.dataset.tab, false);

    var editor = page.querySelector("[data-notes-editor]");
    if (!editor) return;
    var box = editor.querySelector("[data-notes-text]");
    var preview = editor.querySelector("[data-notes-preview]");
    var pending = false;
    function redraw() {
        pending = false;
        preview.replaceChildren(draw(box.value));
    }
    box.addEventListener("input", function () {
        if (!pending) { pending = true; window.requestAnimationFrame(redraw); }
    });
    redraw();

    var saved = box.value, saving = false;
    editor.addEventListener("submit", function () { saving = true; });
    window.addEventListener("beforeunload", function (e) {
        if (!saving && box.value !== saved) e.preventDefault();
    });
})();
