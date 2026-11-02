/* An experiment's notes page (ui/experiment_notes.html): the notes drawn from
 * their Markdown (ui/markdown.js), and the editor's Write and Preview, which
 * asks before the page is left with changes unsaved. */
(function () {
    "use strict";
    var draw = window.codesignerMarkdown;
    document.querySelectorAll("[data-markdown]").forEach(function (el) {
        el.replaceChildren(draw(el.textContent));
        el.classList.add("is-drawn");
    });

    var editor = document.querySelector("[data-notes-editor]");
    if (!editor) return;
    var box = editor.querySelector("[data-notes-text]");
    var preview = editor.querySelector("[data-notes-preview]");
    var modes = editor.querySelectorAll("[data-notes-mode]");
    modes.forEach(function (mode) {
        mode.addEventListener("click", function () {
            var previewing = mode.dataset.notesMode === "preview";
            if (previewing) preview.replaceChildren(draw(box.value));
            preview.hidden = !previewing;
            box.hidden = previewing;
            modes.forEach(function (m) { m.setAttribute("aria-selected", m === mode ? "true" : "false"); });
            if (!previewing) box.focus();
        });
    });

    var saved = box.value, saving = false;
    editor.addEventListener("submit", function () { saving = true; });
    window.addEventListener("beforeunload", function (e) {
        if (!saving && box.value !== saved) e.preventDefault();
    });
})();
