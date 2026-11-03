/* Markdown drawn as elements and text, never as markup, so nothing in it can
 * add to the page beyond what is drawn here: paragraphs, headings, lists,
 * quotes, tables, code, bold, italics, strikethrough and links (http and
 * https only). The assistant's replies (assistant/chat.js) and an
 * experiment's notes (ui/notes.js) are drawn with it.
 *
 * `window.codesignerMarkdown(source)` returns a DocumentFragment. */
(function () {
    "use strict";
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
                h.className = "md-heading md-h" + heading[1].length;
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
                wrap.className = "md-table";
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

    window.codesignerMarkdown = markdown;
})();
