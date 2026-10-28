// The docs site: a file's docstrings and comments as prose, the code between
// them folded away, and every block editable in place. The server does all the
// parsing and every write; this only shows what it sends and posts edits back.
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = s => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem("docsite." + key);
    localStorage.setItem("docsite." + key, value);
  } catch { return null; }
}

const state = {
  index: null,       // {files, names, width, root}
  page: null,        // the file on screen, as the server sent it
  showCode: store("showCode") === "1",
  mode: store("mode") || "flow",
  editor: null,      // the open editor, if any
  unfolded: new Set(),
  focus: -1,
  suffixes: null,
};

// ── talking to the server ───────────────────────────────────────────────────

async function api(path, body) {
  const opts = { headers: { "X-Docsite-Token": window.DOCSITE_TOKEN } };
  if (body !== undefined) {
    opts.method = "POST";
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({ error: res.statusText }));
  if (!res.ok) throw Object.assign(new Error(data.error || res.statusText), { status: res.status, data });
  return data;
}

// ── routing: #/path/to/file.py[:anchor] ────────────────────────────────────

function route() {
  const h = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
  const m = h.match(/^(.*?\.py)(?::(.+))?$/);
  return m ? { path: m[1], anchor: m[2] || null } : { path: null, anchor: null };
}
const href = (path, anchor) => "#/" + path + (anchor ? ":" + anchor : "");

let lastHash = location.hash;
window.addEventListener("hashchange", () => {
  if (state.editor && state.editor.dirty() && !confirm("Leave without saving this edit?")) {
    history.replaceState(null, "", lastHash);
    return;
  }
  closeEditor();
  lastHash = location.hash;
  show();
});

async function show(opts = {}) {
  const { path, anchor } = route();
  markTree(path);
  document.body.classList.remove("nav-open");
  if (!path) return renderHome();
  if (!state.page || state.page.path !== path || opts.reload) {
    try {
      const page = await api("/api/page?path=" + encodeURIComponent(path));
      if (!state.page || state.page.path !== path) state.unfolded.clear();
      state.page = page;
    } catch (e) {
      $("#main").innerHTML = `<p class="empty">${esc(e.message)}</p>`;
      return;
    }
  }
  renderPage();
  if (anchor && !opts.keepScroll) jump(anchor);
  else if (!opts.keepScroll) window.scrollTo(0, 0);
}

function jump(anchor) {
  const el = document.getElementById("s-" + anchor) || document.getElementById(anchor);
  if (!el) return;
  el.scrollIntoView({ block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1400);
  const blocks = targets();
  const i = blocks.indexOf(el.closest(".prose, .add-doc") || el);
  if (i >= 0) setFocus(i, false);
}

// ── the file tree ───────────────────────────────────────────────────────────

function renderTree(filter = "") {
  const tree = $("#tree");
  const files = state.index.files;
  if (filter) {
    const f = filter.toLowerCase();
    tree.innerHTML = files.filter(x => x.path.toLowerCase().includes(f))
      .map(x => fileLink(x, x.path)).join("") || `<p class="flat">No file matches.</p>`;
    return markTree(route().path);
  }
  const root = { dirs: {}, files: [] };
  for (const f of files) {
    const parts = f.path.split("/");
    let node = root;
    for (const p of parts.slice(0, -1)) node = node.dirs[p] ??= { dirs: {}, files: [] };
    node.files.push(f);
  }
  const draw = (node, prefix) =>
    Object.keys(node.dirs).sort().map(d =>
      `<details data-dir="${esc(prefix + d)}"><summary>${esc(d)}/</summary><div>${draw(node.dirs[d], prefix + d + "/")}</div></details>`
    ).join("") + node.files.map(f => fileLink(f, f.path.split("/").pop())).join("");
  tree.innerHTML = draw(root, "");
  markTree(route().path);
}

function fileLink(f, label) {
  const gap = f.symbols - f.documented;
  const cov = f.broken ? `<span class="cov gap">!</span>`
    : f.symbols ? `<span class="cov ${gap ? "gap" : ""}" title="${f.documented} of ${f.symbols} defs documented">${gap ? gap : "✓"}</span>` : "";
  return `<a href="${href(f.path)}" data-path="${esc(f.path)}" title="${esc(f.summary || f.path)}"><span>${esc(label)}</span>${cov}</a>`;
}

function markTree(path) {
  $$("#tree a.cur").forEach(a => a.classList.remove("cur"));
  if (!path) return;
  const a = $(`#tree a[data-path="${CSS.escape(path)}"]`);
  if (!a) return;
  a.classList.add("cur");
  for (let d = a.parentElement.closest("details"); d; d = d.parentElement.closest("details")) d.open = true;
  const nav = $("#nav"), r = a.getBoundingClientRect(), n = nav.getBoundingClientRect();
  if (r.top < n.top + 40 || r.bottom > n.bottom) a.scrollIntoView({ block: "center" });
}

// ── the home page ───────────────────────────────────────────────────────────

function renderHome() {
  state.page = null;
  const files = state.index.files;
  const syms = files.reduce((a, f) => a + (f.symbols || 0), 0);
  const docd = files.reduce((a, f) => a + (f.documented || 0), 0);
  const groups = {};
  for (const f of files) (groups[f.path.includes("/") ? f.path.split("/")[0] : "(root)"] ??= []).push(f);
  $("#main").innerHTML = `<div class="page home">
    <h1>codesigner</h1>
    <p class="lede">The docstrings and comments of every Python file, read from the working tree.
      Click any block of prose to edit it; the save goes straight into the source.
      Press <kbd>?</kbd> for keys.</p>
    <div class="stats">
      <div class="stat"><b>${files.length}</b><span>files</span></div>
      <div class="stat"><b>${docd} / ${syms}</b><span>defs and classes documented</span></div>
    </div>
    ${Object.entries(groups).map(([g, fs]) => `<section class="pkg"><h2>${esc(g)}/</h2><table>${
      fs.map(f => {
        const gap = (f.symbols || 0) - (f.documented || 0);
        return `<tr><td><a href="${href(f.path)}">${esc(f.module)}</a></td>
          <td class="sum">${f.broken ? `<span class="broken">${esc(f.broken)}</span>` : esc(f.summary || "")}</td>
          <td class="n ${gap ? "gap" : ""}">${f.symbols ? `${f.documented}/${f.symbols}` : ""}</td></tr>`;
      }).join("")}</table></section>`).join("")}
  </div>`;
  window.scrollTo(0, 0);
}

// ── a file ──────────────────────────────────────────────────────────────────

function renderPage() {
  const p = state.page;
  const vs = `vscode://file${state.index.root}/${p.path}`;
  const segs = p.segments;
  const undocumented = segs.flatMap(s => s.type === "code" ? s.lines.filter(l => l.add) : []).length;
  const i = state.index.files.findIndex(f => f.path === p.path);
  const prev = state.index.files[i - 1], next = state.index.files[i + 1];
  let html = `<div class="page"><div class="page-head">
    <h1>${esc(p.module)}</h1>
    <div class="meta"><a href="${vs}:1" title="Open in VS Code">${esc(p.path)}</a>
      ${undocumented ? `<span class="chip gap">${undocumented} without a docstring</span>` : ""}
      ${p.broken ? `<span class="broken">${esc(p.broken)}</span>` : ""}</div>
  </div>`;
  segs.forEach((s, k) => { html += s.type === "prose" ? proseHTML(s, k) : codeHTML(s, k); });
  html += `<div class="pager">${prev ? `<a href="${href(prev.path)}">← ${esc(prev.module)}</a>` : "<span></span>"}
    ${next ? `<a href="${href(next.path)}">${esc(next.module)} →</a>` : ""}</div></div>`;
  $("#main").innerHTML = html;
  $("#banner").hidden = true;
  state.focus = -1;
}

function proseHTML(s, k) {
  const cls = `prose k-${s.kind}${s.owner === "" ? " module" : ""}`;
  const label = s.kind === "docstring" ? (s.owner || "module") : s.kind === "attr" ? "#:" : "#";
  return `<section class="${cls}" id="L${s.start}" data-seg="${k}" style="--i:${s.indent}">
    <a class="where" href="vscode://file${state.index.root}/${state.page.path}:${s.start}" title="Open in VS Code">${esc(label)} · L${s.start}</a>
    ${md(s.text) || `<p class="empty-doc">(empty)</p>`}</section>`;
}

const isKept = l => l.role === "header" || l.role === "target";

function lineHTML(l, k) {
  const vs = `vscode://file${state.index.root}/${state.page.path}`;
  const cls = ["ln", "role-" + l.role];
  let id = "";
  if (l.sym) { cls.push("sig", "depth-" + l.depth); id = ` id="s-${esc(l.sym)}"`; }
  let out = `<div class="${cls.join(" ")}"${id}><a class="lnum" href="${vs}:${l.n}" title="Open in VS Code">${l.n}</a><code class="code">${l.html || " "}</code></div>`;
  if (l.add) out += `<button class="add-doc" data-seg="${k}" data-n="${l.n}" style="--i:${l.add.indent}">+ docstring for ${esc(l.add.qualname)}</button>`;
  return out;
}

// The folded run of code that starts at line *n* of segment *k*.
function foldRun(k, n) {
  const lines = state.page.segments[k].lines;
  let i = lines.findIndex(l => l.n === n), j = i;
  while (j < lines.length && !isKept(lines[j])) j++;
  return lines.slice(i, j);
}

function codeHTML(s, k) {
  const line = l => lineHTML(l, k);
  const lines = s.lines;
  let html = `<div class="code">`;
  for (let i = 0; i < lines.length;) {
    const l = lines[i];
    if (state.showCode || isKept(l)) { html += line(l); i++; continue; }
    // A run of body and blank lines, folded unless opened.
    const run = foldRun(k, l.n);
    const j = i + run.length;
    const key = `${run[0].n}`;
    const real = run.filter(x => x.role !== "blank");
    if (state.unfolded.has(key)) html += run.map(line).join("");
    else if (real.length) html += `<button class="fold" data-seg="${k}" data-key="${key}">⋯ ${real.length} line${real.length > 1 ? "s" : ""}</button>`;
    i = j;
  }
  return html + `</div>`;
}

// ── prose as HTML ───────────────────────────────────────────────────────────
// Close enough to Markdown for docstrings that are mostly prose: paragraphs,
// lists, literal blocks, `code`, *emphasis*. Code that names a module or def
// links to it.

const LIST = /^(?:[-*+•]|\d+[.)])\s+/;
const ALIGNED = /\S {3,}\S|[^\s.!?:] {2}\S/;
const LITERAL_START = ["```", ">>>", "|", "+-", "===", "---"];

function md(text) {
  if (!text.trim()) return "";
  return text.split(/\n[ \t]*\n/).map(block => {
    const lines = block.split("\n").filter(l => l.trim());
    if (!lines.length) return "";
    const literal = lines.some(l => LITERAL_START.some(p => l.trimStart().startsWith(p)) || ALIGNED.test(l.trim()))
      || lines.some((l, i) => /^\s/.test(l) && !(i && lines.slice(0, i).some(x => LIST.test(x))));
    if (literal) {
      const pad = Math.min(...lines.map(l => l.match(/^\s*/)[0].length));
      return `<pre>${esc(block.split("\n").map(l => l.slice(pad)).join("\n").replace(/^\n+|\n+$/g, ""))}</pre>`;
    }
    let out = "", para = [], items = [], ordered = false;
    const flushPara = () => { if (para.length) out += `<p>${inline(para.join(" "))}</p>`; para = []; };
    const flushList = () => {
      if (items.length) out += `<${ordered ? "ol" : "ul"}>${items.map(it => `<li>${inline(it)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`;
      items = [];
    };
    for (const l of lines) {
      if (LIST.test(l)) { flushPara(); ordered = /^\d/.test(l); items.push(l.replace(LIST, "")); }
      else if (/^\s/.test(l) && items.length) items[items.length - 1] += " " + l.trim();
      else { flushList(); para.push(l.trim()); }
    }
    flushPara(); flushList();
    return out;
  }).join("");
}

function inline(s) {
  const codes = [];
  s = s.replace(/:[\w:]+:(?=`)/g, "")  // reST roles: :class:`x` reads as `x`
    .replace(/``(.+?)``|`([^`]+)`/g, (_, a, b) => { codes.push(a ?? b); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(\[{"'—–-])\*(?![\s*])([^*]+?)\*(?=$|[\s).,;:!?'"\]}—–-])/g, "$1<em>$2</em>")
    .replace(/\bhttps?:\/\/[^\s<]+[^\s<.,;:)]/g, u => `<a href="${u}" target="_blank" rel="noopener">${u}</a>`);
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => {
    const c = codes[+i];
    const to = resolve(c);
    return to ? `<a href="${to}"><code>${esc(c)}</code></a>` : `<code>${esc(c)}</code>`;
  });
}

// `core.registry`, `registry.MODELS`, `remote/run_experiment.py`, or a def in
// this file, to the page it names.
function resolve(code) {
  const name = code.replace(/\(.*\)$/, "").replace(/^\.+/, "").trim();
  if (!/^[\w.\/]+$/.test(name)) return null;
  const names = state.index.names;
  const target = names[name] || (name.includes(".") || name.includes("/") ? suffix(name) : null)
    || (state.page && state.page.symbols && state.page.symbols.includes(name) ? `${state.page.path}:${name}` : null);
  if (!target) return null;
  const [path, anchor] = target.split(":");
  return href(path, anchor);
}

function suffix(name) {
  if (!state.suffixes) {
    state.suffixes = new Map();
    for (const [full, to] of Object.entries(state.index.names)) {
      if (full.endsWith(".py")) continue;
      const parts = full.split(".");
      for (let i = 1; i < parts.length - 1; i++) {
        const key = parts.slice(i).join(".");
        state.suffixes.set(key, state.suffixes.has(key) && state.suffixes.get(key) !== to ? null : to);
      }
    }
  }
  return state.suffixes.get(name) || null;
}

// ── editing ─────────────────────────────────────────────────────────────────

function openEditor(el) {
  if (state.editor) {
    if (state.editor.el === el) return state.editor.ta.focus();
    if (state.editor.dirty() && !confirm("Discard the edit that is open?")) return;
    closeEditor();
  }
  const p = state.page;
  const seg = p.segments[+el.dataset.seg];
  let target;  // what a save sends, besides the text
  let label, width, first, last, old, flowText;
  if (el.classList.contains("add-doc")) {
    const line = seg.lines.find(l => String(l.n) === el.dataset.n);
    target = { op: "add", qualname: line.add.qualname, start: line.add.start };
    label = `new docstring · ${line.add.qualname}`;
    width = line.add.width; first = 3; last = 3; old = ""; flowText = "";
  } else {
    target = { op: "region", start: seg.start, raw: seg.raw };
    label = `${seg.kind === "docstring" ? (seg.owner || "module") + " docstring" : seg.kind === "attr" ? "#: comment" : "comment"} · L${seg.start}`;
    ({ width, first, last } = seg);
    old = seg.text; flowText = seg.flow;
  }

  const ed = document.createElement("div");
  ed.className = "editor";
  ed.style.setProperty("--i", el.style.getPropertyValue("--i") || 0);
  ed.style.setProperty("--w", width);
  ed.innerHTML = `<div class="ed-bar"><span class="ed-what">${esc(label)}</span>
      <span class="seg"><button data-mode="flow" title="One line per paragraph; wrapped on save (Alt+Q)">Paragraphs</button><button data-mode="lines" title="As in the file; saved as typed (Alt+Q)">Lines</button></span>
      <span class="ed-hint">wraps at ${width} · <kbd>Ctrl</kbd>+<kbd>Enter</kbd> save · <kbd>Esc</kbd> cancel</span></div>
    <textarea spellcheck="true"></textarea>
    <div class="ed-foot"><span class="ed-err"></span><button class="cancel">Cancel</button><button class="save primary">Save</button></div>`;
  const ta = $("textarea", ed);
  let mode = state.mode;
  let base = { flow: flowText, lines: old };   // each mode's text before any typing

  const grow = () => { ta.style.height = "auto"; ta.style.height = Math.max(ta.scrollHeight + 2, 72) + "px"; };
  const setMode = m => {
    mode = m;
    ed.classList.toggle("lines", m === "lines");
    ta.wrap = m === "lines" ? "off" : "soft";
    $$(".seg button", ed).forEach(b => b.classList.toggle("on", b.dataset.mode === m));
    grow();
  };
  const err = msg => { $(".ed-err", ed).textContent = msg || ""; };

  const editor = {
    el, ed, ta,
    dirty: () => ta.value !== base[mode],
    async switchMode(m) {
      if (m === mode) return;
      if (editor.dirty()) {
        ed.classList.add("busy");
        try {
          const r = m === "flow" ? await api("/api/flow", { text: ta.value })
            : await api("/api/reflow", { old, text: ta.value, width, first, last });
          ta.value = r.text;
        } catch (e) { err(e.message); return; }
        finally { ed.classList.remove("busy"); }
      } else {
        ta.value = base[m];
      }
      state.mode = m; store("mode", m);
      setMode(m);
      ta.focus();
    },
    async save() {
      if (!editor.dirty() && target.op === "region") return closeEditor();
      ed.classList.add("busy"); err("");
      try {
        const r = await api("/api/edit", { path: p.path, text: ta.value, flowed: mode === "flow", ...target });
        const top = ed.getBoundingClientRect().top;
        state.page = r.page;
        closeEditor(false);
        renderPage();
        const now = document.getElementById("L" + r.line);
        if (now) {
          window.scrollBy(0, now.getBoundingClientRect().top - top);
          const i = targets().indexOf(now);
          if (i >= 0) setFocus(i, false);
        }
        setUndo(r.undo);
        toast(r.changed ? `Saved ${p.path}:${r.line}` : "No change", r.changed ? { label: "Undo", run: undo } : null);
      } catch (e) {
        ed.classList.remove("busy");
        if (e.status === 409 && e.data.page) {
          // The file moved under the text. Keep what was typed; show where it is now.
          const typed = ta.value;
          state.page = e.data.page;
          closeEditor(false);
          renderPage();
          toast("The file changed on disk, so the page has been reloaded. Your text is on the clipboard.", null, true);
          navigator.clipboard?.writeText(typed).catch(() => {});
        } else err(e.message);
      }
    },
  };

  ta.value = mode === "flow" ? flowText : old;
  ta.addEventListener("input", grow);
  ta.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); editor.save(); }
    else if (e.key === "Escape") { e.preventDefault(); cancelEdit(); }
    else if (e.altKey && e.key.toLowerCase() === "q") { e.preventDefault(); editor.switchMode(mode === "flow" ? "lines" : "flow"); }
  });
  $$(".seg button", ed).forEach(b => b.addEventListener("click", () => editor.switchMode(b.dataset.mode)));
  $(".save", ed).addEventListener("click", () => editor.save());
  $(".cancel", ed).addEventListener("click", cancelEdit);

  el.hidden = true;
  el.after(ed);
  state.editor = editor;
  setMode(mode);
  ta.focus();
  ta.setSelectionRange(0, 0);
  ta.scrollTop = 0;
}

function cancelEdit() {
  if (state.editor && state.editor.dirty() && !confirm("Discard this edit?")) return;
  const el = state.editor && state.editor.el;
  closeEditor();
  if (el) { const i = targets().indexOf(el); if (i >= 0) setFocus(i, false); }
}

function closeEditor(restore = true) {
  const e = state.editor;
  if (!e) return;
  e.ed.remove();
  if (restore) e.el.hidden = false;
  state.editor = null;
}

async function undo() {
  try {
    const r = await api("/api/undo", {});
    setUndo(r.undo);
    toast(`Undid the last save in ${r.path}`);
    if (state.page && state.page.path === r.path && !state.editor) show({ reload: true, keepScroll: true });
  } catch (e) { toast(e.message, null, true); }
}

function setUndo(n) {
  const b = $("#undoBtn");
  b.disabled = !n;
  b.textContent = n ? `Undo (${n})` : "Undo";
}

// ── moving through the prose ────────────────────────────────────────────────

const targets = () => $$("#main .prose, #main .add-doc").filter(el => !el.hidden);

function setFocus(i, scroll = true) {
  const all = targets();
  if (!all.length) return;
  i = Math.max(0, Math.min(all.length - 1, i));
  $$("#main .focus").forEach(el => el.classList.remove("focus"));
  all[i].classList.add("focus");
  state.focus = i;
  if (scroll) all[i].scrollIntoView({ block: "center", behavior: "smooth" });
}

// The block nearest the top of the window, so j starts where you are reading.
function nearest() {
  const all = targets();
  const i = all.findIndex(el => el.getBoundingClientRect().bottom > 70);
  return i < 0 ? all.length - 1 : i;
}

// ── search ──────────────────────────────────────────────────────────────────

let searchTimer, searchSeq = 0;
function search(q) {
  clearTimeout(searchTimer);
  const box = $("#hits");
  if (q.trim().length < 2) { box.hidden = true; return; }
  searchTimer = setTimeout(async () => {
    const seq = ++searchSeq;
    const { hits } = await api("/api/search?q=" + encodeURIComponent(q));
    if (seq !== searchSeq) return;
    const words = q.toLowerCase().split(/\s+/).filter(Boolean).map(w => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const mark = s => esc(s).replace(new RegExp(`(${words.map(esc).join("|")})`, "gi"), "<mark>$1</mark>");
    box.innerHTML = hits.length ? hits.map(h => `<a href="${href(h.path, h.anchor)}">
        <span class="h-title">${mark(h.title)}</span><span class="h-where">${esc(h.path)}:${h.line}</span>
        ${h.snippet ? `<div class="h-snip">${mark(h.snippet)}</div>` : ""}</a>`).join("")
      : `<div class="none">Nothing matches.</div>`;
    box.hidden = false;
  }, 160);
}

// ── telling you things ──────────────────────────────────────────────────────

let toastTimer;
function toast(msg, action, isErr) {
  const t = $("#toast");
  t.className = isErr ? "err" : "";
  t.innerHTML = `<span>${esc(msg)}</span>${action ? `<button>${esc(action.label)}</button>` : ""}`;
  if (action) $("button", t).onclick = () => { t.hidden = true; action.run(); };
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, isErr ? 8000 : 5000);
}

async function showChanges() {
  const { files, undo } = await api("/api/changes");
  setUndo(undo);
  $("#changesBody").innerHTML = files.length ? files.map(f => `<div class="diff"><h3><a href="${href(f.path)}">${esc(f.path)}</a></h3><pre>${
    f.diff.split("\n").slice(2).map(l => `<span class="${l.startsWith("+") ? "d-add" : l.startsWith("-") ? "d-del" : l.startsWith("@@") ? "d-hunk" : ""}">${esc(l) || " "}</span>`).join("")
  }</pre></div>`).join("") : `<p class="empty" style="padding-top:16px">Nothing saved from here yet.</p>`;
  $("#changes").showModal();
}

// The file changed on disk — the IDE saved it, or git did something. Reload
// it, unless an edit is open, in which case say so and leave it.
async function poll() {
  const p = state.page;
  if (!p || document.hidden) return;
  try {
    const { mtime } = await api("/api/stat?path=" + encodeURIComponent(p.path));
    if (mtime === p.mtime || state.page !== p) return;
    if (state.editor) {
      const b = $("#banner");
      b.textContent = "This file changed on disk. Your save will be checked against it as it is now.";
      b.hidden = false;
      return;
    }
    $("#banner").hidden = true;
    await show({ reload: true, keepScroll: true });
  } catch { /* the server stopped; the next poll will say nothing either */ }
}

// ── wiring ──────────────────────────────────────────────────────────────────

document.addEventListener("click", e => {
  const fold = e.target.closest(".fold");
  if (fold) {
    state.unfolded.add(fold.dataset.key);
    const k = +fold.dataset.seg;
    fold.outerHTML = foldRun(k, +fold.dataset.key).map(l => lineHTML(l, k)).join("");
    return;
  }
  const add = e.target.closest(".add-doc");
  if (add) return openEditor(add);
  const prose = e.target.closest("#main .prose");
  if (prose && !e.target.closest("a") && !String(window.getSelection())) return openEditor(prose);
  if (!e.target.closest(".search")) $("#hits").hidden = true;
  if (e.target.closest("[data-close]")) e.target.closest("dialog").close();
});

document.addEventListener("keydown", e => {
  const typing = e.target.closest("input, textarea, select, [contenteditable]");
  if (typing || e.ctrlKey || e.metaKey || e.altKey || $("dialog[open]")) {
    if (e.key === "Escape" && e.target.id === "q") { e.target.blur(); $("#hits").hidden = true; }
    if (e.target.id === "q" && (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "Enter")) {
      const hits = $$("#hits a");
      if (!hits.length) return;
      let i = hits.findIndex(a => a.classList.contains("sel"));
      if (e.key === "Enter") { (hits[i] || hits[0]).click(); $("#hits").hidden = true; e.target.blur(); return; }
      e.preventDefault();
      hits.forEach(a => a.classList.remove("sel"));
      i = e.key === "ArrowDown" ? Math.min(hits.length - 1, i + 1) : Math.max(0, i - 1);
      hits[i].classList.add("sel");
      hits[i].scrollIntoView({ block: "nearest" });
    }
    return;
  }
  const all = targets();
  switch (e.key) {
    case "/": e.preventDefault(); $("#q").focus(); $("#q").select(); break;
    case "j": setFocus(state.focus < 0 ? nearest() : state.focus + 1); break;
    case "k": setFocus(state.focus < 0 ? nearest() : state.focus - 1); break;
    case "Enter": case "e": if (all[state.focus]) { e.preventDefault(); openEditor(all[state.focus]); } break;
    case "c": $("#showCode").click(); break;
    case "?": $("#help").showModal(); break;
    case "[": case "]": {
      const files = state.index.files, i = files.findIndex(f => state.page && f.path === state.page.path);
      const f = files[e.key === "]" ? i + 1 : i - 1];
      if (f) location.hash = href(f.path);
      break;
    }
  }
});

window.addEventListener("beforeunload", e => {
  if (state.editor && state.editor.dirty()) e.preventDefault();
});

async function main() {
  $("#showCode").checked = state.showCode;
  $("#showCode").addEventListener("change", e => {
    state.showCode = e.target.checked;
    store("showCode", state.showCode ? "1" : "0");
    if (state.page && !state.editor) renderPage();
  });
  $("#q").addEventListener("input", e => search(e.target.value));
  $("#q").addEventListener("focus", e => { if (e.target.value.trim().length > 1 && $("#hits").innerHTML) $("#hits").hidden = false; });
  $("#filter").addEventListener("input", e => renderTree(e.target.value));
  $("#undoBtn").addEventListener("click", undo);
  $("#changesBtn").addEventListener("click", showChanges);
  $("#helpBtn").addEventListener("click", () => $("#help").showModal());
  $("#navToggle").addEventListener("click", () => document.body.classList.toggle("nav-open"));
  try {
    state.index = await api("/api/index");
  } catch (e) {
    $("#main").innerHTML = `<p class="empty">Could not reach the server: ${esc(e.message)}. Restart it, and open the address it prints.</p>`;
    return;
  }
  renderTree();
  await show();
  setInterval(poll, 2000);
}

main();
