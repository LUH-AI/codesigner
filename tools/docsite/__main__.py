"""The docsite serves comments as pages editable in place.

The server listens on 127.0.0.1 only, and each API call needs a per-run token
from the page.
"""

from __future__ import annotations

import argparse
import difflib
import json
import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import pages
from . import source as S

STATIC = pages.ROOT / "tools" / "docsite" / "static"
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.Lock()

#: Saves this run as (path, before, after).
HISTORY: list[tuple[str, str, str]] = []
#: Files before this run's first write.
ORIGINALS: dict[str, str] = {}


def _pygments_css():
    try:
        from pygments.formatters import HtmlFormatter
    except ImportError:
        return ""
    def tokens(style):  # token colours only
        css = HtmlFormatter(style=style).get_style_defs(".code")
        return "\n".join(l for l in css.split("\n") if l.startswith(".code .") and ".hll" not in l)
    return f"{tokens('default')}\n@media (prefers-color-scheme: dark) {{\n{tokens('github-dark')}\n}}\n"


class Handler(BaseHTTPRequestHandler):
    server_version = "docsite"

    def log_message(self, fmt, *args):
        pass

    # plumbing

    def _send(self, status, body: bytes, ctype: str):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status=HTTPStatus.OK):
        self._send(status, json.dumps(data).encode(), "application/json")

    def _allowed(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):  # DNS-rebinding
            self._send(HTTPStatus.FORBIDDEN, b"wrong host", "text/plain")
            return False
        if self.path.startswith("/api/") and self.headers.get("X-Docsite-Token") != TOKEN:
            self._send(HTTPStatus.FORBIDDEN, b"bad token", "text/plain")
            return False
        return True

    def _path(self, path):
        if path not in set(pages.files()):
            raise S.EditError(f"not a file this site shows: {path}")
        return path

    # routes

    def do_GET(self):
        if not self._allowed():
            return
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                body = (STATIC / "index.html").read_text().replace("{{TOKEN}}", TOKEN)
                return self._send(HTTPStatus.OK, body.encode(), "text/html; charset=utf-8")
            if url.path == "/pygments.css":
                return self._send(HTTPStatus.OK, _pygments_css().encode(), "text/css")
            if url.path in ("/app.js", "/style.css"):
                ctype = "text/javascript" if url.path.endswith(".js") else "text/css"
                return self._send(HTTPStatus.OK, (STATIC / url.path[1:]).read_bytes(), ctype)
            if url.path == "/api/index":
                return self._json(pages.index())
            if url.path == "/api/page":
                return self._json(pages.page(self._path(q["path"])))
            if url.path == "/api/stat":
                return self._json({"mtime": pages.mtime(self._path(q["path"]))})
            if url.path == "/api/search":
                return self._json({"hits": pages.search(q.get("q", ""))})
            if url.path == "/api/changes":
                return self._json(self._changes())
        except S.EditError as e:
            return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError:
            return self._json({"error": "no such file"}, HTTPStatus.NOT_FOUND)
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self):
        if not self._allowed():
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        route = urlparse(self.path).path
        try:
            if route == "/api/edit":
                return self._json(self._edit(body))
            if route == "/api/undo":
                return self._json(self._undo())
            if route == "/api/reflow":
                return self._json({"text": S.reflow(body["old"], body["text"], body["width"],
                                                    body.get("first", 0), body.get("last", 0))})
            if route == "/api/flow":
                return self._json({"text": S.flow(body["text"])})
        except S.Conflict as e:
            page = pages.page(body["path"]) if body.get("path") else None
            return self._json({"error": str(e), "page": page}, HTTPStatus.CONFLICT)
        except S.EditError as e:
            return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    # writing

    def _edit(self, body):
        path = self._path(body["path"])
        with LOCK:
            before = (pages.ROOT / path).read_text(encoding="utf-8")
            if body["op"] == "add":
                after, line = S.add_docstring(before, body["qualname"], body["start"],
                                              body["text"], body.get("flowed", True))
            else:
                after, line = S.replace_region(before, body["start"], body["raw"],
                                               body["text"], body.get("flowed", True))
            if after != before:
                _write(path, after)
                ORIGINALS.setdefault(path, before)
                HISTORY.append((path, before, after))
                print(f"  saved {path}:{line}", flush=True)
        return {"page": pages.page(path), "line": line, "changed": after != before,
                "undo": len(HISTORY)}

    def _undo(self):
        with LOCK:
            if not HISTORY:
                raise S.EditError("nothing to undo")
            path, before, after = HISTORY[-1]
            now = (pages.ROOT / path).read_text(encoding="utf-8")
            if now != after:
                raise S.EditError(f"{path} has changed since that save, so undoing it "
                                  "would lose those changes — undo it in the editor")
            HISTORY.pop()
            _write(path, before)
            print(f"  undid {path}", flush=True)
        return {"path": path, "undo": len(HISTORY)}

    def _changes(self):
        out = []
        for path, original in ORIGINALS.items():
            now = (pages.ROOT / path).read_text(encoding="utf-8")
            diff = "".join(difflib.unified_diff(
                original.splitlines(True), now.splitlines(True),
                f"a/{path}", f"b/{path}", n=2))
            if diff:
                out.append({"path": path, "diff": diff})
        return {"files": out, "undo": len(HISTORY)}


def _write(path, text):
    with open(pages.ROOT / path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(prog="python -m tools.docsite", description=__doc__.split("\n")[0])
    ap.add_argument("port", type=int, nargs="?", default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"docsite on {url} — saves go straight into the .py files. Ctrl+C to stop.", flush=True)
    if not args.no_open:
        threading.Timer(0.3, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
