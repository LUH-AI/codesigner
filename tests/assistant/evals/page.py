"""A browser stand-in for the evals: it renders Codesigner's pages with the test
client, reports them as assistant/static/assistant/act.js would, and carries out
the assistant's actions on them.

What a page reports follows act.js: named inputs, selects and textareas outside
the chat, except hidden, file, password and button kinds; radio buttons as one
field with options; each element marked `data-assist` as a target. No script
runs here, so a field that a page's script would add or show is missing, and
every field counts as visible.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser

from django.urls import Resolver404, resolve
from django.utils import timezone

SKIP = {"hidden", "file", "password", "submit", "button", "reset", "image"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
        "track", "wbr"}


class _Reader(HTMLParser):
    """`_Reader` collects a page's title, fields, labels, targets and the
    planned fill from its HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.title = ""
        self.fields = []
        self.labels = {}
        self.targets = []
        self.fill = []
        self._label_for = None
        self._label_text = ""
        self._open_text = []
        self._select = None
        self._option = None
        self._textarea = None
        self._script = None
        self._in_title = False

    def _inside_chat(self):
        return any("data-assistant" in attrs for _, attrs in self.stack)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag not in VOID:
            self.stack.append((tag, attrs))
        if tag == "title":
            self._in_title = True
        if tag == "label":
            self._label_for, self._label_text = attrs.get("for"), ""
        if "data-assist" in attrs:
            self._open_text.append([len(self.stack), attrs, ""])
        if tag == "script" and attrs.get("id") == "assistant-fill":
            self._script = ""
        if self._inside_chat() and tag != "script":
            return
        name = attrs.get("name")
        if tag == "input" and name and attrs.get("type", "text") not in SKIP:
            self._input(attrs)
        elif tag == "select" and name:
            self._select = {"name": name, "id": attrs.get("id"), "label": attrs.get("aria-label", ""),
                            "type": "select-one", "fillable": True, "visible": True,
                            "options": [], "value": None}
        elif tag == "option" and self._select is not None:
            self._option = [attrs.get("value"), "", "selected" in attrs]
        elif tag == "textarea" and name:
            self._textarea = {"name": name, "id": attrs.get("id"), "label": attrs.get("aria-label", ""),
                              "type": "textarea", "fillable": True, "visible": True, "value": ""}

    def _input(self, attrs):
        kind = attrs.get("type", "text")
        name = attrs["name"]
        if kind == "radio":
            group = next((f for f in self.fields if f["name"] == name and f["type"] == "radio"), None)
            if group is None:
                group = {"name": name, "id": None, "label": attrs.get("aria-label", ""), "type": "radio",
                         "fillable": True, "visible": True, "options": [], "value": ""}
                self.fields.append(group)
            group["options"].append([attrs.get("value", "on"), attrs.get("id")])
            if "checked" in attrs:
                group["value"] = attrs.get("value", "on")
            return
        field = {"name": name, "id": attrs.get("id"), "label": attrs.get("aria-label", ""),
                 "type": kind, "fillable": True, "visible": True}
        field["value"] = "checked" in attrs if kind == "checkbox" else attrs.get("value", "")
        self.fields.append(field)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.stack and self.stack[-1][0] == tag:
            self.handle_endtag(tag)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._label_for is not None:
            self._label_text += data
        for item in self._open_text:
            item[2] += data
        if self._option is not None:
            self._option[1] += data
        if self._textarea is not None:
            self._textarea["value"] += data
        if self._script is not None:
            self._script += data

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag == "label" and self._label_for:
            self.labels[self._label_for] = " ".join(self._label_text.split())
            self._label_for = None
        if tag == "option" and self._option is not None and self._select is not None:
            value, text, selected = self._option
            text = " ".join(text.split())
            self._select["options"].append([text if value is None else value, text])
            if selected:
                self._select["value"] = self._select["options"][-1][0]
            self._option = None
        if tag == "select" and self._select is not None:
            if self._select["value"] is None:
                options = self._select["options"]
                self._select["value"] = options[0][0] if options else ""
            self.fields.append(self._select)
            self._select = None
        if tag == "textarea" and self._textarea is not None:
            self.fields.append(self._textarea)
            self._textarea = None
        if tag == "script" and self._script is not None:
            self.fill = json.loads(self._script or "[]")
            self._script = None
        if tag in VOID:
            return
        for depth in range(len(self.stack) - 1, -1, -1):
            if self.stack[depth][0] == tag:
                while self._open_text and self._open_text[-1][0] > depth:
                    _, attrs, text = self._open_text.pop()
                    label = attrs.get("aria-label") or " ".join(text.split())
                    self.targets.append({"name": attrs["data-assist"], "label": label[:80]})
                del self.stack[depth:]
                break

    def report(self):
        """`report` returns the fields with their labels, as act.js sends them."""
        out = []
        for field in self.fields:
            field = dict(field)
            if not field["label"] and field.get("id"):
                field["label"] = self.labels.get(field["id"], "")
            if field["type"] == "radio":
                field["options"] = [[value, self.labels.get(rid or "", value)]
                                    for value, rid in field["options"]]
            field.pop("id", None)
            out.append(field)
        return out


class SimulatedPage:
    """`SimulatedPage` is one browser session: the page it has open, kept in
    `BrowserPage` as act.js would keep it, and the actions it carries out."""

    def __init__(self, client, browser: str):
        self.client = client
        self.browser = browser
        self.path = ""
        self.status = 0
        self.title = ""
        self.fields = []
        self.targets = []
        #: Every action carried out, as `(kind, args, done, detail)`.
        self.log = []

    def open(self, url: str) -> int:
        """`open` renders *url*, applies the plan's fill as the page script
        would, reports the page, and returns the status code."""
        response = self.client.get(url, follow=True)
        reader = _Reader()
        reader.feed(response.content.decode("utf-8", "replace"))
        self.path = response.request["PATH_INFO"]
        self.status = response.status_code
        self.title = " ".join(reader.title.split())
        self.fields = reader.report()
        self.targets = reader.targets
        for item in reader.fill:
            self._set(item["name"], item["value"])
        self._report()
        return response.status_code

    def _report(self):
        from assistant.models import BrowserPage

        url_name, experiment_id = "", None
        try:
            match = resolve(self.path)
            url_name = match.view_name
            if getattr(match.func, "experiment_action", None):
                experiment_id = match.kwargs.get("pk")
        except Resolver404:
            pass
        BrowserPage.objects.update_or_create(
            browser=self.browser,
            defaults={"path": self.path, "url_name": url_name, "experiment_id": experiment_id,
                      "title": self.title[:300], "fields": self.fields, "targets": self.targets,
                      "seen_at": timezone.now()})

    def _set(self, name, value) -> bool:
        field = next((f for f in self.fields if f["name"] == name), None)
        if field is None:
            return False
        if field["type"] in ("radio", "select-one"):
            if str(value) not in [str(o[0]) for o in field["options"]]:
                return False
            field["value"] = str(value)
        elif field["type"] == "checkbox":
            field["value"] = value in (True, "true", "on", 1)
        else:
            field["value"] = "" if value is None else str(value)
        return True

    def value(self, name):
        """`value` returns field *name*'s current value, or None without it."""
        field = next((f for f in self.fields if f["name"] == name), None)
        return None if field is None else field["value"]

    def carry_out(self, act) -> None:
        """`carry_out` does *act* as act.js would, and records how it went."""
        args = act.args or {}
        if act.path and act.path != self.path:
            done, detail = False, f"The person is on {self.path} now, not {act.path}."
        elif act.kind == "navigate":
            done, detail = True, f"Opened {args.get('url')}."
            self.open(args["url"])
        elif act.kind == "fill":
            values = args.get("values") or {}
            missed = [name for name, value in values.items() if not self._set(name, value)]
            self._report()
            done = not missed
            detail = (f"Not on the page: {', '.join(missed)}." if missed
                      else f"Filled {', '.join(values)}.")
        elif act.kind == "highlight":
            target = args.get("target")
            done = (any(t["name"] == target for t in self.targets)
                    or any(f["name"] == target for f in self.fields))
            detail = "" if done else f"Nothing called {target} is on the page."
        else:
            done, detail = False, "Not something this page does."
        act.status = act.DONE if done else act.FAILED
        act.detail = detail
        act.save(update_fields=["status", "detail"])
        self.log.append((act.kind, args, done, detail))
