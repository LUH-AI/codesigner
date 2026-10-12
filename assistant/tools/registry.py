"""Tools: what they are called, what they take, and calling one.

The shape is A3S's (`a3s/tools/registry.py`, https://github.com/LUH-AI/A3S,
BSD-3-Clause, Copyright (c) Marcel Wever): a spec with a JSON-schema for its
arguments and a handler, a result that can hand control back to the person, and
a registry that offers the schemas to the model and dispatches its calls. What
is new is that arguments are checked against the schema before a handler sees
them, every call is recorded (`AgentAction`), and a handler is given the
gateway rather than a session — it reaches the application as the person, or
not at all.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """What a tool answers: text for the model, and what the turn should do."""

    output: str
    #: The tool asked the person something and the turn waits for the answer.
    waits_for_person: bool = False
    #: Shown on the page as well as given to the model (a question's choices).
    show: dict = field(default_factory=dict)
    #: The experiment the call touched, for the audit.
    experiment_id: int | None = None


class Refused(Exception):
    """The person this assistant acts for may not do what was asked, or is no
    longer signed in to be acted for. Its message is said to the model."""


@dataclass
class ToolSpec:
    """One tool: its name, what it does, its arguments' JSON schema, and the
    function that does it — called with the turn's context and the arguments."""

    name: str
    description: str
    parameters: dict
    handler: Callable[..., ToolResult]

    @property
    def schema(self) -> dict:
        """The tool in the OpenAI wire format."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


class ToolRegistry:
    """The tools offered for one turn, in order."""

    def __init__(self, specs=()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def schemas(self) -> list[dict]:
        return [spec.schema for spec in self._tools.values()]

    def execute(self, name: str, arguments: dict, context) -> ToolResult:
        """Run tool *name* with *arguments* for *context*'s conversation, and
        record that it was called and what came of it.

        A tool that does not exist, arguments its schema does not allow, a
        refusal and a failure are all answers to the model rather than errors:
        it reads why and can try something else.
        """
        from ..models import AgentAction

        started = time.monotonic()
        spec = self._tools.get(name)
        outcome, detail, result = AgentAction.OK, "", None
        if spec is None:
            outcome, detail = AgentAction.INVALID, f"There is no tool called {name!r}."
        else:
            problem = invalid(arguments, spec.parameters)
            if problem:
                outcome, detail = AgentAction.INVALID, f"Invalid arguments for {name}: {problem}"
            else:
                try:
                    result = spec.handler(context, **arguments)
                except Refused as refusal:
                    outcome, detail = AgentAction.REFUSED, f"Refused: {refusal}"
                except Exception as exc:  # noqa: BLE001 — the model is told, the turn goes on
                    log.exception("tool %s failed", name)
                    outcome, detail = AgentAction.FAILED, f"{name} failed: {exc}"
        if result is None:
            result = ToolResult(output=detail)
        AgentAction.objects.create(
            conversation=context.conversation, user_id=context.conversation.owner_id,
            tool=name, arguments=arguments if isinstance(arguments, dict) else {},
            outcome=outcome, detail=detail or result.output[:2000],
            experiment_id=result.experiment_id,
            duration_ms=int((time.monotonic() - started) * 1000))
        return result


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "array": list, "object": dict}


def invalid(value, schema: dict, where: str = "") -> str:
    """Why *value* does not fit *schema*, or "" if it does.

    The subset of JSON schema tools here are written in: objects with
    properties, required names and nothing else; arrays of items; the plain
    types; enums. Anything a schema leaves out of an object is refused, so a
    model cannot slip in an argument a handler would quietly take — unless the
    object says `additionalProperties`, for a map of names the handler checks
    itself.
    """
    kind = schema.get("type")
    if kind:
        expected = _TYPES[kind]
        if isinstance(value, bool) and kind in ("integer", "number"):
            return f"{where or 'value'} must be a {kind}"
        if not isinstance(value, expected):
            return f"{where or 'value'} must be a{'n' if kind[0] in 'aeiou' else ''} {kind}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where or 'value'} must be one of {schema['enum']}"
    if kind == "object":
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                return f"{name} is required"
        for name, item in value.items():
            if name not in properties:
                if schema.get("additionalProperties") is True:
                    continue
                return f"{name} is not an argument of this tool"
            problem = invalid(item, properties[name], name)
            if problem:
                return problem
    if kind == "array" and "items" in schema:
        for index, item in enumerate(value):
            problem = invalid(item, schema["items"], f"{where}[{index}]")
            if problem:
                return problem
    return ""
