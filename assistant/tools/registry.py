"""The design follows A3S (`a3s/tools/registry.py`,
https://github.com/LUH-AI/A3S, BSD-3-Clause, Copyright (c) Marcel Wever).
`ToolRegistry.execute` checks arguments against the schema and records each call
as `AgentAction`. A handler receives the gateway instead of a session and so
reaches the application only as the person.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class ToolResult:

    output: str
    #: The turn waits for the person's answer.
    waits_for_person: bool = False
    #: The page also shows `show` (a question's choices).
    show: dict = field(default_factory=dict)
    #: The call's experiment, for `AgentAction`.
    experiment_id: int | None = None


class Refused(Exception):
    """Handlers raise `Refused` when the person lacks permission or has signed
    out. `ToolRegistry.execute` passes the message to the model."""


@dataclass
class ToolSpec:
    """`ToolSpec.handler` receives the turn's context, then the arguments as
    keywords."""

    name: str
    description: str
    parameters: dict
    handler: Callable[..., ToolResult]

    @property
    def schema(self) -> dict:
        """`schema` uses OpenAI's format."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


class ToolRegistry:
    """`ToolRegistry` offers one turn's tools."""

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
        """`execute` runs tool *name* and records the call as `AgentAction`.

        An unknown tool, invalid arguments, `Refused` and handler exceptions
        become the result text, so the model reads the reason and can retry.
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
    """`invalid` returns why *value* fails *schema*, or "" when *value* fits.

    An object key missing from `properties` fails unless `additionalProperties`
    is true or a schema the value fits. Thus a model cannot slip in an argument
    a handler would accept unchecked.
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
        extra = schema.get("additionalProperties")
        for name, item in value.items():
            path = f"{where}.{name}" if where else name
            if name not in properties:
                if extra is True:
                    continue
                if isinstance(extra, dict):
                    problem = invalid(item, extra, path)
                    if problem:
                        return problem
                    continue
                return (f"{path} is not one of {where}'s fields" if where
                        else f"{name} is not an argument of this tool")
            problem = invalid(item, properties[name], path)
            if problem:
                return problem
    if kind == "array" and "items" in schema:
        for index, item in enumerate(value):
            problem = invalid(item, schema["items"], f"{where}[{index}]")
            if problem:
                return problem
    return ""
