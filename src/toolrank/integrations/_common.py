"""What the platform integrations share: reading SDK objects and dicts alike, the tool names in a
tool choice, the read-only rule for approvals, and a tool's output as text a model reads."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

RESULT_CHARS = 25_000  # relayed to the model per call, like the proxy's OpenAPI bodies

Approve = Callable[[dict[str, Any], dict[str, Any]], bool]  # (catalogue entry, arguments) -> run it?
OnEvent = Callable[[str, dict[str, Any]], None]  # ("search" | "call" | "turn", details)


def get(obj: Any, key: str, default: Any = None) -> Any:
    """``obj[key]`` for a dict, ``obj.key`` for an SDK object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def names_in(value: Any) -> set[str]:
    """Every string under a ``name`` key, however deep (tool choices, allowed-tool lists)."""
    if isinstance(value, dict):
        out = {value["name"]} if isinstance(value.get("name"), str) else set()
        for v in value.values():
            out |= names_in(v)
        return out
    if isinstance(value, list):
        return set().union(*(names_in(v) for v in value))
    return set()


def read_only(entry: dict[str, Any]) -> bool:
    """Whether a catalogue entry changes nothing: a GET/HEAD operation, or an MCP tool annotated
    ``readOnlyHint``. Examples ask before any other call, since a tool result can steer a model."""
    if entry.get("kind") == "openapi":
        return entry.get("method") in ("GET", "HEAD")
    return bool((entry.get("annotations") or {}).get("readOnlyHint"))


def capped(text: str, limit: int = RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated: {len(text)} characters, {limit} shown]"


def as_text(content: list[dict[str, Any]]) -> str:
    """MCP content blocks as one text: text as is, anything else as JSON."""
    parts = [c["text"] if c.get("type") == "text" else json.dumps(c, ensure_ascii=False) for c in content]
    return capped("\n".join(p for p in parts if p))
