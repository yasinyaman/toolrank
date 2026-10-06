"""Would an agent API take this input schema? (backlog D2.7)

Claude's Messages API checks every tool's ``input_schema`` against JSON Schema 2020-12 and refuses
the whole request over one that fails, and OpenAI refuses function parameters it cannot read. MCP
servers and OpenAPI specs often carry draft-04 and draft-07 habits (``"items": [...]``,
``"exclusiveMinimum": true``) or ``$ref``s that point nowhere. ``problems`` lists what is wrong
with a schema: its root is not an object, a ``$ref`` does not resolve inside it, or it fails the
2020-12 meta-schema (checked with ``jsonschema`` when it is importable, as it is with the ``[mcp]``
extra; without it, the common draft-04/07 habits are checked by hand). Ingest records the problems
in ``Tool.doc["schema_problems"]`` and says so; the platform records (``/v1/tools?full=true``,
``/v1/search``) then carry ``{"type": "object"}`` in its place, so one bad schema costs the agent
that tool's argument hints, not the conversation. MCP's ``search_tools`` still shows it as it is.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import Any

PROBLEMS_KEY = "schema_problems"
_DRAFT_HABITS = ("exclusiveMinimum", "exclusiveMaximum")
_DATA = frozenset({"enum", "const", "default", "examples", "example"})  # values, not schemas
_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})


def problems(schema: Any) -> list[str]:
    """What an agent API would refuse in ``schema`` (a tool's input schema); [] when nothing."""
    if schema is None:
        return []
    if not isinstance(schema, dict):
        return [f"the schema is a {type(schema).__name__}, not an object"]
    schema = {k: v for k, v in schema.items() if k != "$schema"}  # the platform records drop it too
    found: list[str] = []
    root = schema.get("type", "object")
    if root != "object":
        found.append(f"the root is {root!r}, not an object")
    found += [f"$ref {ref!r} {why}" for ref, why in _bad_refs(schema)]
    found += _meta(schema)
    return found


def _walk(node: Any, path: str = "#", *, names: bool = False) -> Iterator[tuple[str, Any]]:
    """Every subschema with its JSON pointer; ``names``: ``node`` maps names to schemas (a
    ``properties`` object), so its own keys are no keywords. Example values are not schemas."""
    if not names:
        yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            if not names and k in _DATA:
                continue
            yield from _walk(v, f"{path}/{k}", names=not names and k in _MAPS)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}/{i}")


def _resolve(schema: dict[str, Any], ref: str) -> bool:
    node: Any = schema
    for part in ref[2:].split("/") if ref != "#" else []:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return False
    return True


def _bad_refs(schema: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for _, node in _walk(schema):
        ref = node.get("$ref") if isinstance(node, dict) else None
        if not isinstance(ref, str):
            continue
        if not ref.startswith("#"):
            out.append((ref, "points outside the schema"))
        elif not _resolve(schema, ref):
            out.append((ref, "points nowhere"))
    return out


@functools.cache
def _validator() -> Any:
    """The 2020-12 meta-schema's validator, built once; None without ``jsonschema``."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return None
    return Draft202012Validator(Draft202012Validator.META_SCHEMA)


def _meta(schema: dict[str, Any]) -> list[str]:
    validator = _validator()
    if validator is None:
        return _habits(schema)
    from jsonschema.exceptions import best_match

    error = best_match(validator.iter_errors(schema))
    if error is None:
        return []
    where = "/".join(str(p) for p in error.absolute_path)
    return [f"not JSON Schema 2020-12 at #/{where}: {error.message[:200]}"]


def _habits(schema: dict[str, Any]) -> list[str]:
    """The meta-schema failures seen most, without ``jsonschema``."""
    out = []
    for path, node in _walk(schema):
        if not isinstance(node, dict):
            continue
        if isinstance(node.get("items"), list):
            out.append(f"not JSON Schema 2020-12 at {path}: 'items' is a list (use prefixItems)")
        for key in _DRAFT_HABITS:
            if isinstance(node.get(key), bool):
                out.append(f"not JSON Schema 2020-12 at {path}: {key!r} is a boolean (draft-04)")
        if "required" in node and not (
            isinstance(node["required"], list) and all(isinstance(x, str) for x in node["required"])
        ):
            out.append(f"not JSON Schema 2020-12 at {path}: 'required' is not a list of names")
        if "type" in node and not isinstance(node["type"], str | list):
            out.append(f"not JSON Schema 2020-12 at {path}: 'type' is not a name")
    return out[:3]
