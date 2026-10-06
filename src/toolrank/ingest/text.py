"""The one text an ingested tool is indexed by: the shape Phase 0 measured.

``{"server", "name", "description", "inputSchema"}`` as JSON, server first: the MCP-Zero and
LiveMCPBench ablation where the server name added ~8 top-1 points (``docs/reports/faz0-week4.md``);
it is what the ``documentation`` tool format returns for an ingested tool. A text over the
character budget loses schema detail first and description text second, never its server or
name, so the text is the same whatever serves the embeddings: vLLM 0.13's
``truncate_prompt_tokens`` happens to keep the first tokens (``scripts/truncation_side.py``),
other endpoints may cut elsewhere or refuse. MCP tools rarely come near the budget (LiveMCPBench:
p99 3.9K characters); large OpenAPI request bodies do (Stripe: 61 of 612 operations), and so do
pydantic-made schemas (FastMCP servers) whose models sit in ``$defs``. When no schema step is small
enough, the schema goes before the description is cut: what the tool does matters more to retrieval
than how it is called.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

MAX_CHARS = 6000
_KEEP_TOP = 8  # top-level properties kept when even a flat schema is over budget
_NESTED = ("anyOf", "oneOf", "allOf")  # lists of schemas
_MAPS = ("properties", "patternProperties", "$defs", "definitions")  # name -> schema
_ONE = ("items", "additionalProperties")  # a schema (or a list of them, or a bool)
_DEFS = ("$defs", "definitions")


def _nested(schema: dict[str, Any], out: dict[str, Any], fn: Any) -> dict[str, Any]:
    """``out`` with ``fn`` applied to every subschema of ``schema`` one level down."""
    for key in _MAPS:
        if isinstance(schema.get(key), dict):
            out[key] = {k: fn(v) for k, v in schema[key].items()}
    for key in _ONE:
        if isinstance(schema.get(key), dict):
            out[key] = fn(schema[key])
        elif isinstance(schema.get(key), list):
            out[key] = [fn(v) for v in schema[key]]
    for key in _NESTED:
        if isinstance(schema.get(key), list):
            out[key] = [fn(s) for s in schema[key]]
    return out


def _dump(doc: dict[str, Any]) -> str:
    return json.dumps(doc, ensure_ascii=False)


def _without_meta_schema(schema: Any) -> Any:
    """Drop ``$schema`` keys (``"http://json-schema.org/draft-07/schema#"`` on every tool of many
    TypeScript servers): tokens without meaning for retrieval."""
    if isinstance(schema, dict):
        return {k: _without_meta_schema(v) for k, v in schema.items() if k != "$schema"}
    if isinstance(schema, list):
        return [_without_meta_schema(v) for v in schema]
    return schema


def _cap_depth(schema: Any, depth: int) -> Any:
    """Below ``depth`` levels, object and array schemas keep only their type and description."""
    if not isinstance(schema, dict):
        return schema
    if depth <= 0:
        kept = {k: schema[k] for k in ("type", "description", "$ref") if k in schema}
        return kept or {"type": "object"}
    return _nested(schema, dict(schema), lambda s: _cap_depth(s, depth - 1))


def _drop_descriptions(schema: Any, keep_level: int, level: int = 0) -> Any:
    """Descriptions deeper than ``keep_level`` go (0 = the input schema, 1 = its parameters)."""
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items() if k != "description" or level <= keep_level}
    return _nested(schema, out, lambda s: _drop_descriptions(s, keep_level, level + 1))


def _first_properties(schema: Any, keep: int) -> Any:
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict) or len(props) <= keep:
        return schema
    names = list(props)[:keep]
    out = {**schema, "properties": {n: props[n] for n in names}, "…": f"{len(props) - keep} more properties"}
    if "required" in schema:
        out["required"] = [n for n in schema["required"] or [] if n in names]
    return out


def _without_defs(schema: Any) -> Any:
    """The schema without its ``$defs`` / ``definitions``: the ``$ref`` names stay as the hint."""
    if not isinstance(schema, dict):
        return schema
    return {k: v for k, v in schema.items() if k not in _DEFS}


def _names_only(schema: Any, keep: int) -> Any:
    """The first ``keep`` top-level parameters with their types and nothing else."""
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return {"type": "object"}
    names = list(props)[:keep]
    out: dict[str, Any] = {"type": "object", "properties": {}}
    for n in names:
        t = props[n].get("type") if isinstance(props[n], dict) else None
        out["properties"][n] = {"type": t} if isinstance(t, str) else {}
    if len(props) > keep:
        out["…"] = f"{len(props) - keep} more properties"
    return out


def _smaller_schemas(schema: Any) -> Iterator[Any]:
    """Ever smaller versions of ``schema``: depth 4 → 1, then descriptions below the parameters
    dropped, then the parameters' own, then only the first top-level properties, then those without
    the ``$defs``, and last their names and types only."""
    for depth in (4, 3, 2, 1):
        yield _cap_depth(schema, depth)
    flat = _cap_depth(schema, 1)
    yield _drop_descriptions(flat, keep_level=1)
    yield _drop_descriptions(flat, keep_level=0)
    first = _first_properties(_drop_descriptions(flat, keep_level=0), _KEEP_TOP)
    yield first
    yield _without_defs(first)
    yield _names_only(first, _KEEP_TOP)


def shrink_schema(schema: Any, max_chars: int) -> tuple[Any, bool]:
    """``schema`` within ``max_chars`` of JSON (the same steps as ``fit_text``); -> (schema, shrunk)."""
    schema = _without_meta_schema(schema)
    if len(_dump(schema)) <= max_chars:
        return schema, False
    smaller = schema
    for smaller in _smaller_schemas(schema):
        if len(_dump(smaller)) <= max_chars:
            break
    return smaller, True


def fit_text(
    server: str,
    name: str,
    description: str,
    input_schema: dict[str, Any] | None = None,
    *,
    max_chars: int | None = MAX_CHARS,
) -> tuple[str, bool]:
    """-> (text, whether the tool had to be shrunk to fit ``max_chars``)."""
    doc: dict[str, Any] = {"server": server, "name": name, "description": description}
    if input_schema is not None:
        input_schema = doc["inputSchema"] = _without_meta_schema(input_schema)
    text = _dump(doc)
    if max_chars is None or len(text) <= max_chars:
        return text, False
    if input_schema is not None:
        for smaller in _smaller_schemas(input_schema):
            doc["inputSchema"] = smaller
            text = _dump(doc)
            if len(text) <= max_chars:
                return text, True
        del doc["inputSchema"]  # the description says more about what the tool is for
        text = _dump(doc)
        if len(text) <= max_chars:
            return text, True
    while len(text) > max_chars and doc["description"]:  # JSON escapes make lengths approximate
        keep = max(len(doc["description"]) - (len(text) - max_chars) - 1, 0)
        doc["description"] = doc["description"][:keep].rstrip() + "…" if keep else ""
        text = _dump(doc)
    return text, True


def tool_text(
    server: str,
    name: str,
    description: str,
    input_schema: dict[str, Any] | None = None,
    *,
    max_chars: int | None = MAX_CHARS,
) -> str:
    """JSON of server, name, description and (when given) the input schema, within ``max_chars``."""
    return fit_text(server, name, description, input_schema, max_chars=max_chars)[0]
