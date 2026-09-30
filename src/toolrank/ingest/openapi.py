"""OpenAPI 3.x specs as tool sources: one operation, one tool.

A tool is what an agent would call. Its name is the ``operationId`` (else ``METHOD_path``), its
description summary + description, and its ``inputSchema`` one flat object: path, query, header
and cookie parameters plus the request body's top-level properties. The body's media type is
JSON if offered, else form (Stripe's bodies are all form-encoded), else multipart, else the first
listed; ``allOf`` is merged and top-level ``oneOf``/``anyOf`` alternatives are unioned. A
parameter that clashes with a body property becomes ``name__<in>`` (FastMCP's convention).
Local ``$ref``s are inlined per operation with a cycle guard and a depth cap; external ones become
a stub and are counted; examples, ``xml``, ``externalDocs`` and ``x-`` extensions are dropped and
HTML tags stripped from descriptions. How to call the operation (method, path, base URL, where
each argument goes, body media type) sits in ``Tool.doc["http"]`` for the proxy, outside the
indexed text. Swagger 2.0 is refused: convert it to OpenAPI 3 first.
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from toolrank.domain import Tool
from toolrank.ingest.text import MAX_CHARS, fit_text

HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
SKIP_HEADERS = {"accept", "content-type", "authorization"}
MEDIA_ORDER = ("application/json", "application/x-www-form-urlencoded", "multipart/form-data")
MAX_DEPTH = 8
_DROP = {"example", "examples", "xml", "externalDocs", "discriminator"}
_SCHEMA_MAPS = ("properties", "patternProperties")
_SCHEMA_ONE = ("items", "additionalProperties", "not")
_SCHEMA_LISTS = ("allOf", "anyOf", "oneOf")


def _plain(text: str) -> str:
    """Description text without HTML tags or entities (Stripe writes ``<p>…</p>``)."""
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _yaml(raw: str) -> Any:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("YAML specs need PyYAML: pip install 'toolrank[openapi]'") from e
    base = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

    class Loader(base):  # type: ignore[misc, valid-type]
        pass

    # dates stay strings: "2022-11-28" in a spec is an API version, not a datetime
    Loader.yaml_implicit_resolvers = {
        k: [r for r in v if r[0] != "tag:yaml.org,2002:timestamp"]
        for k, v in base.yaml_implicit_resolvers.items()
    }
    return yaml.load(raw, Loader=Loader)  # noqa: S506 - a SafeLoader subclass


def load_spec(source: str) -> dict[str, Any]:
    """An OpenAPI 3.x document from a path or an http(s) URL, JSON or YAML."""
    if re.match(r"https?://", source):
        with urllib.request.urlopen(source, timeout=120) as r:
            raw = r.read().decode("utf-8")
    else:
        raw = Path(source).read_text(encoding="utf-8")
    spec = json.loads(raw) if raw.lstrip()[:1] in ("{", "[") else _yaml(raw)
    if not isinstance(spec, dict):
        raise ValueError(f"{source}: not an OpenAPI document")
    if "swagger" in spec:
        raise ValueError(
            f"{source}: Swagger {spec['swagger']}: convert it to OpenAPI 3 first "
            f"(e.g. `npx swagger2openapi {source} -o spec3.json`)"
        )
    if not str(spec.get("openapi", "")).startswith("3."):
        raise ValueError(f"{source}: not an OpenAPI 3.x document (no 'openapi: 3.x' field)")
    return spec


class _Resolver:
    """Local ``$ref`` resolution against one spec; counts external and broken refs."""

    def __init__(self, spec: dict[str, Any], stats: Counter):
        self.spec, self.stats = spec, stats

    def _target(self, ref: str) -> Any:
        node: Any = self.spec
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                self.stats["broken_refs"] += 1
                return None
            node = node[part]
        return node

    def shallow(self, node: Any, stack: tuple[str, ...] = ()) -> Any:
        """Follow ``$ref`` chains at this node only (parameters, bodies, path items)."""
        while isinstance(node, dict) and "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/"):
                self.stats["external_refs"] += 1
                return None
            if ref in stack:
                return None
            stack = (*stack, ref)
            target = self._target(ref)
            if not isinstance(target, dict):
                return None
            node = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
        return node

    def inline(self, node: Any, stack: tuple[str, ...] = (), depth: int = 0) -> Any:
        """A schema with every local ``$ref`` inlined, noise keys dropped, cycles and depth cut."""
        if isinstance(node, list):
            return [self.inline(x, stack, depth) for x in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/"):
                self.stats["external_refs"] += 1
                return {"description": f"external schema {ref}"}
            if ref in stack:
                return {"type": "object"}  # recursive schema: cut here
            target = self._target(ref)
            if not isinstance(target, dict):
                return {"type": "object"}
            merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
            return self.inline(merged, (*stack, ref), depth)
        if depth > MAX_DEPTH:
            return {k: node[k] for k in ("type",) if k in node} or {"type": "object"}
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _DROP or key.startswith("x-"):
                continue
            if key == "description" and isinstance(value, str):
                out[key] = _plain(value)
            elif key in _SCHEMA_MAPS and isinstance(value, dict):
                out[key] = {k: self.inline(v, stack, depth + 1) for k, v in value.items()}
            elif key in _SCHEMA_ONE and isinstance(value, dict):
                out[key] = self.inline(value, stack, depth + 1)
            elif key in _SCHEMA_LISTS and isinstance(value, list):
                out[key] = [self.inline(v, stack, depth + 1) for v in value]
            else:
                out[key] = value
        return out


def _merge_all_of(schema: Any) -> Any:
    if not isinstance(schema, dict) or not isinstance(schema.get("allOf"), list):
        return schema
    merged = {k: v for k, v in schema.items() if k != "allOf"}
    props = dict(merged.get("properties") or {})
    required = list(merged.get("required") or [])
    for part in schema["allOf"]:
        part = _merge_all_of(part)
        if not isinstance(part, dict):
            continue
        props.update(part.get("properties") or {})
        required += [r for r in part.get("required") or [] if r not in required]
        for key in ("type", "description"):
            if part.get(key) and key not in merged:
                merged[key] = part[key]
    if props:
        merged["properties"] = props
    if required:
        merged["required"] = required
    return merged


def _body_properties(schema: Any, body_required: bool) -> tuple[dict[str, Any], list[str]] | None:
    """Top-level properties of an object body (``allOf`` merged, ``oneOf``/``anyOf`` unioned);
    None when the body is not an object with properties."""
    schema = _merge_all_of(schema)
    if not isinstance(schema, dict):
        return None
    props = dict(schema.get("properties") or {})
    for key in ("oneOf", "anyOf"):
        for alt in schema.get(key) or []:
            alt = _merge_all_of(alt)
            if isinstance(alt, dict):
                for name, sub in (alt.get("properties") or {}).items():
                    props.setdefault(name, sub)
    if not props:
        return None
    required = [r for r in schema.get("required") or [] if r in props] if body_required else []
    return props, required


def _empty_object(schema: Any) -> bool:
    """``{"type": "object", "properties": {}}``: the body Stripe declares on every GET."""
    return (
        isinstance(schema, dict)
        and schema.get("type", "object") == "object"
        and schema.get("properties") == {}
        and not schema.get("additionalProperties")
        and not any(schema.get(k) for k in ("oneOf", "anyOf", "allOf"))
    )


def _param_schema(p: dict[str, Any]) -> Any:
    """A parameter's schema: ``schema``, or the first media type's under ``content``."""
    if p.get("schema") is not None:
        return p["schema"]
    content = p.get("content")
    if isinstance(content, dict) and content:
        return (next(iter(content.values())) or {}).get("schema") or {}
    return {}


def _server_url(servers: Any) -> str:
    if not isinstance(servers, list) or not servers or not isinstance(servers[0], dict):
        return ""
    url, variables = str(servers[0].get("url") or ""), servers[0].get("variables") or {}
    return re.sub(
        r"\{(\w+)\}", lambda m: str((variables.get(m.group(1)) or {}).get("default", m.group(0))), url
    )


class OpenAPISource:
    """``ToolSource`` over one OpenAPI 3.x document; ``stats`` counts operations, capped texts,
    external or broken refs and skipped paths (no leading ``/``) after ``list_tools``."""

    kind = "openapi"

    def __init__(
        self,
        spec: dict[str, Any],
        name: str | None = None,
        *,
        origin: str = "",
        max_chars: int | None = MAX_CHARS,
    ):
        title = str((spec.get("info") or {}).get("title") or "")
        self.spec, self.name = spec, name or slug(title) or "api"
        self.origin, self.max_chars = origin.split("?")[0], max_chars
        self.base_url = _server_url(spec.get("servers"))
        self.stats: Counter = Counter()

    def list_tools(self) -> list[Tool]:
        self.stats = Counter()
        r = _Resolver(self.spec, self.stats)
        tools: list[Tool] = []
        names: Counter = Counter()
        for path, item in (self.spec.get("paths") or {}).items():
            if not str(path).startswith("/"):  # an "x-" extension, or no path at all: the proxy refuses it
                self.stats["bad_paths"] += not str(path).startswith("x-")
                continue
            item = r.shallow(item)
            if not isinstance(item, dict):
                continue
            for method in HTTP_METHODS:
                op = item.get(method)
                if isinstance(op, dict):
                    tools.append(self._tool(r, method, path, item, op, names))
        return tools

    def _parameters(self, r: _Resolver, item: dict[str, Any], op: dict[str, Any]) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for p in [*(item.get("parameters") or []), *(op.get("parameters") or [])]:
            p = r.shallow(p)
            if isinstance(p, dict) and p.get("name") and p.get("in"):
                merged[(str(p["name"]), str(p["in"]))] = p
        return [
            p
            for (name, where), p in merged.items()
            if not (where == "header" and name.lower() in SKIP_HEADERS)
        ]

    def _body(self, r: _Resolver, op: dict[str, Any]) -> tuple[str, Any, bool]:
        rb = r.shallow(op.get("requestBody"))
        content = rb.get("content") if isinstance(rb, dict) else None
        if not isinstance(content, dict) or not content:
            return "", None, False
        media = next((m for m in MEDIA_ORDER if m in content), None) or next(
            (m for m in content if "json" in m), next(iter(content))
        )
        schema = r.inline((content.get(media) or {}).get("schema") or {})
        if _empty_object(schema):
            return "", None, False
        return media, schema, bool(rb.get("required"))

    def _tool(
        self, r: _Resolver, method: str, path: str, item: dict[str, Any], op: dict[str, Any], names: Counter
    ) -> Tool:
        self.stats["operations"] += 1
        name = str(op.get("operationId") or "") or re.sub(r"[^A-Za-z0-9]+", "_", f"{method}_{path}").strip(
            "_"
        )
        names[name] += 1
        if names[name] > 1:
            name = f"{name}_{names[name]}"
        summary, desc = _plain(str(op.get("summary") or "")), _plain(str(op.get("description") or ""))
        parts = [desc] if summary and desc.startswith(summary) else [x for x in (summary, desc) if x]
        description = "\n\n".join(parts) or f"{method.upper()} {path}"

        media, body, body_required = self._body(r, op)
        flat = _body_properties(body, body_required) if body is not None else None
        body_props, body_req = flat or ({}, [])
        props: dict[str, Any] = {}
        required: list[str] = []
        args: dict[str, dict[str, str]] = {}
        taken = set(body_props) if flat is not None else ({"body"} if body is not None else set())
        for p in self._parameters(r, item, op):
            pname, where = str(p["name"]), str(p["in"])
            schema = r.inline(_param_schema(p))
            if p.get("description"):
                schema = {**schema, "description": _plain(str(p["description"]))}
            key = pname if pname not in taken and pname not in props else f"{pname}__{where}"
            props[key] = schema
            args[key] = {"in": where, "name": pname}
            if where == "query":  # how the proxy serialises lists and objects (OpenAPI defaults)
                style = str(p.get("style") or "form")
                args[key].update(style=style, explode=bool(p.get("explode", style == "form")))
            if p.get("required") or where == "path":
                required.append(key)
        if flat is not None:
            for key, schema in body_props.items():
                props[key] = schema
                args[key] = {"in": "body", "name": key}
            required += [k for k in body_req if k not in required]
        elif body is not None:
            props["body"] = body
            args["body"] = {"in": "body", "name": ""}
            if body_required:
                required.append("body")
        input_schema: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            input_schema["required"] = required

        text, capped = fit_text(self.name, name, description, input_schema, max_chars=self.max_chars)
        self.stats["capped"] += capped
        base_url = _server_url(op.get("servers")) or _server_url(item.get("servers")) or self.base_url
        http = {"method": method.upper(), "path": path, "base_url": base_url, "args": args}
        if media:
            http["body_media_type"] = media
        doc = {
            "server": self.name,
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "http": http,
        }
        if op.get("deprecated"):
            doc["deprecated"] = True
        return Tool(id=f"{self.name}/{name}", doc=doc, documentation=text, category=self.name)
