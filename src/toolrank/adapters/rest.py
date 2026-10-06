"""The REST face: the retriever behind the MCP proxy, for platforms that call a search endpoint
themselves and run the tools on their side (Anthropic's custom tool search, OpenAI's client-side
tool_search).

``POST /v1/search`` {query, instruction?, k?, full_schemas?} -> {search_id, mode, took_ms, rule, tools}
``POST /v1/rank`` {query, instruction?, tools: [MCP tool objects] | tool_ids: [...]} -> scores
``POST /v1/call`` {name, arguments?, search_id?} -> {name, call_id, outcome, isError, content, ...}
``GET /v1/tools[?server=&full=true]`` -> the catalogue; ``GET /v1/tools/{id}`` -> one tool's record
``GET /openapi.json`` -> this API; ``GET /healthz`` -> 200 once the index is ready, else 503

Served next to ``/mcp`` by ``mcp_proxy.http_app``, whose ``Guard`` puts the bearer token and the
Host and Origin checks on ``/v1``; bodies are capped at ``MAX_BODY`` here too (a chunked body has
no Content-Length). Searches and calls go to the usage log with ``via: rest``: the
``X-Session-Id`` header is the session (under the API key's name, the tenant). While the first
index builds, searches return keyword matches (``mode: lexical``). ``/v1/call`` runs a catalogue
tool exactly as MCP ``call_tool`` does (``mcp_proxy.dispatch_call``, same write policy); a tool
that fails is still a 200 with ``isError``. Errors are JSON ``{"error": ...}``: 400 bad input, 404
unknown tool, 413 body too large, 415 a call that is not JSON, 503 index not ready, embedding
endpoint down or backends not running. ``/v1/rank`` scores up to ``MAX_RANK`` tools by cosine (the
semantic arm also under ``--hybrid``), best first, each with its ``index`` in the request; supplied
tools get the text ingest gives MCP tools. Every tool carries its ``api_name`` (``toolrank.names``),
the name to give it on an agent API; ``/v1/tools?full=true`` is the one download a platform client
needs: each tool's description, input schema (always an object, no ``$schema``), annotations and,
for OpenAPI operations, the HTTP method, plus the catalogue's hash.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from toolrank import __version__
from toolrank.adapters.backends import Backends
from toolrank.adapters.mcp_proxy import (
    DESCRIPTION_CHARS,
    FULL_SCHEMAS,
    MAX_BODY,
    TENANT_KEY,
    dispatch_call,
    hit_json,
)
from toolrank.domain import Tool
from toolrank.ingest.mcp import tool_from_mcp
from toolrank.metrics import arm_kind
from toolrank.names import api_name
from toolrank.retriever import IndexNotReady, Retriever
from toolrank.usage import UsageLog

log = logging.getLogger("toolrank.serve")

MAX_RANK = 200
MAX_K = 50


class _Reject(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _error(status: int, message: str) -> Any:
    from starlette.responses import JSONResponse

    return JSONResponse({"error": message}, status_code=status)


async def _body(request: Any) -> dict[str, Any]:
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > MAX_BODY:
            raise _Reject(413, f"body over {MAX_BODY} bytes")
    try:
        body = json.loads(bytes(data))
    except ValueError:
        raise _Reject(400, "the body is not JSON") from None
    if not isinstance(body, dict):
        raise _Reject(400, "the body must be a JSON object")
    return body


def _string(body: dict[str, Any], key: str, *, required: bool = False) -> str | None:
    value = body.get(key)
    if value is None or (required and isinstance(value, str) and not value.strip()):
        if required:
            raise _Reject(400, f"{key} is required")
        return None
    if not isinstance(value, str):
        raise _Reject(400, f"{key} must be a string")
    return value.strip() if required else value


def identity(request: Any) -> tuple[str | None, str | None, str | None]:
    """(session, client, tenant) of a REST request, as ``mcp_proxy.identity`` for MCP ones."""
    tenant = request.scope.get(TENANT_KEY)
    header = (request.headers.get("x-session-id") or "")[:128]
    agent = (request.headers.get("user-agent") or "-").split("/")[0][:64]
    host = request.client.host if request.client else None
    # a named key's under its name (names hold no ':'), so no session id another key or a request
    # without a key picks can be the same: rest@alice:abc, rest:abc
    session = (f"rest@{tenant}:{header}" if tenant else f"rest:{header}") if header else None
    return session, f"{tenant or '-'}|{agent}|{host or '-'}", tenant


def _kind(tool: Tool) -> str:
    return "openapi" if "http" in tool.doc else "mcp"


def _platform_fields(tool: Tool) -> dict[str, Any]:
    out = {"api_name": api_name(tool.id), "kind": _kind(tool)}
    if "http" in tool.doc:
        out["method"] = str(tool.doc["http"].get("method") or "GET").upper()
    return out


def platform_record(tool: Tool) -> dict[str, Any]:
    """A catalogue tool as a platform client needs it (``/v1/tools?full=true``)."""
    schema = dict(tool.doc.get("inputSchema") or {})
    schema.pop("$schema", None)  # a draft-07 id on an API that validates against 2020-12
    schema.setdefault("type", "object")
    desc = tool.description
    out: dict[str, Any] = {
        "name": tool.id,
        "server": tool.category,
        "description": desc if len(desc) <= DESCRIPTION_CHARS else desc[:DESCRIPTION_CHARS] + "…",
        "inputSchema": schema,
        **_platform_fields(tool),
    }
    if tool.doc.get("annotations"):
        out["annotations"] = tool.doc["annotations"]
    return out


def _supplied_tool(n: int, record: Any) -> Any:
    if not isinstance(record, dict) or not isinstance(record.get("name"), str) or not record["name"]:
        raise _Reject(400, f"tools[{n}] needs a string name")
    for key in ("server", "title", "description"):
        if record.get(key) is not None and not isinstance(record[key], str):
            raise _Reject(400, f"tools[{n}].{key} must be a string")
    if record.get("inputSchema") is not None and not isinstance(record["inputSchema"], dict):
        raise _Reject(400, f"tools[{n}].inputSchema must be an object")
    return tool_from_mcp(record.get("server") or "", record)


def rest_routes(retriever: Retriever, usage: UsageLog, backends: Backends | None = None) -> list[Any]:
    """The routes; ``/v1/call`` only with ``backends`` (running in the app's lifespan)."""
    import anyio
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    limiter: list[Any] = []  # created inside the event loop: searches, lookups, calls; then /v1/rank's own
    if usage.metrics.catalogue is None:  # what a search is compared against, for the token estimate
        usage.metrics.catalogue = retriever.catalogue

    async def in_thread(fn: Any, *args: Any, own_slots: bool = False, **kw: Any) -> Any:
        """``fn`` in a worker thread: four for searches, lookups and calls, two of their own for
        ``/v1/rank`` (``own_slots``), whose tools a slow endpoint may take seconds to embed."""
        if not limiter:
            limiter.extend([anyio.CapacityLimiter(4), anyio.CapacityLimiter(2)])
        return await anyio.to_thread.run_sync(
            functools.partial(fn, *args, **kw), limiter=limiter[1 if own_slots else 0], abandon_on_cancel=True
        )

    def endpoint(fn: Callable[[Any], Awaitable[Any]]) -> Callable[[Any], Awaitable[Any]]:
        @functools.wraps(fn)
        async def wrapped(request: Any) -> Any:
            try:
                return await fn(request)
            except _Reject as e:
                return _error(e.status, str(e))
            except IndexNotReady as e:
                return _error(503, str(e))

        return wrapped

    @endpoint
    async def search(request: Any) -> Any:
        body = await _body(request)
        query = _string(body, "query", required=True)
        inst = _string(body, "instruction")
        k = body.get("k")
        if k is not None and (not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K):
            raise _Reject(400, f"k must be an integer from 1 to {MAX_K}")
        session, client, tenant = identity(request)
        try:
            res = await in_thread(
                retriever.search, query, k=k, instruction=inst, arm_key=session or client, tenant=tenant
            )
        except IndexNotReady:
            raise
        except Exception as e:  # the embedding endpoint is down, ...
            log.warning("search failed: %s: %s", type(e).__name__, e)  # the endpoint's address stays here
            raise _Reject(503, f"search failed ({type(e).__name__}); the server's log says why") from e
        sid = usage.search(
            res, session=session, via="rest", heads=res.heads, client=client, tenant=tenant, arm=res.arm
        )
        full = body.get("full_schemas") is True
        tools = [
            {**hit_json(h, full=full or n < FULL_SCHEMAS), **_platform_fields(h.tool)}
            for n, h in enumerate(res.hits)
        ]
        out = {"search_id": sid, "mode": res.mode, "took_ms": round(res.took_ms, 1), "rule": res.rule}
        return JSONResponse({**out, "tools": tools})

    @endpoint
    async def rank(request: Any) -> Any:
        body = await _body(request)
        query = _string(body, "query", required=True)
        inst = _string(body, "instruction")
        ids, given = body.get("tool_ids"), body.get("tools")
        if (ids is None) == (given is None):
            raise _Reject(400, "give either tools (MCP tool objects) or tool_ids")
        items = ids if ids is not None else given
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_RANK:
            raise _Reject(400, f"{'tool_ids' if ids is not None else 'tools'}: a list of 1 to {MAX_RANK}")
        session, client, tenant = identity(request)
        if ids is not None:
            if not all(isinstance(i, str) for i in ids):
                raise _Reject(400, "tool_ids must be strings")
            found = await in_thread(lambda: [retriever.get(i, tenant) for i in ids])
            missing = [i for i, t in zip(ids, found, strict=True) if t is None]
            if missing:
                raise _Reject(404, f"unknown tool ids: {', '.join(missing[:5])}")
            tools = found
            labels = [{"name": t.id} for t in found]
        else:
            tools = [_supplied_tool(n, r) for n, r in enumerate(given)]
            labels = [
                {"name": r["name"], **({"server": r["server"]} if r.get("server") else {})} for r in given
            ]
        # position ids: the scores map back to the request even when names repeat
        tools = [dataclasses.replace(t, id=str(n)) for n, t in enumerate(tools)]
        try:
            scored = await in_thread(
                retriever.rank,
                query,
                tools,
                instruction=inst,
                arm_key=session or client,
                tenant=tenant,
                own_slots=True,
            )
        except IndexNotReady:
            raise
        except ValueError as e:  # a scorer that cannot score tools outside its index
            raise _Reject(400, str(e)) from e
        except Exception as e:
            log.warning("rank failed: %s: %s", type(e).__name__, e)
            raise _Reject(503, f"rank failed ({type(e).__name__}); the server's log says why") from e
        out = [{"index": int(n), **labels[int(n)], "score": round(s, 6)} for n, s in scored]
        return JSONResponse({"tools": out})

    @endpoint
    async def list_tools(request: Any) -> Any:
        server = request.query_params.get("server")
        full = request.query_params.get("full", "").lower() in ("1", "true", "yes")
        tools, catalog = await in_thread(retriever.catalogue, identity(request)[2])
        chosen = [t for t in tools if not server or t.category == server]
        if full:
            items = [platform_record(t) for t in chosen]
        else:
            items = [
                {
                    "name": t.id,
                    "api_name": api_name(t.id),
                    "server": t.category,
                    "kind": _kind(t),
                    "description": (t.description.strip().splitlines() or [""])[0][:300],
                }
                for t in chosen
            ]
        return JSONResponse({"count": len(items), "catalog": catalog, "tools": items})

    @endpoint
    async def get_tool(request: Any) -> Any:
        tool = await in_thread(retriever.get, request.path_params["tool_id"], identity(request)[2])
        if tool is None:
            raise _Reject(404, "no such tool")
        keep = ("title", "description", "inputSchema", "outputSchema", "annotations", "http")
        record = {k: tool.doc[k] for k in keep if k in tool.doc}
        head = {"name": tool.id, "api_name": api_name(tool.id), "server": tool.category, "kind": _kind(tool)}
        return JSONResponse({**head, **record})

    @endpoint
    async def call(request: Any) -> Any:
        assert backends is not None
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            raise _Reject(415, "send the call as application/json")  # a web form cannot
        body = await _body(request)
        name = _string(body, "name", required=True)
        arguments = body.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise _Reject(400, "arguments must be an object")
        search_id = _string(body, "search_id")
        try:
            done = await dispatch_call(
                retriever, backends, usage, in_thread, name=name, arguments=arguments,
                search_id=search_id, who=identity(request), via="rest",
            )  # fmt: skip
        except IndexNotReady:
            raise
        except RuntimeError as e:  # the backends are not running
            raise _Reject(503, str(e)) from e
        if done.tool is None:
            raise _Reject(404, f"no tool named {name!r}; GET /v1/tools lists them")
        result = done.result
        out: dict[str, Any] = {
            "name": done.tool.id,
            "call_id": done.call_id,
            "outcome": done.outcome,
            "isError": bool(result.is_error),
            "http_status": done.http_status,
            "content": [c.model_dump(mode="json", by_alias=True, exclude_none=True) for c in result.content],
        }
        if result.structured_content is not None:
            out["structuredContent"] = result.structured_content
        return JSONResponse(out)

    async def openapi(request: Any) -> Any:
        return JSONResponse(OPENAPI)

    async def metrics(request: Any) -> Any:
        from starlette.responses import PlainTextResponse

        if (
            identity(request)[2] in retriever.allowed
        ):  # server-wide counts: not for a key limited to some sources
            return _error(403, "this key is limited to some sources; metrics are server-wide")
        st = retriever.status()
        gauge = "gauge"
        extra: list[tuple[str, str, str, dict[str, str], float]] = [
            ("toolrank_build_info", gauge, "The running version.", {"version": __version__}, 1),
            ("toolrank_index_ready", gauge, "1 once the semantic index answers.", {}, float(st["ready"])),
            ("toolrank_catalog_tools", gauge, "Tools in the catalogue.", {}, st["tools"]),
            ("toolrank_catalog_sources", gauge, "Servers and APIs in the catalogue.", {}, st["sources"]),
        ]
        whole = await in_thread(usage.metrics.catalog_tokens) if st["tools"] else None
        if whole is not None:
            text = "Estimated tokens of every tool's name, description and input schema."
            extra.append(("toolrank_catalog_tokens", gauge, text, {}, whole))
        for arm, heads in (st.get("heads") or {}).items():
            ready = heads is not None if arm == "base" else bool(heads.get("ready"))
            text = "Heads files the server follows (1: answering)."
            extra.append(("toolrank_heads", gauge, text, {"arm": arm_kind(arm)}, float(ready)))
        enc = getattr(retriever, "encoder", None)
        for (kind, source), n in sorted(getattr(enc, "texts", {}).items()):
            text = "Texts embedded, by where the vector came from."
            extra.append(
                ("toolrank_embedding_texts_total", "counter", text, {"kind": kind, "source": source}, n)
            )
        if enc is not None and hasattr(enc, "tokens_spent"):
            text = "Tokens sent to the embedding endpoint."
            extra.append(("toolrank_embedding_tokens_total", "counter", text, {}, enc.tokens_spent))
        return PlainTextResponse(usage.metrics.render(extra), media_type="text/plain; version=0.0.4")

    async def healthz(request: Any) -> Any:  # no token, no Host check (probes): what a probe needs only
        st = retriever.status()
        body = {k: st[k] for k in ("ready", "mode")}
        return JSONResponse(body, status_code=200 if st["ready"] else 503)

    return [
        Route("/v1/search", search, methods=["POST"]),
        Route("/v1/rank", rank, methods=["POST"]),
        *([Route("/v1/call", call, methods=["POST"])] if backends is not None else []),
        Route("/v1/tools", list_tools, methods=["GET"]),
        Route("/v1/tools/{tool_id:path}", get_tool, methods=["GET"]),
        Route("/v1/metrics", metrics, methods=["GET"]),
        Route("/openapi.json", openapi, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
    ]


# -- OpenAPI 3.1 --------------------------------------------------------------------------------
def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{name}"}


def _json(schema: dict[str, Any]) -> dict[str, Any]:
    return {"content": {"application/json": {"schema": schema}}}


def _responses(ok: dict[str, Any], *errors: int) -> dict[str, Any]:
    out: dict[str, Any] = {"200": {"description": "OK", **_json(ok)}}
    for code in errors:
        out[str(code)] = {"description": _ERRORS[code], **_json(_ref("Error"))}
    return out


_ERRORS = {
    400: "Bad input",
    401: "Missing or wrong bearer token",
    403: "Origin not allowed",
    404: "Unknown tool",
    413: "Body too large",
    415: "Not application/json",
    421: "Host not allowed",
    503: "Index not ready, the embedding endpoint is down, or the tool backends are not running",
}
_API_NAME = {
    "type": "string",
    "pattern": "^[A-Za-z0-9_-]{1,64}$",
    "description": "The tool's name on an agent API (Anthropic, OpenAI); a pure function of the id.",
}
_QUERY = {
    "query": {"type": "string", "description": "The request, in plain words."},
    "instruction": {"type": "string", "description": "Overrides the server's retrieval instruction."},
}
_SCHEMAS: dict[str, Any] = {
    "Error": {"type": "object", "properties": {"error": {"type": "string"}}, "required": ["error"]},
    "CallRequest": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Tool id, as /v1/search and /v1/tools return it."},
            "arguments": {"type": "object", "description": "Arguments matching the tool's inputSchema."},
            "search_id": {"type": "string", "description": "The search that found the tool, for the log."},
        },
        "required": ["name"],
    },
    "CallResult": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "call_id": {"type": "string"},
            "outcome": {"enum": ["ok", "tool_error", "protocol_error", "timeout", "refused"]},
            "isError": {"type": "boolean"},
            "http_status": {"type": ["integer", "null"], "description": "OpenAPI tools: the API's status."},
            "content": {"type": "array", "items": {"type": "object"}, "description": "MCP content blocks."},
            "structuredContent": {"type": "object"},
        },
        "required": ["name", "call_id", "outcome", "isError", "content"],
    },
    "Hit": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Tool id, `<server>/<tool name>`."},
            "api_name": _API_NAME,
            "server": {"type": "string"},
            "kind": {"enum": ["mcp", "openapi"]},
            "method": {"type": "string", "description": "OpenAPI operations: the HTTP method."},
            "score": {"type": "number"},
            "description": {"type": "string"},
            "inputSchema": {"type": "object"},
            "inputSchemaShrunk": {"type": "boolean"},
            "annotations": {"type": "object"},
            "used_with": {
                "type": "string",
                "description": "Set on a tool that was not ranked into the list: the returned tool that "
                "agents call it together with (`serve --co-use`).",
            },
        },
        "required": ["name", "api_name", "server", "kind", "score", "description", "inputSchema"],
    },
    "SearchRequest": {
        "type": "object",
        "properties": {
            **_QUERY,
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_K,
                "description": "Fixed number of tools; default: the server's cut (adaptive K).",
            },
            "full_schemas": {
                "type": "boolean",
                "description": f"Full inputSchema for every hit, not only the first {FULL_SCHEMAS}.",
            },
        },
        "required": ["query"],
    },
    "SearchResponse": {
        "type": "object",
        "properties": {
            "search_id": {"type": "string"},
            "mode": {
                "enum": ["semantic", "lexical"],
                "description": "lexical: keyword matches while the semantic index is being built.",
            },
            "took_ms": {"type": "number"},
            "rule": {"type": "string"},
            "tools": {"type": "array", "items": _ref("Hit")},
        },
        "required": ["search_id", "mode", "took_ms", "rule", "tools"],
    },
    "RankRequest": {
        "type": "object",
        "properties": {
            **_QUERY,
            "tools": {
                "type": "array",
                "maxItems": MAX_RANK,
                "description": "MCP tool objects (name, description, inputSchema) plus an optional server.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "server": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "inputSchema": {"type": "object"},
                    },
                    "required": ["name"],
                },
            },
            "tool_ids": {"type": "array", "maxItems": MAX_RANK, "items": {"type": "string"}},
        },
        "required": ["query"],
    },
    "RankResponse": {
        "type": "object",
        "properties": {
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "Position in the request."},
                        "name": {"type": "string"},
                        "server": {"type": "string"},
                        "score": {"type": "number"},
                    },
                    "required": ["index", "name", "score"],
                },
            }
        },
        "required": ["tools"],
    },
    "ToolList": {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "catalog": {"type": "string", "description": "Hash of the catalogue these tools come from."},
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "description": "With full=true, also inputSchema, annotations and (OpenAPI) method.",
                    "properties": {
                        "name": {"type": "string"},
                        "api_name": _API_NAME,
                        "server": {"type": "string"},
                        "kind": {"enum": ["mcp", "openapi"]},
                        "description": {
                            "type": "string",
                            "description": "First line, at most 300 characters (full=true: up to 1,500).",
                        },
                        "inputSchema": {"type": "object"},
                        "annotations": {"type": "object"},
                        "method": {"type": "string"},
                    },
                    "required": ["name", "api_name", "server", "kind", "description"],
                },
            },
        },
    },
    "ToolRecord": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "api_name": _API_NAME,
            "server": {"type": "string"},
            "kind": {"enum": ["mcp", "openapi"]},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "inputSchema": {"type": "object"},
            "outputSchema": {"type": "object"},
            "annotations": {"type": "object"},
            "http": {"type": "object", "description": "OpenAPI operations: method, path, base_url, args."},
        },
        "required": ["name", "server", "kind"],
    },
    "Health": {
        "type": "object",
        "properties": {
            "ready": {"type": "boolean"},
            "mode": {"enum": ["semantic", "lexical", "starting", "failed"]},
        },
    },
}
OPENAPI: dict[str, Any] = {
    "openapi": "3.1.0",
    "info": {
        "title": "toolrank",
        "version": __version__,
        "description": (
            "Tool retrieval for LLM agents: search and rank the tools behind this server. The bearer "
            "token is required when the server was started with an API key."
        ),
    },
    "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}}, "schemas": _SCHEMAS},
    "security": [{"bearer": []}, {}],
    "paths": {
        "/v1/search": {
            "post": {
                "operationId": "search",
                "summary": "The tools a request needs",
                "requestBody": {"required": True, **_json(_ref("SearchRequest"))},
                "responses": _responses(_ref("SearchResponse"), 400, 401, 403, 413, 421, 503),
            }
        },
        "/v1/rank": {
            "post": {
                "operationId": "rank",
                "summary": f"Score up to {MAX_RANK} given tools, or catalogue tools by id, for a request",
                "requestBody": {"required": True, **_json(_ref("RankRequest"))},
                "responses": _responses(_ref("RankResponse"), 400, 401, 403, 404, 413, 421, 503),
            }
        },
        "/v1/call": {
            "post": {
                "operationId": "call",
                "summary": "Run a tool (a failing tool is a 200 with isError)",
                "requestBody": {"required": True, **_json(_ref("CallRequest"))},
                "responses": _responses(_ref("CallResult"), 400, 401, 403, 404, 413, 415, 421, 503),
            }
        },
        "/v1/tools": {
            "get": {
                "operationId": "listTools",
                "summary": "The catalogue",
                "parameters": [
                    {"name": "server", "in": "query", "schema": {"type": "string"}},
                    {"name": "full", "in": "query", "schema": {"type": "boolean"}},
                ],
                "responses": _responses(_ref("ToolList"), 401, 403, 421, 503),
            }
        },
        "/v1/tools/{tool_id}": {
            "get": {
                "operationId": "getTool",
                "summary": "One tool's record",
                "parameters": [
                    {"name": "tool_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": _responses(_ref("ToolRecord"), 401, 403, 404, 421, 503),
            }
        },
        "/v1/metrics": {
            "get": {
                "operationId": "metrics",
                "summary": "Prometheus metrics: searches, calls, latency, token estimate, cache hits",
                "responses": {
                    "200": {
                        "description": "Text exposition format 0.0.4",
                        "content": {"text/plain": {"schema": {"type": "string"}}},
                    },
                    "401": {"description": "Missing or wrong bearer token"},
                    "403": {"description": "A key limited to some sources"},
                },
            }
        },
        "/healthz": {
            "get": {
                "operationId": "health",
                "summary": "Whether the index is ready (503 until it is)",
                "security": [],
                "responses": {
                    "200": {"description": "Ready", **_json(_ref("Health"))},
                    "503": {"description": "Not ready", **_json(_ref("Health"))},
                },
            }
        },
    },
}
