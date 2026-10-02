"""The MCP face of toolrank: one server, two tools, in front of every ingested tool.

``search_tools(query, k?)`` finds the tools a request needs (``toolrank.retriever``) and returns
their names, descriptions and input schemas; ``call_tool(name, arguments, search_id?)`` forwards the
call to the tool's MCP server or OpenAPI operation (``adapters/backends.py``). An agent sees two
tool definitions instead of hundreds. Built on the SDK's low-level ``Server``: ``MCPServer``
derives input schemas from Python signatures, and a proxy passes results through untouched.

Search results carry the full ``inputSchema`` for the first ``FULL_SCHEMAS`` tools and a shrunk one
(``ingest.text.shrink_schema``) for the rest, so ten Stripe operations do not fill a context
window; a failed call returns the tool's full schema. Searches run in worker threads under their
own capacity limiter (the retriever may wait on the embedding endpoint), and every search and call
goes to the usage log under the caller's ``identity``. While the first index builds, searches get
keyword matches with a note saying so. ``serve_stdio`` runs the server for desktop clients;
``http_app`` mounts it at ``/mcp`` in a Starlette app with bearer tokens (anonymous or named, a name
becoming the caller's tenant), a Host check and a body limit on our own routes.
"""

from __future__ import annotations

import functools
import hmac
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from toolrank import __version__
from toolrank.adapters.backends import Backends, error_result
from toolrank.domain import Tool
from toolrank.ingest.text import shrink_schema
from toolrank.retriever import IndexNotReady, Retriever
from toolrank.usage import UsageLog

SEARCH_TOOL, CALL_TOOL = "search_tools", "call_tool"
FULL_SCHEMAS = 3  # search hits that carry their whole inputSchema
SHRUNK_CHARS = 1500
DESCRIPTION_CHARS = 1500
MAX_BODY = 1 << 20  # bytes, on our own routes
TENANT_KEY = "toolrank.tenant"  # ASGI scope key: the name of the API key the request used
LEXICAL_NOTE = (
    "The semantic index is still being built, so these are keyword matches; search again shortly "
    "for better ones."
)
EMPTY_NOTE = (
    "No tool in this catalogue is close enough to the request. Answer without a tool, or search again "
    "in other words."
)
INSTRUCTIONS = (
    "This server fronts many tools. Call search_tools with what you want to do, then call_tool with "
    "a returned tool name and arguments that match its inputSchema."
)
_LOOPBACK = ("127.0.0.1", "localhost", "::1")
_WILDCARD = ("0.0.0.0", "::")


def hit_json(hit: Any, *, full: bool) -> dict[str, Any]:
    schema, shrunk = (hit.input_schema, False) if full else shrink_schema(hit.input_schema, SHRUNK_CHARS)
    desc = hit.tool.description
    out: dict[str, Any] = {
        "name": hit.id,
        "server": hit.server,
        "score": round(hit.score, 4),
        "description": desc if len(desc) <= DESCRIPTION_CHARS else desc[:DESCRIPTION_CHARS] + "…",
        "inputSchema": schema,
    }
    if getattr(hit, "used_with", None):
        out["used_with"] = hit.used_with  # not a match by itself: agents call it along with that tool
    if shrunk:
        out["inputSchemaShrunk"] = True
    if hit.tool.doc.get("annotations"):
        out["annotations"] = hit.tool.doc["annotations"]
    return out


def identity(ctx: Any) -> tuple[str | None, str | None, str | None]:
    """(session, client, tenant) of an MCP request. A stdio server serves one client: ``stdio``.
    Over HTTP: the MCP session id when the client has one (2026-07-28 clients have none), the API
    key's name as tenant, and ``tenant|app|host`` as the client key that ties a call to the same
    client's searches."""
    req = getattr(ctx, "request", None)
    if req is None:
        return "stdio", "stdio", None
    tenant = req.scope.get(TENANT_KEY)
    params = getattr(getattr(ctx, "session", None), "client_params", None)
    app = getattr(getattr(params, "client_info", None), "name", None)
    host = req.client.host if req.client else None
    return req.headers.get("mcp-session-id"), f"{tenant or '-'}|{app or '-'}|{host or '-'}", tenant


@dataclass(frozen=True)
class Dispatched:
    tool: Tool | None  # None: no tool by that name
    result: Any  # mcp_types.CallToolResult
    outcome: str
    http_status: int | None
    call_id: str


async def dispatch_call(
    retriever: Retriever,
    backends: Backends,
    usage: UsageLog,
    in_thread: Callable[..., Awaitable[Any]],
    *,
    name: str,
    arguments: dict[str, Any],
    search_id: str | None,
    who: tuple[str | None, str | None, str | None],
    via: str,
) -> Dispatched:
    """One tool call for MCP ``call_tool`` and REST ``/v1/call``: route it to its backend, log it,
    and on a tool or protocol error append the tool's full input schema to the result. Raises
    ``IndexNotReady``, and ``RuntimeError`` when the backends are not running."""
    from mcp_types import TextContent

    session, client, tenant = who
    tool = await in_thread(retriever.get, name, tenant)  # None as well for a tool outside the key's sources
    if tool is None:
        cid = usage.call(
            tool=name, kind=None, session=session, via=via, outcome="unknown_tool",
            took_ms=0.0, arguments=arguments, search_id=search_id, client=client, tenant=tenant,
        )  # fmt: skip
        result = error_result(
            f"no tool named {name!r}: call search_tools first and pass a returned name exactly"
        )
        return Dispatched(None, result, "unknown_tool", None, cid)
    t0 = time.perf_counter()
    out = await backends.call(tool, arguments, tenant)
    result = out.result
    cid = usage.call(
        tool=tool.id, kind="openapi" if "http" in tool.doc else "mcp", session=session, via=via,
        outcome=out.outcome, took_ms=(time.perf_counter() - t0) * 1000.0, arguments=arguments,
        search_id=search_id, http_status=out.http_status, client=client, tenant=tenant,
        error=result.content[0].text if result.is_error and result.content else None,
    )  # fmt: skip
    if result.is_error and out.outcome in ("tool_error", "protocol_error"):
        schema = json.dumps(tool.doc.get("inputSchema") or {}, ensure_ascii=False)
        hint = TextContent(type="text", text=f"inputSchema of {tool.id}: {schema}")
        result = result.model_copy(update={"content": [*result.content, hint]})
    return Dispatched(tool, result, out.outcome, out.http_status, cid)


def tool_definitions(retriever: Retriever) -> list[Any]:
    from mcp_types import Tool as MCPTool

    st = retriever.status()
    where = f"the {st['tools']} tools of {st['sources']} servers and APIs" if st["tools"] else "the tools"
    return [
        MCPTool(
            name=SEARCH_TOOL,
            description=(
                f"Find the tools you need among {where} behind this server. Describe the task, or the "
                "one step you are about to take, in plain words. You get the best-matching tools with "
                f"their names, descriptions and input schemas (complete for the first {FULL_SCHEMAS}, "
                "shortened for the rest), and a search_id. Then use call_tool."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What you want to do, in plain words."},
                    "k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "How many tools to return; default: as many as the request needs, up to 10.",
                    },
                },
                "required": ["query"],
            },
        ),
        MCPTool(
            name=CALL_TOOL,
            description=(
                "Call a tool that search_tools returned: name exactly as returned, arguments matching "
                "its inputSchema. Pass the search_id you got so the call is tied to that search. If "
                "the call fails, the error includes the tool's complete inputSchema."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "A tool name returned by search_tools."},
                    "arguments": {"type": "object", "description": "The tool's arguments."},
                    "search_id": {"type": "string", "description": "The search_id of that search."},
                },
                "required": ["name"],
            },
        ),
    ]


def build_proxy(retriever: Retriever, backends: Backends, usage: UsageLog, *, name: str = "toolrank") -> Any:
    """The low-level MCP ``Server``; its lifespan keeps the backends' sessions."""
    import anyio
    from mcp.server.lowlevel.server import Server
    from mcp_types import CallToolResult, ListToolsResult, TextContent

    limiter: list[Any] = []  # created inside the event loop

    async def in_thread(fn: Any, *args: Any, **kw: Any) -> Any:
        if not limiter:
            limiter.append(anyio.CapacityLimiter(4))
        return await anyio.to_thread.run_sync(
            functools.partial(fn, *args, **kw), limiter=limiter[0], abandon_on_cancel=True
        )

    async def search(args: dict[str, Any], who: tuple[str | None, str | None, str | None]) -> Any:
        query = str(args.get("query") or "").strip()
        k = args.get("k")
        if not query:
            return error_result("search_tools needs a non-empty query")
        if k is not None and (not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 50):
            return error_result("k must be an integer from 1 to 50")
        session, client, tenant = who
        try:
            res = await in_thread(retriever.search, query, k=k, arm_key=session or client, tenant=tenant)
        except IndexNotReady as e:
            return error_result(str(e))
        except Exception as e:  # the embedding endpoint is down, ...
            return error_result(f"search failed: {type(e).__name__}: {e}")
        sid = usage.search(
            res, session=session, via="mcp", heads=res.heads, client=client, tenant=tenant, arm=res.arm
        )
        payload: dict[str, Any] = {
            "search_id": sid,
            "tools": [hit_json(h, full=n < FULL_SCHEMAS) for n, h in enumerate(res.hits)],
        }
        if res.mode == "lexical":
            payload["note"] = LEXICAL_NOTE
        elif not res.hits:  # a threshold with --cut-min 0 turned every tool away
            payload["note"] = EMPTY_NOTE
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
        )

    async def call(args: dict[str, Any], who: tuple[str | None, str | None, str | None]) -> Any:
        arguments = args.get("arguments") or {}
        if not isinstance(arguments, dict):
            return error_result("arguments must be an object")
        search_id = args.get("search_id")
        if search_id is not None and not isinstance(search_id, str):  # before the tool runs, as REST does
            return error_result("search_id must be a string")
        try:
            done = await dispatch_call(
                retriever, backends, usage, in_thread, name=str(args.get("name") or ""),
                arguments=arguments, search_id=search_id, who=who, via="mcp",
            )  # fmt: skip
        except IndexNotReady as e:
            return error_result(str(e))
        return done.result

    async def on_list_tools(ctx: Any, params: Any) -> Any:
        return ListToolsResult(tools=tool_definitions(retriever))

    async def on_call_tool(ctx: Any, params: Any) -> Any:
        args = dict(params.arguments or {})
        if params.name == SEARCH_TOOL:
            return await search(args, identity(ctx))
        if params.name == CALL_TOOL:
            return await call(args, identity(ctx))
        return error_result(f"unknown tool {params.name!r}: this server has {SEARCH_TOOL} and {CALL_TOOL}")

    @asynccontextmanager
    async def lifespan(server: Any) -> AsyncIterator[dict[str, Any]]:
        async with backends.running():
            yield {}

    return Server(
        name,
        version=__version__,
        instructions=INSTRUCTIONS,
        lifespan=lifespan,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def serve_stdio(server: Any) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def allowed_hosts(host: str, extra: Sequence[str] = ()) -> list[str]:
    """Host header patterns for the bind address and ``--allowed-host``: a name given with a port
    as is; one without, both bare (a reverse proxy on 80/443 sends no port) and on any port
    (``name:*``)."""
    base = ["127.0.0.1", "localhost", "[::1]"] if host in _LOOPBACK else [host]
    if host in _WILDCARD:  # every interface: a published container port is reached as localhost too
        base = ["127.0.0.1", "localhost", "[::1]", host]
    out: list[str] = []
    for h in [*base, *extra]:
        out += [h] if ":" in h.strip("[]") and not h.startswith("[") else [h, f"{h}:*"]
    return out


def _origin_ok(value: str, patterns: Sequence[str]) -> bool:
    """The SDK's rule for ``/mcp``: an exact match, or ``scheme://host:*`` for any port."""
    return any(value == p or (p.endswith(":*") and value.startswith(p[:-1])) for p in patterns)


def _host_ok(value: str, patterns: Sequence[str]) -> bool:
    for p in patterns:
        if p.endswith(":*") and (value == p[:-2] or value.startswith(p[:-1])):
            return True
        if value == p:
            return True
    return False


class Guard:
    """ASGI middleware: bearer tokens on ``/mcp`` and ``/v1`` when any are set — ``api_key``
    (anonymous) and ``named_keys`` ({name: key}; the name goes into the scope as the tenant) — and
    on ``/v1`` the checks the SDK applies to ``/mcp`` only: Host, Origin (a web page must not reach
    a loopback server that has no token) and the body limit."""

    def __init__(
        self,
        app: Any,
        *,
        api_key: str | None = None,
        named_keys: Mapping[str, str] | None = None,
        hosts: Sequence[str],
        origins: Sequence[str] = (),
        max_body: int = MAX_BODY,
    ):
        self.app, self.hosts, self.origins, self.max_body = app, list(hosts), list(origins), max_body
        self.tokens: list[tuple[bytes, str | None]] = (
            [(f"Bearer {api_key}".encode(), None)] if api_key else []
        )
        self.tokens += [(f"Bearer {key}".encode(), name) for name, key in (named_keys or {}).items()]

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        headers = dict(scope.get("headers") or [])
        given = headers.get(b"authorization", b"")
        # every token is compared, so the time taken does not tell which one matched
        matched = [name for token, name in self.tokens if hmac.compare_digest(given, token)]
        if self.tokens and not matched and path.startswith(("/mcp", "/v1")):
            await _json(
                send, 401, {"error": "missing or wrong bearer token"}, [(b"www-authenticate", b"Bearer")]
            )
            return
        if matched and matched[0] is not None:
            scope[TENANT_KEY] = matched[0]
        if path.startswith("/v1"):
            if not _host_ok(headers.get(b"host", b"").decode("latin-1"), self.hosts):
                await _json(send, 421, {"error": "Host not allowed"})
                return
            origin = headers.get(b"origin")
            if origin is not None and not _origin_ok(origin.decode("latin-1"), self.origins):
                await _json(send, 403, {"error": "Origin not allowed"})
                return
            length = headers.get(b"content-length")
            if length is not None and length.isdigit() and int(length) > self.max_body:
                await _json(send, 413, {"error": f"body over {self.max_body} bytes"})
                return
        await self.app(scope, receive, send)


async def _json(
    send: Any, status: int, body: dict[str, Any], extra: Sequence[tuple[bytes, bytes]] = ()
) -> None:
    data = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode()), *extra]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": data})


def http_app(
    server: Any,
    *,
    host: str = "127.0.0.1",
    extra_hosts: Sequence[str] = (),
    api_key: str | None = None,
    named_keys: Mapping[str, str] | None = None,
    routes: Sequence[Any] = (),
) -> Any:
    """Streamable HTTP at ``/mcp`` plus ``routes``, behind ``Guard``."""
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = allowed_hosts(host, extra_hosts)
    origins = [f"{scheme}://{h}" for h in hosts for scheme in ("http", "https")]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
    )
    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        transport_security=security,
        host=host,
        custom_starlette_routes=list(routes),
    )
    return Guard(app, api_key=api_key, named_keys=named_keys, hosts=hosts, origins=origins)
