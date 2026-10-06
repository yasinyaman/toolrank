"""Where ``call_tool`` goes: MCP servers (long-lived client sessions) and OpenAPI operations (HTTP).

MCP: one ``MCPBackend`` per configured server. Its task, started on the first call inside the
proxy's task group (``Backends.running()``), owns the ``mcp.Client``: connect (bounded by
``connect_timeout``), list the tools once, stay connected, and after a failure wait for the next
caller before reconnecting (with backoff). Callers use the session directly with
``read_timeout_seconds``, so a timed-out call stops there, and a call in flight when the backend
failed is never resent. Every failure becomes an ``isError`` result naming the server; the
backend's ``serverInfo`` stamp is removed so clients see the proxy's.

OpenAPI: ``OpenAPIExecutor`` turns a tool's ``doc["http"]`` routing and the agent's arguments into
one HTTP request (``httpx2``): path parameters percent-encoded, query parameters by their OpenAPI
style (``form`` explode → repeated keys, ``deepObject`` → ``k[sub]=v`` / ``k[0]=v``), booleans as
``true``/``false``, JSON bodies, and form bodies nested with brackets the way Stripe reads them.
Safety: only GET and HEAD unless ``allow_write``; configured headers (credentials) go only to the
configured base URL (a path that does not start with ``/`` is refused: appended to the base URL it
could name another host; and the composed URL must have the base URL's origin); redirects are not followed; responses are cut at ``max_chars``. Multipart,
header and cookie parameters are refused for now.

Tenants (``tenants.Tenant``, ``serve --api-keys``): a call made with a named key gets that key's
headers for the source on top of the config's (OpenAPI: only towards the configured ``base_url``),
and an MCP server for which the key has headers or env is reached over a connection of the key's
own, opened lazily like the shared one: one tenant's credentials never carry another's call.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import math
import re
import tempfile
import urllib.parse
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from toolrank.domain import Tool
from toolrank.ingest.mcp import OpenAPIBackend, ServerConfig

SAFE_METHODS = ("GET", "HEAD")
MAX_RESPONSE_CHARS = 25_000


def merge_headers(base: dict[str, str], own: dict[str, str]) -> dict[str, str]:
    """``base`` with ``own`` on top, names compared case-insensitively (HTTP's rule): a key's
    ``authorization`` replaces the config's ``Authorization`` instead of travelling next to it."""
    mine = {k.lower() for k in own}
    return {**{k: v for k, v in base.items() if k.lower() not in mine}, **own}


@dataclass(frozen=True)
class CallOutcome:
    result: Any  # mcp_types.CallToolResult
    outcome: str  # one of toolrank.usage.OUTCOMES
    http_status: int | None = None


class Refused(ValueError):
    """The proxy will not make this call (policy or an unsupported shape), whatever the backend says."""


def error_result(text: str) -> Any:
    from mcp_types import CallToolResult, TextContent

    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def _strip_server_info(result: Any) -> Any:
    from mcp_types import SERVER_INFO_META_KEY

    meta = getattr(result, "meta", None)
    if isinstance(meta, dict) and SERVER_INFO_META_KEY in meta:
        rest = {k: v for k, v in meta.items() if k != SERVER_INFO_META_KEY}
        return result.model_copy(update={"meta": rest or None})
    return result


# -- MCP --------------------------------------------------------------------------------------
class MCPBackend:
    """One MCP server behind the proxy. State changes happen under ``_cond``: callers raise
    ``_want`` and wait until a connection exists or an attempt that started after them failed; the
    worker connects when wanted. A caller that sees the transport break sets that connection's own
    ``broken`` event, so a late failure never tears down a newer connection."""

    def __init__(self, cfg: ServerConfig, *, connect_timeout: float = 60.0):
        self.cfg, self.connect_timeout = cfg, connect_timeout
        self._session: tuple[Any, Any] | None = None  # (client, its broken event)
        self._error: str | None = None
        self._delay = 0.0
        self._want = False
        self._attempt = 0  # attempts started
        self._failed = 0  # the latest attempt that failed
        self._started = False

    def _start(self, tg: Any) -> None:
        if self._started:
            return
        import anyio

        self._cond = anyio.Condition()
        self._started = True
        tg.start_soon(self._run, name=f"toolrank-backend-{self.cfg.name}")

    async def _run(self) -> None:
        import shutil

        import anyio
        import httpx2
        from mcp import Client

        from toolrank.adapters.mcp_client import _reason, _stderr_tail, _transport, list_all

        while True:
            async with self._cond:
                while not self._want:
                    await self._cond.wait()
                self._want = False
                self._attempt += 1
                attempt = self._attempt
            if self._delay:
                await anyio.sleep(self._delay)
            failure = None
            broken = anyio.Event()
            with tempfile.TemporaryFile("w+", encoding="utf-8") as errlog:
                try:
                    if self.cfg.transport == "stdio" and shutil.which(self.cfg.command) is None:
                        raise FileNotFoundError(f"command not found: {self.cfg.command}")
                    # the deadline bounds the connection only; it is lifted once connected
                    with anyio.CancelScope(deadline=anyio.current_time() + self.connect_timeout) as scope:
                        async with AsyncExitStack() as stack:
                            http = None
                            if self.cfg.transport == "http":
                                http = await stack.enter_async_context(
                                    httpx2.AsyncClient(
                                        headers=dict(self.cfg.headers), timeout=httpx2.Timeout(30, read=None)
                                    )
                                )
                            client = await stack.enter_async_context(
                                Client(_transport(self.cfg, errlog, http), cache=None)
                            )
                            await list_all(client)
                            scope.deadline = math.inf
                            async with self._cond:
                                self._session, self._error, self._delay = (client, broken), None, 0.0
                                self._cond.notify_all()
                            await broken.wait()
                    if scope.cancelled_caught:
                        failure = _reason(self.cfg, TimeoutError(), self.connect_timeout) + _stderr_tail(
                            errlog
                        )
                except FileNotFoundError as e:
                    failure = f"{self.cfg.name}: {e}"
                except Exception as e:  # the SDK raises exception groups, transport and validation errors
                    failure = _reason(self.cfg, e, self.connect_timeout) + _stderr_tail(errlog)
            async with self._cond:
                self._session = None
                if failure:
                    self._error, self._failed = failure, attempt
                    self._delay = min(max(self._delay * 2, 1.0), 30.0)
                self._cond.notify_all()

    async def _connection(self) -> tuple[Any, Any] | None:
        import anyio

        session = self._session
        if session is not None:
            return session
        async with self._cond:
            self._want = True
            first = self._attempt + 1  # the next attempt to start is (also) ours
            self._cond.notify_all()
            with anyio.move_on_after(self.connect_timeout + self._delay + 5):
                while self._session is None and self._failed < first:
                    await self._cond.wait()
            return self._session

    async def call(self, tg: Any, name: str, arguments: dict[str, Any], *, timeout: float) -> CallOutcome:
        from mcp.shared.exceptions import MCPError
        from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT

        self._start(tg)
        server = self.cfg.name
        session = await self._connection()
        if session is None:
            return CallOutcome(error_result(self._error or f"{server}: not connected yet"), "protocol_error")
        client, broken = session
        try:
            result = await client.call_tool(name, arguments, read_timeout_seconds=timeout)
        except MCPError as e:
            if e.code == REQUEST_TIMEOUT:
                return CallOutcome(
                    error_result(f"{server}: {name} did not answer within {timeout:g} s"), "timeout"
                )
            if e.code == CONNECTION_CLOSED:
                self._drop(session)
                return CallOutcome(
                    error_result(f"{server}: the connection closed during the call; it was not retried"),
                    "protocol_error",
                )
            return CallOutcome(error_result(f"{server}: {e.message}"), "protocol_error")
        except (RuntimeError, ValueError) as e:  # output-schema checks, validation: the session is fine
            return CallOutcome(error_result(f"{server}: {type(e).__name__}: {e}"), "protocol_error")
        except Exception as e:  # a broken transport: reconnect on the next call
            self._drop(session)
            return CallOutcome(
                error_result(f"{server}: {type(e).__name__}: {e}; the call was not retried"), "protocol_error"
            )
        return CallOutcome(_strip_server_info(result), "tool_error" if result.is_error else "ok")

    def _drop(self, session: tuple[Any, Any]) -> None:
        """Retire ``session`` (only if it is still current); the next caller reconnects."""
        if self._session is session:
            self._session = None
        session[1].set()


# -- OpenAPI ----------------------------------------------------------------------------------
def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return "" if v is None else str(v)


def _bracket(prefix: str, value: Any) -> list[tuple[str, str]]:
    """``{"a": {"b": 1}, "c": [2]}`` -> ``a[b]=1``, ``c[0]=2`` (Stripe's nesting, deepObject)."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _bracket(f"{prefix}[{k}]", v)]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in _bracket(f"{prefix}[{i}]", v)]
    return [(prefix, _scalar(value))]


def _query_pairs(name: str, value: Any, style: str, explode: bool) -> list[tuple[str, str]]:
    if style == "deepObject":
        return _bracket(name, value)
    if isinstance(value, list):
        if style == "form" and explode:
            return [(name, _scalar(v)) for v in value]
        sep = {"spaceDelimited": " ", "pipeDelimited": "|"}.get(style, ",")
        return [(name, sep.join(_scalar(v) for v in value))]
    if isinstance(value, dict):
        if style == "form" and explode:
            return [(str(k), _scalar(v)) for k, v in value.items()]
        return [(name, ",".join(f"{k},{_scalar(v)}" for k, v in value.items()))]
    return [(name, _scalar(value))]


# hosts a spec's own ``servers`` may not send the proxy to (a configured base_url still can): cloud
# instance metadata, which hands out credentials to whoever asks from inside the machine
METADATA_HOSTS = {"metadata.google.internal", "metadata", "metadata.azure.internal", "instance-data"}


def spec_host_refused(url: str) -> bool:
    """Whether a base URL that came from the spec, not the config, points at link-local or
    metadata addresses (169.254.169.254 and friends)."""
    host = (urllib.parse.urlsplit(url).hostname or "").rstrip(".").lower()
    if host in METADATA_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_link_local
    except ValueError:
        return False


def same_origin(url: str, base: str) -> bool:
    """Whether a request to ``url`` goes to ``base``'s scheme, host and port, as the HTTP client
    reads the two: the guard that keeps a source's headers (credentials) on its own base URL."""
    import httpx2

    try:
        a, b = httpx2.URL(url), httpx2.URL(base)
    except httpx2.InvalidURL:
        return False
    return (a.scheme, a.host, a.port) == (b.scheme, b.host, b.port)


class OpenAPIExecutor:
    def __init__(
        self,
        configs: dict[str, OpenAPIBackend],
        *,
        allow_write: bool = False,
        timeout: float = 60.0,
        transport: Any = None,
        max_chars: int = MAX_RESPONSE_CHARS,
    ):
        self.configs, self.allow_write, self.timeout = configs, allow_write, timeout
        self.transport, self.max_chars = transport, max_chars
        self._client: Any = None

    def build(
        self, tool: Tool, arguments: dict[str, Any] | None, extra_headers: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        """The request (``httpx2.AsyncClient.request`` keyword arguments) for one call; raises
        ``Refused`` for policy and unsupported shapes, ``ValueError`` for bad arguments.
        ``extra_headers`` (a tenant's) join the config's, under the same rule: configured base URL only."""
        http = tool.doc.get("http") or {}
        method = str(http.get("method") or "GET").upper()
        source = tool.category
        if method not in SAFE_METHODS and not self.allow_write:
            raise Refused(
                f"{tool.id} is a {method} operation; this proxy only sends GET and HEAD "
                "(start it with --allow-write to allow changes)"
            )
        cfg = self.configs.get(source)
        base = cfg.base_url if cfg is not None and cfg.base_url else str(http.get("base_url") or "")
        if not re.match(r"https?://", base):
            raise Refused(
                f"{source}: no absolute base URL ({base!r}); set openapi.{source}.base_url in the config"
            )
        if not (cfg is not None and cfg.base_url) and spec_host_refused(base):
            raise Refused(
                f"{source}: the spec sends this call to {base!r}, a link-local or metadata address; "
                f"set openapi.{source}.base_url in the config if that is really where the API lives"
            )
        headers = merge_headers(cfg.headers, extra_headers or {}) if cfg is not None and cfg.base_url else {}
        spec = http.get("args") or {}
        path = str(http.get("path") or "")
        if not path.startswith("/"):  # "@host/x" or ".host/x" after the base URL names another host
            raise Refused(f"{tool.id}: the path {path!r} does not start with '/'")
        query: list[tuple[str, str]] = []
        body: dict[str, Any] = {}
        whole, has_whole = None, False
        for key, value in (arguments or {}).items():
            arg = spec.get(key)
            if arg is None:
                raise ValueError(f"{tool.id} has no argument {key!r}")
            where, name = arg.get("in"), str(arg.get("name", key))
            if where == "path":
                text = _scalar(value)
                # a dot segment moves the request up the path; so does "../x", once an upstream
                # decodes the %2F that quoting makes of its slash
                parts = re.split(r"[/\\]", urllib.parse.unquote(text))
                if any(p in (".", "..") for p in parts):
                    raise ValueError(f"{tool.id}: path parameter {name!r} cannot be {text!r} (a dot segment)")
                path = path.replace("{" + name + "}", urllib.parse.quote(text, safe=""))
            elif where == "query":
                if isinstance(value, (list, dict)) and "style" not in arg:
                    raise Refused(f"{tool.id}: re-ingest the spec to pass a list or an object in {key!r}")
                query += _query_pairs(
                    name, value, str(arg.get("style", "form")), bool(arg.get("explode", True))
                )
            elif where == "body":
                if name:
                    body[name] = value
                else:
                    whole, has_whole = value, True
            else:
                raise Refused(f"{tool.id}: {where} parameters are not supported by the proxy yet")
        missing = re.findall(r"\{([^}]+)\}", path)
        if missing:
            raise ValueError(f"{tool.id} is missing path parameter(s): {', '.join(missing)}")
        url = base.rstrip("/") + path
        if not same_origin(url, base):  # the URL that is sent, however it came to be composed
            raise Refused(f"{tool.id}: {url!r} is not a URL on {base}")
        req: dict[str, Any] = {"method": method, "url": url, "params": query, "headers": headers}
        payload = whole if has_whole else (body or None)
        if payload is not None:
            media = str(http.get("body_media_type") or "application/json")
            if "json" in media:
                req["json"] = payload
            elif media == "application/x-www-form-urlencoded" and isinstance(payload, dict):
                pairs = [p for k, v in payload.items() for p in _bracket(k, v)]
                req["content"] = urllib.parse.urlencode(pairs).encode("utf-8")
                req["headers"] = {**headers, "Content-Type": media}
            else:
                raise Refused(f"{tool.id}: {media} bodies are not supported by the proxy yet")
        return req

    def _http(self) -> Any:
        import httpx2

        if self._client is None:
            from http.cookiejar import CookieJar, DefaultCookiePolicy

            # one client serves every caller: a cookie one response sets must not ride along on the
            # next caller's request, so no cookie is ever kept
            jar = CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))
            self._client = httpx2.AsyncClient(
                timeout=httpx2.Timeout(self.timeout),
                follow_redirects=False,
                transport=self.transport,
                cookies=jar,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def call(
        self, tool: Tool, arguments: dict[str, Any] | None, extra_headers: Mapping[str, str] | None = None
    ) -> CallOutcome:
        import httpx2
        from mcp_types import CallToolResult, TextContent

        try:
            req = self.build(tool, arguments, extra_headers)
        except Refused as e:
            return CallOutcome(error_result(str(e)), "refused")
        except ValueError as e:
            return CallOutcome(error_result(str(e)), "tool_error")
        try:  # the body is read only as far as the answer will show (a large one stays on the wire)
            async with self._http().stream(**req) as resp:
                status = resp.status_code
                if 300 <= status < 400:
                    text = f"HTTP {status}: redirect to {resp.headers.get('location', '?')} (not followed)"
                else:
                    raw, more = bytearray(), False
                    async for chunk in resp.aiter_bytes():
                        raw += chunk
                        if len(raw) > 4 * self.max_chars:  # UTF-8: at most 4 bytes a character
                            more = True
                            break
                    body = bytes(raw).decode(resp.encoding or "utf-8", errors="replace")
                    if more or len(body) > self.max_chars:
                        size = resp.headers.get("content-length")
                        body = (
                            body[: self.max_chars]
                            + "\n… truncated"
                            + (f": {size} bytes in total" if size else "")
                        )
                    text = f"HTTP {status}\n{body}"
        except httpx2.TimeoutException:
            return CallOutcome(
                error_result(f"{tool.category}: no answer within {self.timeout:g} s"), "timeout"
            )
        except httpx2.HTTPError as e:
            return CallOutcome(error_result(f"{tool.category}: {type(e).__name__}: {e}"), "protocol_error")
        failed = status >= 300
        result = CallToolResult(content=[TextContent(type="text", text=text)], is_error=failed)
        return CallOutcome(result, "tool_error" if failed else "ok", status)


# -- router -----------------------------------------------------------------------------------
class Backends:
    """``call_tool``'s router: MCP tools to their server's session, OpenAPI tools over HTTP."""

    def __init__(
        self,
        servers: Sequence[ServerConfig] = (),
        openapi: dict[str, OpenAPIBackend] | None = None,
        *,
        allow_write: bool = False,
        call_timeout: float = 60.0,
        connect_timeout: float = 60.0,
        transport: Any = None,
        tenants: Mapping[str, Any] | None = None,
    ):
        self.mcp = {c.name: MCPBackend(c, connect_timeout=connect_timeout) for c in servers}
        self.tenants = dict(tenants or {})  # name -> tenants.Tenant
        self.connect_timeout = connect_timeout
        self._own: dict[tuple[str, str], MCPBackend] = {}  # (server, tenant) -> its own connection
        self.http = OpenAPIExecutor(
            openapi or {}, allow_write=allow_write, timeout=call_timeout, transport=transport
        )
        self.call_timeout = call_timeout
        self._tg: Any = None

    @asynccontextmanager
    async def running(self) -> AsyncIterator[Backends]:
        """The task group the MCP sessions live in; leaving it closes every backend."""
        import anyio

        try:
            async with anyio.create_task_group() as tg:
                self._tg = tg
                try:
                    yield self
                finally:
                    tg.cancel_scope.cancel()
        finally:
            self._tg = None
            await self.http.aclose()

    def backend_for(self, server: str, tenant: str | None) -> MCPBackend | None:
        """The shared connection to ``server``, or the tenant's own when it has credentials for it."""
        shared = self.mcp.get(server)
        t = self.tenants.get(tenant) if tenant else None
        if shared is None or t is None:
            return shared
        headers, env = t.credentials(server)
        if not headers and not env:
            return shared
        own = self._own.get((server, t.name))
        if own is None:
            cfg = dataclasses.replace(
                shared.cfg, headers=merge_headers(shared.cfg.headers, headers), env={**shared.cfg.env, **env}
            )
            own = self._own[(server, t.name)] = MCPBackend(cfg, connect_timeout=self.connect_timeout)
        return own

    async def call(
        self, tool: Tool, arguments: dict[str, Any] | None, tenant: str | None = None
    ) -> CallOutcome:
        if "http" in tool.doc:
            t = self.tenants.get(tenant) if tenant else None
            extra = t.credentials(tool.category)[0] if t is not None else None
            return await self.http.call(tool, arguments, extra)
        backend = self.backend_for(tool.category, tenant)
        if backend is None:
            return CallOutcome(
                error_result(
                    f"{tool.category}: this MCP server is not configured for the proxy (--config or --server)"
                ),
                "refused",
            )
        if self._tg is None:
            raise RuntimeError("Backends.call() outside Backends.running()")
        name = str(tool.doc.get("name") or tool.id)
        return await backend.call(self._tg, name, dict(arguments or {}), timeout=self.call_timeout)
