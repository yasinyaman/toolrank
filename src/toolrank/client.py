"""A client for ``toolrank serve``'s REST API, standard library only, for code that runs its own
agent loop: ``toolrank.integrations`` (Anthropic, OpenAI) and framework adapters.

    tr = ToolrankClient("http://127.0.0.1:8765", api_key=os.environ.get("TOOLRANK_API_KEY"))
    catalog = tr.catalog()                     # every tool as a platform client needs it
    found = tr.search("refund the last payment of this customer")
    out = tr.call(found["tools"][0]["name"], {"payment_intent": "pi_1"}, search_id=found["search_id"])

``AsyncToolrankClient`` is the same API for asyncio code (``await tr.search(...)``): each call
runs the standard-library client in a worker thread, so the event loop never waits on the network
and nothing more is installed.

Redirects are not followed, so the bearer token never reaches another host; a call is never retried,
since it may have had effects; errors raise ``ToolrankError`` with the HTTP status (0 when the
server was not reached) and the server's message. ``session`` becomes the ``X-Session-Id`` header,
which groups one conversation's searches and calls in the usage log. ``transport`` replaces the
HTTP layer: ``(method, url, headers, body, timeout) -> (status, body)``.
"""

from __future__ import annotations

import asyncio
import functools
import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from toolrank import __version__

Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


RANK_CHUNK = 200  # /v1/rank's limit per request
RANK_BYTES = 900_000  # under its 1 MiB body limit


def _chunks(tools: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    """(offset, tools) pieces of at most ``RANK_CHUNK`` tools and about ``RANK_BYTES`` of JSON."""
    out: list[tuple[int, list[dict[str, Any]]]] = []
    start, size, current = 0, 0, []
    for n, tool in enumerate(tools):
        weight = len(json.dumps(tool, ensure_ascii=False).encode("utf-8")) + 2
        if current and (len(current) >= RANK_CHUNK or size + weight > RANK_BYTES):
            out.append((start, current))
            start, size, current = n, 0, []
        current.append(tool)
        size += weight
    if current:
        out.append((start, current))
    return out


class ToolrankError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"toolrank {status}: {message}" if status else f"toolrank: {message}")
        self.status, self.message = status, message


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # the 3xx comes back as an HTTPError


_OPENER = urllib.request.build_opener(_NoRedirect)


def _urllib(
    method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except TimeoutError as e:
        raise ToolrankError(0, f"{method} {url} timed out after {timeout:g} s") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise ToolrankError(0, f"{method} {url} timed out after {timeout:g} s") from e
        raise ToolrankError(0, f"{method} {url}: server unreachable ({e.reason})") from e
    except (http.client.HTTPException, ConnectionError) as e:  # never retried: a call may have run
        raise ToolrankError(
            0, f"{method} {url}: connection lost before the answer ({type(e).__name__})"
        ) from e


class ToolrankClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        api_key: str | None = None,
        *,
        session: str | None = None,
        timeout: float = 30.0,
        call_timeout: float = 150.0,
        transport: Transport | None = None,
        user_agent: str = f"toolrank-client/{__version__}",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key, self.session = api_key, session
        self.timeout, self.call_timeout = timeout, call_timeout
        self.transport = transport or _urllib
        self.user_agent = user_agent

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        session: str | None = None,
        ok: tuple[int, ...] = (200,),
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if session or self.session:
            headers["X-Session-Id"] = str(session or self.session)
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        status, raw = self.transport(method, self.base_url + path, headers, data, timeout or self.timeout)
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError:
            payload = None
        if status not in ok:
            message = payload.get("error") if isinstance(payload, dict) else None
            raise ToolrankError(status, str(message or raw[:300].decode("utf-8", "replace") or "no body"))
        if not isinstance(payload, dict):
            raise ToolrankError(status, "the answer is not a JSON object")
        return payload

    def health(self) -> dict[str, Any]:
        """``{ready, mode}``; a server still building its index is no error."""
        return self._request("GET", "/healthz", ok=(200, 503))

    def catalog(self, server: str | None = None) -> dict[str, Any]:
        """``{count, catalog, tools}``: each tool's id (``name``), ``api_name``, description, input
        schema, annotations and, for OpenAPI operations, the HTTP method."""
        query = {"full": "true", **({"server": server} if server else {})}
        return self._request("GET", "/v1/tools?" + urllib.parse.urlencode(query))

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        instruction: str | None = None,
        full_schemas: bool = False,
        session: str | None = None,
    ) -> dict[str, Any]:
        """``{search_id, mode, took_ms, rule, tools}``, with ``confidence`` when the server has a
        calibration and ``note`` when the agent should know something about the list."""
        body: dict[str, Any] = {"query": query}
        if k is not None:
            body["k"] = k
        if instruction is not None:
            body["instruction"] = instruction
        if full_schemas:
            body["full_schemas"] = True
        return self._request("POST", "/v1/search", body, session=session)

    def rank(
        self,
        query: str,
        tools: list[dict[str, Any]],
        *,
        instruction: str | None = None,
        session: str | None = None,
    ) -> list[dict[str, Any]]:
        """Score tools the caller has (MCP records: ``name``, ``description``, ``inputSchema``,
        optional ``server``) for ``query``; -> ``[{index, name, server?, score}]``, best first,
        ``index`` being the tool's position in ``tools``. Sent in chunks that stay under the
        server's limits (``RANK_CHUNK`` tools, ``RANK_BYTES`` of JSON)."""
        out: list[dict[str, Any]] = []
        for start, chunk in _chunks(tools):
            body: dict[str, Any] = {"query": query, "tools": chunk}
            if instruction is not None:
                body["instruction"] = instruction
            for row in self._request("POST", "/v1/rank", body, session=session)["tools"]:
                out.append({**row, "index": start + int(row["index"])})
        return sorted(out, key=lambda r: -r["score"])

    def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        search_id: str | None = None,
        session: str | None = None,
    ) -> dict[str, Any]:
        """``{name, call_id, outcome, isError, http_status, content, structuredContent?}``; a tool
        that fails is a result with ``isError``, not an exception."""
        body: dict[str, Any] = {"name": name, "arguments": arguments or {}}
        if search_id:
            body["search_id"] = search_id
        return self._request("POST", "/v1/call", body, timeout=self.call_timeout, session=session)

    def feedback(
        self,
        name: str,
        outcome: str,
        *,
        search_id: str | None = None,
        took_ms: float | None = None,
        session: str | None = None,
    ) -> dict[str, Any]:
        """Report a call the client ran itself (``outcome``: ``ok`` or ``tool_error``), so the server's
        usage log, and with it ``toolrank learn``, sees it; -> ``{name, call_id}``."""
        body: dict[str, Any] = {"name": name, "outcome": outcome}
        if search_id:
            body["search_id"] = search_id
        if took_ms is not None:
            body["took_ms"] = took_ms
        return self._request("POST", "/v1/feedback", body, session=session)


class AsyncToolrankClient:
    """``ToolrankClient``'s API as coroutines: each method runs the standard-library client in a
    worker thread (``asyncio.to_thread``, which carries the caller's context). Takes the same
    arguments, or ``client=`` an existing one. Cancelling a ``call`` drops its answer; the request
    was sent, so the tool may still run."""

    def __init__(self, *args: Any, client: ToolrankClient | None = None, **kwargs: Any):
        if client is not None and (args or kwargs):
            raise TypeError("give either client= or ToolrankClient's arguments, not both")
        self.sync = client or ToolrankClient(*args, **kwargs)

    async def _run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))

    async def health(self) -> dict[str, Any]:
        return await self._run(self.sync.health)

    async def catalog(self, server: str | None = None) -> dict[str, Any]:
        return await self._run(self.sync.catalog, server)

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """``ToolrankClient.search``: ``k``, ``instruction``, ``full_schemas``, ``session``."""
        return await self._run(self.sync.search, query, **kwargs)

    async def rank(self, query: str, tools: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        """``ToolrankClient.rank``: ``instruction``, ``session``."""
        return await self._run(self.sync.rank, query, tools, **kwargs)

    async def feedback(self, name: str, outcome: str, **kwargs: Any) -> dict[str, Any]:
        """``ToolrankClient.feedback``: ``search_id``, ``took_ms``, ``session``."""
        return await self._run(self.sync.feedback, name, outcome, **kwargs)

    async def call(self, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """``ToolrankClient.call``: ``search_id``, ``session``."""
        return await self._run(self.sync.call, name, arguments, **kwargs)
