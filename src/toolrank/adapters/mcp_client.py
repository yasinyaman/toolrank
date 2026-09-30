"""A ``ToolSource`` over one MCP server, stdio or streamable HTTP, through the official ``mcp`` SDK.

``mcp`` (the ``[mcp]`` extra) is imported inside functions, like torch in ``adapters/clm.py``.
``fetch_tools`` is the async core, which the proxy calls from its own event loop;
``fetch_many`` lists several servers concurrently and ``MCPServerSource.list_tools`` wraps one for
synchronous callers. The high-level ``mcp.Client`` negotiates the protocol version with old and new
servers alike. One timeout covers the whole listing (process start or connection, handshake and
every page); a stdio server's stderr goes to a temp file and its tail is shown when listing fails.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import Any, TextIO

from toolrank.domain import Tool
from toolrank.ingest.mcp import ServerConfig, tools_from_listing

MAX_PAGES = 1000
_DOWNLOADERS = {"npx", "uvx", "bunx", "pipx", "pnpm", "dlx"}


class MCPListingError(RuntimeError):
    """A server could not be listed; an ingest keeps that server's previous tools."""


async def list_all(client: Any, *, max_pages: int = MAX_PAGES) -> list[dict[str, Any]]:
    """Every tool of a connected ``mcp.Client``, spec field names, following ``next_cursor``."""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(max_pages):
        page = await client.list_tools(cursor=cursor)
        out += [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in page.tools]
        cursor = page.next_cursor
        if cursor is None:
            return out
        if cursor in seen:
            raise MCPListingError(f"the server repeated the page cursor {cursor!r}")
        seen.add(cursor)
    raise MCPListingError(f"more than {max_pages} pages of tools")


def _stderr_tail(errlog: TextIO, chars: int = 600) -> str:
    errlog.flush()
    errlog.seek(0)
    tail = errlog.read()[-chars:].strip()
    return f"\n  server stderr: …{tail}" if tail else ""


def _leaves(e: BaseException) -> list[BaseException]:
    if isinstance(e, BaseExceptionGroup):
        return [x for sub in e.exceptions for x in _leaves(sub)]
    return [e]


def _reason(cfg: ServerConfig, e: Exception, timeout: float) -> str:
    leaves = _leaves(e)
    if any(isinstance(x, TimeoutError) for x in leaves):
        hint = (
            f" (the first `{cfg.command}` run downloads the package: raise --timeout or run it once by hand)"
            if cfg.transport == "stdio" and cfg.command in _DOWNLOADERS
            else ""
        )
        return f"{cfg.name}: no tool listing within {timeout:g} s{hint}"
    ours = [x for x in leaves if isinstance(x, MCPListingError)]
    detail = "; ".join(f"{type(x).__name__}: {x}".rstrip(": ") for x in (ours or leaves))
    return f"{cfg.name}: {detail}"


def _transport(cfg: ServerConfig, errlog: TextIO, http: Any) -> Any:
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client

    if cfg.transport == "stdio":
        params = StdioServerParameters(
            command=cfg.command, args=list(cfg.args), env=dict(cfg.env) or None, cwd=cfg.cwd
        )
        return stdio_client(params, errlog=errlog)
    return streamable_http_client(cfg.url, http_client=http)


async def fetch_tools(cfg: ServerConfig, *, timeout: float = 60.0) -> list[Tool]:
    """List one server's tools -> ``Tool``s; raises ``MCPListingError`` with the reason."""
    import anyio
    import httpx2
    from mcp import Client

    if cfg.transport == "stdio" and shutil.which(cfg.command) is None:
        raise MCPListingError(f"{cfg.name}: command not found: {cfg.command}")
    with tempfile.TemporaryFile("w+", encoding="utf-8") as errlog:
        try:
            with anyio.fail_after(timeout):
                async with AsyncExitStack() as stack:
                    http = None
                    if cfg.transport == "http":
                        http = await stack.enter_async_context(
                            httpx2.AsyncClient(
                                headers=dict(cfg.headers), timeout=httpx2.Timeout(30, read=timeout)
                            )
                        )
                    client = await stack.enter_async_context(
                        Client(_transport(cfg, errlog, http), cache=None)
                    )
                    records = await list_all(client)
        except Exception as e:  # the SDK's task groups may wrap what went wrong in exception groups
            raise MCPListingError(_reason(cfg, e, timeout) + _stderr_tail(errlog)) from e
    try:
        return tools_from_listing(cfg.name, records)
    except ValueError as e:
        raise MCPListingError(str(e)) from e


async def fetch_many(
    cfgs: Sequence[ServerConfig], *, timeout: float = 60.0
) -> dict[str, list[Tool] | MCPListingError]:
    """List every server concurrently -> {name: tools, or the error that server raised}."""
    import anyio

    results: dict[str, list[Tool] | MCPListingError] = {}

    async def one(cfg: ServerConfig) -> None:
        try:
            results[cfg.name] = await fetch_tools(cfg, timeout=timeout)
        except MCPListingError as e:
            results[cfg.name] = e

    async with anyio.create_task_group() as tg:
        for cfg in cfgs:
            tg.start_soon(one, cfg)
    return {cfg.name: results[cfg.name] for cfg in cfgs}


class MCPServerSource:
    """``ToolSource`` for one server (the CLI lists several at once with ``fetch_many``)."""

    kind = "mcp"

    def __init__(self, cfg: ServerConfig, *, timeout: float = 60.0):
        self.cfg, self.name, self.timeout = cfg, cfg.name, timeout

    def list_tools(self) -> list[Tool]:
        return asyncio.run(fetch_tools(self.cfg, timeout=self.timeout))
