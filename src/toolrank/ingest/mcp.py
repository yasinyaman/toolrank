"""MCP servers as tool sources: server configs, and MCP tools to toolrank ``Tool``s.

Pure (no SDK import); ``adapters/mcp_client.py`` does the talking. A server comes from a
``--server NAME=TARGET`` flag (an http(s) URL is streamable HTTP, anything else a stdio command
line) or from the JSON file MCP clients already keep: ``{"mcpServers": {...}}`` (Claude Desktop,
Cursor) or ``{"servers": {...}}`` (VS Code). Env vars and headers stay in the config: they are
never written next to the tools and never shown in a ``repr``.

String values may reference the environment as ``${VAR}`` or ``${env:VAR}`` (VS Code's form); a
missing variable is an error that names it. The same file can carry an ``"openapi"`` section for
``toolrank serve``: per OpenAPI source a ``base_url`` and ``headers`` (credentials), which are only
ever sent to that base URL.

A tool is ``<server>/<tool name>`` with ``category`` = the server name, indexed by
``ingest.text.tool_text``. Ids are never parsed back (names may contain ``/``); code that needs the
server reads ``Tool.category`` and the name ``Tool.doc["name"]``.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolrank.domain import Tool
from toolrank.ingest.text import MAX_CHARS, tool_text

# what Tool.doc keeps of an MCP tool (the spec's field names); icons and _meta are dropped
DOC_KEYS = ("name", "title", "description", "inputSchema", "outputSchema", "annotations")
_HTTP_TYPES = {"http", "streamable-http", "streamablehttp", "streamable_http"}


@dataclass(frozen=True)
class ServerConfig:
    name: str
    transport: str  # "stdio" | "http"
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict, repr=False)
    cwd: str | None = None
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError(f"server name {self.name!r}: empty or padded with spaces")
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"{self.name}: a stdio server needs a command")
        if self.transport == "http" and not re.match(r"https?://", self.url):
            raise ValueError(f"{self.name}: a streamable HTTP server needs an http(s) URL")
        if self.transport not in ("stdio", "http"):
            raise ValueError(f"{self.name}: unknown transport {self.transport!r}")


def parse_server(spec: str) -> ServerConfig:
    """``"github=https://host/mcp"`` (streamable HTTP) or ``"time=uvx mcp-server-time"`` (stdio)."""
    name, sep, target = spec.partition("=")
    name, target = name.strip(), target.strip()
    if not sep or not name or not target:
        raise ValueError(f"--server wants NAME=URL or NAME=COMMAND, got {spec!r}")
    if re.match(r"https?://", target):
        return ServerConfig(name, "http", url=target)
    command, *args = shlex.split(target)
    return ServerConfig(name, "stdio", command=command, args=tuple(args))


_ENV = re.compile(r"\$\{(?:env:)?([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any, where: str = "") -> Any:
    """``${VAR}`` / ``${env:VAR}`` in strings (also inside lists and dicts) from ``os.environ``."""
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            if m.group(1) not in os.environ:
                raise ValueError(f"{where or 'config'}: environment variable {m.group(1)} is not set")
            return os.environ[m.group(1)]

        return _ENV.sub(sub, value)
    if isinstance(value, list):
        return [expand_env(v, where) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v, where) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class OpenAPIBackend:
    """How ``toolrank serve`` reaches one OpenAPI source: base URL and headers (credentials)."""

    name: str
    base_url: str = ""
    headers: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.headers and not re.match(r"https?://", self.base_url):
            raise ValueError(f"openapi.{self.name}: headers need an absolute base_url to be sent to")


def _read_json(path: str | Path) -> Any:
    """A config file as JSON; one that cannot be read or parsed is a ``ValueError`` naming the file
    and, when it is a matter of permissions, the user toolrank runs as (in a container, not you)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as e:  # a ValueError of its own, but one that does not name the file
        raise ValueError(f"{path}: not UTF-8 ({e.reason} at byte {e.start})") from e
    except OSError as e:
        who = ""
        if isinstance(e, PermissionError) and hasattr(os, "getuid"):
            who = f": toolrank runs as uid {os.getuid()}, which must be able to read it"
        raise ValueError(f"{path}: {e.strerror or e}{who}") from e
    try:
        return json.loads(text)
    except ValueError as e:
        raise ValueError(f"{path}: not JSON ({e})") from e


def load_openapi(path: str | Path) -> dict[str, OpenAPIBackend]:
    """The ``"openapi"`` section of a config file (empty when there is none)."""
    data = _read_json(path)
    section = data.get("openapi") if isinstance(data, dict) else None
    out: dict[str, OpenAPIBackend] = {}
    for name, entry in (section or {}).items():
        entry = expand_env(entry, f"openapi.{name}")
        headers = {str(k): str(v) for k, v in (entry.get("headers") or {}).items()}
        out[str(name)] = OpenAPIBackend(str(name), str(entry.get("base_url") or ""), headers)
    return out


def config_from_entry(name: str, entry: dict[str, Any]) -> ServerConfig:
    entry = expand_env(entry, name)
    kind = str(entry.get("type") or entry.get("transport") or "").lower()
    if kind == "sse":
        raise ValueError(f"{name}: the SSE transport is not supported yet; use streamable HTTP")
    url = entry.get("url") or entry.get("serverUrl")
    if url or kind in _HTTP_TYPES:
        headers = {str(k): str(v) for k, v in (entry.get("headers") or {}).items()}
        return ServerConfig(name, "http", url=str(url or ""), headers=headers)
    env = {str(k): str(v) for k, v in (entry.get("env") or {}).items()}
    return ServerConfig(
        name,
        "stdio",
        command=str(entry.get("command") or ""),
        args=tuple(str(a) for a in entry.get("args") or ()),
        env=env,
        cwd=entry.get("cwd"),
    )


def load_config(path: str | Path) -> list[ServerConfig]:
    """Servers from an MCP client config file; entries marked ``"disabled": true`` are skipped."""
    data = _read_json(path)
    servers = (data.get("mcpServers") or data.get("servers")) if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        if isinstance(data, dict) and data.get("openapi"):
            return []  # a serve config with only OpenAPI sources
        raise ValueError(f"{path}: no 'mcpServers' (Claude Desktop, Cursor) or 'servers' (VS Code) object")
    return [config_from_entry(str(n), e) for n, e in servers.items() if not e.get("disabled")]


def tool_from_mcp(server: str, record: dict[str, Any], *, max_chars: int | None = MAX_CHARS) -> Tool:
    """One entry of a ``tools/list`` result (spec field names) -> ``Tool``."""
    name = str(record.get("name") or "")
    if not name:
        raise ValueError(f"{server}: a tool without a name")
    description = str(record.get("description") or record.get("title") or "")
    doc = {"server": server, **{k: record[k] for k in DOC_KEYS if record.get(k) is not None}}
    return Tool(
        id=f"{server}/{name}",
        doc=doc,
        documentation=tool_text(server, name, description, record.get("inputSchema"), max_chars=max_chars),
        category=server,
    )


def tools_from_listing(server: str, records: Iterable[dict[str, Any]]) -> list[Tool]:
    """A whole listing; a repeated tool name keeps its first entry."""
    out: dict[str, Tool] = {}
    for record in records:
        tool = tool_from_mcp(server, record)
        out.setdefault(tool.id, tool)
    return list(out.values())
