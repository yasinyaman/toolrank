"""Tenants: the named API keys of ``serve --api-keys`` and what each may reach.

The file is a JSON object {name: entry}. An entry is the key itself (every source, the config's
credentials: the form before tenants), or an object:

    {"team": {"key": "${TEAM_KEY}",
              "sources": ["github", "time"],
              "headers": {"github": {"Authorization": "Bearer ${TEAM_GITHUB_TOKEN}"}},
              "env": {"time": {"TZ": "Europe/Istanbul"}}}}

``sources`` limits the key to those sources' tools: searches, ``/v1/tools``, ``/v1/rank`` by id
and calls see nothing else, and a tool outside them is answered like one that does not exist (its
name is not confirmed). Without ``sources`` the key reaches every source. ``headers`` are sent with
the key's calls to that source on top of the config's (an OpenAPI source's configured
``base_url`` only; a streamable HTTP MCP server's connection), ``env`` is added to a stdio MCP
server's environment; a source with either gets a connection of its own per key, so one tenant's
credentials never serve another's call. ``${VAR}`` is read from the environment, as in the config.

The catalogue, the index and the embedding cache are shared: tenants differ in what they may see
and call, in their heads (``DATA/heads/tenants/<name>/``), their co-use table and their lines of
the usage log (``tenant``), not in their copy of the tools.

Pure: nothing here imports an adapter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolrank.ingest.mcp import expand_env


@dataclass(frozen=True)
class Tenant:
    name: str
    key: str = field(repr=False)
    sources: frozenset[str] | None = None  # None: every source
    headers: Mapping[str, Mapping[str, str]] = field(default_factory=dict, repr=False)
    env: Mapping[str, Mapping[str, str]] = field(default_factory=dict, repr=False)

    def allows(self, source: str) -> bool:
        return self.sources is None or source in self.sources

    def credentials(self, source: str) -> tuple[dict[str, str], dict[str, str]]:
        """(headers, env) this tenant adds to its calls to ``source``."""
        return dict(self.headers.get(source) or {}), dict(self.env.get(source) or {})


def _strings(where: str, value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError(f"{where}: expected an object of strings")
    return {k: expand_env(v, where) for k, v in value.items()}


def _per_source(where: str, value: Any) -> dict[str, dict[str, str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object {{source: {{name: value}}}}")
    return {str(source): _strings(f"{where}.{source}", inner) for source, inner in value.items()}


def parse_tenants(raw: Any, where: str = "--api-keys") -> dict[str, Tenant]:
    """The parsed file's object -> {name: Tenant}; raises ``ValueError`` naming what is wrong."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{where}: expected a JSON object {{name: key}} or {{name: {{key, sources, ...}}}}")
    out: dict[str, Tenant] = {}
    for name, entry in raw.items():
        at = f"{where} {name}"
        # the name goes into paths (DATA/heads/tenants/<name>), session keys and arm labels
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name)
            or name in (".", "..")
        ):
            raise ValueError(
                f"{at}: a name is 1-64 letters, digits, '_', '-' and '.' (not '.' or '..' alone)"
            )
        if isinstance(entry, str):
            entry = {"key": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"{at}: expected a key string or an object with a key")
        unknown = set(entry) - {"key", "sources", "headers", "env"}
        if unknown:
            raise ValueError(f"{at}: unknown field(s) {sorted(unknown)}")
        key = entry.get("key")
        key = expand_env(key, at) if isinstance(key, str) else key
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{at}: needs a non-empty string key")
        sources = entry.get("sources")
        if sources is not None:
            if not isinstance(sources, list) or not all(isinstance(s, str) and s for s in sources):
                raise ValueError(f"{at}.sources: expected a list of source names")
            sources = frozenset(sources)
        headers, env = (
            _per_source(f"{at}.headers", entry.get("headers")),
            _per_source(f"{at}.env", entry.get("env")),
        )
        if sources is not None:
            outside = sorted((set(headers) | set(env)) - sources)
            if outside:
                raise ValueError(f"{at}: credentials for {outside}, which are not among its sources")
        out[name] = Tenant(name, key, sources, headers, env)
    keys = [t.key for t in out.values()]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{where}: two names share a key")
    return out


def load_tenants(path: str | Path) -> dict[str, Tenant]:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"--api-keys {path}: {e}") from e
    return parse_tenants(raw, f"--api-keys {path}")


def check_sources(
    tenants: Mapping[str, Tenant], catalogue: Iterable[str], mcp: Mapping[str, str], openapi: Iterable[str]
) -> list[str]:
    """Raise when a tenant has credentials for a source the config cannot call with them (``mcp``
    maps a configured MCP server to its transport); -> warnings for allowed sources that are not
    in the catalogue (yet)."""
    known, apis = set(catalogue), set(openapi)
    warnings = []
    for t in tenants.values():
        for source in t.headers:
            if source not in apis and mcp.get(source) != "http":
                raise ValueError(
                    f"--api-keys {t.name}: headers for {source!r}, which is neither an OpenAPI source of the "
                    "config nor a streamable HTTP MCP server"
                )
        for source in t.env:
            if mcp.get(source) != "stdio":
                raise ValueError(f"--api-keys {t.name}: env for {source!r}, which is not a stdio MCP server")
        missing = sorted((t.sources or set()) - known)
        if missing:
            warnings.append(f"--api-keys {t.name}: sources not in the catalogue (yet): {', '.join(missing)}")
    return warnings
