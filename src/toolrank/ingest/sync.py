"""Incremental sync of an ingest directory: ``tools.jsonl`` plus a ``sources.json`` manifest.

A source that listed its tools this run replaces exactly its own tools (``Tool.category`` is the
source name); every other source stays as it is, and a source whose listing failed keeps its
previous tools. Two guards, each lifted by a flag: a source name stays bound to its kind (an MCP
server and an OpenAPI spec never replace each other by accident; ``replace``), and a listing of
0 tools does not wipe a source that had some (``allow_empty``). The embedding cache is keyed by
text, so an unchanged tool is never embedded again and a removed one simply leaves the corpus.

``sources.json`` holds each source's kind, tool count, last sync time and, for a spec, where it
came from; never env vars or headers. Both files are written through a temp file and
``os.replace``, tools sorted by source name and in listing order within a source.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from toolrank.datasets.jsonl import load_tools, write_tools
from toolrank.domain import Tool

TOOLS, SOURCES = "tools.jsonl", "sources.json"


@dataclass
class Listing:
    """What one source returned this run; ``tools`` is None when the listing failed."""

    name: str
    kind: str
    tools: list[Tool] | None
    error: str = ""
    info: dict[str, Any] = field(default_factory=dict)  # manifest extras: transport, origin, base_url


@dataclass
class SourceDiff:
    name: str
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    total: int = 0
    error: str = ""  # set when nothing changed for this source

    def line(self) -> str:
        if self.error:
            return f"{self.name}  failed: {self.error} (kept {self.total} tools)"
        return f"{self.name}  +{self.added} ~{self.changed} -{self.removed} ={self.unchanged}  ({self.total} tools)"


def _row(t: Tool) -> str:
    return json.dumps(
        {"id": t.id, "doc": t.doc, "documentation": t.documentation, "category": t.category},
        sort_keys=True,
        ensure_ascii=False,
    )


def load_dir(out: str | Path) -> tuple[list[Tool], dict[str, dict[str, Any]]]:
    out = Path(out)
    tools = load_tools(out / TOOLS) if (out / TOOLS).exists() else []
    manifest = json.loads((out / SOURCES).read_text(encoding="utf-8")) if (out / SOURCES).exists() else {}
    return tools, manifest


def _grouped(tools: Iterable[Tool]) -> dict[str, list[Tool]]:
    by_source: dict[str, list[Tool]] = defaultdict(list)
    for t in tools:
        by_source[t.category].append(t)
    return by_source


def _flatten(by_source: dict[str, list[Tool]]) -> list[Tool]:
    return [t for name in sorted(by_source) for t in by_source[name]]


def sync(
    tools: Sequence[Tool],
    manifest: dict[str, dict[str, Any]],
    listings: Sequence[Listing],
    *,
    replace: bool = False,
    allow_empty: bool = False,
    now: str | None = None,
) -> tuple[list[Tool], dict[str, dict[str, Any]], list[SourceDiff]]:
    """-> (the new corpus, the new manifest, one diff per listing)."""
    by_source = _grouped(tools)
    manifest = {k: dict(v) for k, v in manifest.items()}
    stamp = now or datetime.now(UTC).isoformat(timespec="seconds")
    diffs: list[SourceDiff] = []
    for listing in listings:
        old = by_source.get(listing.name, [])
        diff = SourceDiff(listing.name, total=len(old))
        known = (manifest.get(listing.name) or {}).get("kind")
        if listing.tools is None:
            diff.error = listing.error or "listing failed"
        elif known and known != listing.kind and not replace:
            diff.error = f"{listing.name!r} is already a {known} source (--replace to overwrite it, or pick another name)"
        elif not listing.tools and old and not allow_empty:
            diff.error = f"listed 0 tools where there were {len(old)} (--allow-empty to accept)"
        else:
            strays = [t.id for t in listing.tools if t.category != listing.name]
            if strays:
                raise ValueError(f"{listing.name}: tools of another source: {strays[:3]}")
            before = {t.id: _row(t) for t in old}
            after = {t.id: _row(t) for t in listing.tools}
            diff.added = sum(i not in before for i in after)
            diff.removed = sum(i not in after for i in before)
            diff.changed = sum(i in before and before[i] != row for i, row in after.items())
            diff.unchanged = len(after) - diff.added - diff.changed
            diff.total = len(listing.tools)
            by_source[listing.name] = list(listing.tools)
            manifest[listing.name] = {
                "kind": listing.kind,
                "tools": diff.total,
                "synced_at": stamp,
                **listing.info,
            }
        diffs.append(diff)
    return _flatten(by_source), manifest, diffs


def drop(
    tools: Sequence[Tool], manifest: dict[str, dict[str, Any]], names: Sequence[str]
) -> tuple[list[Tool], dict[str, dict[str, Any]], list[SourceDiff]]:
    """Remove whole sources; an unknown name is an error."""
    by_source = _grouped(tools)
    unknown = [n for n in names if n not in manifest and n not in by_source]
    if unknown:
        raise ValueError(f"no such source: {', '.join(unknown)}")
    diffs = [SourceDiff(n, removed=len(by_source.pop(n, []))) for n in names]
    manifest = {k: v for k, v in manifest.items() if k not in names}
    return _flatten(by_source), manifest, diffs


def write_dir(out: str | Path, tools: Sequence[Tool], manifest: dict[str, dict[str, Any]]) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{TOOLS}.tmp"
    write_tools(tmp, tools)
    os.replace(tmp, out / TOOLS)
    tmp = out / f"{SOURCES}.tmp"
    tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, out / SOURCES)
