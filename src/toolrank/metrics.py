"""Prometheus metrics of a running server (``GET /v1/metrics``, text exposition format 0.0.4).

``Metrics`` counts what the usage log sees, in the same two places (``UsageLog.search`` and
``UsageLog.call``), so MCP and REST traffic are counted alike and nothing is counted that is not
logged; ``--no-usage-log`` turns the files off, not the counters. No request text, tool arguments
or API key names reach a label: ``arm`` is reduced to its kind (``tenant`` for any key's own heads).

What is exposed:

* ``toolrank_searches_total{via,mode,arm}``, ``toolrank_search_duration_seconds`` (ranking, the
  request's embedding included), ``toolrank_search_tools_returned``, ``toolrank_search_empty_total``,
  ``toolrank_search_co_use_added_total``;
* ``toolrank_calls_total{kind,outcome,via}``, ``toolrank_call_duration_seconds``,
  ``toolrank_calls_linked_total{link}`` and ``toolrank_called_tool_rank`` (where the called tool
  stood in the search it is linked to: its ``le="5"`` bucket over its count is "top 5");
* the token estimate: ``toolrank_search_returned_tokens_total`` (the tools a search handed over)
  and ``toolrank_search_saved_tokens_total`` (the catalogue minus that, per search): what an agent
  did not have to read because it searched instead of loading every tool. A tool is counted as its
  name, description and input schema in JSON at ``CHARS_PER_TOKEN`` characters a token, with full
  schemas, and the searches answered while a new catalogue is still being sized claim no saving:
  an estimate, on the low side;
* whatever the caller passes to ``render`` (the server adds the catalogue's size, index readiness,
  the heads in use and the embedding cache's hits).

Stdlib only; nothing here imports an adapter.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any

CHARS_PER_TOKEN = 4
SECONDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
TOOLS = (0, 1, 2, 3, 5, 10, 20, 50)
RANKS = (1, 2, 3, 5, 10, 20)

# name -> (type, help, buckets)
FAMILIES: dict[str, tuple[str, str, tuple[float, ...]]] = {
    "toolrank_searches_total": ("counter", "Searches answered.", ()),
    "toolrank_search_duration_seconds": (
        "histogram",
        "Time to rank one search, its embedding included.",
        SECONDS,
    ),
    "toolrank_search_tools_returned": ("histogram", "Tools a search handed over.", TOOLS),
    "toolrank_search_empty_total": ("counter", "Searches that returned no tool.", ()),
    "toolrank_search_co_use_added_total": ("counter", "Tools appended to results as co-use partners.", ()),
    "toolrank_search_returned_tokens_total": (
        "counter",
        "Estimated tokens of the tools searches handed over.",
        (),
    ),
    "toolrank_search_saved_tokens_total": (
        "counter",
        "Estimated tokens not handed over: the whole catalogue minus what each search returned.",
        (),
    ),
    "toolrank_calls_total": ("counter", "Tool calls forwarded, by how they ended.", ()),
    "toolrank_call_duration_seconds": ("histogram", "Time a forwarded call took.", SECONDS),
    "toolrank_calls_linked_total": ("counter", "Calls by how they were tied to a search.", ()),
    "toolrank_called_tool_rank": ("histogram", "Rank of a called tool in the search it is linked to.", RANKS),
}

Sample = tuple[str, str, str, dict[str, str], float]  # name, type, help, labels, value


def tool_tokens(tool: Any) -> int:
    """A tool's rough size as an agent API receives it: name, description and input schema."""
    doc = getattr(tool, "doc", None) or {}
    schema = doc.get("inputSchema") or doc.get("parameters") or {}
    text = json.dumps(
        {"name": tool.name, "description": tool.description, "input_schema": schema}, default=str
    )
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def arm_kind(arm: str | None) -> str:
    """``base``, ``current``, ``candidate``, ``tenant`` or ``tenant-candidate``: never a key's name."""
    arm = arm or "base"
    if arm.startswith("tenant:"):
        return "tenant-candidate" if arm.endswith(":candidate") else "tenant"
    return arm


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(labels: dict[str, str] | Sequence[tuple[str, str]]) -> str:
    items = labels.items() if isinstance(labels, dict) else labels
    inner = ",".join(f'{k}="{_escape(str(v))}"' for k, v in items)
    return "{" + inner + "}" if inner else ""


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


class Metrics:
    """Thread-safe counters and histograms, rendered as Prometheus text."""

    def __init__(self, catalogue: Callable[[], tuple[Sequence[Any], str]] | None = None):
        self.catalogue = catalogue  # -> (tools, catalogue hash): what a search is compared against
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}
        self._catalog_tokens: tuple[str, int] | None = None
        self._sizing = False

    # -- recording ----------------------------------------------------------------------------
    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        buckets = FAMILIES[name][2]
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            h = self._histograms.setdefault(key, [0.0] * (len(buckets) + 2))  # buckets, sum, count
            for n, le in enumerate(buckets):
                if value <= le:
                    h[n] += 1
            h[-2] += value
            h[-1] += 1

    def catalog_tokens(self, catalog: str | None = None, *, wait: bool = True) -> int | None:
        """The estimated tokens of the whole catalogue, cached per catalogue hash. None without a
        ``catalogue`` to ask, or when ``catalog`` names another catalogue than the current one.
        With ``wait=False`` (a search being logged) an estimate that is not there yet is computed in
        the background and this call returns None: sizing 40,000 tools must not hold a request up."""
        if self.catalogue is None:
            return None
        with self._lock:
            cached = self._catalog_tokens
            if cached is not None and catalog == cached[0]:
                return cached[1]
            if not wait:
                if not self._sizing:
                    self._sizing = True
                    threading.Thread(target=self._size, name="toolrank-metrics", daemon=True).start()
                return None
        cached = self._size()
        return cached[1] if cached is not None and catalog in (None, cached[0]) else None

    def _size(self) -> tuple[str, int] | None:
        """Size the current catalogue (unless it is the cached one) -> (hash, tokens)."""
        try:
            assert self.catalogue is not None
            tools, current = self.catalogue()
            with self._lock:
                cached = self._catalog_tokens
            if cached is None or cached[0] != current:
                cached = (current, sum(tool_tokens(t) for t in tools))
            with self._lock:
                self._catalog_tokens = cached
            return cached
        except Exception:  # the index is not there yet: no estimate, searches are still counted
            return None
        finally:
            with self._lock:
                self._sizing = False

    def search(self, result: Any, *, via: str, arm: str | None = None) -> None:
        """One answered search (a ``retriever.SearchResult``)."""
        hits = list(result.hits)
        self.inc(
            "toolrank_searches_total", via=via, mode=getattr(result, "mode", "semantic"), arm=arm_kind(arm)
        )
        self.observe("toolrank_search_duration_seconds", float(result.took_ms) / 1000.0)
        self.observe("toolrank_search_tools_returned", len(hits))
        if not hits:
            self.inc("toolrank_search_empty_total")
        added = int(getattr(result, "added", 0) or 0)
        if added:
            self.inc("toolrank_search_co_use_added_total", added)
        returned = sum(tool_tokens(h.tool) for h in hits)
        self.inc("toolrank_search_returned_tokens_total", returned)
        whole = self.catalog_tokens(getattr(result, "catalog", None), wait=False)
        if whole is not None:
            self.inc("toolrank_search_saved_tokens_total", max(0, whole - returned))

    def call(
        self, *, kind: str | None, outcome: str, via: str, took_ms: float, link: str, rank: int | None
    ) -> None:
        """One forwarded call and how it was tied to a search."""
        self.inc("toolrank_calls_total", kind=kind or "unknown", outcome=outcome, via=via)
        self.observe("toolrank_call_duration_seconds", float(took_ms) / 1000.0)
        self.inc("toolrank_calls_linked_total", link=link)
        if rank is not None:
            self.observe("toolrank_called_tool_rank", rank)

    # -- reading ------------------------------------------------------------------------------
    def render(self, extra: Iterable[Sample] = ()) -> str:
        """Everything recorded, then ``extra`` samples (name, type, help, labels, value)."""
        with self._lock:
            counters = dict(self._counters)
            histograms = {k: list(v) for k, v in self._histograms.items()}
        lines: list[str] = []
        for name, (kind, text, buckets) in FAMILIES.items():
            rows = sorted(k for k in (counters if kind == "counter" else histograms) if k[0] == name)
            if not rows and kind == "histogram":
                continue
            lines += [f"# HELP {name} {text}", f"# TYPE {name} {kind}"]
            if kind == "counter":
                if not rows:  # a counter nobody touched is 0, not absent
                    lines.append(f"{name} 0")
                lines += [
                    f"{name}{_labels(labels)} {_number(counters[(name, labels)])}" for _, labels in rows
                ]
                continue
            for _, labels in rows:
                h = histograms[(name, labels)]
                for le, n in zip(buckets, h, strict=False):
                    lines.append(f"{name}_bucket{_labels([*labels, ('le', _number(le))])} {_number(n)}")
                lines.append(f"{name}_bucket{_labels([*labels, ('le', '+Inf')])} {_number(h[-1])}")
                lines.append(f"{name}_sum{_labels(labels)} {_number(h[-2])}")
                lines.append(f"{name}_count{_labels(labels)} {_number(h[-1])}")
        seen: set[str] = set()
        for name, kind, text, labels, value in extra:
            if name not in seen:
                seen.add(name)
                lines += [f"# HELP {name} {text}", f"# TYPE {name} {kind}"]
            lines.append(f"{name}{_labels(labels)} {_number(value)}")
        return "\n".join(lines) + "\n"
