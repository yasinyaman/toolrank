"""Co-use: the tools agents call together, counted from the usage log.

A request that needs several tools is only served well when all of them are in the list
(``Comprehensiveness``). Ranking scores each tool on its own; the log knows something ranking does
not: which tools were called after the same request. ``co_use`` turns the log's events into a
table, tool -> the partners called along with it in at least ``min_count`` requests and in at
least ``min_p`` of the requests where the tool itself was called, and ``partners`` / ``expand``
append such partners of the tools shown to a result list. Only tool ids are read: no request text,
no arguments.

``CoUseTable`` is what a server holds (``serve --co-use N``): the tables of its own log's latest
``DAYS`` daily files, built when the server starts and again in the background once they are older
than ``every`` seconds, so a search never waits for them. Each API key gets the table of its own
requests (``tenant``; requests without a named key share one), so one tenant's traffic never shapes
another's results.

Measured on a simulated ToolRet log (``scripts/couse_sweep.py``, ``docs/reports/faz2-week5.md``):
count 2 / share 0.5 / 2 partners changed 5% of the lists and raised completeness by 0.6 points for
0.05 more tools a list, the same with heads learned from that log; spending the tools on a longer
ranked list buys a seventh of that per tool.

Nothing here imports an adapter.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from collections.abc import Container, Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from toolrank.usage import may_learn_from, read_events

Partners = dict[str, list[tuple[str, float, int]]]  # tool -> [(partner, p(partner | tool), count)]
MIN_COUNT, MIN_P = 2, 0.5
DAYS = 30  # daily log files a server's table is built from


def co_use(
    events: Iterable[dict[str, Any]],
    *,
    min_count: int = MIN_COUNT,
    min_p: float = MIN_P,
    since: str | None = None,
    tenant: str | None = None,
    exact_tenant: bool = False,
    counts: Counter[str] | None = None,
) -> Partners:
    """Tool -> its partners, most likely first. A request is one ``emb_hmac`` (its searches merge;
    a search without one counts alone) and its tools are those of the linked calls that ended
    ``ok``; ``since`` (an ISO date or timestamp) and ``tenant`` narrow the searches (with
    ``exact_tenant``, ``tenant=None`` means the requests without a named key, not all of them).
    Searches a Jev second stage answered shape no table (the provider's terms); when ``counts`` is
    given they are tallied in it."""
    events = list(events)
    request_of: dict[str, str] = {}
    for e in events:
        if e.get("event") != "search":
            continue
        if since and str(e.get("ts", "")) < since:
            continue
        if (exact_tenant or tenant) and e.get("tenant") != tenant:
            continue
        if not may_learn_from(e.get("scorer")):
            if counts is not None:
                counts["searches_with_jev"] += 1
            continue
        if not isinstance(e.get("id"), str):  # a line no version of the log writes: it says nothing
            continue
        request_of[e["id"]] = e.get("emb_hmac") or e["id"]
    called: dict[str, set[str]] = defaultdict(set)
    for e in events:
        ok = e.get("event") == "call" and e.get("outcome") == "ok" and isinstance(e.get("tool"), str)
        if ok and e.get("search_id") in request_of:
            called[request_of[e["search_id"]]].add(e["tool"])
    alone: Counter[str] = Counter()
    together: Counter[tuple[str, str]] = Counter()
    for tools in called.values():
        alone.update(tools)
        together.update(combinations(sorted(tools), 2))
    table: Partners = defaultdict(list)
    for (a, b), n in together.items():
        if n < min_count:
            continue
        for x, y in ((a, b), (b, a)):
            p = n / alone[x]
            if p >= min_p:
                table[x].append((y, p, n))
    return {t: sorted(rows, key=lambda r: (-r[1], -r[2], r[0])) for t, rows in table.items()}


def partners(
    shown: Sequence[str],
    table: Mapping[str, Sequence[tuple[str, float, int]]],
    extra: int,
    catalogue: Container[str] | None = None,
) -> list[tuple[str, str]]:
    """Up to ``extra`` (partner, the shown tool it is called with) pairs for tools not in ``shown``,
    the most likely first: a partner of several shown tools counts with its best share, and ties
    keep the order in which the shown tools brought them up. With ``catalogue``, a partner that is
    no longer in it (the log is older than the catalogue) is passed over."""
    if extra <= 0:
        return []
    have = set(shown)
    best: dict[str, tuple[float, str]] = {}
    for tool in shown:
        for partner, p, _ in table.get(tool, ()):
            if partner in have or (catalogue is not None and partner not in catalogue):
                continue
            if p > best.get(partner, (0.0, ""))[0]:
                best[partner] = (p, tool)
    more = sorted(best, key=lambda t: -best[t][0])[:extra]  # stable: first brought up, first kept
    return [(t, best[t][1]) for t in more]


def expand(
    shown: Sequence[str], table: Mapping[str, Sequence[tuple[str, float, int]]], extra: int
) -> list[str]:
    """``shown`` followed by its ``partners``."""
    return [*shown, *(t for t, _ in partners(shown, table, extra))]


class CoUseTable:
    """A usage log's co-use table, kept fresh without making a search wait (see the module's text)."""

    def __init__(
        self,
        directory: str | Path,
        *,
        min_count: int = MIN_COUNT,
        min_p: float = MIN_P,
        every: float = 300.0,
        days: int = DAYS,
    ):
        self.dir, self.min_count, self.min_p = Path(directory), min_count, min_p
        self.every, self.days = every, days
        self._lock = threading.Lock()
        self._building = False
        self._tables: dict[str | None, Partners] = {}
        self._counts: Counter[str] = Counter()
        self._at = 0.0
        self._build()

    def _build(self) -> None:
        tables: dict[str | None, Partners] = {}
        counts: Counter[str] = Counter()
        try:
            events = read_events(self.dir, newest=self.days)
            owner = {e["id"]: e.get("tenant") for e in events if e.get("event") == "search" and "id" in e}
            by_tenant: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
            for e in events:  # a call goes with the search it is linked to, whoever's key it names
                if e.get("event") == "search":
                    by_tenant[e.get("tenant")].append(e)
                elif e.get("event") == "call" and e.get("search_id") in owner:
                    by_tenant[owner[e["search_id"]]].append(e)
            for tenant, own in by_tenant.items():
                tables[tenant] = co_use(own, min_count=self.min_count, min_p=self.min_p, counts=counts)
        except (
            Exception
        ):  # an unreadable directory, or a line no code expected: no partners until the next try
            tables, counts = {}, Counter()
        with self._lock:
            self._tables, self._at, self._building = tables, time.monotonic(), False
            self._counts = counts

    def counts(self) -> Counter[str]:
        """What the latest build passed over (e.g. ``searches_with_jev``)."""
        with self._lock:
            return Counter(self._counts)

    def table(self, tenant: str | None = None) -> Partners:
        """``tenant``'s table (None: requests without a named key); tables older than ``every``
        seconds are rebuilt in the background."""
        with self._lock:
            stale = time.monotonic() - self._at >= self.every and not self._building
            if stale:
                self._building = True
            table = self._tables.get(tenant, {})
        if stale:
            threading.Thread(target=self._build, name="toolrank-co-use", daemon=True).start()
        return table
