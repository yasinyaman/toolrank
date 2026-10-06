"""How sure a search is: ``confidence``, the share of answerable requests whose best score was at or
below this one's (backlog D2.14).

A best cosine of 0.6 means different things on different catalogues and backbones: the answerable
requests of MCP-Zero average 0.71, LiveMCPBench's 0.51 (``docs/reports/faz2-week5.md``), so no score
threshold carries over. A calibration fixes the meaning per catalogue. ``toolrank data gen-queries``
writes requests for the catalogue's own tools, each answerable by construction; ``toolrank
calibrate`` ranks them the way a server does and keeps their best first-stage scores, sorted, in
``DATA/calibration.json``. A search's ``confidence`` is the empirical distribution function at its
best score: 0.05 says only 5% of the answerable requests scored lower.

It separates answerable requests from the rest no better than the raw score does (same order, same
AUROC). What it buys is a threshold that means the same everywhere: ``--min-confidence 0.05`` turns
away about 5% of answerable requests on any catalogue, so the false refusals are chosen, not
discovered. What it catches is measured, not promised: each request is also ranked with its gold
tools' sources hidden (a "twin", the same request on a catalogue that lacks the answer), and the
share of twins below the band is the catch rate ``calibrate`` reports. Formulas that rescale by the
number of options, (N·p − 1)/(N − 1), have no meaning for cosines and are not used.

An entry holds for one first stage (the scorer's name: backbone, formats, heads, server weight), one
heads file (sha256 prefix) and one serving instruction; a server whose first stage matches no entry
gives no confidence and turns nothing away. Promoting new heads (``toolrank ab``) or changing the
backbone therefore needs a new calibration. The catalogue may change: the distribution moves with
the backbone far more than with a few tools, and ``catalog`` records which one was measured.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import os
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolrank.domain import Query, Tool

FILE = "calibration.json"
VERSION = 1
MIN_REQUESTS = 50  # fewer, and the 5% band rests on two requests


@dataclass(frozen=True)
class Calibration:
    scorer: str  # the first stage's name
    heads: str | None  # sha256 prefix of the heads file, None without heads
    instruction: str  # the serving instruction the requests were ranked with
    best: tuple[float, ...]  # the answerable requests' best scores, ascending
    twins: tuple[float, ...] = ()  # the best scores with the gold tools' sources hidden, ascending
    recall_at_5: float | None = None  # a gold tool in the first five: is the set sane at all
    catalog: str | None = None  # the tools.jsonl hash it was measured on
    requests: str = ""  # where the requests came from
    made: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str | None, str]:
        return self.scorer, self.heads, self.instruction

    def confidence(self, score: float) -> float:
        """The share of the calibration's requests whose best score is at or below ``score``."""
        return bisect.bisect_right(self.best, score) / len(self.best)

    def refused(self, band: float) -> float:
        """The share of the answerable requests that ``--min-confidence band`` would turn away."""
        return sum(self.confidence(s) < band for s in self.best) / len(self.best)

    def caught(self, band: float) -> float | None:
        """The share of the twins (no answer in the catalogue) that it would turn away."""
        if not self.twins:
            return None
        return sum(self.confidence(s) < band for s in self.twins) / len(self.twins)

    def to_json(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["best"], out["twins"] = list(self.best), list(self.twins)
        return out

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Calibration:
        best = tuple(sorted(float(x) for x in d["best"]))
        if not best:
            raise ValueError("a calibration without requests")
        return cls(
            scorer=str(d["scorer"]),
            heads=d.get("heads"),
            instruction=str(d.get("instruction") or ""),
            best=best,
            twins=tuple(sorted(float(x) for x in d.get("twins") or ())),
            recall_at_5=d.get("recall_at_5"),
            catalog=d.get("catalog"),
            requests=str(d.get("requests") or ""),
            made=str(d.get("made") or ""),
            extra=dict(d.get("extra") or {}),
        )


def load(path: str | Path) -> list[Calibration]:
    """The entries of a calibration file; [] when there is none."""
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("version") != VERSION:
        raise ValueError(f"{p}: calibration version {raw.get('version')!r}, this toolrank reads {VERSION}")
    return [Calibration.from_json(e) for e in raw.get("entries") or []]


def save(path: str | Path, entry: Calibration) -> None:
    """Add ``entry`` to the file, replacing one for the same first stage, heads and instruction; the
    file is replaced whole, so a running server never reads half of it."""
    p = Path(path)
    kept = [e for e in load(p) if e.key != entry.key]
    body = json.dumps({"version": VERSION, "entries": [e.to_json() for e in [*kept, entry]]}, indent=1)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".calibration-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body + "\n")
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Calibrations:
    """A calibration file that may be rewritten while a server runs: read again when its size or
    modification time changes; a file that cannot be read counts as none (``error`` says why)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.error: str | None = None
        self._stamp: tuple[int, int] | None = None
        self._entries: dict[tuple[str, str | None, str], Calibration] = {}
        self._lock = threading.Lock()

    def find(self, scorer: str, heads: str | None, instruction: str) -> Calibration | None:
        try:
            st = self.path.stat()
            stamp: tuple[int, int] | None = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None
        if stamp != self._stamp:
            with self._lock:
                if stamp != self._stamp:
                    try:
                        entries = {e.key: e for e in load(self.path)} if stamp else {}
                        self.error = None
                    except (OSError, ValueError, KeyError, TypeError) as e:
                        entries, self.error = {}, f"{type(e).__name__}: {e}"
                    self._entries, self._stamp = entries, stamp
        return self._entries.get((scorer, heads, instruction))

    def __len__(self) -> int:
        return len(self._entries)


def measure(
    scorer: Any, tools: Sequence[Tool], queries: Sequence[Query], instruction: str, *, depth: int = 20
) -> tuple[list[float], list[float], float, int]:
    """Rank ``queries`` (requests written for ``tools``' own tools) with ``scorer``, a first stage,
    under the serving ``instruction`` -> (best scores, twins' best scores, Recall@5, requests whose
    gold tools are no longer in the catalogue and were skipped). The best scores come from a list as
    deep as a server's (``depth``: a server vote re-scores the top of it, so depth can matter); a
    twin's is that of the first tool, in the whole catalogue's order, from a source none of the
    request's gold tools belongs to."""
    source = {t.id: t.category for t in tools}
    kept = [q for q in queries if q.qrels and all(t in source for t in q.qrels)]
    asked = [dataclasses.replace(q, instruction=instruction) for q in kept]
    if not asked:
        return [], [], 0.0, len(queries)
    best: list[float] = []
    twins: list[float] = []
    hits = 0
    short, deep = scorer.rank(asked, min(depth, len(tools))), scorer.rank(asked, len(tools))
    for q, r, whole in zip(asked, short, deep, strict=True):
        # a shared index (pgvector) may hold another catalogue's rows: a server skips them too
        ranked = [(t, s) for t, s in zip(r.tool_ids, r.scores, strict=True) if t in source]
        if not ranked:
            continue
        best.append(float(ranked[0][1]))
        hits += any(t in q.qrels for t, _ in ranked[:5])
        gold = {source[t] for t in q.qrels}
        others = (
            s
            for t, s in zip(whole.tool_ids, whole.scores, strict=True)
            if t in source and source[t] not in gold
        )
        twin = next(others, None)
        if twin is not None:
            twins.append(float(twin))
    return best, twins, (hits / len(best) if best else 0.0), len(queries) - len(kept)


def band_score(cal: Calibration, band: float) -> float:
    """The lowest best score that ``--min-confidence band`` still lets through."""
    return next(s for s in cal.best if cal.confidence(s) >= band)
