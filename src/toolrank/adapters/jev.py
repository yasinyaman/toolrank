"""TypeSafe AI's Jev, a "System One" model, as a reranker over any scorer and as a chunked
standalone scorer.

Jev generates no text. A Choice question over up to 255 options returns a probability for every
option in one call (https://docs.typesafe.ai/primitives/choice): a ranking of those options for
the request in the state. Two adapters:

* ``JevReranker`` wraps a scorer. The base list to ``depth`` becomes one Choice question, the
  probabilities become the scores (ties, which Jev rounds to 0.0, keep the base order) and the base
  list continues below ``depth`` with negative scores, so a top-100 list stays complete and the
  metrics below ``depth`` are the base scorer's.
* ``JevScorer`` ranks a whole corpus the way TypeSafe's docs suggest past 255 options: the corpus
  in chunks of ``chunk`` tools, one Choice per chunk, the top ``per_chunk`` of every chunk into one
  final Choice (``per_chunk`` shrinks so the final round has at most 255 candidates); the final
  order first, then the rest of round one by probability. Feasible for a few thousand tools, not
  for ToolRet's 44k.

The state is ``{"request": ...}``; the benchmark instruction (w/ inst) leads the question. Option
keys are ``t000``...; the option text is the tool in ``tool_format`` cut to ``max_chars`` (Jev's
budget is 32k tokens for the state plus the longest question). Standard library only. The key is
``TYPESAFE_API_KEY``; responses are cached in SQLite by model and request body, so a rerun costs
nothing and ranks exactly the same. Pin a versioned model id (``jev-1.13.0``): aliases move.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from toolrank.domain import Query, RankedList, Tool
from toolrank.formats import QUERY_FORMATS, TOOL_FORMATS, NamedFormatter

__all__ = ["MAX_OPTIONS", "JevClient", "JevReranker", "JevScorer", "choice_question"]

MAX_OPTIONS = 255  # a Choice question's limit (https://docs.typesafe.ai/api)
DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_URL = "https://api.typesafe.ai/v1"
LEAD = "Which of the listed tools should be called to carry out `request`?"


def option_key(n: int) -> str:
    return f"t{n:03d}"


def choice_question(instruction: str, texts: Sequence[str], max_chars: int) -> dict[str, Any]:
    """One Choice over ``texts`` (keys ``t000``..), the instruction first when there is one."""
    if not 1 <= len(texts) <= MAX_OPTIONS:
        raise ValueError(f"a Choice takes 1..{MAX_OPTIONS} options, not {len(texts)}")
    head = instruction.strip()
    return {
        "type": "choice",
        "instructions": f"{head}\n\n{LEAD}" if head else LEAD,
        "criteria": {option_key(n): x[:max_chars] for n, x in enumerate(texts)},
    }


def _order(probabilities: dict[str, float], n: int) -> list[tuple[int, float]]:
    """Option positions best first: by probability, ties by the position (the base order)."""
    p = [float(probabilities.get(option_key(i), 0.0)) for i in range(n)]
    return [(i, p[i]) for i in sorted(range(n), key=lambda i: (-p[i], i))]


class JevCache:
    """request body -> answers, one SQLite file; a rerun asks nothing and ranks the same."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=60.0, check_same_thread=False)
        self._lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS jev (key TEXT PRIMARY KEY, model TEXT, answers TEXT)")
        self.db.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.db.execute("SELECT answers FROM jev WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, answers: dict[str, Any]) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO jev (key, model, answers) VALUES (?, ?, ?)",
                (key, model, json.dumps(answers, ensure_ascii=False)),
            )
            self.db.commit()


def _error_body(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode("utf-8", errors="replace")[:500]
    except Exception:  # noqa: BLE001 - a synthetic error without a body
        return ""


class JevClient:
    """``POST {base_url}/systemone``: one state, a map of questions, the answers back.

    ``ask_many`` runs requests in ``workers`` threads (TypeSafe's limit is 40 requests/s). Every
    call that reaches the endpoint counts in ``calls``, its input tokens in ``tokens_spent`` and its
    wall-clock in ``call_ms``; cache hits count in ``cached``. Tests replace ``_post``.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        *,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
        workers: int = 8,
    ):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.cache = JevCache(Path(cache_dir) / "jev.sqlite") if cache_dir else None
        self.timeout, self.max_retries, self.workers = timeout, max_retries, max(1, workers)
        self.calls = self.cached = self.tokens_spent = 0
        self.call_ms: list[float] = []
        self._lock = threading.Lock()

    def ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """The ``answers`` for one evaluation, from the cache when the same body was sent before."""
        body = {"state": state, "model": self.model, "questions": questions}
        key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            with self._lock:
                self.cached += 1
            return hit
        t0 = time.perf_counter()
        resp = self._post(body)
        dt = (time.perf_counter() - t0) * 1000.0
        answers = resp["answers"]
        with self._lock:
            self.calls += 1
            self.tokens_spent += int((resp.get("usage") or {}).get("input_tokens") or 0)
            self.call_ms.append(dt)
        if self.cache is not None:
            self.cache.put(key, str(resp.get("model") or self.model), answers)
        return answers

    def ask_many(self, items: Sequence[tuple[Any, dict[str, Any]]]) -> list[dict[str, Any]]:
        if self.workers == 1 or len(items) <= 1:
            return [self.ask(s, q) for s, q in items]
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda x: self.ask(*x), items))

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Jev needs a key: set TYPESAFE_API_KEY (https://console.typesafe.ai/keys)")
        req = urllib.request.Request(
            f"{self.base_url}/systemone",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        req.add_unredirected_header("Authorization", f"Bearer {self.api_key}")  # never onto a redirect
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500:  # the body says what was wrong with the request
                    raise RuntimeError(f"Jev refused the request: HTTP {e.code} {_error_body(e)}") from e
                last = e
                retry_after = (getattr(e, "headers", None) or {}).get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else 1.5 * (attempt + 1)
                except ValueError:
                    wait = 1.5 * (attempt + 1)
                time.sleep(min(wait, 60.0))
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Jev endpoint {self.base_url} unreachable: {last}") from last

    def stats(self) -> dict[str, Any]:
        ms = sorted(self.call_ms)
        return {
            "model": self.model,
            "url": self.base_url,
            "calls": self.calls,
            "cached": self.cached,
            "input_tokens": self.tokens_spent,
            "call_ms_p50": round(statistics.median(ms), 1) if ms else None,
            "call_ms_p95": round(ms[min(len(ms) - 1, int(0.95 * (len(ms) - 1)))], 1) if ms else None,
        }


def _formatter(f: NamedFormatter | str) -> NamedFormatter:
    return TOOL_FORMATS[f] if isinstance(f, str) else f


class JevReranker:
    """Any scorer's list to ``depth``, reordered by one Jev Choice per query."""

    score_kind = "jev"

    def __init__(
        self,
        base: Any,
        client: JevClient,
        *,
        depth: int = 100,
        tool_format: NamedFormatter | str = "name_desc",
        max_chars: int = 1000,
    ):
        if not 2 <= depth <= MAX_OPTIONS:
            raise ValueError(f"--rerank-depth must be 2..{MAX_OPTIONS} (one Choice question)")
        self.base, self.client, self.depth, self.max_chars = base, client, depth, max_chars
        self.jev_format = _formatter(tool_format)
        self.tool_format, self.query_format = base.tool_format, base.query_format
        self.name = f"jev[{client.model},d{depth},{self.jev_format.name}]/{base.name}"
        self.tools: dict[str, Tool] = {}
        self.last_base: dict[str, RankedList] = {}

    def index(self, tools: Iterable[Tool]) -> None:
        tools = list(tools)
        self.tools = {t.id: t for t in tools}
        self.base.index(tools)

    def rank(self, queries: Sequence[Query], k: int) -> list[RankedList]:
        base = self.base.rank(queries, max(k, self.depth))
        self.last_base = {r.query_id: r for r in base}
        heads = [r.tool_ids[: self.depth] for r in base]
        asked = [i for i, h in enumerate(heads) if len(h) >= 2]  # one option needs no question
        items = [
            (
                {"request": queries[i].text},
                {
                    "tool": choice_question(
                        queries[i].instruction,
                        [self.jev_format(self.tools[t]) for t in heads[i]],
                        self.max_chars,
                    )
                },
            )
            for i in asked
        ]
        answers = dict(zip(asked, self.client.ask_many(items), strict=True))
        out = []
        for i, (r, head) in enumerate(zip(base, heads, strict=True)):
            if i not in answers:
                out.append(RankedList(r.query_id, r.tool_ids[:k], r.scores[:k]))
                continue
            order = _order(answers[i]["tool"]["probabilities"], len(head))
            ids = [head[n] for n, _ in order] + r.tool_ids[self.depth :]
            scores = [p for _, p in order] + [-float(n + 1) for n in range(len(r.tool_ids) - len(head))]
            out.append(RankedList(r.query_id, ids[:k], scores[:k]))
        return out

    def score_tools(self, query: Query, tools: Sequence[Tool]) -> list[float]:
        return self.base.score_tools(query, tools)


class JevScorer:
    """Jev alone over a corpus: chunked Choice questions, the chunk winners re-ranked once."""

    score_kind = "jev"

    def __init__(
        self,
        client: JevClient,
        tool_format: NamedFormatter | str = "name_desc",
        *,
        chunk: int = 200,
        per_chunk: int = 20,
        max_chars: int = 1000,
    ):
        if not 2 <= chunk <= MAX_OPTIONS:
            raise ValueError(f"--jev-chunk must be 2..{MAX_OPTIONS}")
        if per_chunk < 1:
            raise ValueError("--jev-per-chunk must be at least 1")
        self.client, self.chunk, self.per_chunk, self.max_chars = client, chunk, per_chunk, max_chars
        self.tool_format = _formatter(tool_format)
        self.query_format = QUERY_FORMATS["plain"]  # the instruction goes into the question, not the state
        self.name = f"jev[{client.model},c{chunk}x{per_chunk},{self.tool_format.name}]"
        self.ids: list[str] = []
        self.texts: list[str] = []

    def index(self, tools: Iterable[Tool]) -> None:
        tools = list(tools)
        self.ids = [t.id for t in tools]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate tool ids")
        self.texts = [self.tool_format(t)[: self.max_chars] for t in tools]

    def _chunks(self) -> list[range]:
        return [range(i, min(i + self.chunk, len(self.ids))) for i in range(0, len(self.ids), self.chunk)]

    def _question(self, q: Query, members: Sequence[int]) -> tuple[Any, dict[str, Any]]:
        return {"request": q.text}, {
            "tool": choice_question(q.instruction, [self.texts[i] for i in members], self.max_chars)
        }

    def rank(self, queries: Sequence[Query], k: int) -> list[RankedList]:
        if not queries or not self.ids:
            return [RankedList(q.id, [], []) for q in queries]
        chunks = self._chunks()
        if len(self.ids) == 1:
            return [RankedList(q.id, self.ids[:k], [1.0][:k]) for q in queries]
        first = self.client.ask_many([self._question(q, ch) for q in queries for ch in chunks])
        if len(chunks) == 1:  # one chunk: round one is the final order
            return [
                self._assemble(q, _order(first[qi]["tool"]["probabilities"], len(self.ids)), [], k)
                for qi, q in enumerate(queries)
            ]
        per = max(1, min(self.per_chunk, MAX_OPTIONS // len(chunks)))
        winners: list[list[int]] = []  # per query: the chunk winners, global indexes
        rest: list[list[tuple[int, float]]] = []  # the others by round-one probability
        for qi in range(len(queries)):
            w: list[int] = []
            others: list[tuple[int, float]] = []
            for ci, ch in enumerate(chunks):
                order = _order(first[qi * len(chunks) + ci]["tool"]["probabilities"], len(ch))
                w.extend(ch[n] for n, _ in order[:per])
                others.extend((ch[n], p) for n, p in order[per:])
            winners.append(w)
            rest.append(sorted(others, key=lambda x: (-x[1], x[0])))
        final = self.client.ask_many([self._question(q, winners[qi]) for qi, q in enumerate(queries)])
        out = []
        for qi, q in enumerate(queries):
            order = _order(final[qi]["tool"]["probabilities"], len(winners[qi]))
            out.append(self._assemble(q, [(winners[qi][n], p) for n, p in order], rest[qi], k))
        return out

    def _assemble(
        self, q: Query, top: Sequence[tuple[int, float]], rest: Sequence[tuple[int, float]], k: int
    ) -> RankedList:
        ids = [self.ids[i] for i, _ in top] + [self.ids[i] for i, _ in rest]
        scores = [p for _, p in top] + [-float(n + 1) for n in range(len(rest))]
        return RankedList(q.id, ids[:k], scores[:k])
