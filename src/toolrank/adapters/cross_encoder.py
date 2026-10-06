"""A cross-encoder (reranker) behind vLLM's score API, in Jev's role: the local alternative.

A cross-encoder reads the request and one candidate together and returns a relevance score, which
is what a bi-encoder cannot do and what Jev's gain comes from. ``CrossEncoderScorer`` scores a
query against candidates with ``POST {base_url}/score`` (``text_1`` the query prompt, ``text_2``
the candidate prompts; vLLM joins each pair), so it serves ``--rerank cross`` through
``score_tools``; ``rank`` over a whole corpus scores every tool and is only for small sets.

Two prompt templates (``template``): ``qwen3`` for Qwen/Qwen3-Reranker-* loaded as the original
model (vLLM's recipe: the system prefix, ``<Instruct>:`` / ``<Query>:`` on the query side,
``<Document>:`` and the assistant suffix on the document side), ``bge`` for
BAAI/bge-reranker-v2-gemma (FlagEmbedding's ``A: query`` / ``B: passage`` / prompt, which vLLM
joins by plain concatenation). The benchmark instruction is the ``<Instruct>`` (Qwen3) or leads the
query (bge); without one, the serving instruction. Standard library only; scores cached in SQLite
by model, query prompt and document prompt, so a rerun asks nothing.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from toolrank.adapters.embeddings_api import env_api_key
from toolrank.domain import Query, RankedList, Tool
from toolrank.formats import QUERY_FORMATS, TOOL_FORMATS, NamedFormatter
from toolrank.ports import scoped

__all__ = ["TEMPLATES", "CrossEncoderScorer", "ScoreClient", "prompts"]

DEFAULT_INSTRUCTION = "Given an agent's request for a tool, retrieve the MCP tool that fulfills it."

_QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_BGE_PROMPT = (
    "Given a query A and a passage B, determine whether the passage contains an answer to the query by "
    "providing a prediction of either 'Yes' or 'No'."
)
TEMPLATES = ("qwen3", "bge")


def prompts(template: str, request: str, instruction: str, docs: Sequence[str]) -> tuple[str, list[str]]:
    """-> (text_1, text_2 list) for one query against ``docs``, in the model's own format."""
    inst = instruction.strip() or DEFAULT_INSTRUCTION
    if template == "qwen3":
        return f"{_QWEN3_PREFIX}<Instruct>: {inst}\n<Query>: {request}\n", [
            f"<Document>: {d}{_QWEN3_SUFFIX}" for d in docs
        ]
    if template == "bge":
        return f"A: {inst}\n{request}\n", [f"B: {d}\n{_BGE_PROMPT}" for d in docs]
    raise ValueError(f"unknown template {template!r}; choose from {TEMPLATES}")


class ScoreCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=60.0, check_same_thread=False)
        self._lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS score (key TEXT PRIMARY KEY, score REAL)")
        self.db.commit()

    @staticmethod
    def key(base_url: str, model: str, text_1: str, text_2: str) -> str:
        return hashlib.sha256(scoped(f"{base_url}\x00{model}\x00{text_1}\x00{text_2}").encode()).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in range(0, len(keys), 900):
            chunk = keys[s : s + 900]
            q = f"SELECT key, score FROM score WHERE key IN ({','.join('?' * len(chunk))})"
            with self._lock:
                out.update(self.db.execute(q, chunk).fetchall())
        return out

    def put_many(self, rows: Sequence[tuple[str, float]]) -> None:
        with self._lock:
            self.db.executemany("INSERT OR REPLACE INTO score (key, score) VALUES (?, ?)", rows)
            self.db.commit()


class ScoreClient:
    """``POST {base_url}/score``: one ``text_1`` against a list of ``text_2``, scores back in order.
    ``calls`` and ``pairs`` count what reached the endpoint, ``cached`` the pairs the cache answered;
    ``call_ms`` the wall-clock of each call. Tests replace ``_post``."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        batch: int = 64,
    ):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or env_api_key(self.base_url, "TOOLRANK_EMB_API_KEY")
        self.cache = ScoreCache(Path(cache_dir) / "scores.sqlite") if cache_dir else None
        self.timeout, self.max_retries, self.batch = timeout, max_retries, batch
        self.calls = self.pairs = self.cached = 0
        self.call_ms: list[float] = []
        self._lock = threading.Lock()

    def score(self, text_1: str, text_2: Sequence[str]) -> list[float]:
        keys = [ScoreCache.key(self.base_url, self.model, text_1, t) for t in text_2]
        have = self.cache.get_many(keys) if self.cache is not None else {}
        out: list[float | None] = [have.get(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        with self._lock:
            self.cached += len(text_2) - len(todo)
        for s in range(0, len(todo), self.batch):
            idx = todo[s : s + self.batch]
            t0 = time.perf_counter()
            scores = self._post({"model": self.model, "text_1": text_1, "text_2": [text_2[i] for i in idx]})
            dt = (time.perf_counter() - t0) * 1000.0
            if len(scores) != len(idx):
                raise RuntimeError(f"score endpoint returned {len(scores)} scores for {len(idx)} pairs")
            for i, v in zip(idx, scores, strict=True):
                out[i] = float(v)
            with self._lock:
                self.calls += 1
                self.pairs += len(idx)
                self.call_ms.append(dt)
            if self.cache is not None:
                self.cache.put_many([(keys[i], float(v)) for i, v in zip(idx, scores, strict=True)])
        return [float(v) for v in out]  # type: ignore[arg-type]

    def _post(self, body: dict[str, Any]) -> list[float]:
        req = urllib.request.Request(
            f"{self.base_url}/score",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            req.add_unredirected_header("Authorization", f"Bearer {self.api_key}")
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    j = json.loads(r.read().decode("utf-8"))
                return [float(d["score"]) for d in j["data"]]
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    try:
                        text = e.read().decode("utf-8", errors="replace")[:500]
                    except Exception:  # noqa: BLE001
                        text = ""
                    raise RuntimeError(f"score endpoint refused the request: HTTP {e.code} {text}") from e
                last = e
            except (OSError, http.client.HTTPException) as e:  # unreachable, slow, or the connection dropped
                last = e
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"score endpoint {self.base_url} unreachable: {last}") from last

    def stats(self) -> dict[str, Any]:
        ms = sorted(self.call_ms)
        return {
            "model": self.model,
            "url": self.base_url,
            "calls": self.calls,
            "pairs": self.pairs,
            "cached_pairs": self.cached,
            "call_ms_p50": round(ms[len(ms) // 2], 1) if ms else None,
            "call_ms_p95": round(ms[min(len(ms) - 1, int(0.95 * (len(ms) - 1)))], 1) if ms else None,
        }


class CrossEncoderScorer:
    """A cross-encoder as a ``Scorer``: ``score_tools`` for ``--rerank cross``, ``rank`` for small
    corpora (every tool is scored for every query)."""

    score_kind = "cross"

    def __init__(
        self,
        client: ScoreClient,
        tool_format: NamedFormatter | str = "name_desc",
        *,
        template: str = "qwen3",
        max_chars: int | None = None,
        max_query_chars: int = 6000,
        max_tools: int = 5000,
    ):
        if template not in TEMPLATES:
            raise ValueError(f"unknown template {template!r}; choose from {TEMPLATES}")
        self.client, self.template, self.max_chars, self.max_tools = client, template, max_chars, max_tools
        self.max_query_chars = max_query_chars  # the request is in every pair: one 18k-character
        # LiveMCPBench task blew a 4096-token window; the head of a request carries the task
        self.tool_format = TOOL_FORMATS[tool_format] if isinstance(tool_format, str) else tool_format
        self.query_format = QUERY_FORMATS["plain"]  # the instruction has its own slot in the prompt
        self.name = f"cross[{client.model},{template}]/{self.tool_format.name}"
        self.tools: list[Tool] = []

    def _text(self, t: Tool) -> str:
        x = self.tool_format(t)
        return x[: self.max_chars] if self.max_chars else x

    def index(self, tools: Iterable[Tool]) -> None:
        self.tools = list(tools)

    def score_tools(self, query: Query, tools: Sequence[Tool]) -> list[float]:
        if not tools:
            return []
        request = query.text[: self.max_query_chars] if self.max_query_chars else query.text
        text_1, text_2 = prompts(self.template, request, query.instruction, [self._text(t) for t in tools])
        return self.client.score(text_1, text_2)

    def rank(self, queries: Sequence[Query], k: int) -> list[RankedList]:
        if len(self.tools) > self.max_tools:
            raise ValueError(
                f"{len(self.tools)} tools: a cross-encoder scores every pair; use --rerank cross over a "
                f"retriever (or raise max_tools)"
            )
        out = []
        for q in queries:
            s = self.score_tools(q, self.tools)
            order = sorted(range(len(s)), key=lambda i: (-s[i], i))[:k]
            out.append(RankedList(q.id, [self.tools[i].id for i in order], [s[i] for i in order]))
        return out
