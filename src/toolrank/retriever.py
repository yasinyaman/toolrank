"""Query-time retrieval over an ingest dir, shared by ``toolrank search``, the MCP proxy and REST.

A ``Retriever`` holds one immutable state — the tools, an id map and a scorer indexed on them — and
replaces it whole when ``tools.jsonl`` changes: a reload makes a fresh scorer with ``make_scorer``
(encoder and heads are shared), indexes it (incremental against a persistent index, so only new or
changed tools are embedded), then swaps the state in one assignment. Searches read the state once,
so any number of them can run in threads while a reload happens.

With ``background=True`` the first index is built in a thread: a server answers the protocol
handshake at once and ``search`` waits for the index up to ``ready_timeout`` seconds. A server also
passes ``fallback``, a keyword scorer (BM25) that indexes in about a second: until the semantic
index is ready — minutes when every tool has to be embedded — searches get its matches (``mode``
``lexical``) instead of an error, and ``get``/``tools`` work, so ``call_tool`` does too. A first
build that fails is retried every ``retry_s`` seconds when a request comes in. ``notify`` receives
one line per index event (keyword index up, index ready, build failed).
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toolrank.cut import AdaptiveK
from toolrank.datasets.jsonl import tools_from_lines
from toolrank.domain import Query, RankedList, Tool

LOG_TOP = 20  # ranked tools kept for the usage log, whatever the cut returns
SETTLE_S = 0.05  # a tools.jsonl modified more recently than this may still be being written


class IndexNotReady(RuntimeError):
    """The first index is still being built, or building it failed (the message says which)."""


@dataclass(frozen=True)
class Hit:
    tool: Tool
    score: float

    @property
    def id(self) -> str:
        return self.tool.id

    @property
    def server(self) -> str:
        return self.tool.category

    @property
    def kind(self) -> str:
        return "openapi" if "http" in self.tool.doc else "mcp"

    @property
    def input_schema(self) -> dict[str, Any]:
        return dict(self.tool.doc.get("inputSchema") or {"type": "object"})


@dataclass(frozen=True)
class SearchResult:
    query: str
    instruction: str
    hits: list[Hit]
    ranked: list[tuple[str, float]]  # top LOG_TOP before the cut
    took_ms: float
    rule: str
    emb_key: str | None
    scorer: str
    catalog: str
    mode: str = "semantic"  # "lexical": the keyword stand-in answered, the index is still building
    # whether the request brought its instruction (request text, a digest in the usage log) or it is
    # the server's own; True unless the retriever says otherwise, so a result built elsewhere hides it
    own_instruction: bool = True


@dataclass(frozen=True)
class _State:
    stamp: tuple[int, int]
    tools: list[Tool]
    by_id: dict[str, Tool]
    scorer: Any
    catalog: str
    index_s: float
    sync: dict[str, int]
    lexical: bool = False


class Retriever:
    def __init__(
        self,
        data_dir: str | Path,
        make_scorer: Callable[[], Any],
        *,
        rule: AdaptiveK | None = None,
        instruction: str = "",
        fixed_k: int = 0,
        depth: int = 10,
        cache_key: Callable[[str], str | None] | None = None,
        heads_sha: str | None = None,
        background: bool = False,
        ready_timeout: float = 120.0,
        fallback: Callable[[], Any] | None = None,
        retry_s: float = 30.0,
        notify: Callable[[str], None] | None = None,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.path = self.data_dir / "tools.jsonl"
        self.make_scorer, self.rule, self.instruction = make_scorer, rule, instruction
        self.fixed_k, self.depth, self.cache_key, self.heads_sha = fixed_k, depth, cache_key, heads_sha
        self.ready_timeout, self.make_fallback, self.retry_s = ready_timeout, fallback, retry_s
        self.notify = notify or (lambda msg: None)
        self._state: _State | None = None
        self._fallback: _State | None = None  # only until the first semantic state exists
        self._error: BaseException | None = None
        self._retry_at = 0.0
        self._ready = threading.Event()  # the first semantic build finished, or failed
        self._fallback_done = threading.Event()  # the keyword stand-in is built, failed, or not wanted
        self._changed = threading.Condition()
        self._reload_lock = threading.Lock()
        if background:
            threading.Thread(target=self._reload_quietly, name="toolrank-index", daemon=True).start()
            if fallback is not None:  # in parallel: a warm start must not wait for it
                threading.Thread(target=self._build_fallback, name="toolrank-keywords", daemon=True).start()
        if not background or fallback is None:
            self._fallback_done.set()
        if not background:
            self.reload()

    # -- state ---------------------------------------------------------------------------------
    def _stamp(self) -> tuple[int, int]:
        st = self.path.stat()
        return st.st_mtime_ns, st.st_size

    def _read(self) -> tuple[tuple[int, int], bytes]:
        """``tools.jsonl`` read whole once it has stopped changing: ``ingest`` replaces it atomically,
        but a file written in place must not be indexed half-written."""
        for _ in range(200):
            stamp = self._stamp()
            age = time.time() - stamp[0] / 1e9
            if 0 <= age < SETTLE_S:
                time.sleep(SETTLE_S - age)
                continue
            data = self.path.read_bytes()
            if self._stamp() == stamp:
                return stamp, data
        raise RuntimeError(f"{self.path} kept changing while it was read")

    def _build(self, make: Callable[[], Any], *, lexical: bool = False) -> _State:
        stamp, data = self._read()
        catalog = hashlib.sha256(data).hexdigest()[:16]
        tools = tools_from_lines(data.decode("utf-8").splitlines())
        t0 = time.perf_counter()
        scorer = make()
        scorer.index(tools)
        sync = dict(getattr(getattr(scorer, "semantic", scorer), "last_sync", {}) or {})
        by_id = {t.id: t for t in tools}
        return _State(stamp, tools, by_id, scorer, catalog, time.perf_counter() - t0, sync, lexical)

    def _publish(self, *, ready: bool = False, **fields: Any) -> None:
        """Set attributes (and ``_ready``) under the condition, then wake the waiting searches."""
        with self._changed:
            for k, v in fields.items():
                setattr(self, k, v)
            if ready:
                self._ready.set()
            self._changed.notify_all()

    def reload(self) -> bool:
        """Build and swap in a fresh state if ``tools.jsonl`` changed; -> whether it did."""
        with self._reload_lock:
            if self._state is not None and self._state.stamp == self._stamp():
                return False
            st = self._build(self.make_scorer)
            # the semantic index is up: the keyword stand-in is no longer needed
            self._publish(ready=True, _state=st, _error=None, _fallback=None)
        self.notify(
            f"index ready: {len(st.tools)} tools from {len({t.category for t in st.tools})} sources in "
            f"{st.index_s:.1f} s (embedded {st.sync.get('embedded', 0)}, kept {st.sync.get('kept', 0)})"
        )
        return True

    def _build_fallback(self) -> None:
        assert self.make_fallback is not None
        try:
            fb = self._build(self.make_fallback, lexical=True)
        except Exception as e:  # the semantic build reports its own errors
            with self._changed:
                self._fallback_done.set()
                self._changed.notify_all()
            self.notify(f"keyword index failed: {type(e).__name__}: {e}")
            return
        with self._changed:
            self._fallback_done.set()
            if self._state is not None:  # the semantic index came first: no stand-in needed
                self._changed.notify_all()
                return
            self._fallback = fb
            self._changed.notify_all()
        self.notify(
            f"keyword index ready: {len(fb.tools)} tools in {fb.index_s:.1f} s; searches use it until "
            "the semantic index is built"
        )

    def _reload_quietly(self) -> None:
        try:
            self.reload()
        except Exception as e:  # reported by the next search / status, the old state keeps serving
            first = self._state is None
            self._publish(ready=True, _error=e, _retry_at=time.monotonic() + self.retry_s)
            self.notify(
                f"index {'build' if first else 'reload'} failed: {type(e).__name__}: {e}"
                + (f"; retried on a request after {self.retry_s:.0f} s" if first else "")
            )

    def _retry_if_due(self) -> None:
        if self._error is not None and time.monotonic() >= self._retry_at and not self._reload_lock.locked():
            self._retry_at = time.monotonic() + self.retry_s
            threading.Thread(target=self._reload_quietly, name="toolrank-retry", daemon=True).start()

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until the first index build has finished or failed; -> False on timeout."""
        return self._ready.wait(timeout)

    def _settled(self, fallback: bool) -> bool:
        if self._state is not None or self._fallback is not None:
            return True
        # the semantic build failed: a caller that can use the stand-in waits for it as well
        return self._ready.is_set() and (not fallback or self._fallback_done.is_set())

    def state(self, *, fallback: bool = False) -> _State:
        """The semantic state; with ``fallback``, the keyword stand-in while there is none yet.
        Waits up to ``ready_timeout`` for the first build, but not once the stand-in is up: then
        the build is the slow kind, and a caller that needs the semantic index is told at once."""
        with self._changed:
            self._changed.wait_for(lambda: self._settled(fallback), timeout=self.ready_timeout)
        st = self._state
        if st is None:
            self._retry_if_due()
            fb = self._fallback
            if fallback and fb is not None:
                return fb
            if self._error is None:
                meanwhile = " (keyword search works meanwhile)" if fb is not None else ""
                raise IndexNotReady(
                    f"the tool index for {self.data_dir} is still being built{meanwhile}; try again shortly"
                )
            raise IndexNotReady(f"building the tool index failed: {self._error}")
        with contextlib.suppress(OSError):
            due = time.monotonic() >= self._retry_at  # a failed reload waits retry_s before the next
            if due and self._stamp() != st.stamp and not self._reload_lock.locked():
                threading.Thread(target=self._reload_quietly, name="toolrank-reload", daemon=True).start()
        return st

    def status(self) -> dict[str, Any]:
        sem, fb = self._state, self._fallback
        st = sem or fb
        if sem is not None:
            mode = "semantic"
        elif fb is not None:
            mode = "lexical"
        else:
            mode = "failed" if self._error is not None else "starting"
        return {
            "ready": sem is not None,
            "mode": mode,
            "tools": len(st.tools) if st else 0,
            "sources": len({t.category for t in st.tools}) if st else 0,
            "catalog": st.catalog if st else None,
            "scorer": st.scorer.name if st else None,
            "index_s": round(st.index_s, 3) if st else None,
            "sync": dict(st.sync) if st else None,
            "error": str(self._error) if self._error else None,
        }

    # -- queries -------------------------------------------------------------------------------
    def describe(self, k: int | None = None) -> str:
        fixed = k or self.fixed_k
        if fixed:
            return f"top {fixed}"
        return f"adaptive K ({self.rule.describe()})" if self.rule else f"top {self.depth}"

    def search(self, query: str, *, k: int | None = None, instruction: str | None = None) -> SearchResult:
        st = self.state(fallback=self.make_fallback is not None)
        inst = self.instruction if instruction is None else instruction
        q = Query(id=uuid.uuid4().hex, text=query, qrels={}, instruction=inst)
        fixed = k or self.fixed_k
        rule = None if fixed else self.rule
        top = fixed or (rule.max_k if rule else self.depth)
        t0 = time.perf_counter()
        if hasattr(st.scorer, "rank_pairs"):
            fused, semantic = st.scorer.rank_pairs([q], max(top, LOG_TOP))[0]
        else:
            fused, semantic = st.scorer.rank([q], max(top, LOG_TOP))[0], None
        if st.lexical:  # BM25 pads its list with zero scores: those tools share no term with the request
            keep = [n for n, s in enumerate(fused.scores) if s > 0]
            fused = RankedList(
                fused.query_id, [fused.tool_ids[n] for n in keep], [fused.scores[n] for n in keep]
            )
        if rule is not None and getattr(st.scorer, "score_kind", "cosine") != "bm25":
            shown = rule.cut(fused, semantic)
        else:
            shown = RankedList(fused.query_id, fused.tool_ids[:top], fused.scores[:top])
        took = (time.perf_counter() - t0) * 1000.0
        fmt = getattr(getattr(st.scorer, "semantic", st.scorer), "query_format", None)
        emb_key = None
        if self.cache_key is not None and fmt is not None and not st.lexical:
            emb_key = self.cache_key(fmt(q))
        return SearchResult(
            query=query,
            instruction=inst,
            hits=[Hit(st.by_id[t], s) for t, s in zip(shown.tool_ids, shown.scores, strict=True)],
            ranked=list(zip(fused.tool_ids[:LOG_TOP], fused.scores[:LOG_TOP], strict=True)),
            took_ms=took,
            rule=f"top {top} keyword matches" if st.lexical else self.describe(fixed),
            emb_key=emb_key,
            scorer=st.scorer.name,
            catalog=st.catalog,
            mode="lexical" if st.lexical else "semantic",
            own_instruction=inst != self.instruction,
        )

    def rank(
        self, query: str, tools: Sequence[Tool], *, instruction: str | None = None
    ) -> list[tuple[str, float]]:
        """Cosine of caller-supplied tools (not necessarily in the index), best first."""
        st = self.state()
        if not hasattr(st.scorer, "score_tools"):
            raise ValueError(f"{st.scorer.name} cannot score tools outside its index")
        inst = self.instruction if instruction is None else instruction
        q = Query(id=uuid.uuid4().hex, text=query, qrels={}, instruction=inst)
        scores = st.scorer.score_tools(q, list(tools))
        return sorted(zip([t.id for t in tools], scores, strict=True), key=lambda x: -x[1])

    def tools(self) -> list[Tool]:
        return self.state(fallback=self.make_fallback is not None).tools

    def catalogue(self) -> tuple[list[Tool], str]:
        """The tools and the hash of the ``tools.jsonl`` they came from, from one state."""
        st = self.state(fallback=self.make_fallback is not None)
        return st.tools, st.catalog

    def get(self, tool_id: str) -> Tool | None:
        return self.state(fallback=self.make_fallback is not None).by_id.get(tool_id)
