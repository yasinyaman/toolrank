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

Heads can change while the server runs (``toolrank learn`` writes them). With ``heads_dir`` (serve
and search: ``DATA/heads``), ``current.npz`` there replaces the heads the flags chose, ``candidate.npz``
answers a sticky ``candidate_share`` of the requests (by ``arm_key``: the session, else the client)
next to it, and ``tenants/<name>/current.npz`` / ``candidate.npz`` do the same for one API key's
requests. Each is a variant: its own scorer over the same tools, built in the background when the
file appears or changes and dropped when it goes, its index snapshot under ``index/variants/<name>``.
Until a variant is built, requests get the one before it. ``SearchResult.arm`` and ``heads`` say
which answered, for the usage log and ``toolrank ab``.

A second stage (``--rerank``) that fails during a search gives way to the first stage's list:
``SearchResult.rerank_error`` says why, for the server's log, and the answer carries a note.

With a calibration (``DATA/calibration.json``, ``toolrank.calibration``) that matches the answering
first stage, heads and instruction, every search carries a ``confidence``, and ``min_confidence``
turns away a request whose confidence is lower (an empty list) unless it named its own ``k``.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toolrank.calibration import Calibration, Calibrations
from toolrank.couse import partners
from toolrank.cut import AdaptiveK
from toolrank.datasets.jsonl import tools_from_lines
from toolrank.domain import Query, RankedList, Tool
from toolrank.ports import cache_scope, rerank_failures, supplied, visible_ids

LOG_TOP = 20  # ranked tools kept for the usage log, whatever the cut returns
SETTLE_S = 0.05  # a tools.jsonl modified more recently than this may still be being written
_BLANKS = re.compile(r"[^\S\n]+")  # whitespace other than a line break
_GAPS = re.compile(r"\n{3,}")


def tidy(text: str) -> str:
    """A request as it is embedded, cached and logged: NFC, CRLF as LF, runs of spaces and tabs as one
    space, no space at either end of a line, at most one blank line in a row, nothing around it.
    Requests that differ only there are one cache entry and one digest in the usage log."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = (_BLANKS.sub(" ", line).strip() for line in text.split("\n"))
    return _GAPS.sub("\n\n", "\n".join(lines)).strip()


class IndexNotReady(RuntimeError):
    """The first index is still being built, or building it failed (the message says which)."""


@dataclass(frozen=True)
class Hit:
    tool: Tool
    score: float
    used_with: str | None = None  # not ranked into the list: agents call it together with this hit

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
    arm: str = "base"  # which heads answered: base, current, candidate, tenant:<name>[:candidate]
    heads: str | None = None  # sha256 prefix of that heads file
    # whether the request brought its instruction (request text, a digest in the usage log) or it is
    # the server's own; True unless the retriever says otherwise, so a result built elsewhere hides it
    own_instruction: bool = True
    added: int = 0  # the last ``added`` hits are co-use partners of the hits before them
    model: str | None = None  # the backbone that embedded the request (None: keyword, or unknown)
    # why the second stage failed, so the first stage's order answered; for the server's log only
    # (it may name the reranker's address), the client gets a note
    rerank_error: str | None = None
    # the share of the calibration's answerable requests that scored at or below this one's best;
    # None without a calibration for the first stage, heads and instruction that answered
    confidence: float | None = None


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


def _only(ranked: RankedList, ids: frozenset[str]) -> RankedList:
    keep = [n for n, t in enumerate(ranked.tool_ids) if t in ids]
    return RankedList(ranked.query_id, [ranked.tool_ids[n] for n in keep], [ranked.scores[n] for n in keep])


@dataclass
class _Variant:
    """A heads file next to the log and the state built from it (``None`` until built)."""

    name: str
    path: Path
    state: _State | None = None
    stamp: tuple[Any, ...] | None = (
        None  # the heads file's (mtime, size) and the tools stamp it was built for
    )
    sha: str | None = None
    building: bool = False
    error: str | None = None


@contextlib.contextmanager
def _scope(tenant: str | None) -> Iterator[None]:
    token = cache_scope.set(tenant) if tenant is not None else None
    try:
        yield
    finally:
        if token is not None:
            cache_scope.reset(token)


def innermost(scorer: Any) -> Any:
    """The first stage under the wrappers: a hybrid's ``semantic`` scorer, a second stage's ``base``."""
    for _ in range(4):
        inner = getattr(scorer, "semantic", None) or getattr(scorer, "base", None)
        if inner is None:
            break
        scorer = inner
    return scorer


def bucket(key: str | None, share: float) -> bool:
    """Whether ``key`` falls in the first ``share`` of a stable hash: the same session or client
    keeps getting the same arm."""
    if share <= 0:
        return False
    if key is None:
        key = uuid.uuid4().hex
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10_000 < share * 10_000


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
        heads_dir: str | Path | None = None,
        variants: Callable[[Path, str], Callable[[], Any]] | None = None,
        candidate_share: float = 0.1,
        use_current: bool = True,
        co_use: Any | None = None,
        co_use_extra: int = 0,
        allowed: dict[str, frozenset[str]] | None = None,
        calibration: str | Path | None = None,
        min_confidence: float | None = None,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.path = self.data_dir / "tools.jsonl"
        self.make_scorer, self.rule, self.instruction = make_scorer, rule, instruction
        self.fixed_k, self.depth, self.cache_key, self.heads_sha = fixed_k, depth, cache_key, heads_sha
        self.ready_timeout, self.make_fallback, self.retry_s = ready_timeout, fallback, retry_s
        self.notify = notify or (lambda msg: None)
        # heads files that come and go while serving; ``variants(path, name)`` makes their scorers
        self.heads_dir = Path(heads_dir).resolve() if heads_dir and variants else None
        self.make_variant, self.candidate_share, self.use_current = variants, candidate_share, use_current
        self._variants: dict[str, _Variant] = {}
        self._variants_lock = threading.Lock()
        # a ``couse.CoUseTable`` (anything with ``table()``) and how many partners a result may gain
        self.co_use, self.co_use_extra = co_use, co_use_extra
        self.encoder: Any = None  # the composition root may leave the encoder here for the metrics
        self.second: Any = None  # and the second stage's client (``scored()``)
        # tenant -> the sources its key may reach (``tenants.Tenant.sources``); absent: every source
        self.allowed: dict[str, frozenset[str]] = dict(allowed or {})
        self._visible: dict[tuple[str, str], frozenset[str]] = {}  # (catalogue, tenant) -> tool ids
        # DATA/calibration.json: what a best score means on this catalogue (``toolrank calibrate``)
        self.calibrations = Calibrations(calibration) if calibration else None
        self.min_confidence = min_confidence
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
        sync = dict(getattr(innermost(scorer), "last_sync", {}) or {})
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
        self._say_calibration(st)
        return True

    def _say_calibration(self, st: _State) -> None:
        """One line on whether searches carry a confidence (and the gate works), when it matters."""
        if self.calibrations is None:
            return
        cal = self.calibration(st, self.heads_sha)
        if cal is not None:
            gate = f"; requests below {self.min_confidence:g} are turned away" if self.min_confidence else ""
            self.notify(
                f"confidence: calibrated on {len(cal.best)} requests ({cal.requests or 'unnamed'}){gate}"
            )
        elif self.calibrations.error or len(self.calibrations) or self.min_confidence:
            why = self.calibrations.error or (
                "it has no entry for this first stage, heads and instruction"
                if self.calibrations.path.exists()
                else "there is none"
            )
            self.notify(
                f"no confidence: {self.calibrations.path.name}: {why} (toolrank calibrate)"
                + ("; --min-confidence turns nothing away" if self.min_confidence else "")
            )

    def calibration(self, st: _State, heads: str | None) -> Calibration | None:
        """The calibration of ``st``'s first stage with ``heads`` under the serving instruction."""
        if self.calibrations is None or st.lexical:
            return None
        return self.calibrations.find(innermost(st.scorer).name, heads, self.instruction)

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
                raise IndexNotReady(f"the tool index is still being built{meanwhile}; try again shortly")
            # the reason (often the embedding endpoint's address) went to the server's log
            raise IndexNotReady("building the tool index failed; the server's log says why")
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
        out = {
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
        if self.heads_dir is not None:
            with self._variants_lock:
                out["heads"] = {
                    "base": self.heads_sha,
                    **{
                        v.name: {"sha": v.sha, "ready": v.state is not None, "error": v.error}
                        for v in self._variants.values()
                    },
                }
        return out

    # -- heads that change while serving ------------------------------------------------------------
    def _variant(self, name: str, path: Path) -> _Variant | None:
        """The variant for ``path``, built (again) in the background when the file is new or changed;
        None when there is no such file. What comes back may be the previous build while a new one
        runs, or None while the first one does."""
        try:
            st = path.stat()
        except OSError:
            with self._variants_lock:
                if self._variants.pop(name, None) is not None:
                    self.notify(f"heads {name}: {path.name} is gone")
            return None
        stamp = (st.st_mtime_ns, st.st_size, self._stamp())
        with self._variants_lock:
            v = self._variants.setdefault(name, _Variant(name, path))
            if v.stamp != stamp and not v.building:
                v.building = True
                threading.Thread(
                    target=self._build_variant, args=(v, stamp), name=f"toolrank-heads-{name}", daemon=True
                ).start()
        return v

    def _build_variant(self, v: _Variant, stamp: tuple[Any, ...]) -> None:
        assert self.make_variant is not None
        try:
            state = self._build(self.make_variant(v.path, v.name))
            # the bytes the heads were loaded from: the file may have been replaced since
            sha = getattr(getattr(innermost(state.scorer), "heads", None), "sha256", None)
            sha = (sha or hashlib.sha256(v.path.read_bytes()).hexdigest())[:16]
        except Exception as e:  # the file may be half-written, or not heads at all
            with self._variants_lock:
                v.error, v.stamp, v.building = f"{type(e).__name__}: {e}", stamp, False
            self.notify(f"heads {v.name}: {v.path.name} could not be loaded: {v.error}")
            return
        with self._variants_lock:
            v.state, v.sha, v.error, v.stamp, v.building = state, sha, None, stamp, False
        self.notify(f"heads {v.name}: {v.path.name} ({sha[:8]}) in {state.index_s:.1f} s")

    def pick(
        self, *, arm_key: str | None = None, tenant: str | None = None, candidate: bool = True
    ) -> tuple[_State, str, str | None]:
        """The state a request is answered with -> (state, arm, heads sha): the tenant's current
        heads if it has some, else ``current.npz``, else the base; and the matching candidate for
        the share of ``arm_key`` values that fall in the candidate bucket. A key with heads files of
        its own runs its own experiment; any other key's requests are part of the shared one (its
        candidate's share included), as requests without a key are. A variant that is not built
        yet is skipped, so a request never waits for one. ``candidate=False`` leaves the candidate
        out (``settled``)."""
        base = self.state(fallback=self.make_fallback is not None)
        chosen, arm, sha = base, "base", self.heads_sha
        if self.heads_dir is None or base.lexical:
            return chosen, arm, sha
        where, prefix = self.heads_dir, ""
        if tenant:
            own, mine = self.heads_dir / "tenants" / tenant, f"tenant:{tenant}"
            current = self._variant(mine, own / "current.npz")
            if current is not None and current.state is not None:
                chosen, arm, sha = current.state, mine, current.sha
            if current is not None or (own / "candidate.npz").exists():
                where, prefix = own, mine
        if arm == "base" and self.use_current:
            current = self._variant("current", self.heads_dir / "current.npz")
            if current is not None and current.state is not None:
                chosen, arm, sha = current.state, "current", current.sha
        name = f"{prefix}:candidate" if prefix else "candidate"
        trial = self._variant(name, where / "candidate.npz")  # built as soon as it appears
        if (
            candidate
            and trial is not None
            and trial.state is not None
            and bucket(arm_key, self.candidate_share)
        ):
            chosen, arm, sha = trial.state, name, trial.sha
        return chosen, arm, sha

    def settled(self, timeout: float = 600.0) -> tuple[_State, str, str | None]:
        """What a request without a key gets once ``DATA/heads/current.npz`` (if there is one) is
        built, the candidate left out: the arm ``toolrank calibrate`` measures."""
        deadline = time.monotonic() + timeout
        while True:
            st, arm, sha = self.pick(candidate=False)
            current = self._variants.get("current")
            waiting = current is not None and current.state is None and current.error is None
            if not waiting or time.monotonic() > deadline:
                return st, arm, sha
            time.sleep(0.05)

    # -- queries -------------------------------------------------------------------------------
    def describe(self, k: int | None = None, *, gated: bool = False) -> str:
        fixed = k or self.fixed_k
        gate = f"; confidence at least {self.min_confidence:g}" if gated else ""
        if fixed:
            return f"top {fixed}{gate}"
        return (f"adaptive K ({self.rule.describe()})" if self.rule else f"top {self.depth}") + gate

    def _ranked(
        self, st: _State, q: Query, depth: int, want: int, visible: frozenset[str] | None
    ) -> tuple[RankedList, RankedList | None, int]:
        """The state's scorer's (fused, first-stage) lists for ``q``, ranked deeper until a key
        limited to some sources has ``want`` of its tools in them (or there are no more)."""
        while True:
            if hasattr(st.scorer, "rank_pairs"):
                fused, semantic = st.scorer.rank_pairs([q], depth)[0]
            else:
                fused, semantic = st.scorer.rank([q], depth)[0], None
            if visible is None:
                break
            # a key limited to some sources: rank deeper until enough of its tools are in the list
            fused, semantic = _only(fused, visible), (_only(semantic, visible) if semantic else None)
            if len(fused.tool_ids) >= want or depth >= len(st.tools):
                break
            depth = min(len(st.tools), 4 * depth)
        return fused, semantic, depth

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        instruction: str | None = None,
        arm_key: str | None = None,
        tenant: str | None = None,
    ) -> SearchResult:
        """A named key's request runs in its own ``cache_scope``: what another key asked before
        neither speeds it up nor shows. Request and instruction are ``tidy``'d first."""
        if instruction is not None:
            instruction = tidy(instruction)
        with _scope(tenant):
            return self._search(tidy(query), k=k, instruction=instruction, arm_key=arm_key, tenant=tenant)

    def _search(
        self,
        query: str,
        *,
        k: int | None,
        instruction: str | None,
        arm_key: str | None,
        tenant: str | None,
    ) -> SearchResult:
        st, arm, heads = self.pick(arm_key=arm_key, tenant=tenant)
        inst = self.instruction if instruction is None else instruction
        q = Query(id=uuid.uuid4().hex, text=query, qrels={}, instruction=inst)
        fixed = k or self.fixed_k
        rule = None if fixed else self.rule
        top = fixed or (rule.max_k if rule else self.depth)
        t0 = time.perf_counter()
        want = max(top, LOG_TOP)
        visible = self.visible(st, tenant)
        depth = want  # the first-stage scorers go deeper themselves for a key limited to some sources
        failures: list[str] = []
        rank_within = visible_ids.set(visible)  # the scorers rank among the key's own tools only
        tolerate = rerank_failures.set(failures)  # a failing second stage: the first stage's list
        try:
            fused, semantic, depth = self._ranked(st, q, depth, want, visible)
        finally:
            rerank_failures.reset(tolerate)
            visible_ids.reset(rank_within)
        if not all(t in st.by_id for t in fused.tool_ids):  # a shared index (pgvector) holding other rows
            known = frozenset(st.by_id)
            fused, semantic = _only(fused, known), (_only(semantic, known) if semantic else None)
        if st.lexical:  # BM25 pads its list with zero scores: those tools share no term with the request
            keep = [n for n, s in enumerate(fused.scores) if s > 0]
            fused = RankedList(
                fused.query_id, [fused.tool_ids[n] for n in keep], [fused.scores[n] for n in keep]
            )
        if rule is not None and getattr(st.scorer, "score_kind", "cosine") != "bm25":
            shown = rule.cut(fused, semantic)
        else:
            shown = RankedList(fused.query_id, fused.tool_ids[:top], fused.scores[:top])
        cal, confidence = self.calibration(st, heads), None
        # a request that names its own k asked for that many tools: it is never turned away
        gated = cal is not None and k is None and self.min_confidence is not None
        first = (semantic or fused).scores  # the first stage's: the scores the calibration holds
        if cal is not None and first:
            confidence = cal.confidence(first[0])
            if gated and confidence < self.min_confidence:
                shown = RankedList(fused.query_id, [], [])
        took = (time.perf_counter() - t0) * 1000.0
        fmt = getattr(innermost(st.scorer), "query_format", None)
        emb_key = None
        if self.cache_key is not None and fmt is not None and not st.lexical:
            emb_key = self.cache_key(fmt(q))
        hits = [Hit(st.by_id[t], s) for t, s in zip(shown.tool_ids, shown.scores, strict=True)]
        added = 0
        # not when the request names its own k: it asked for that many tools
        if self.co_use is not None and self.co_use_extra > 0 and k is None and hits and not st.lexical:
            known = dict(zip(fused.tool_ids, fused.scores, strict=True))
            table = self.co_use.table(tenant)
            reach = st.by_id if visible is None else visible
            for tool, owner in partners(shown.tool_ids, table, self.co_use_extra, reach):
                hits.append(Hit(st.by_id[tool], known.get(tool, 0.0), used_with=owner))
                added += 1
        encoder = getattr(innermost(st.scorer), "encoder", None)  # under a hybrid or a second stage too
        return SearchResult(
            query=query,
            instruction=inst,
            hits=hits,
            ranked=list(zip(fused.tool_ids[:LOG_TOP], fused.scores[:LOG_TOP], strict=True)),
            took_ms=took,
            rule=f"top {top} keyword matches" if st.lexical else self.describe(fixed, gated=gated),
            emb_key=emb_key,
            scorer=st.scorer.name,
            catalog=st.catalog,
            mode="lexical" if st.lexical else "semantic",
            own_instruction=inst != self.instruction,
            arm=arm,
            heads=heads,
            added=added,
            model=getattr(encoder, "model", None),
            rerank_error=failures[-1] if failures else None,
            confidence=None if confidence is None else round(confidence, 4),
        )

    def rank(
        self,
        query: str,
        tools: Sequence[Tool],
        *,
        instruction: str | None = None,
        arm_key: str | None = None,
        tenant: str | None = None,
    ) -> list[tuple[str, float]]:
        """Cosine of caller-supplied tools (not necessarily in the index), best first. The heads are
        picked as a search's are: the tenant's or the promoted ones, and the candidate's share; a
        named key's tools and request are embedded in its own ``cache_scope`` (an exact catalogue
        tool text it was not given never comes back faster)."""
        if instruction is not None:
            instruction = tidy(instruction)
        with _scope(tenant):
            return self._rank(tidy(query), tools, instruction=instruction, arm_key=arm_key, tenant=tenant)

    def _rank(
        self,
        query: str,
        tools: Sequence[Tool],
        *,
        instruction: str | None,
        arm_key: str | None,
        tenant: str | None,
    ) -> list[tuple[str, float]]:
        st, _arm, _heads = self.pick(arm_key=arm_key, tenant=tenant)
        if st.lexical:
            st = self.state()  # the keyword stand-in cannot score tools it has not indexed: wait
        if not hasattr(st.scorer, "score_tools"):
            raise ValueError(f"{st.scorer.name} cannot score tools outside its index")
        inst = self.instruction if instruction is None else instruction
        q = Query(id=uuid.uuid4().hex, text=query, qrels={}, instruction=inst)
        mark = supplied.set(True)  # the caller's tools: a request's limits, kept in memory only
        try:
            scores = st.scorer.score_tools(q, list(tools))
        finally:
            supplied.reset(mark)
        return sorted(zip([t.id for t in tools], scores, strict=True), key=lambda x: -x[1])

    def visible(self, st: _State, tenant: str | None) -> frozenset[str] | None:
        """The ids of ``st``'s tools that ``tenant`` may reach; None when it may reach them all."""
        allowed = self.allowed.get(tenant) if tenant else None
        if allowed is None:
            return None
        key = (st.catalog, tenant or "")
        found = self._visible.get(key)
        if found is None:
            found = frozenset(t.id for t in st.tools if t.category in allowed)
            if len(self._visible) > 64:  # catalogues come and go: keep the cache small
                self._visible.clear()
            self._visible[key] = found
        return found

    def tools(self, tenant: str | None = None) -> list[Tool]:
        return self.catalogue(tenant)[0]

    def catalogue(self, tenant: str | None = None) -> tuple[list[Tool], str]:
        """The tools (those ``tenant`` may reach) and the hash of the ``tools.jsonl`` they came from,
        from one state."""
        st = self.state(fallback=self.make_fallback is not None)
        visible = self.visible(st, tenant)
        return (st.tools if visible is None else [t for t in st.tools if t.id in visible]), st.catalog

    def get(self, tool_id: str, tenant: str | None = None) -> Tool | None:
        """The tool, or None when there is none or ``tenant`` may not reach it (alike, on purpose)."""
        st = self.state(fallback=self.make_fallback is not None)
        visible = self.visible(st, tenant)
        return st.by_id.get(tool_id) if visible is None or tool_id in visible else None
