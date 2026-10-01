"""The usage log: what the proxy retrieved for a request and what the agent then called (schema v3).

It is also future training data: tools retrieved but not called are hard-negative candidates,
tools called successfully are positives, failed calls weak positives. One JSON object per line, in
daily files ``<dir>/usage-YYYY-MM-DD.jsonl``; every event is a single ``os.write`` on an
``O_APPEND`` descriptor, so a stdio and an HTTP server can share the directory. Request and error
text are kept only with ``log_text`` (``mask_pii`` then replaces e-mail addresses, phone, card and
IBAN numbers in them); otherwise requests and arguments are HMAC-SHA256 digests under a per-install
key (``<dir>/.key``, mode 0600): repeats are recognisable, guesses are not.
``emb_hmac`` is the digest of the request's embedding-cache key: whoever holds the key can match it
to the cache's keys and use the request's backbone vector without its text. The server's own
instruction is configuration and is logged as text; one sent with a request is request text, so
only ``instruction_hmac`` is kept. The name of a tool that does not exist is what the agent typed:
it is logged as text, cut at ``UNKNOWN_TOOL_CHARS``.

``search``: v, event, ts, id, session, client, via, tenant, query_hmac, query, emb_hmac,
instruction_hmac, instruction, rule, results ([[tool, score], ...], the top 20 before the cut),
shown (tools returned), took_ms, scorer, heads, arm (which heads answered: base, current, candidate,
tenant:<name>[:candidate]; added to v3 with ``toolrank learn``), catalog.

``call``: v, event, ts, id, session, client, via, tenant, tool, kind (mcp | openapi), search_id,
rank, link, outcome (ok | tool_error | protocol_error | timeout | refused | unknown_tool),
http_status, took_ms, args_hmac, error.

Who asked: ``session`` is the MCP session (``stdio`` for a stdio server, which serves one client;
``null`` over HTTP for 2026-07-28 clients, which have none); ``client`` is a coarser caller key —
API key name, client app and remote host — stored as a keyed digest; ``tenant`` is the API key's
name (``--api-keys``). ``link`` says how a call was tied to a search: ``search_id`` (the agent
passed it back), ``session`` (the session's latest search that returned the tool), ``client`` (for
a call without a session: the same client's latest such search through the same interface,
``via``) or ``none``. A call that has a session is never tied to another session's search.

v2 (30 Sep 2026): ``client`` added; v1's ``recent`` link (the latest such search of any caller)
is gone, since it tied calls to other clients' searches.

v3 (30 Sep 2026): ``emb_key`` became ``emb_hmac`` (the cache key is an unkeyed hash of endpoint,
model and request, so the log alone confirmed a guess at a request); a request's own instruction is
no longer logged as text; unknown tool names are cut.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
# what ``mask_pii`` replaces, in this order: an IBAN or card number must not be left as a "phone"
_PII = (
    ("<email>", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("<iban>", re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]){11,30}\b")),
    ("<card>", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("<phone>", re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d\b")),
)
OUTCOMES = ("ok", "tool_error", "protocol_error", "timeout", "refused", "unknown_tool")
_KEEP = 512  # recent searches kept in memory for linking calls
UNKNOWN_TOOL_CHARS = 200


@dataclass(frozen=True)
class _Seen:
    session: str | None
    client: str | None
    via: str
    ranks: dict[str, int]


def mask_pii(text: str) -> str:
    """``text`` with e-mail addresses, IBANs, card numbers and phone numbers (nine digits or more)
    replaced by tags. A pattern, not an understanding: names and addresses pass."""
    for tag, rx in _PII:
        if tag == "<phone>":  # nine digits or more: a date or a small number stays
            text = rx.sub(
                lambda m: "<phone>" if sum(c.isdigit() for c in m.group()) >= 9 else m.group(), text
            )
        else:
            text = rx.sub(tag, text)
    return text


class UsageLog:
    def __init__(
        self,
        directory: str | Path | None,
        *,
        log_text: bool = False,
        mask_pii: bool = False,
        tenant: str | None = None,
    ):
        self.dir = Path(directory).resolve() if directory else None
        self.log_text, self.mask_pii, self.tenant = log_text, mask_pii, tenant
        self._key = self._load_key() if self.dir is not None else secrets.token_bytes(32)
        self._lock = threading.Lock()
        self._searches: OrderedDict[str, _Seen] = OrderedDict()

    # -- storage -------------------------------------------------------------------------------
    def _load_key(self) -> bytes:
        assert self.dir is not None
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / ".key"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return path.read_bytes()
        with os.fdopen(fd, "wb") as f:
            f.write(secrets.token_bytes(32))
        return path.read_bytes()

    def digest(self, text: str) -> str:
        return hmac.new(self._key, text.encode("utf-8"), sha256).hexdigest()

    def _write(self, event: dict[str, Any]) -> None:
        if self.dir is None:
            return
        day = event["ts"][:10]
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(self.dir / f"usage-{day}.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="milliseconds")

    def _text(self, text: str | None) -> str | None:
        """Request or error text as the log keeps it: not at all, as is, or masked."""
        if text is None or not self.log_text:
            return None
        return mask_pii(text) if self.mask_pii else text

    def _client(self, client: str | None) -> str | None:
        return self.digest(client)[:16] if client else None  # holds a remote address: never in clear

    # -- events --------------------------------------------------------------------------------
    def search(
        self,
        result: Any,
        *,
        session: str | None,
        via: str,
        heads: str | None = None,
        client: str | None = None,
        tenant: str | None = None,
        arm: str | None = None,
    ) -> str:
        """Log a ``retriever.SearchResult``; -> its search id (returned to the agent)."""
        sid = "s-" + uuid.uuid4().hex[:16]
        ranks = {h.id: n for n, h in enumerate(result.hits, 1)}
        with self._lock:
            self._searches[sid] = _Seen(session, client, via, ranks)
            while len(self._searches) > _KEEP:
                self._searches.popitem(last=False)
        self._write(
            {
                "v": SCHEMA_VERSION,
                "event": "search",
                "ts": self._now(),
                "id": sid,
                "session": session,
                "client": self._client(client),
                "via": via,
                "tenant": tenant or self.tenant,
                "query_hmac": self.digest(result.query),
                "query": self._text(result.query),
                "emb_hmac": self.digest(result.emb_key) if result.emb_key else None,
                "instruction_hmac": self.digest(result.instruction),
                "instruction": result.instruction
                if not result.own_instruction
                else self._text(result.instruction),
                "rule": result.rule,
                "results": [[t, round(s, 6)] for t, s in result.ranked],
                "shown": len(result.hits),
                "took_ms": round(result.took_ms, 2),
                "scorer": result.scorer,
                "heads": heads,
                "arm": arm,
                "catalog": result.catalog,
            }
        )
        return sid

    def link(
        self,
        tool: str,
        session: str | None,
        search_id: str | None = None,
        *,
        via: str | None = None,
        client: str | None = None,
    ) -> tuple[str | None, int | None, str]:
        """(search id, rank of ``tool`` in it, how it was found) for a call."""
        with self._lock:
            # a call must never fail on its log entry: anything but a known id is no link
            seen = self._searches.get(search_id) if isinstance(search_id, str) else None
            if seen is not None and tool in seen.ranks:
                return search_id, seen.ranks[tool], "search_id"
            by_client: tuple[str | None, int | None, str] | None = None
            for sid, s in reversed(self._searches.items()):
                if tool not in s.ranks:
                    continue
                if session is not None and s.session == session:
                    return sid, s.ranks[tool], "session"
                if (
                    by_client is None
                    and session is None
                    and client is not None
                    and s.client == client
                    and s.via == via
                ):
                    by_client = (sid, s.ranks[tool], "client")
        return by_client or (None, None, "none")

    def call(
        self,
        *,
        tool: str,
        kind: str | None,
        session: str | None,
        via: str,
        outcome: str,
        took_ms: float,
        arguments: Any = None,
        search_id: str | None = None,
        http_status: int | None = None,
        error: str | None = None,
        client: str | None = None,
        tenant: str | None = None,
    ) -> str:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        if outcome == "unknown_tool":  # not a catalogue id: whatever the agent typed
            tool = tool[:UNKNOWN_TOOL_CHARS]
        cid = "c-" + uuid.uuid4().hex[:16]
        sid, rank, how = self.link(tool, session, search_id, via=via, client=client)
        args = json.dumps(
            arguments if arguments is not None else {}, sort_keys=True, ensure_ascii=False, default=str
        )
        self._write(
            {
                "v": SCHEMA_VERSION,
                "event": "call",
                "ts": self._now(),
                "id": cid,
                "session": session,
                "client": self._client(client),
                "via": via,
                "tenant": tenant or self.tenant,
                "tool": tool,
                "kind": kind,
                "search_id": sid,
                "rank": rank,
                "link": how,
                "outcome": outcome,
                "http_status": http_status,
                "took_ms": round(took_ms, 2),
                "args_hmac": self.digest(args),
                "error": self._text(error[:200]) if error else None,
            }
        )
        return cid
