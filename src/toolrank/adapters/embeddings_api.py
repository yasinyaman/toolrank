"""A ``TextEncoder`` over any OpenAI-compatible ``/v1/embeddings`` endpoint.

Works unchanged against vLLM in pooling mode (``vllm serve Qwen/Qwen3-8B --runner pooling`` for
the CLM backbone, or ``Qwen/Qwen3-Embedding-8B``), against Ollama and against hosted APIs.
Embeddings are cached on disk (SQLite, keyed by model + text) so re-running with another tool
format or another scorer never re-encodes what it already has - the "embed once, iterate on the
heads" workflow that makes CLM fine-tuning cheap.

Standard library only (``urllib``); no client SDK needed.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from toolrank.ports import scoped


def env_api_key(base_url: str, own: str) -> str | None:
    """The key to send to ``base_url`` from the environment: ``$own`` (``TOOLRANK_EMB_API_KEY``,
    ``TOOLRANK_CHAT_API_KEY``) to any endpoint, ``OPENAI_API_KEY`` only to OpenAI itself. An agent's
    OpenAI key sits in many shells; it must not travel to a local or third-party endpoint."""
    if key := os.environ.get(own):
        return key
    url = urllib.parse.urlsplit(base_url)
    return (
        os.environ.get("OPENAI_API_KEY")
        if (url.scheme, url.hostname) == ("https", "api.openai.com")
        else None
    )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class EmbeddingCache:
    """text -> float32 vector, persisted in one SQLite file per cache directory."""

    def __init__(self, path: str | Path, namespace: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ns = namespace
        # several encoders may write at once (timeout); a server searches from several threads (lock)
        self.db = sqlite3.connect(str(self.path), timeout=60.0, check_same_thread=False)
        self._lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
        self.db.commit()

    def _key(self, text: str) -> str:
        return hashlib.sha256(scoped(self.ns + "\x00" + text).encode("utf-8")).hexdigest()

    def get_many(self, texts: Sequence[str]) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        keys = [self._key(t) for t in texts]
        for start in range(0, len(keys), 900):  # SQLite variable limit
            chunk = keys[start : start + 900]
            q = f"SELECT key, dim, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})"
            with self._lock:
                rows = self.db.execute(q, chunk).fetchall()
            found = {k: np.frombuffer(v, dtype=np.float32)[:d] for k, d, v in rows}
            for i, k in enumerate(chunk, start):
                if k in found:
                    out[i] = found[k]
        return out

    def missing(self, texts: Sequence[str]) -> list[int]:
        """Indices of ``texts`` not cached yet (reads keys only, no vectors)."""
        keys = [self._key(t) for t in texts]
        have: set[str] = set()
        for start in range(0, len(keys), 900):
            chunk = keys[start : start + 900]
            q = f"SELECT key FROM emb WHERE key IN ({','.join('?' * len(chunk))})"
            with self._lock:
                have.update(k for (k,) in self.db.execute(q, chunk).fetchall())
        return [i for i, k in enumerate(keys) if k not in have]

    def put_many(self, texts: Sequence[str], vecs: np.ndarray) -> None:
        rows = [
            (self._key(t), int(v.shape[0]), np.ascontiguousarray(v, dtype=np.float32).tobytes())
            for t, v in zip(texts, vecs, strict=True)
        ]
        with self._lock:
            self.db.executemany("INSERT OR REPLACE INTO emb (key, dim, vec) VALUES (?, ?, ?)", rows)
            self.db.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self.db.execute("SELECT COUNT(*) FROM emb").fetchone()[0])


class OpenAIEmbeddings:
    """``TextEncoder`` for ``POST {base_url}/embeddings`` (OpenAI wire format).

    ``truncate_prompt_tokens`` is a vLLM extension (the CLM reference setup uses 2048); leave it
    ``None`` for providers that reject unknown fields. ``normalize=True`` L2-normalises rows,
    which is what CLM's embedder does before the heads.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8090/v1",
        *,
        api_key: str | None = None,
        batch: int = 32,
        truncate_prompt_tokens: int | None = None,
        normalize: bool = True,
        cache_dir: str | Path | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        query_timeout: float | None = None,
        query_max_retries: int | None = None,
    ):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or env_api_key(self.base_url, "TOOLRANK_EMB_API_KEY")
        self.batch, self.truncate, self.normalize = batch, truncate_prompt_tokens, normalize
        self.timeout, self.max_retries = timeout, max_retries  # max_retries = attempts per request
        # a server gives queries a short leash; indexing a catalogue keeps the long one
        self.query_limits: dict[str, Any] = {}
        if query_timeout is not None:
            self.query_limits["timeout"] = query_timeout
        if query_max_retries is not None:
            self.query_limits["attempts"] = query_max_retries
        self.name = f"emb/{model}"
        ns = f"{self.base_url}|{model}|trunc={truncate_prompt_tokens}|norm={normalize}"
        self.cache = EmbeddingCache(Path(cache_dir) / "embeddings.sqlite", ns) if cache_dir else None
        self.tokens_spent = 0
        # texts answered from the cache and texts sent to the endpoint, per kind (a server's metrics)
        self.texts: dict[tuple[str, str], int] = {}
        self._texts_lock = threading.Lock()

    # -- wire ---------------------------------------------------------------------------------
    def _post(
        self, texts: Sequence[str], *, timeout: float | None = None, attempts: int | None = None
    ) -> tuple[list[np.ndarray], int]:
        body: dict = {"model": self.model, "input": list(texts), "encoding_format": "base64"}
        if self.truncate:
            body["truncate_prompt_tokens"] = self.truncate
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:  # never copied onto a redirect, which may point anywhere
            req.add_unredirected_header("Authorization", f"Bearer {self.api_key}")
        last: Exception | None = None
        attempts = attempts or self.max_retries
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    j = json.loads(r.read().decode("utf-8"))
                break
            # OSError: URLError, HTTPError, timeouts, and a keep-alive the server dropped (RemoteDisconnected)
            except (OSError, http.client.HTTPException) as e:  # noqa: PERF203
                last = e
                if attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
        else:
            hint = ""
            if isinstance(last, urllib.error.HTTPError) and last.code in (401, 403) and not self.api_key:
                hint = " (no API key was sent: set TOOLRANK_EMB_API_KEY)"
            raise RuntimeError(f"embeddings endpoint {self.base_url} unreachable: {last}{hint}") from last
        out: list[np.ndarray | None] = [None] * len(texts)
        for d in j["data"]:
            e = d["embedding"]
            v = (
                np.frombuffer(base64.b64decode(e), dtype=np.float32)
                if isinstance(e, str)
                else np.asarray(e, dtype=np.float32)
            )
            out[int(d["index"])] = v.astype(np.float32, copy=False)
        if any(v is None for v in out):
            raise RuntimeError("embeddings response is missing rows")
        return out, int((j.get("usage") or {}).get("prompt_tokens") or 0)  # type: ignore[return-value]

    def cache_key(self, text: str) -> str | None:
        """The embedding-cache key of ``text`` (what a usage log can store instead of the text)."""
        return self.cache._key(text if text else " ") if self.cache is not None else None

    # -- port ---------------------------------------------------------------------------------
    def encode(self, texts: Sequence[str], *, kind: str = "document") -> np.ndarray:
        # vLLM rejects an empty prompt ("The decoder prompt cannot be empty"); ToolRet has one
        # empty query (mnms_query_17, w/o inst), so send a single space: one token, no content.
        texts = [t if t else " " for t in texts]
        # `is not None`: EmbeddingCache has __len__, so an empty cache is falsy
        have = self.cache.get_many(texts) if self.cache is not None else {}
        todo = [i for i in range(len(texts)) if i not in have]
        with self._texts_lock:
            for source, n in (("cache", len(texts) - len(todo)), ("endpoint", len(todo))):
                self.texts[(kind, source)] = self.texts.get((kind, source), 0) + n
        # de-duplicate identical strings within the request
        uniq: dict[str, list[int]] = {}
        for i in todo:
            uniq.setdefault(texts[i], []).append(i)
        pending = list(uniq)
        limits = self.query_limits if kind == "query" else {}
        for s in range(0, len(pending), self.batch):
            chunk = pending[s : s + self.batch]
            vecs, spent = self._post(chunk, **limits)
            self.tokens_spent += spent
            if self.cache is not None:
                self.cache.put_many(chunk, np.stack(vecs))
            for t, v in zip(chunk, vecs, strict=True):
                for i in uniq[t]:
                    have[i] = v
        m = np.stack([have[i] for i in range(len(texts))]).astype(np.float32)
        return l2_normalize(m) if self.normalize else m

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False
