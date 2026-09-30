"""Lexical baseline: BM25 via ``bm25s`` (the ToolRet paper's "BM25s" row).

Tokenisation: lowercase, English stopwords, optional Snowball stemming when ``PyStemmer`` is
installed (``pip install toolrank[stem]``). Anthropic's Tool Search Tool BM25 variant and most
gateways' "keyword search" are this class of model, so this row is the floor every learned
scorer must clear.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence

import numpy as np

from toolrank.domain import Query, RankedList, Tool
from toolrank.formats import QUERY_FORMATS, TOOL_FORMATS, NamedFormatter


def _stemmer(lang: str = "english"):
    try:
        import Stemmer  # PyStemmer

        return Stemmer.Stemmer(lang)
    except ImportError:
        return None


class BM25Scorer:
    name = "bm25"
    score_kind = "bm25"

    def __init__(
        self,
        tool_format: NamedFormatter | str = "documentation",
        query_format: NamedFormatter | str = "plain",
        *,
        k1: float = 1.5,
        b: float = 0.75,
        stem: bool = True,
        stopwords: str | None = "en",
    ):
        self.tool_format = TOOL_FORMATS[tool_format] if isinstance(tool_format, str) else tool_format
        self.query_format = QUERY_FORMATS[query_format] if isinstance(query_format, str) else query_format
        self.k1, self.b, self.stopwords = k1, b, stopwords
        self.stemmer = _stemmer() if stem else None
        self.name = f"bm25/{self.tool_format.name}/{self.query_format.name}" + (
            "" if self.stemmer else "/nostem"
        )
        self._ids: list[str] = []
        self._bm25 = None
        self._lock = threading.Lock()  # a server ranks from several threads; the stemmer is not shared-safe

    def index(self, tools: Iterable[Tool]) -> None:
        import bm25s

        tools = list(tools)
        self._ids = [t.id for t in tools]
        texts = [self.tool_format(t) for t in tools]
        toks = bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer, show_progress=False)
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self._bm25.index(toks, show_progress=False)

    def rank(self, queries: Sequence[Query], k: int) -> list[RankedList]:
        import bm25s

        if self._bm25 is None:
            raise RuntimeError("call index() before rank()")
        texts = [self.query_format(q) for q in queries]
        kk = min(k, len(self._ids))
        with self._lock:
            toks = bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer, show_progress=False)
            docs, scores = self._bm25.retrieve(toks, k=kk, show_progress=False, n_threads=1)
        docs, scores = np.asarray(docs), np.asarray(scores, dtype=np.float32)
        return [
            RankedList(q.id, [self._ids[int(j)] for j in docs[i]], [float(s) for s in scores[i]])
            for i, q in enumerate(queries)
        ]
