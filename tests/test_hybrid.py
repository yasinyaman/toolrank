import pytest

from toolrank.adapters.bm25 import BM25Scorer
from toolrank.adapters.hybrid import HybridScorer
from toolrank.cli import main
from toolrank.domain import Query, RankedList
from toolrank.formats import TOOL_FORMATS


class _Fixed:
    """A scorer that returns preset lists (best first), whatever the query."""

    def __init__(self, name, ids, scores, score_kind="cosine"):
        self.name, self.ids, self.scores, self.score_kind = name, ids, scores, score_kind
        self.tool_format = self.query_format = TOOL_FORMATS["name_desc"]
        self.indexed = None

    def index(self, tools):
        self.indexed = list(tools)

    def rank(self, queries, k):
        return [RankedList(q.id, self.ids[:k], self.scores[:k]) for q in queries]


def _q():
    return [Query(id="q", text="send an email", qrels={})]


def test_rrf_sums_reciprocal_ranks_and_breaks_ties_by_semantic_rank():
    sem = _Fixed("sem", ["a", "b", "c", "d"], [0.9, 0.8, 0.7, 0.6])
    lex = _Fixed("bm25", ["c", "x", "a"], [5.0, 4.0, 3.0], "bm25")
    h = HybridScorer(sem, lex, k_rrf=60, depth=10)
    (r,) = h.rank(_q(), k=4)
    # a: 1/61 + 1/63, c: 1/63 + 1/61 (tie, a has the better semantic rank), b: 1/62, x: 1/62 (tie, b is in sem)
    assert r.tool_ids == ["a", "c", "b", "x"]
    assert r.scores[0] == pytest.approx(1 / 61 + 1 / 63) and r.scores[2] == pytest.approx(1 / 62)
    assert h.last_semantic["q"].tool_ids == ["a", "b", "c", "d"]
    assert h.name == "hybrid[rrf60,d10]/sem+bm25" and h.score_kind == "rrf"


def test_lexical_weight_scales_the_bm25_term():
    sem = _Fixed("sem", ["a", "b"], [0.9, 0.8])
    lex = _Fixed("bm25", ["b", "a"], [5.0, 4.0], "bm25")
    h = HybridScorer(sem, lex, lexical_weight=0.25)
    (r,) = h.rank(_q(), k=2)
    assert r.tool_ids == ["a", "b"] and r.scores[0] == pytest.approx(1 / 61 + 0.25 / 62)
    assert h.name == "hybrid[rrf60,d100,w0.25]/sem+bm25"


def test_zero_score_bm25_padding_is_ignored():
    sem = _Fixed("sem", ["a", "b"], [0.9, 0.8])
    lex = _Fixed("bm25", ["z", "y"], [0.0, 0.0], "bm25")  # bm25s pads with 0-score hits when nothing matches
    (r,) = HybridScorer(sem, lex).rank(_q(), k=5)
    assert r.tool_ids == ["a", "b"]


def test_hybrid_end_to_end_on_the_synthetic_set(tmp_path, capsys):
    from toolrank.datasets.synthetic import write_synthetic

    write_synthetic(tmp_path, n_tools=60, n_queries=20, seed=3)
    sem = BM25Scorer("name_desc", "plain")  # stands in for a cosine scorer: any list will do here
    sem.score_kind = "cosine"
    h = HybridScorer(sem, BM25Scorer("documentation", "plain"), depth=20)
    from toolrank.datasets.jsonl import load_queries, load_tools

    h.index(load_tools(tmp_path / "tools.jsonl"))
    ranked = h.rank(load_queries(tmp_path / "queries.jsonl"), k=10)
    assert all(len(r.tool_ids) <= 10 and len(set(r.tool_ids)) == len(r.tool_ids) for r in ranked)
    with pytest.raises(SystemExit, match="dense or clm"):
        main(["eval", "--data", str(tmp_path), "--scorer", "bm25", "--hybrid"])
