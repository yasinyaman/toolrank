import json

import pytest

from toolrank.adapters.bm25 import BM25Scorer
from toolrank.cli import main
from toolrank.cut import AdaptiveK, cutter
from toolrank.domain import Query, RankedList, Tool
from toolrank.eval.metrics import evaluate_cut
from toolrank.eval.runner import run_eval


def _r(scores, qid="q"):
    return RankedList(qid, [f"t{i}" for i in range(len(scores))], list(scores))


def test_count_margin_threshold_and_bounds():
    scores = [0.80, 0.78, 0.74, 0.60, 0.59]
    assert AdaptiveK(margin=0.05).count(scores) == 2
    assert AdaptiveK(margin=0.25).count(scores) == 5
    assert AdaptiveK(margin=0.25, max_k=3).count(scores) == 3
    assert AdaptiveK(threshold=0.75).count(scores) == 2
    assert AdaptiveK(margin=0.01, min_k=3).count(scores) == 3
    assert AdaptiveK(threshold=0.9).count(scores) == 1  # min_k 1 by default
    assert AdaptiveK(threshold=0.9, min_k=0).count(scores) == 0
    assert AdaptiveK(max_k=4, min_k=4).count(scores) == 4  # a fixed top-4
    assert AdaptiveK(margin=0.1).count([]) == 0


def test_hybrid_lists_take_the_count_from_the_semantic_cosines():
    fused = RankedList("q", ["x", "a", "b"], [0.03, 0.029, 0.02])  # RRF units
    semantic = _r([0.8, 0.79, 0.5])
    assert AdaptiveK(margin=0.05).cut(fused, semantic).tool_ids == ["x", "a"]

    class _Hybrid:
        name, score_kind = "hybrid", "rrf"
        last_semantic = {"q": semantic}

    assert cutter(_Hybrid(), AdaptiveK(margin=0.05))(fused).tool_ids == ["x", "a"]
    with pytest.raises(ValueError, match="cosine"):
        cutter(BM25Scorer(), AdaptiveK(margin=0.05))


def test_evaluate_cut_metrics():
    m = evaluate_cut(["a", "x", "b"], {"a": 1, "b": 2, "c": 1, "z": 0})
    assert m == {"K@cut": 3.0, "Recall@cut": 2 / 3, "Precision@cut": 2 / 3, "Comprehensiveness@cut": 0.0}
    assert evaluate_cut([], {"a": 1}) == {
        "K@cut": 0.0,
        "Recall@cut": 0.0,
        "Precision@cut": 0.0,
        "Comprehensiveness@cut": 0.0,
    }


class _Scored:
    """A cosine-scored scorer with preset scores per tool."""

    name, score_kind = "fixed", "cosine"

    def __init__(self, scores):
        self.scores = scores

    def index(self, tools):
        pass

    def rank(self, queries, k):
        order = sorted(self.scores, key=lambda t: -self.scores[t])[:k]
        return [RankedList(q.id, order, [self.scores[t] for t in order]) for q in queries]


def test_run_eval_adds_cut_metrics_and_compare_shows_k_unscaled(tmp_path, capsys):
    tools = [Tool(id=t) for t in "abcd"]
    queries = [Query(id="q", text="x", qrels={"a": 1, "b": 1})]
    scorer = _Scored({"a": 0.9, "b": 0.88, "c": 0.5, "d": 0.4})
    report = run_eval(
        scorer, tools, queries, dataset="toy", k=4, ks=(1, 2), cut=cutter(scorer, AdaptiveK(margin=0.1))
    )
    assert report.overall["K@cut"] == 2.0 and report.overall["Comprehensiveness@cut"] == 1.0
    assert report.overall["Recall@2"] == 1.0  # fixed-k metrics still read the full list
    (tmp_path / "r.json").write_text(json.dumps(report.to_dict()))
    main(["compare", "--metrics", "K@cut,Recall@cut,NDCG@99", "--cat-macro", "", str(tmp_path / "r.json")])
    assert "| 2.00 | 100.00 | — |" in capsys.readouterr().out
