import json
import random
import zlib

import numpy as np
import pytest

from toolrank.adapters.bm25 import BM25Scorer
from toolrank.adapters.dense import DenseScorer, topk_dot
from toolrank.cli import main
from toolrank.datasets.jsonl import (
    load_pairs,
    load_queries,
    load_tools,
    write_pairs,
    write_queries,
    write_tools,
)
from toolrank.datasets.synthetic import make_queries, make_tools
from toolrank.domain import Query, RankedList, Tool, TrainPair
from toolrank.eval.runner import format_table, run_eval, save_report


def test_pairs_roundtrip(tmp_path):
    p = TrainPair(
        id="t0", text="q", positives=('{"name": "a"}',), negatives=('{"name": "b"}', "c"), instruction="i"
    )
    write_pairs(tmp_path / "pairs.jsonl", [p, TrainPair(id="t1", text="r", positives=("x",))])
    back = load_pairs(tmp_path / "pairs.jsonl")
    assert back[0] == p and back[1].negatives == () and back[1].instruction == ""
    assert load_pairs(tmp_path / "pairs.jsonl", limit=1) == [p]


def _corpus():
    tools = [
        Tool(id="w", doc={"name": "get_weather", "description": "Current weather and forecast for a city."}),
        Tool(
            id="f",
            doc={"name": "search_flights", "description": "Find flights between two airports on a date."},
        ),
        Tool(id="m", doc={"name": "send_email", "description": "Send an email message to a recipient."}),
    ]
    queries = [
        Query(id="q1", text="will it rain in Istanbul tomorrow? forecast please", qrels={"w": 1}, task="a"),
        Query(id="q2", text="book me flights from Ankara to Berlin", qrels={"f": 1}, task="a"),
        Query(id="q3", text="email the report to Ayşe", qrels={"m": 1}, task="b"),
    ]
    return tools, queries


def test_bm25_ranks_obvious_matches_first():
    tools, queries = _corpus()
    s = BM25Scorer("name_desc", "plain")
    s.index(tools)
    ranked = s.rank(queries, k=3)
    assert [r.tool_ids[0] for r in ranked] == ["w", "f", "m"]
    assert all(len(r.tool_ids) == 3 for r in ranked)


def test_runner_report_and_table(tmp_path):
    tools, queries = _corpus()
    report = run_eval(
        BM25Scorer("name_desc", "plain"), tools, queries, dataset="mini", k=3, ks=(1, 3), batch=2
    )
    assert report.n_queries == 3 and report.n_tools == 3
    assert report.overall["NDCG@1"] == 1.0
    assert set(report.per_task) == {"a", "b"}
    assert report.latency_ms["per_query_p50"] >= 0
    p = save_report(report, tmp_path / "r.json")
    back = json.loads(p.read_text())
    assert back["overall"]["Recall@3"] == 1.0
    table = format_table(report, metrics=("NDCG@1",))
    assert "**Avg**" in table and "| a |" in table
    assert report.category_macro == {} and "Cat-macro" not in table  # no categories in _corpus


class _FixedScorer:
    """Returns a preset ranking per query, so aggregation can be checked exactly."""

    name = "fixed"

    def __init__(self, rankings):
        self.rankings = rankings

    def index(self, tools):
        pass

    def rank(self, queries, k):
        out = []
        for q in queries:
            ids = self.rankings[q.id][:k]
            out.append(RankedList(q.id, ids, [1.0] * len(ids)))
        return out


def _categorized_report():
    # web task "a": 3 queries, 1 hit; code task "b": 1 query, 1 hit
    queries = [Query(id=f"a{i}", text="x", qrels={"t": 1}, task="a", category="web") for i in range(3)]
    queries.append(Query(id="b0", text="x", qrels={"t": 1}, task="b", category="code"))
    ranking = {"a0": ["t"], "a1": ["u"], "a2": ["u"], "b0": ["t"]}
    tools = [Tool(id="t"), Tool(id="u")]
    return run_eval(_FixedScorer(ranking), tools, queries, dataset="mini", k=1, ks=(1,))


def test_runner_category_macro_ignores_task_sizes():
    report = _categorized_report()
    assert report.overall["NDCG@1"] == 0.5  # micro: 2 hits over 4 queries
    per_cat = {c: m["NDCG@1"] for c, m in report.per_category.items()}
    assert per_cat == pytest.approx({"code": 1.0, "web": 1 / 3})
    assert report.category_macro["NDCG@1"] == pytest.approx(2 / 3)  # (1/3 + 1) / 2
    table = format_table(report, metrics=("NDCG@1",))
    assert "| *web* |" in table and "| **Cat-macro** | 2 cat. | **66.67** |" in table
    assert report.to_dict()["category_macro"] == report.category_macro


def test_compare_adds_cat_macro_column(tmp_path, capsys):
    new = save_report(_categorized_report(), tmp_path / "new.json")
    legacy = json.loads(new.read_text())
    del legacy["category_macro"]  # results written before the field existed
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    main(["compare", "--metrics", "NDCG@1", str(new), str(tmp_path / "legacy.json")])
    head, _, row_new, row_legacy = capsys.readouterr().out.splitlines()
    assert head == "| Run | dataset | inst | n | NDCG@1 | NDCG@1 cat-macro | p50 ms |"
    assert "| 50.00 | 66.67 |" in row_new and "| 50.00 | — |" in row_legacy
    main(["compare", "--metrics", "NDCG@1", "--cat-macro", "", str(new)])
    assert "cat-macro" not in capsys.readouterr().out


def test_jsonl_roundtrip(tmp_path):
    rng = random.Random(1)
    tools = make_tools(40, rng)
    queries = make_queries(tools, 25, rng)
    write_tools(tmp_path / "tools.jsonl", tools)
    write_queries(tmp_path / "queries.jsonl", queries)
    t2, q2 = load_tools(tmp_path / "tools.jsonl"), load_queries(tmp_path / "queries.jsonl")
    assert [t.id for t in t2] == [t.id for t in tools]
    assert t2[0].doc["name"] == tools[0].doc["name"]
    assert [q.qrels for q in q2] == [q.qrels for q in queries]
    assert load_queries(tmp_path / "queries.jsonl", tasks=[queries[0].task])
    write_queries(
        tmp_path / "cat.jsonl", [Query(id="x", text="t", qrels={"a": 1}, task="t1", category="web")]
    )
    assert load_queries(tmp_path / "cat.jsonl")[0].category == "web"


def test_synthetic_bm25_is_good_but_not_trivial():
    rng = random.Random(3)
    tools = make_tools(300, rng)
    queries = make_queries(tools, 200, rng)
    report = run_eval(BM25Scorer("schema", "plain"), tools, queries, dataset="syn", k=20, ks=(5, 10))
    assert 0.3 < report.overall["NDCG@10"] < 0.999


class _ToyEncoder:
    """Bag-of-words hashing encoder: enough to exercise DenseScorer without a model."""

    name = "toy"

    STOP = {"a", "an", "the", "to", "for", "on", "in", "me", "from", "and", "it", "will", "please", "of"}

    def encode(self, texts, *, kind="document"):
        m = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(":", " ").replace(".", " ").replace("?", " ").split():
                if w not in self.STOP:
                    m[i, zlib.crc32(w.encode()) % 256] += 1.0  # deterministic across processes
        return m


def test_dense_scorer_with_toy_encoder():
    tools, queries = _corpus()
    s = DenseScorer(_ToyEncoder(), "name_desc", "plain")
    s.index(tools)
    ranked = s.rank(queries, k=2)
    assert [r.tool_ids[0] for r in ranked] == ["w", "f", "m"]
    assert len(ranked[0].tool_ids) == 2 and s.name == "dense/toy/name_desc/plain"


def test_topk_dot_is_sorted_and_exact():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    m = np.array([[0.1, 1.0], [0.9, 0.0], [0.5, 0.5]], dtype=np.float32)
    idx, sc = topk_dot(q, m, 2)
    assert idx.tolist() == [[1, 2]] and np.allclose(sc, [[0.9, 0.5]])


def test_summarize_is_the_reports_aggregation_with_and_without_categories():
    from toolrank.eval.runner import summarize

    report = _categorized_report()
    rows = [(Query(id="a", text="x", qrels={}, task="a", category="web"), {"NDCG@1": 1.0})]
    rows += [
        (Query(id=f"b{i}", text="x", qrels={}, task="b", category="code"), {"NDCG@1": 0.0}) for i in range(3)
    ]
    s = summarize(rows)
    assert (s.overall["NDCG@1"], s.category_macro["NDCG@1"]) == (0.25, 0.5)
    assert set(s.per_task) == {"a", "b"} and set(s.per_category) == {"web", "code"}
    plain = summarize([(Query(id="q", text="x", qrels={}), {"NDCG@1": 1.0})])
    assert (plain.per_task.keys(), plain.per_category, plain.category_macro) == ({"all": 0}.keys(), {}, {})
    assert report.category_macro and report.overall["NDCG@1"] == 0.5  # run_eval goes through it


def test_a_ranked_list_or_a_corpus_that_repeats_an_id_is_refused():
    """trec_eval reads a run as a doc -> score map: a doc named twice would count twice here
    (['a', 'a'] with gold {a}: Recall 2.0), so it is an error, not a number."""
    from toolrank.eval.metrics import evaluate_cut, evaluate_query
    from toolrank.eval.runner import run_eval

    assert evaluate_query(["a", "b"], {"a": 1, "c": 1}, ks=(5,))["Recall@5"] == 0.5
    with pytest.raises(ValueError, match="more than once"):
        evaluate_query(["a", "a", "b"], {"a": 1, "c": 1}, ks=(5,))
    with pytest.raises(ValueError, match="more than once"):
        evaluate_cut(["a", "a"], {"a": 1})
    tools = [
        Tool(id="x", doc={"name": "x", "description": "one"}),
        Tool(id="x", doc={"name": "x", "description": "two"}),
    ]
    with pytest.raises(ValueError, match="duplicate tool ids"):
        run_eval(BM25Scorer("name_desc", "plain"), tools, [], dataset="dupes")
