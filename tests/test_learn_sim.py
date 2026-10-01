"""``scripts/learn_sim.py``: a benchmark split into traffic and held-out queries, and the simulated
agent whose calls end up in the usage log the way ``toolrank learn`` reads them."""

import importlib.util
import random
from pathlib import Path

import pytest

from toolrank.datasets.jsonl import load_queries, load_tools, write_queries, write_tools
from toolrank.datasets.synthetic import make_queries, make_tools
from toolrank.learn import judge, mine, read_events
from toolrank.retriever import Hit, SearchResult
from toolrank.usage import UsageLog

ROOT = Path(__file__).resolve().parents[1]


def _script():
    path = ROOT / "scripts" / "learn_sim.py"
    if not path.exists():  # the sdist ships no scripts
        pytest.skip("scripts/ is not here")
    spec = importlib.util.spec_from_file_location("learn_sim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Retriever:
    """Shows every query the same three tools after its gold one at ``rank`` (0: not shown); sessions
    alternate between the arms."""

    def __init__(self, tools, rank):
        self.tools, self.rank, self.n = tools, rank, 0

    def search(self, query, *, instruction=None, arm_key=None):
        gold = next(t for t in self.tools if t.id == query.split()[-1])
        others = [t for t in self.tools if t is not gold][:3]
        shown = others if not self.rank else [*others[: self.rank - 1], gold, *others[self.rank - 1 :]]
        self.n += 1
        return SearchResult(
            query=query,
            instruction=instruction or "",
            hits=[Hit(t, 0.5) for t in shown],
            ranked=[(t.id, 0.5) for t in shown],
            took_ms=1.0,
            rule="test",
            emb_key=f"key of {query}",
            scorer="test",
            catalog="c",
            arm="candidate" if self.n % 2 == 0 else "base",
        )


def test_split_links_the_tools_and_holds_queries_out(tmp_path):
    sim = _script()
    tools = make_tools(20, random.Random(0))
    queries = make_queries(tools, 10, random.Random(0))
    write_tools(tmp_path / "bench" / "tools.jsonl", tools)
    write_queries(tmp_path / "bench" / "queries.jsonl", queries)
    n = sim.split(tmp_path / "bench", tmp_path / "out", 0.7, 0)
    traffic = load_queries(tmp_path / "out" / "traffic.jsonl")
    held = load_queries(tmp_path / "out" / "heldout" / "queries.jsonl")
    assert sorted(q.id for q in traffic + held) == sorted(q.id for q in queries)
    asked = {t for q in traffic for t in q.qrels}  # heldout_new: no gold tool the traffic asks for
    new = [q for q in held if not set(q.qrels) & asked]
    assert load_queries(tmp_path / "out" / "heldout_new" / "queries.jsonl") == new
    assert n == {"traffic": 7, "heldout": 3, "heldout_new": len(new)}
    assert load_tools(tmp_path / "out" / "tools.jsonl") == load_tools(
        tmp_path / "out" / "heldout" / "tools.jsonl"
    )
    assert sim.split(tmp_path / "bench", tmp_path / "out", 0.7, 0) == n  # again, over the links
    assert [q.id for q in load_queries(tmp_path / "out" / "traffic.jsonl")] == [q.id for q in traffic]


def test_play_logs_searches_and_the_calls_of_the_gold_tools_shown(tmp_path):
    sim = _script()
    tools = make_tools(8, random.Random(0))
    queries = make_queries(tools, 6, random.Random(0))
    gold = [next(iter(q.qrels)) for q in queries]  # one gold tool each, named by the request
    queries = [type(q)(q.id, f"find {g}", {g: 1}, "inst") for q, g in zip(queries, gold, strict=True)]
    arms = sim.play(_Retriever(tools, 2), UsageLog(tmp_path / "usage"), queries)
    assert arms["base"] == {
        "searches": 3,
        "gold_shown": 1.0,
        "complete": 1.0,
        "top1": 0.0,
        "mrr": 0.5,
        "shown": 4.0,
    }
    events = read_events(tmp_path / "usage")
    pairs, counts = mine(events)
    assert (
        counts["searches"] == 6
        and counts["calls_linked"] == 6
        and len(pairs) == len({q.text for q in queries})
    )
    assert {p.positives for p in pairs} == {(g,) for g in gold} and all(len(p.negatives) == 3 for p in pairs)
    stats = judge(events)  # what toolrank ab reads agrees with the script's own summary
    assert stats["candidate"]["searches"] == 3 and stats["candidate"]["mrr"] == arms["candidate"]["mrr"]

    # a gold tool that was not shown is never called; a noisy agent calls a wrong tool instead
    arms = sim.play(_Retriever(tools, 0), UsageLog(tmp_path / "unseen"), queries)
    assert arms["base"]["gold_shown"] == 0.0 and "calls" not in mine(read_events(tmp_path / "unseen"))[1]
    sim.play(_Retriever(tools, 1), UsageLog(tmp_path / "noisy"), queries, noise=1.0)
    noisy, _ = mine(read_events(tmp_path / "noisy"))
    assert noisy and all(not p.positives and len(p.weak) == 1 for p in noisy)  # the gold one was passed over
