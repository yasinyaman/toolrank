import hashlib
import json
import random
import shutil
import sys

import numpy as np
import pytest

from toolrank.adapters.embeddings_api import OpenAIEmbeddings
from toolrank.cli import main
from toolrank.datasets.jsonl import write_pairs, write_queries, write_tools
from toolrank.datasets.synthetic import make_queries, make_tools
from toolrank.domain import TrainPair

URL = "http://unused/v1"


@pytest.fixture
def sent(monkeypatch):
    """16-d vectors from a text hash; records every batch sent to the endpoint."""
    calls: list[list[str]] = []

    def fake_post(self, texts, **kw):
        calls.append(list(texts))
        seeds = [int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) for t in texts]
        return [np.random.default_rng(s).standard_normal(16).astype(np.float32) for s in seeds], 3 * len(
            texts
        )

    monkeypatch.setattr(OpenAIEmbeddings, "_post", fake_post)
    return calls


def _sets(tmp_path):
    """A shared synthetic catalogue: a dev set, an eval set with other queries, training pairs."""
    tools = make_tools(40, random.Random(5))
    dev_q = make_queries(tools, 30, random.Random(11))
    dev_texts = {q.text.lower() for q in dev_q}
    eval_q = [q for q in make_queries(tools, 30, random.Random(12)) if q.text.lower() not in dev_texts]
    for name, queries in (("dev", dev_q), ("evalset", eval_q)):
        write_tools(tmp_path / name / "tools.jsonl", tools)
        write_queries(tmp_path / name / "queries.jsonl", queries)
    by_id = {t.id: t for t in tools}
    train_q = make_queries(tools, 150, random.Random(13))
    pairs = [
        TrainPair(
            id=q.id,
            text=q.text,
            positives=tuple(by_id[t].documentation for t in q.qrels),
            instruction=q.instruction,
        )
        for q in train_q
    ]
    write_pairs(tmp_path / "pairs.jsonl", pairs)
    return tmp_path / "pairs.jsonl", tmp_path / "dev", tmp_path / "evalset"


def _args(tmp_path, pairs, dev, ev, *extra):
    return [
        "finetune",
        "--data", str(pairs), "--dev", str(dev), "--eval", str(ev), "--out", str(tmp_path / "heads.pt"),
        "--emb-url", URL, "--cache-dir", str(tmp_path / "cache"), "--results", str(tmp_path / "results"),
        *extra,
    ]  # fmt: skip


def _report(tmp_path, name="heads"):
    return json.loads((tmp_path / "results" / f"finetune_{name}.json").read_text())


def test_embed_only_fills_the_cache_once_without_torch(tmp_path, sent):
    pairs, dev, ev = _sets(tmp_path)
    assert main(_args(tmp_path, pairs, dev, ev, "--embed-only")) == 0
    first = _report(tmp_path)
    assert first["texts"]["embedded"] == first["texts"]["total"] > 0 and sent
    assert first["n_train"] + sum(first["dropped"].values()) == 150  # leaks counted per source
    sent.clear()
    assert main(_args(tmp_path, pairs, dev, ev, "--embed-only")) == 0
    assert _report(tmp_path)["texts"]["embedded"] == 0 and sent == []  # everything came from the cache
    assert not (tmp_path / "heads.pt").exists()


def test_a_dev_set_that_is_also_reported_is_refused(tmp_path, sent):
    pairs, dev, ev = _sets(tmp_path)
    with pytest.raises(SystemExit, match="both the dev set and an eval set"):
        main(_args(tmp_path, pairs, dev, dev, "--embed-only"))
    shutil.copytree(dev, tmp_path / "copy")  # like mcp_zero and mcp_zero_server
    with pytest.raises(SystemExit, match="share 30 queries"):
        main(_args(tmp_path, pairs, dev, tmp_path / "copy", "--embed-only"))
    with pytest.raises(SystemExit, match="needs categories"):
        main(_args(tmp_path, pairs, dev, ev, "--embed-only", "--select", "ndcg10-cat"))
    with pytest.raises(SystemExit, match="shape fresh heads"):
        main(_args(tmp_path, pairs, dev, ev, "--init-ckpt", "x.pt", "--width", "8"))


def test_training_without_torch_says_what_to_install(tmp_path, sent, monkeypatch):
    pairs, dev, ev = _sets(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit, match=r"\[clm\]"):
        main(_args(tmp_path, pairs, dev, ev))


def test_finetune_selects_on_dev_saves_serving_heads_and_evaluates(tmp_path, sent):
    torch = pytest.importorskip("torch")
    from toolrank.adapters.heads_np import NumpyHeads
    from toolrank.build import DEFAULT_SERVING

    pairs, dev, ev = _sets(tmp_path)
    npz = tmp_path / "heads.npz"
    train = [
        "--epochs",
        "2",
        "--batch",
        "16",
        "--lr",
        "1e-3",
        "--width",
        "16",
        "--depth",
        "2",
        "--device",
        "cpu",
    ]
    assert main(_args(tmp_path, pairs, dev, ev, "--npz", str(npz), "--curve", *train)) == 0
    rep = _report(tmp_path)
    dev_curve = [h["dev.NDCG@10"] for h in rep["history"]]
    assert len(dev_curve) == 3 and all("evalset.NDCG@10" in h for h in rep["history"])
    assert rep["best_epoch"] == dev_curve.index(max(dev_curve))  # the earliest best dev epoch
    gaps = rep["curve_vs_official"]
    assert set(gaps) == {"dev.NDCG@10", "evalset.NDCG@10"} and all(abs(g) < 1e-6 for g in gaps.values())

    cfg = torch.load(tmp_path / "heads.pt")["cfg"]
    assert (cfg["tool_format"], cfg["query_format"], cfg["truncate"]) == (
        "documentation",
        "instruct_query",
        8192,
    )
    assert cfg["instruction"] == DEFAULT_SERVING["instruction"] and cfg["trained_from"] == "scratch"
    assert cfg["selected_on"]["set"] == "dev" and cfg["selected_on"]["epoch"] == rep["best_epoch"]
    assert NumpyHeads(npz).cfg["selected_on"] == cfg["selected_on"]

    later = tmp_path / "later.json"  # eval reads the formats and the truncation from the heads
    flags = ["--scorer", "clm", "--clm-ckpt", str(tmp_path / "heads.pt"), "--emb-model", "qwen3-emb"]
    flags += ["--emb-url", URL, "--cache-dir", str(tmp_path / "cache"), "--with-inst", "--out", str(later)]
    assert main(["eval", "--data", str(ev), *flags]) == 0
    got = json.loads(later.read_text())
    assert (got["config"]["tool_format"], got["config"]["truncate"]) == ("documentation", 8192)
    assert got["overall"]["NDCG@10"] == pytest.approx(rep["official"]["evalset"]["NDCG@10"])
