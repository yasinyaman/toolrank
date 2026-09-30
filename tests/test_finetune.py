import numpy as np
import pytest

from toolrank.domain import Query, TrainPair
from toolrank.finetune import Batches, TrainConfig, pair_texts, split_pairs, train_heads
from toolrank.formats import query_format, tool_format


def test_split_drops_benchmark_requests_and_is_deterministic():
    pairs = [TrainPair(id=str(i), text=f"request {i}", positives=("a",)) for i in range(20)]
    bench = [Query(id="q", text="Request   3", qrels={})]  # whitespace and case do not hide a leak
    train, val, dropped = split_pairs(pairs, bench, n_train=10, n_val=5, seed=1)
    assert dropped == 1 and len(train) == 10 and len(val) == 5
    assert "request 3" not in {p.text for p in train + val}
    assert split_pairs(pairs, bench, 10, 5, seed=1) == (train, val, 1)


def test_pair_texts_format_like_the_benchmark():
    doc = '{"name": "get_weather", "description": "Weather for a city.", "parameters": {"city": {"type": "string"}}}'
    pair = TrainPair(id="0", text="rain in Oslo?", positives=(doc,), instruction="Given a weather task")
    states, pos, neg = pair_texts([pair], tool_format("example_call"), query_format("clm"))
    assert states == ["rain in Oslo?\n\nGiven a weather task"]
    assert pos == [["get_weather(city=<string>)\nWeather for a city."]] and neg == [[]]


def test_train_heads_learns_a_toy_mapping_and_saves_clm_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    from toolrank.adapters.clm import CLMHeads

    rng = np.random.default_rng(0)
    dim, n_tools = 32, 64
    tools = rng.standard_normal((n_tools, dim)).astype(np.float32)
    rot = np.linalg.qr(rng.standard_normal((dim, dim)))[0].astype(np.float32)

    def make(n: int) -> Batches:  # a request is a noisy, rotated copy of its tool's vector
        pos = rng.integers(0, n_tools, n)
        states = tools[pos] @ rot + 0.1 * rng.standard_normal((n, dim)).astype(np.float32)
        neg = [[int(x) for x in rng.choice(n_tools, 4, replace=False) if x != p] for p in pos]
        return Batches(states, tools, [[int(p)] for p in pos], neg)

    head_cfg = {"width": 64, "depth": 2, "projection_dim": 16}
    cfg = TrainConfig(epochs=25, batch=64, lr=3e-3, head_cfg=head_cfg, device="cpu")
    ck, hist = train_heads(make(512), make(64), cfg, log=lambda _: None)
    assert hist[-1]["val_recall@10"] > hist[0]["val_recall@10"] + 0.2
    assert ck["cfg"]["best_epoch"] > 0 and ck["cfg"]["trained_from"] == "scratch"
    torch.save(ck, tmp_path / "heads.pt")
    assert CLMHeads(tmp_path / "heads.pt", device="cpu").proj_dim == 16  # loads like a CLM checkpoint

    again, _ = train_heads(
        make(64),
        make(16),
        TrainConfig(epochs=1, batch=32, device="cpu"),
        init=str(tmp_path / "heads.pt"),
        log=lambda _: None,
    )
    from toolrank.adapters.heads_np import sha256_file

    # a file name and its hash, never a local path (the cfg ships inside packaged heads)
    assert (again["cfg"]["trained_from"], again["cfg"]["init_sha256"]) == (
        "heads.pt",
        sha256_file(tmp_path / "heads.pt"),
    )


def test_skip_heads_start_as_the_identity(tmp_path):
    torch = pytest.importorskip("torch")
    from toolrank.adapters.clm import CLMHeads

    rng = np.random.default_rng(1)
    tools = rng.standard_normal((32, 16)).astype(np.float32)
    b = Batches(tools[:8] + 0.01, tools, [[i] for i in range(8)], [[] for _ in range(8)])
    cfg = TrainConfig(
        epochs=1, batch=8, lr=0.0, head_cfg={"width": 8, "depth": 2, "skip": True}, device="cpu"
    )
    ck, hist = train_heads(b, b, cfg, log=lambda _: None)
    assert hist[0]["val_recall@10"] == 1.0  # the raw geometry already ranks each request's tool first
    torch.save(ck, tmp_path / "skip.pt")
    heads = CLMHeads(tmp_path / "skip.pt", device="cpu")
    x = rng.standard_normal((4, 16)).astype(np.float32)
    assert heads.proj_dim == 16
    assert np.allclose(heads.project_states(x), x / np.linalg.norm(x, axis=1, keepdims=True), atol=1e-5)


def test_freeze_action_trains_the_state_head_only():
    pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    tools = rng.standard_normal((32, 16)).astype(np.float32)
    rot = np.linalg.qr(rng.standard_normal((16, 16)))[0].astype(np.float32)
    # requests are rotated tools: the identity ranks them badly, only the state head can fix it
    b = Batches(tools @ rot, tools, [[i] for i in range(32)], [[] for _ in range(32)])
    cfg = TrainConfig(
        epochs=30,
        batch=8,
        lr=1e-2,
        head_cfg={"width": 32, "depth": 2, "skip": True},
        device="cpu",
        freeze_action=True,
    )
    ck, hist = train_heads(b, b, cfg, log=lambda _: None)
    assert ck["cfg"]["best_epoch"] > 0 and hist[-1]["val_recall@10"] > hist[0]["val_recall@10"]
    out = lambda head: [float(v.abs().max()) for k, v in ck[head].items() if k.startswith("out.")]  # noqa: E731
    assert max(out("action_head")) == 0.0  # frozen: still the identity
    assert max(out("state_head")) > 0.0


def test_neg_filter_drops_negatives_the_start_scores_like_the_positive():
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    tools = rng.standard_normal((4, 16)).astype(np.float32)
    tools[1] = tools[0]  # tool 1 is a copy of the positive: a false negative
    b = Batches(tools[:1].copy(), tools, [[0]], [[1, 2, 3]])
    lines: list[str] = []
    cfg = TrainConfig(
        epochs=1,
        batch=1,
        lr=0.0,
        head_cfg={"width": 8, "depth": 2, "skip": True},
        device="cpu",
        neg_filter=0.95,
    )
    train_heads(b, b, cfg, log=lines.append)
    assert "kept 2 of 3 mined negatives" in lines[0]


def test_selection_on_a_dev_metric_keeps_the_earliest_best_and_never_ships_worse():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    dim = 16
    tools = rng.standard_normal((20, dim)).astype(np.float32)
    train = Batches(tools[:8] + 0.1, tools, [[i] for i in range(8)], [[] for _ in range(8)])
    cfg = TrainConfig(epochs=3, batch=4, lr=1e-3, head_cfg={"width": 16, "depth": 2, "projection_dim": 8})

    def scripted(values):
        return lambda epoch, sh, ah: {"dev.NDCG@10": values[epoch]}

    quiet = {"log": lambda _: None, "select": "dev.NDCG@10"}
    ck, hist = train_heads(train, None, cfg, on_epoch=scripted([0.5, 0.7, 0.6, 0.7]), **quiet)
    assert ck["cfg"]["best_epoch"] == 1 and "val_recall@10" not in hist[0]  # the tie at 3 keeps 1
    ck, _ = train_heads(train, None, cfg, on_epoch=scripted([0.5, 0.4, 0.3, 0.2]), **quiet)
    assert ck["cfg"]["best_epoch"] == 0  # training never beat the start: the start comes back
    with pytest.raises(ValueError, match="not an epoch metric"):
        train_heads(train, None, cfg, log=lambda _: None, select="dev.NDCG@10")


def test_training_continues_from_packaged_npz_heads(tmp_path):
    torch = pytest.importorskip("torch")
    from test_heads_np import GOLDEN, _case_npz
    from toolrank.adapters.heads_np import NumpyHeads, sha256_file
    from toolrank.finetune import _heads_from

    npz = _case_npz(tmp_path, "gelu_layernorm_skip", dtype="float16", serving={"truncate": 8192})
    with np.load(GOLDEN, allow_pickle=False) as z:
        x = z["x"].astype(np.float32)
    sh, ah, scale, head_cfg, origin = _heads_from(str(npz), TrainConfig(), x.shape[1])
    with torch.no_grad():
        ours = torch.nn.functional.normalize(sh(torch.from_numpy(x)), dim=-1).numpy()
    assert np.allclose(ours, NumpyHeads(npz).project_states(x), atol=1e-5)  # the weights users run
    assert origin == {"trained_from": npz.name, "init_sha256": sha256_file(npz)}
    assert float(scale.detach()) == pytest.approx(2.0) and "trained_from" not in head_cfg
    with pytest.raises(ValueError, match="backbone vectors"):
        _heads_from(str(npz), TrainConfig(), x.shape[1] + 1)


def test_leaks_per_source_and_missing_instructions_filled():
    from toolrank.finetune import leaks, with_instruction

    pairs = [
        TrainPair(id="1", text="Find a hotel", positives=("a",)),
        TrainPair(id="2", text="book a flight", positives=("b",), instruction="Given a travel task"),
    ]
    sets = {
        "dev": [Query(id="d", text="find  a HOTEL", qrels={})],
        "eval": [Query(id="e", text="x", qrels={})],
    }
    assert leaks(pairs, sets) == {"dev": 1, "eval": 0}
    filled, n = with_instruction(pairs, "Given an agent's request")
    assert n == 1 and [p.instruction for p in filled] == ["Given an agent's request", "Given a travel task"]


def test_dev_curve_scores_exactly_like_toolrank_eval():
    import dataclasses
    import hashlib
    import random

    from toolrank.adapters.dense import DenseScorer
    from toolrank.datasets.synthetic import make_queries, make_tools
    from toolrank.eval.runner import run_eval
    from toolrank.finetune import EvalSet, curve_metrics
    from toolrank.formats import QUERY_FORMATS, TOOL_FORMATS

    class _Hash:
        name = "hash"

        def encode(self, texts, *, kind="document"):
            seed = [int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) for t in texts]
            m = np.asarray([np.random.default_rng(s).standard_normal(16) for s in seed], dtype=np.float32)
            return m / np.linalg.norm(m, axis=1, keepdims=True)

    rng = random.Random(3)
    tools = make_tools(60, rng)
    queries = [
        dataclasses.replace(q, category="a" if q.task < "g" else "b") for q in make_queries(tools, 40, rng)
    ]
    enc = _Hash()
    report = run_eval(DenseScorer(enc, "name_desc", "instruct_query"), tools, queries, dataset="syn")
    tf, qf = TOOL_FORMATS["name_desc"], QUERY_FORMATS["instruct_query"]
    dev = EvalSet(
        "dev",
        queries,
        [t.id for t in tools],
        enc.encode([tf(t) for t in tools]),
        enc.encode([qf(q) for q in queries]),
    )
    got = curve_metrics("dev", dev.score(lambda x: x, lambda x: x))  # identity heads = the dense baseline
    assert got["dev.NDCG@10"] == pytest.approx(report.overall["NDCG@10"], abs=1e-9)
    assert got["dev.NDCG@10-cat"] == pytest.approx(report.category_macro["NDCG@10"], abs=1e-9)
