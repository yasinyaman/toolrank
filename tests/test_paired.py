import json
import random

import pytest

from toolrank.cli import main
from toolrank.datasets.jsonl import write_queries, write_tools
from toolrank.datasets.synthetic import make_queries, make_tools
from toolrank.eval.paired import permutation_test, sign_test


def test_sign_test_known_example():
    # 8 wins, 2 losses: two-sided binomial p = 2 * (C(10,0) + C(10,1) + C(10,2)) / 2**10
    wins, losses, p = sign_test([1.0] * 8 + [0.0] * 2, [0.0] * 8 + [1.0] * 2)
    assert (wins, losses) == (8, 2)
    assert p == pytest.approx(2 * (1 + 10 + 45) / 1024)


def test_sign_test_drops_ties_and_checks_lengths():
    assert sign_test([1.0, 1.0, 2.0], [1.0, 0.0, 2.0]) == (1, 0, 1.0)  # two ties drop out
    with pytest.raises(ValueError):
        sign_test([1.0], [1.0, 2.0])


def test_permutation_test_is_exact_up_to_twelve_queries():
    # differences +1, +1: of the 4 sign assignments only ++ and -- reach |sum| 2 -> p = 0.5
    assert permutation_test([1.0, 1.0], [0.0, 0.0]) == 0.5
    assert permutation_test([1.0], [1.0]) == 1.0  # no difference: every assignment reaches


def test_permutation_test_is_deterministic_and_spots_a_real_difference():
    b = [0.7, 0.6, 0.5, 0.75, 0.4, 0.65, 0.55, 0.7, 0.6, 0.5, 0.68, 0.57, 0.46, 0.72]
    a = [x + 0.2 for x in b]  # 14 queries: above the exact range, Monte Carlo
    assert permutation_test(a, b, seed=1) == permutation_test(a, b, seed=1)
    assert permutation_test(a, b, seed=1) < 0.01
    with pytest.raises(ValueError):
        permutation_test([1.0], [1.0, 2.0])


def _eval_runs(tmp_path, name, *extra):
    rng = random.Random(3)
    tools = make_tools(60, rng)
    queries = make_queries(tools, 30, rng)
    data = tmp_path / name
    data.mkdir()
    write_tools(data / "tools.jsonl", tools)
    write_queries(data / "queries.jsonl", queries)
    out = tmp_path / f"{name}.runs.json"
    main(["eval", "--data", str(data), "--scorer", "bm25", *extra, "--runs-out", str(out)])
    return json.loads(out.read_text())


def test_eval_runs_out_writes_one_row_per_query(tmp_path):
    r = _eval_runs(tmp_path, "d")
    assert r["scorer"].startswith("bm25") and r["dataset"] == "d"
    assert len(r["rows"]) == 30
    row = r["rows"][0]
    assert set(row) == {"id", "top", "P@1", "hit@5", "NDCG@10"}
    assert len(row["top"]) <= 20 and row["P@1"] in (0.0, 1.0)
    # P@1 and hit@5 agree with the ids the row carries
    queries = {
        q["id"]: {d for d, g in q["qrels"].items() if g > 0}
        for q in map(json.loads, (tmp_path / "d" / "queries.jsonl").read_text().splitlines())
    }
    for row in r["rows"]:
        rel = queries[row["id"]]
        assert row["P@1"] == float(row["top"][:1] != [] and row["top"][0] in rel)
        assert row["hit@5"] == float(any(t in rel for t in row["top"][:5]))


def _write_runs(path, scorer, rows):
    path.write_text(json.dumps({"dataset": "d", "scorer": scorer, "rows": rows}))


def test_compare_paired_reports_sign_and_permutation_p(tmp_path, capsys):
    ids = [f"q{i}" for i in range(10)]
    rows_a = [{"id": i, "top": [], "P@1": 1.0, "hit@5": 1.0, "NDCG@10": 0.8} for i in ids]
    rows_b = [{**r, "P@1": 0.0, "NDCG@10": 0.6} for r in rows_a[:8]] + rows_a[8:]
    _write_runs(tmp_path / "a.json", "alpha", rows_a)
    _write_runs(tmp_path / "b.json", "beta", rows_b)
    main(["compare", "--paired", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    out = capsys.readouterr().out
    # P@1: 8 up, 0 down -> p = 2/2**8; NDCG@10: the 2 ties copy each extreme assignment 4x
    assert "| P@1 (8 up, 0 down) | 100.00 | 20.00 | +80.00 | 0.0078 |" in out
    assert "| hit@5 (0 up, 0 down) | 100.00 | 100.00 | +0.00 | 1.0000 |" in out
    assert "| NDCG@10 (mean diff) | 80.00 | 64.00 | +16.00 | 0.0078 |" in out
    assert "n = 10 queries paired by id" in out


def test_compare_paired_needs_two_runs_files(tmp_path):
    _write_runs(tmp_path / "a.json", "alpha", [])
    _write_runs(tmp_path / "b.json", "beta", [])
    with pytest.raises(SystemExit):  # three files
        main(["compare", "--paired", "a", "b", "c"])
    with pytest.raises(SystemExit):  # no shared query id
        main(["compare", "--paired", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    (tmp_path / "r.json").write_text(json.dumps({"scorer": "x", "overall": {}}))  # a results file
    with pytest.raises(SystemExit):
        main(["compare", "--paired", str(tmp_path / "r.json"), str(tmp_path / "a.json")])


def test_data_seed_reaches_the_split_not_the_training_seed(monkeypatch, tmp_path):
    import toolrank.finetune as ft

    captured = {}

    def fake_run(job, log=print):
        captured["split_seed"] = job.seed
        captured["train_seed"] = job.train.seed
        return {"embed_only": True}

    monkeypatch.setattr(ft, "run", fake_run)
    args = [
        "finetune",
        "--data",
        str(tmp_path / "pairs.jsonl"),
        "--dev",
        str(tmp_path / "dev"),
        "--out",
        str(tmp_path / "h.npz"),
        "--embed-only",
        "--results",
        str(tmp_path),
    ]
    main(args + ["--seed", "5", "--data-seed", "11"])
    assert captured == {"split_seed": 11, "train_seed": 5}
    main(args + ["--seed", "5"])  # without --data-seed the split follows --seed, as before
    assert captured == {"split_seed": 5, "train_seed": 5}
