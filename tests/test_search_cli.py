import hashlib
import json

import numpy as np
import pytest

from toolrank.adapters.embeddings_api import OpenAIEmbeddings
from toolrank.cli import main
from toolrank.datasets.jsonl import write_tools
from toolrank.domain import Tool
from toolrank.ingest.text import tool_text


@pytest.fixture
def fake_endpoint(monkeypatch):
    """16-d vectors from a text hash, plus the texts each call sent."""
    sent: list[list[str]] = []

    def fake_post(self, texts):
        sent.append(list(texts))
        rows = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(16).astype(np.float32))
        return rows, 5 * len(texts)

    monkeypatch.setattr(OpenAIEmbeddings, "_post", fake_post)
    return sent


@pytest.fixture
def ingest_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TOOLRANK_HEADS", raising=False)
    monkeypatch.setenv("TOOLRANK_CACHE", str(tmp_path / "no-heads"))
    tools = [
        Tool(
            id=f"mail/{n}",
            doc={"server": "mail", "name": n, "description": d},
            documentation=tool_text("mail", n, d),
            category="mail",
        )
        for n, d in [("send", "Send an email"), ("list", "List the inbox"), ("delete", "Delete a message")]
    ]
    write_tools(tmp_path / "tools" / "tools.jsonl", tools)
    return tmp_path / "tools"


def _args(ingest_dir, tmp_path, *extra):
    return [
        "search",
        "send an email to Ada",
        "--data",
        str(ingest_dir),
        "--cache-dir",
        str(tmp_path / "c"),
        *extra,
    ]


def test_search_uses_raw_qwen3_without_heads_and_keeps_its_index(ingest_dir, tmp_path, fake_endpoint, capsys):
    assert main(_args(ingest_dir, tmp_path, "--k", "2")) == 0
    out = capsys.readouterr().out
    assert out.startswith("dense/emb/toolrank-emb-v0.2/documentation/instruct_query")
    assert "3 tools; index" in out and "(embedded 3, kept 0)" in out and "top 2" in out
    assert len([line for line in out.splitlines() if line.strip().startswith(("1.", "2.", "3."))]) == 2
    assert (ingest_dir / "index" / "index.npz").exists()
    main(_args(ingest_dir, tmp_path, "--json"))
    got = json.loads(capsys.readouterr().out)
    assert got["instruction"].startswith("Given an agent's request for a tool")
    assert {t["server"] for t in got["tools"]} == {"mail"} and 1 <= len(got["tools"]) <= 3
    main(_args(ingest_dir, tmp_path))
    assert "(embedded 0, kept 3)" in capsys.readouterr().out  # index and query both from caches


def test_search_picks_up_heads_from_toolrank_heads(ingest_dir, tmp_path, fake_endpoint, monkeypatch, capsys):
    from test_heads_np import _case_npz

    serving = {"tool_format": "documentation", "query_format": "instruct_query", "instruction": "Find tools."}
    monkeypatch.setenv("TOOLRANK_HEADS", str(_case_npz(tmp_path, "gelu_layernorm_skip", serving=serving)))
    main(_args(ingest_dir, tmp_path, "--json", "--no-cut"))
    got = json.loads(capsys.readouterr().out)
    assert got["scorer"].startswith("clm[gelu_layernorm_skip]/") and got["instruction"] == "Find tools."
    assert len(got["tools"]) == 3


def test_ingest_warms_the_cache_that_search_reads(tmp_path, fake_endpoint, monkeypatch):
    from test_ingest_cli import _write_spec

    monkeypatch.delenv("TOOLRANK_HEADS", raising=False)
    monkeypatch.setenv("TOOLRANK_CACHE", str(tmp_path / "no-heads"))
    out, url = tmp_path / "tools", ["--emb-url", "http://unused/v1"]
    assert main(["ingest", "openapi", _write_spec(tmp_path / "billing.json"), "--out", str(out), *url]) == 0
    assert (out / "cache" / "embeddings.sqlite").exists() and sum(len(b) for b in fake_endpoint) == 2
    fake_endpoint.clear()
    assert main(["search", "list my invoices", "--data", str(out), *url]) == 0
    assert [len(b) for b in fake_endpoint] == [1]  # only the request: both tools came from DATA/cache


def test_a_server_without_heads_says_so(ingest_dir, tmp_path, fake_endpoint, monkeypatch):
    from test_heads_np import _case_npz
    from toolrank.build import build_retriever
    from toolrank.cli import build_parser

    events: list[str] = []
    base_model = _args(ingest_dir, tmp_path, "--emb-model", "qwen3-emb")
    build_retriever(build_parser().parse_args(base_model), notify=events.append)
    assert [e for e in events if e.startswith("no heads found")] != []
    events.clear()  # the default backbone needs no heads: nothing to say
    build_retriever(build_parser().parse_args(_args(ingest_dir, tmp_path)), notify=events.append)
    assert [e for e in events if e.startswith("no heads found")] == []
    monkeypatch.setenv("TOOLRANK_HEADS", str(_case_npz(tmp_path, "gelu_layernorm_skip")))
    build_retriever(build_parser().parse_args(_args(ingest_dir, tmp_path)), notify=events.append)
    assert [e for e in events if e.startswith("no heads found")] == []


def test_search_appends_what_the_usage_log_shows_is_called_together(
    ingest_dir, tmp_path, fake_endpoint, capsys
):
    from toolrank.usage import SCHEMA_VERSION

    events = []
    for n in range(2):  # two requests after which send and delete were both called
        events.append(
            {"v": SCHEMA_VERSION, "event": "search", "id": f"s{n}", "ts": "2026-10-01", "emb_hmac": f"r{n}"}
        )
        events += [
            {"v": SCHEMA_VERSION, "event": "call", "tool": t, "search_id": f"s{n}", "outcome": "ok"}
            for t in ("mail/send", "mail/delete")
        ]
    (ingest_dir / "usage").mkdir()
    (ingest_dir / "usage" / "usage-2026-10-01.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )
    main(_args(ingest_dir, tmp_path, "--k", "1", "--json"))
    (first,) = json.loads(capsys.readouterr().out)["tools"]
    other = "mail/delete" if first["id"] == "mail/send" else "mail/send"
    assert first["id"] != "mail/list"  # the fake vectors put a paired tool first
    main(_args(ingest_dir, tmp_path, "--k", "1", "--co-use", "2", "--json"))
    tools = json.loads(capsys.readouterr().out)["tools"]
    assert [t["id"] for t in tools] == [first["id"], other] and tools[1]["used_with"] == first["id"]
    main(_args(ingest_dir, tmp_path, "--k", "1", "--co-use", "2"))
    assert f"(used with {first['id']})" in capsys.readouterr().out


def test_co_use_needs_the_usage_log(ingest_dir, tmp_path, fake_endpoint):
    pytest.importorskip("mcp")
    with pytest.raises(SystemExit, match="--co-use reads the usage log"):
        main(
            [
                "serve",
                "--data",
                str(ingest_dir),
                "--cache-dir",
                str(tmp_path / "c"),
                "--co-use",
                "2",
                "--no-usage-log",
            ]
        )


def test_serve_refuses_credentials_for_a_source_it_cannot_send_them_to(ingest_dir, tmp_path, fake_endpoint):
    pytest.importorskip("mcp")
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"team": {"key": "k", "headers": {"mail": {"Authorization": "Bearer x"}}}}))
    with pytest.raises(SystemExit, match="neither an OpenAPI source"):
        main(
            ["serve", "--data", str(ingest_dir), "--cache-dir", str(tmp_path / "c"), "--api-keys", str(keys)]
        )


def test_cached_heads_go_only_on_the_backbone_they_were_trained_on(
    ingest_dir, tmp_path, fake_endpoint, monkeypatch, capsys
):
    import shutil

    from test_heads_np import _case_npz
    from toolrank.adapters.heads_np import HEADS_FILE

    cache = tmp_path / "no-heads" / "heads"
    cache.mkdir(parents=True)
    shutil.copy(_case_npz(tmp_path, "gelu_layernorm_skip"), cache / HEADS_FILE)
    main(_args(ingest_dir, tmp_path, "--json"))
    assert json.loads(capsys.readouterr().out)["scorer"].startswith("dense/emb/toolrank-emb-v0.2/")
    main(_args(ingest_dir, tmp_path, "--json", "--emb-model", "qwen3-emb"))
    assert json.loads(capsys.readouterr().out)["scorer"].startswith("clm[")
    main(_args(ingest_dir, tmp_path, "--json", "--emb-model", "my-own-model"))  # unknown: as before
    assert json.loads(capsys.readouterr().out)["scorer"].startswith("clm[")
    monkeypatch.setenv("TOOLRANK_HEADS", str(cache / HEADS_FILE))  # asked for by name: always
    main(_args(ingest_dir, tmp_path, "--json"))
    assert json.loads(capsys.readouterr().out)["scorer"].startswith("clm[")


def test_clm_ckpt_none_drops_only_the_packaged_heads(ingest_dir, tmp_path, fake_endpoint):
    """--clm-ckpt none: the packaged heads stay off, but a learned DATA/heads/current.npz
    (and a candidate) still serve."""
    import shutil
    import time

    from test_heads_np import _case_npz
    from toolrank.build import build_retriever
    from toolrank.cli import build_parser

    home = ingest_dir / "heads"
    home.mkdir()
    shutil.copy(_case_npz(tmp_path, "gelu_layernorm_skip"), home / "current.npz")
    r = build_retriever(build_parser().parse_args(_args(ingest_dir, tmp_path, "--clm-ckpt", "none")))
    res = r.search("send an email")
    deadline = time.time() + 10
    while res.arm != "current" and time.time() < deadline:  # the variant builds in the background
        time.sleep(0.02)
        res = r.search("send an email")
    assert res.arm == "current" and res.scorer.startswith("clm[current]")


def test_heads_replaced_under_the_same_name_reproject_the_persistent_index(
    ingest_dir, tmp_path, fake_endpoint
):
    """learn rewrites candidate.npz and ab renames it to current.npz while a server runs: new bytes
    under an old path must give a new fingerprint, or the index keeps tool rows of the old heads."""
    from test_retriever import _npz_heads
    from toolrank.build import build_retriever, file_sha256
    from toolrank.cli import build_parser

    heads = tmp_path / "h" / "candidate.npz"
    _npz_heads(heads, seed=1)
    args = _args(
        ingest_dir, tmp_path, "--clm-ckpt", str(heads), "--k", "3", "--index-dir", str(tmp_path / "ix")
    )
    first = build_retriever(build_parser().parse_args(args))
    first.search("send an email")
    assert first.status()["sync"]["embedded"] == 3
    old = file_sha256(str(heads))
    _npz_heads(heads, seed=2)  # same name, same size
    assert file_sha256(str(heads)) != old
    second = build_retriever(build_parser().parse_args(args))
    res = second.search("send an email")
    assert second.status()["sync"] == {"embedded": 3, "removed": 0, "kept": 0}
    again = build_retriever(build_parser().parse_args(args))
    assert [h.id for h in again.search("send an email").hits] == [h.id for h in res.hits]
    assert again.status()["sync"]["kept"] == 3  # unchanged heads: nothing to re-project
