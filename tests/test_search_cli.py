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


def test_heads_go_only_on_the_backbone_they_name(ingest_dir, tmp_path, fake_endpoint, capsys):
    """Heads trained on Qwen3-Embedding-8B and the v0.2 backbone are both 4096 wide: the width check
    passes, so the heads' cfg decides. A defaulted --emb-model stops; one given by name is a warning."""
    from test_retriever import _npz_heads
    from toolrank.build import build_retriever, heads_mismatch
    from toolrank.cli import build_parser

    assert heads_mismatch("Qwen/Qwen3-Embedding-8B", "toolrank-emb-v0.2")
    assert not heads_mismatch("Qwen/Qwen3-Embedding-8B", "qwen3-emb-fp8")
    assert not heads_mismatch("yasinyaman/toolrank-emb-8b", "toolrank-emb-v0.2-q4_k_m")  # its GGUF build
    assert not heads_mismatch("Qwen/Qwen3-Embedding-8B", "my-own-name") and not heads_mismatch(
        None, "qwen3-emb"
    )
    heads = tmp_path / "h.npz"
    _npz_heads(heads, seed=1, backbone="Qwen/Qwen3-Embedding-8B")
    parse = build_parser().parse_args
    with pytest.raises(ValueError, match="trained on Qwen/Qwen3-Embedding-8B, served on toolrank-emb-v0.2"):
        build_retriever(parse(_args(ingest_dir, tmp_path, "--clm-ckpt", str(heads))))
    build_retriever(parse(_args(ingest_dir, tmp_path, "--clm-ckpt", str(heads), "--emb-model", "qwen3-emb")))
    assert "warning" not in capsys.readouterr().err
    named = _args(ingest_dir, tmp_path, "--clm-ckpt", str(heads), "--emb-model", "toolrank-emb-v0.2")
    build_retriever(parse(named))
    assert "warning: h.npz: heads trained on Qwen/Qwen3-Embedding-8B" in capsys.readouterr().err


def test_pgvector_variants_and_eval_keep_tables_of_their_own(
    ingest_dir, tmp_path, fake_endpoint, monkeypatch
):
    """A candidate's vectors must not overwrite the control arm's rows in one shared table, and an
    eval must not delete a served catalogue's rows: each variant gets a table, eval another default."""
    from test_retriever import _npz_heads
    from toolrank.adapters import index_pgvector
    from toolrank.adapters.index_numpy import NumpyIndex
    from toolrank.build import build_index, build_retriever
    from toolrank.cli import build_parser

    class _InMemory(NumpyIndex):  # the table name is what matters here, not Postgres
        name = "pgvector"

        def __init__(self, dsn, table="toolrank_tools"):
            super().__init__(None)
            self.table = table

    monkeypatch.setattr(index_pgvector, "PgVectorIndex", _InMemory)
    args = _args(ingest_dir, tmp_path, "--index", "pgvector", "--pg-dsn", "postgresql://unused")
    r = build_retriever(build_parser().parse_args(args))
    heads = tmp_path / "h" / "candidate.npz"
    _npz_heads(heads, seed=1)
    tables = {name: r.make_variant(heads, name)().vindex.table for name in ("candidate", "tenant:acme")}
    assert len(set(tables.values())) == 2 and all(t.startswith("toolrank_tools_") for t in tables.values())
    base = build_index(build_parser().parse_args(args))
    assert base.table == "toolrank_tools"
    ev = build_parser().parse_args(["eval", "--data", str(tmp_path), "--index", "pgvector"])
    assert ev.pg_table == "toolrank_eval"


def test_clm_ckpt_none_ranks_without_the_packaged_heads_even_when_they_are_cached(
    ingest_dir, tmp_path, fake_endpoint, monkeypatch
):
    from test_retriever import _npz_heads
    from toolrank.adapters.heads_np import HEADS_FILE
    from toolrank.build import build_retriever
    from toolrank.cli import build_parser

    cache = tmp_path / "cached"
    monkeypatch.setenv("TOOLRANK_CACHE", str(cache))
    _npz_heads(cache / "heads" / HEADS_FILE, seed=1, backbone="Qwen/Qwen3-Embedding-8B")
    base = _args(ingest_dir, tmp_path, "--emb-model", "qwen3-emb")  # the backbone the packaged heads fit
    assert build_retriever(build_parser().parse_args(base)).search("send an email").scorer.startswith("clm[")
    off = build_retriever(build_parser().parse_args([*base, "--clm-ckpt", "none"]))
    assert off.search("send an email").scorer.startswith("dense/")


def test_a_named_keys_requests_never_hit_another_keys_cached_embeddings(ingest_dir, tmp_path, fake_endpoint):
    """A shared query cache answers a request another key made before in milliseconds instead of
    150: that is how bob could confirm alice's exact request. Each named key gets its own scope;
    the catalogue stays shared, and the usage log's emb_key follows the scope (learn finds it)."""
    import sqlite3

    from toolrank.build import build_retriever
    from toolrank.cli import build_parser
    from toolrank.domain import Tool

    r = build_retriever(build_parser().parse_args(_args(ingest_dir, tmp_path)))
    fake_endpoint.clear()

    def sent_for(**kw):
        before = sum(len(b) for b in fake_endpoint)
        res = r.search("refund the last payment of alice@example.com", **kw)
        return sum(len(b) for b in fake_endpoint) - before, res

    (alice_first, a1), (bob, b1), (alice_again, a2) = (sent_for(tenant=t) for t in ("alice", "bob", "alice"))
    assert (alice_first, bob, alice_again) == (1, 1, 0)
    assert [sent_for()[0] for _ in range(2)] == [1, 0]  # requests without a named key share one scope
    assert a1.emb_key == a2.emb_key != b1.emb_key
    keys = {
        k for (k,) in sqlite3.connect(tmp_path / "c" / "embeddings.sqlite").execute("SELECT key FROM emb")
    }
    assert {a1.emb_key, b1.emb_key} <= keys
    # /v1/rank: a supplied tool whose text is a catalogue tool's is embedded again for a named key
    mail = r.state().tools[0]
    before = sum(len(b) for b in fake_endpoint)
    r.rank("x", [Tool(id="mine", doc=mail.doc, documentation=mail.documentation)], tenant="bob")
    assert sum(len(b) for b in fake_endpoint) - before == 2  # the request and the tool
