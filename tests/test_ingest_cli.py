import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from toolrank.adapters.embeddings_api import OpenAIEmbeddings
from toolrank.cli import main

SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Billing API", "version": "1"},
    "servers": [{"url": "https://billing.example.com"}],
    "paths": {
        "/invoices": {
            "get": {"operationId": "listInvoices", "summary": "List invoices"},
            "post": {
                "operationId": "createInvoice",
                "summary": "Create an invoice",
                "requestBody": {
                    "content": {
                        "application/json": {"schema": {"properties": {"amount": {"type": "integer"}}}}
                    }
                },
            },
        }
    },
}


def _write_spec(path: Path, spec=SPEC) -> str:
    path.write_text(json.dumps(spec))
    return str(path)


def test_ingest_openapi_writes_tools_and_manifest_then_syncs(tmp_path, capsys):
    spec, out = _write_spec(tmp_path / "billing.json"), tmp_path / "tools"
    assert main(["ingest", "openapi", spec, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "billing-api: 2 operations, 0 texts shrunk" in printed and "billing-api  +2 ~0 -0 =0" in printed
    rows = [json.loads(line) for line in (out / "tools.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["billing-api/listInvoices", "billing-api/createInvoice"]
    manifest = json.loads((out / "sources.json").read_text())["billing-api"]
    assert manifest["kind"] == "openapi" and manifest["origin"] == spec
    assert manifest["base_url"] == "https://billing.example.com"

    changed = json.loads(json.dumps(SPEC))
    changed["paths"]["/invoices"]["get"]["summary"] = "List all invoices"
    del changed["paths"]["/invoices"]["post"]
    _write_spec(tmp_path / "billing.json", changed)
    assert main(["ingest", "openapi", spec, "--out", str(out), "--dry-run"]) == 0
    assert "billing-api  +0 ~1 -1 =0" in capsys.readouterr().out
    assert len((out / "tools.jsonl").read_text().splitlines()) == 2  # dry run wrote nothing
    assert main(["ingest", "drop", "billing-api", "--out", str(out)]) == 0
    assert (out / "tools.jsonl").read_text() == "" and json.loads((out / "sources.json").read_text()) == {}
    with pytest.raises(SystemExit, match="no such source"):
        main(["ingest", "drop", "billing-api", "--out", str(out)])


def test_warm_up_embeds_only_new_and_changed_tools(tmp_path, monkeypatch, capsys):
    sent: list[list[str]] = []

    def fake_post(self, texts):
        sent.append(list(texts))
        return [np.ones(4, dtype=np.float32) for _ in texts], 10 * len(texts)

    monkeypatch.setattr(OpenAIEmbeddings, "_post", fake_post)
    spec, out = _write_spec(tmp_path / "billing.json"), str(tmp_path / "tools")
    emb = ["--emb-url", "http://unused/v1", "--emb-model", "m", "--cache-dir", str(tmp_path / "cache")]
    main(["ingest", "openapi", spec, "--out", out, *emb])
    assert (
        sum(len(b) for b in sent) == 2
        and "embedded 2 new or changed texts (20 tokens" in capsys.readouterr().out
    )
    main(["ingest", "openapi", spec, "--out", out, *emb])
    assert sum(len(b) for b in sent) == 2 and "embedded 0 new or changed texts" in capsys.readouterr().out
    changed = json.loads(json.dumps(SPEC))
    changed["paths"]["/invoices"]["get"]["summary"] = "List all invoices"
    _write_spec(tmp_path / "billing.json", changed)
    main(["ingest", "openapi", spec, "--out", out, *emb])
    assert len(sent[-1]) == 1 and '"List all invoices"' in sent[-1][0]
    # in a container the endpoint comes from the environment, as it does for serve
    monkeypatch.setenv("TOOLRANK_EMB_URL", "http://unused/v1")
    monkeypatch.setenv("TOOLRANK_EMB_MODEL", "m")
    changed["paths"]["/invoices"]["get"]["summary"] = "List every invoice"
    _write_spec(tmp_path / "billing.json", changed)
    main(["ingest", "openapi", spec, "--out", out, "--cache-dir", str(tmp_path / "cache")])
    assert len(sent[-1]) == 1 and '"List every invoice"' in sent[-1][0]


def test_ingest_mcp_stdio_server_end_to_end(tmp_path, capsys):
    pytest.importorskip("mcp")
    fixture = Path(__file__).parent / "fixtures" / "mcp_server.py"
    server = f"fixture={shlex.quote(sys.executable)} {shlex.quote(str(fixture))}"
    broken = "gone=no-such-command-xyz"
    out = str(tmp_path / "tools")
    assert main(["ingest", "mcp", "--server", server, "--server", broken, "--out", out]) == 1
    printed = capsys.readouterr().out
    assert "fixture  +2 ~0 -0 =0" in printed and "gone  failed: gone: command not found" in printed
    rows = [json.loads(line) for line in Path(out, "tools.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["fixture/add", "fixture/search_issues"]
    manifest = json.loads(Path(out, "sources.json").read_text())
    assert set(manifest) == {"fixture"}  # the failed server left nothing behind
    assert (manifest["fixture"]["kind"], manifest["fixture"]["transport"]) == ("mcp", "stdio")
    with pytest.raises(SystemExit, match="duplicate server names"):
        main(["ingest", "mcp", "--server", server, "--server", server, "--out", out])
