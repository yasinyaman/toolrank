import json

import pytest

from toolrank.ingest.mcp import tool_from_mcp
from toolrank.ingest.sync import Listing, drop, load_dir, sync, write_dir


def _mcp(server, *names, desc="d"):
    return [
        tool_from_mcp(server, {"name": n, "description": f"{n} {desc}", "inputSchema": {"type": "object"}})
        for n in names
    ]


def _run(out, listings, **kw):
    tools, manifest = load_dir(out)
    tools, manifest, diffs = sync(tools, manifest, listings, now="2026-09-29T00:00:00+00:00", **kw)
    write_dir(out, tools, manifest)
    return {d.name: d for d in diffs}


def test_a_source_replaces_only_its_tools_and_a_failed_one_keeps_them(tmp_path):
    _run(
        tmp_path,
        [
            Listing("github", "mcp", _mcp("github", "a", "b", "c")),
            Listing("slack", "mcp", _mcp("slack", "x")),
        ],
    )
    new = _mcp("github", "a", "b", "d")
    new[1] = _mcp("github", "b", desc="changed")[0]
    diffs = _run(tmp_path, [Listing("github", "mcp", new), Listing("slack", "mcp", None, error="timeout")])
    g, s = diffs["github"], diffs["slack"]
    assert (g.added, g.changed, g.removed, g.unchanged, g.total) == (1, 1, 1, 1, 3)
    assert s.error == "timeout" and s.line() == "slack  failed: timeout (kept 1 tools)"
    tools, manifest = load_dir(tmp_path)
    assert [t.id for t in tools] == ["github/a", "github/b", "github/d", "slack/x"]
    assert manifest["github"] == {"kind": "mcp", "tools": 3, "synced_at": "2026-09-29T00:00:00+00:00"}


def test_reingesting_the_same_listing_is_byte_identical(tmp_path):
    listing = [Listing("fs", "mcp", _mcp("fs", "read", "write"), info={"transport": "stdio"})]
    _run(tmp_path, listing)
    first = (tmp_path / "tools.jsonl").read_bytes()
    diffs = _run(tmp_path, listing)
    assert (tmp_path / "tools.jsonl").read_bytes() == first
    assert (diffs["fs"].unchanged, diffs["fs"].added, diffs["fs"].changed) == (2, 0, 0)
    assert json.loads((tmp_path / "sources.json").read_text())["fs"]["transport"] == "stdio"


def test_kind_and_empty_guards_and_their_overrides(tmp_path):
    _run(tmp_path, [Listing("github", "mcp", _mcp("github", "a"))])
    spec_tools = _mcp("github", "repos/get")
    diffs = _run(tmp_path, [Listing("github", "openapi", spec_tools)])
    assert "already a mcp source" in diffs["github"].error
    assert _run(tmp_path, [Listing("github", "mcp", [])])["github"].error.startswith("listed 0 tools")
    assert [t.id for t in load_dir(tmp_path)[0]] == ["github/a"]
    assert not _run(tmp_path, [Listing("github", "openapi", spec_tools)], replace=True)["github"].error
    assert _run(tmp_path, [Listing("github", "openapi", [])], allow_empty=True)["github"].removed == 1
    tools, manifest = load_dir(tmp_path)
    assert tools == [] and manifest["github"]["kind"] == "openapi"


def test_drop_removes_whole_sources_and_rejects_unknown_names(tmp_path):
    _run(tmp_path, [Listing("a", "mcp", _mcp("a", "x", "y")), Listing("b", "mcp", _mcp("b", "z"))])
    tools, manifest = load_dir(tmp_path)
    kept, manifest, diffs = drop(tools, manifest, ["a"])
    assert [t.id for t in kept] == ["b/z"] and set(manifest) == {"b"} and diffs[0].removed == 2
    with pytest.raises(ValueError, match="no such source: nope"):
        drop(kept, manifest, ["nope"])


def test_a_listing_cannot_carry_another_sources_tools(tmp_path):
    with pytest.raises(ValueError, match="another source"):
        sync([], {}, [Listing("a", "mcp", _mcp("b", "x"))])
