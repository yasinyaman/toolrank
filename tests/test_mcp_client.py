import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from toolrank.adapters.mcp_client import (  # noqa: E402
    MCPListingError,
    MCPServerSource,
    fetch_many,
    fetch_tools,
    list_all,
)
from toolrank.ingest.mcp import ServerConfig  # noqa: E402

FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_server.py")


def _stdio(name="fixture", mode="ok"):
    return ServerConfig(name, "stdio", command=sys.executable, args=(FIXTURE, "--mode", mode))


def _paged_server(pages):
    """A low-level server whose tools/list follows the given pages; a page is (names, next cursor)."""
    import mcp_types as types
    from mcp.server.lowlevel.server import Server

    async def on_list_tools(ctx, params):
        index = int(params.cursor) if params and params.cursor else 0
        names, cursor = pages[index]
        tools = [types.Tool(name=n, description=f"tool {n}", input_schema={"type": "object"}) for n in names]
        return types.ListToolsResult(tools=tools, next_cursor=cursor)

    return Server("paged", on_list_tools=on_list_tools)


def _list_in_process(server):
    from mcp import Client

    async def main():
        async with Client(server, cache=None) as client:
            return await list_all(client)

    try:
        return asyncio.run(main())
    except BaseExceptionGroup as group:  # the session's task group wraps what the body raised
        leaf = group
        while isinstance(leaf, BaseExceptionGroup):
            leaf = leaf.exceptions[0]
        raise leaf from None


def test_list_all_follows_cursors_and_rejects_a_repeated_one():
    records = _list_in_process(_paged_server([(["a", "b"], "1"), (["c"], "2"), (["d"], None)]))
    assert [r["name"] for r in records] == ["a", "b", "c", "d"]
    assert records[0] == {"name": "a", "description": "tool a", "inputSchema": {"type": "object"}}
    with pytest.raises(MCPListingError, match="repeated"):
        _list_in_process(_paged_server([(["a"], "1"), (["b"], "1")]))


def test_in_process_mcpserver_tools_carry_the_ingest_text():
    sys.path.insert(0, str(Path(FIXTURE).parent))
    try:
        from mcp_server import build
    finally:
        sys.path.pop(0)
    records = _list_in_process(build())
    by_name = {r["name"]: r for r in records}
    assert by_name["search_issues"]["title"] == "Search issues"
    assert set(by_name["search_issues"]["inputSchema"]["properties"]) == {"query", "labels", "limit"}


def test_stdio_server_end_to_end():
    tools = MCPServerSource(_stdio(), timeout=30).list_tools()
    assert [t.id for t in tools] == ["fixture/add", "fixture/search_issues"]
    text = json.loads(tools[0].documentation)
    assert text["server"] == "fixture" and text["description"] == "Add two integers."
    assert set(text["inputSchema"]["properties"]) == {"a", "b"}


def test_failures_name_the_server_and_show_its_stderr():
    results = asyncio.run(
        fetch_many([_stdio("crash", "crash"), _stdio("hang", "hang"), _stdio("empty", "empty")], timeout=3)
    )
    assert isinstance(results["crash"], MCPListingError) and "missing API key" in str(results["crash"])
    assert isinstance(results["hang"], MCPListingError) and "within 3 s" in str(results["hang"])
    assert results["empty"] == []
    with pytest.raises(MCPListingError, match="command not found"):
        asyncio.run(fetch_tools(ServerConfig("x", "stdio", command="no-such-command-xyz")))


def test_streamable_http_server_end_to_end():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, FIXTURE, "--http", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                if time.monotonic() > deadline or proc.poll() is not None:
                    pytest.fail("fixture HTTP server did not start")
                time.sleep(0.2)
        cfg = ServerConfig("web", "http", url=f"http://127.0.0.1:{port}/mcp", headers={"X-Test": "1"})
        tools = asyncio.run(fetch_tools(cfg, timeout=30))
        assert [t.id for t in tools] == ["web/add", "web/search_issues"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)
