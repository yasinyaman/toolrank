import json
import subprocess
import sys

import pytest

from toolrank.ingest.mcp import ServerConfig, load_config, parse_server, tool_from_mcp, tools_from_listing


def test_parse_server_url_is_http_anything_else_a_stdio_command():
    http = parse_server("github=https://api.example.com/mcp")
    assert (http.name, http.transport, http.url) == ("github", "http", "https://api.example.com/mcp")
    stdio = parse_server('time = uvx mcp-server-time --local-timezone "Europe/Istanbul"')
    assert (stdio.name, stdio.transport, stdio.command) == ("time", "stdio", "uvx")
    assert stdio.args == ("mcp-server-time", "--local-timezone", "Europe/Istanbul")
    for bad in ("no-equals", "=uvx x", "name="):
        with pytest.raises(ValueError):
            parse_server(bad)


def test_load_config_reads_claude_desktop_and_vscode_files_and_hides_secrets(tmp_path):
    claude = tmp_path / "claude.json"
    claude.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fs": {"command": "npx", "args": ["-y", "server-fs", "/tmp"], "env": {"TOKEN": "s3cret"}},
                    "remote": {
                        "url": "https://mcp.example.com/mcp",
                        "headers": {"Authorization": "Bearer s3cret"},
                    },
                    "off": {"command": "x", "disabled": True},
                }
            }
        )
    )
    fs, remote = load_config(claude)
    assert (fs.transport, fs.command, fs.args, fs.env) == (
        "stdio",
        "npx",
        ("-y", "server-fs", "/tmp"),
        {"TOKEN": "s3cret"},
    )
    assert (remote.transport, remote.headers["Authorization"]) == ("http", "Bearer s3cret")
    assert "s3cret" not in repr(fs) + repr(remote)
    vscode = tmp_path / "mcp.json"
    vscode.write_text(
        json.dumps({"servers": {"io.github.x/y": {"type": "http", "url": "http://127.0.0.1:9/mcp"}}})
    )
    assert [c.name for c in load_config(vscode)] == ["io.github.x/y"]
    vscode.write_text(json.dumps({"servers": {"legacy": {"type": "sse", "url": "http://h/sse"}}}))
    with pytest.raises(ValueError, match="SSE"):
        load_config(vscode)
    with pytest.raises(ValueError):
        ServerConfig("x", "stdio")


def test_tool_from_mcp_text_doc_and_ids():
    record = {
        "name": "repos/get",
        "title": "Get a repository",
        "inputSchema": {"type": "object", "properties": {"repo": {"type": "string"}}},
        "annotations": {"readOnlyHint": True},
        "icons": [{"src": "https://x/icon.png"}],
        "_meta": {"x": 1},
    }
    tool = tool_from_mcp("github", record)
    assert (tool.id, tool.category, tool.name) == ("github/repos/get", "github", "repos/get")
    text = json.loads(tool.documentation)
    assert list(text) == ["server", "name", "description", "inputSchema"]
    assert text["description"] == "Get a repository"  # no description: the title stands in
    assert set(tool.doc) == {"server", "name", "title", "inputSchema", "annotations"}
    with pytest.raises(ValueError):
        tool_from_mcp("github", {"description": "no name"})
    twice = tools_from_listing("s", [{"name": "a", "description": "1"}, {"name": "a", "description": "2"}])
    assert [t.doc["description"] for t in twice] == ["1"]


def test_importing_the_cli_loads_no_optional_dependency():
    heavy = (
        "{'mcp', 'mcp_types', 'yaml', 'torch', 'httpx2', 'faiss', 'psycopg', 'pgvector', 'starlette', "
        "'uvicorn', 'anyio', 'datasets', 'anthropic', 'openai', 'langchain_core', 'langgraph', "
        "'llama_index', 'litellm', 'langchain'}"
    )
    modules = (
        "toolrank.cli, toolrank.adapters.mcp_proxy, toolrank.adapters.rest, toolrank.adapters.backends, "
        "toolrank.client, toolrank.names, toolrank.integrations.anthropic, toolrank.integrations.openai, "
        "toolrank.integrations.langgraph, toolrank.integrations.litellm, toolrank.integrations.langchain"
    )
    code = f"import sys, {modules}; print(sorted({heavy} & set(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
