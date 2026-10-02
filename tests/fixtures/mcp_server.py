"""A tiny MCP server for the ingestion tests.

python mcp_server.py [--http PORT] [--mode ok|empty|hang|crash|extra]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import anyio
from mcp.server.mcpserver import MCPServer


def build(mode: str = "ok") -> MCPServer:
    server = MCPServer("fixture", version="0.1.0")
    if mode == "empty":
        return server

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool(title="Search issues")
    def search_issues(query: str, labels: list[str] | None = None, limit: int = 10) -> list[str]:
        """Search the issue tracker by text and labels."""
        return []

    if mode == "extra":  # tools the proxy tests need: slow, crashing, identifiable

        @server.tool()
        async def slow(seconds: float) -> str:
            """Wait, then answer."""
            await anyio.sleep(seconds)
            return "done"

        @server.tool()
        def crash() -> str:
            """Exit the server process in the middle of the call."""
            os._exit(7)

        @server.tool()
        def pid() -> int:
            """The server's process id."""
            return os.getpid()

        @server.tool()
        def env(name: str) -> str:
            """One environment variable of the server process (empty when unset)."""
            return os.environ.get(name, "")

    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, default=0, help="serve streamable HTTP on this port (default: stdio)")
    ap.add_argument("--mode", default="ok", choices=["ok", "empty", "hang", "crash", "extra"])
    a = ap.parse_args()
    if a.mode == "crash":
        print("fixture: missing API key", file=sys.stderr, flush=True)
        sys.exit(3)
    if a.mode == "hang":
        time.sleep(3600)
    server = build(a.mode)
    if a.http:
        server.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=a.http,
            stateless_http=True,
            json_response=True,
        )
    else:
        server.run()
