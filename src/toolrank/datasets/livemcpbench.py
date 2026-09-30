"""LiveMCPBench (icip-cas/LiveMCPBench): 95 agent tasks over LiveMCPTool (69 MCP servers, 525 tools).

Each task's annotation lists, by name, the tools a human used to solve it; every corpus tool with
such a name is relevant (13 names exist on more than one server). Tasks with no annotated tool in
the corpus are dropped. ``Tool.category`` is the MCP server, ``Query.task`` / ``category`` the task
category (Office, Finance, ...). The benchmark ships no instruction, so queries carry one generic
retrieval instruction for the w/ inst setting. Files come from the repository at a pinned commit;
no Hub access needed.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from toolrank.datasets.jsonl import write_queries, write_tools
from toolrank.domain import Query, Tool

COMMIT = "36a51a1065dda49c1e503b87edf33a1e0b331a74"  # main, 2025-12-18
RAW = f"https://raw.githubusercontent.com/icip-cas/LiveMCPBench/{COMMIT}"
TOOLS_FILE = "tools/LiveMCPTool/tools.json"
TASKS_FILE = "annotated_data/all_annotations.json"
INSTRUCTION = "Given an agent task, retrieve the MCP tools needed to complete it."


def tool_names(annotation: str) -> list[str]:
    """``"1. get-weread-rank\\n2. create_document"`` -> ``["get-weread-rank", "create_document"]``."""
    out = []
    for line in re.split(r"[\n;]+", annotation or ""):
        name = re.sub(r"^\s*\d+[.)]\s*", "", line).strip().strip("`'\"")
        if name:
            out.append(name)
    return out


def convert(
    servers: list[dict[str, Any]], tasks: list[dict[str, Any]]
) -> tuple[list[Tool], list[Query], int]:
    """-> (tools, queries, tasks dropped because none of their tools is in the corpus)."""
    tools: list[Tool] = []
    by_name: dict[str, list[str]] = defaultdict(list)
    for srv in servers:
        for key, block in (srv.get("tools") or {}).items():
            for t in block.get("tools") or []:
                doc = {k: t[k] for k in ("name", "description", "inputSchema") if t.get(k) is not None}
                tool = Tool(
                    id=f"{key}/{t['name']}",
                    doc=doc,
                    documentation=json.dumps(doc, ensure_ascii=False),
                    category=str(srv.get("name") or key),
                )
                tools.append(tool)
                by_name[t["name"]].append(tool.id)
    queries: list[Query] = []
    for task in tasks:
        names = tool_names((task.get("Annotator Metadata") or {}).get("Tools", ""))
        qrels = {tid: 1 for n in names for tid in by_name.get(n, ())}
        if not qrels:
            continue
        cat = str(task.get("category") or "")
        queries.append(
            Query(
                id=str(task["task_id"]),
                text=str(task["Question"]),
                qrels=qrels,
                instruction=INSTRUCTION,
                task=cat,
                category=cat,
            )
        )
    return tools, queries, len(tasks) - len(queries)


def _fetch(path: str) -> Any:
    with urllib.request.urlopen(f"{RAW}/{path}", timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def pull_livemcpbench(out_dir: str | Path) -> tuple[int, int, int]:
    """Download LiveMCPBench and write ``tools.jsonl`` + ``queries.jsonl``; -> (tools, queries, dropped)."""
    tools, queries, dropped = convert(_fetch(TOOLS_FILE), _fetch(TASKS_FILE))
    out = Path(out_dir)
    n_tools, n_queries = (
        write_tools(out / "tools.jsonl", tools),
        write_queries(out / "queries.jsonl", queries),
    )
    (out / "SOURCE.md").write_text(f"LiveMCPBench - https://github.com/icip-cas/LiveMCPBench at {COMMIT}\n")
    return n_tools, n_queries, dropped
