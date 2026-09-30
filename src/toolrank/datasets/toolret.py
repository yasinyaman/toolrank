"""ToolRet (Shi et al., ACL 2025 Findings): three tool categories; the Hub has 7,961 queries over
44,453 tools (the paper's text says 7.6k / 43k, but its per-category numbers are on this data).

``pull_toolret`` downloads the Hub datasets once (needs ``pip install toolrank[data]`` and Hub
access) and writes the JSONL pair ``toolrank.datasets.jsonl`` reads. The Hub tool rows carry only
``id`` and ``documentation`` (a JSON string); ``doc`` is parsed from it. The task list and the
task -> category map are copied from the official ``toolret/config.py`` so per-task numbers line
up with the paper's tables. Evaluation protocol (``toolret/eval.py``): retrieve top-100 over the
WHOLE corpus (all categories), trec_eval cut-offs 5/10/20, "Avg" = micro-average over queries.
The paper's tables report a different "Average": the mean of the three category scores, each
the mean over its tasks (``EvalReport.category_macro``; pulled queries carry ``category`` for it).
"""

from __future__ import annotations

import json
from pathlib import Path

from toolrank.datasets.jsonl import write_queries, write_tools
from toolrank.domain import Query, Tool
from toolrank.formats import parse_doc

QUERY_REPO = "mangopy/ToolRet-Queries"
TOOL_REPO = "mangopy/ToolRet-Tools"

TASK_TO_CATEGORY: dict[str, str] = {
    "craft-math-algebra": "code",
    "craft-tabmwp": "code",
    "craft-vqa": "code",
    "gorilla-huggingface": "code",
    "gorilla-pytorch": "code",
    "gorilla-tensor": "code",
    "toolink": "code",
    "apibank": "web",
    "apigen": "web",
    "mnms": "web",
    "reversechain": "web",
    "rotbench": "web",
    "t-eval-dialog": "web",
    "t-eval-step": "web",
    "taskbench-daily": "web",
    "toolace": "web",
    "toolbench": "web",
    "toolemu": "web",
    "tooleyes": "web",
    "toollens": "web",
    "ultratool": "web",
    "autotools-food": "web",
    "autotools-music": "web",
    "autotools-weather": "web",
    "restgpt-spotify": "web",
    "restgpt-tmdb": "web",
    "appbench": "customized",
    "gpt4tools": "customized",
    "gta": "customized",
    "taskbench-huggingface": "customized",
    "taskbench-multimedia": "customized",
    "metatool": "customized",
    "tool-be-honest": "customized",
    "toolalpaca": "customized",
    "toolbench-sam": "customized",
}
TOOLRET_TASKS: tuple[str, ...] = tuple(TASK_TO_CATEGORY)
TOOLRET_CATEGORIES: tuple[str, ...] = ("web", "code", "customized")


def _labels(raw) -> dict[str, int]:
    labels = json.loads(raw) if isinstance(raw, str) else (raw or [])
    return {str(x["id"]): int(x.get("relevance", 1)) for x in labels}


def pull_toolret(
    out_dir: str | Path, tasks: list[str] | None = None, categories: list[str] | None = None
) -> tuple[int, int]:
    """Download ToolRet from the Hub and write ``<out_dir>/tools.jsonl`` + ``queries.jsonl``.

    Returns ``(n_tools, n_queries)``. Requires the ``datasets`` library (``toolrank[data]``).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise SystemExit("ToolRet download needs the 'datasets' library: pip install 'toolrank[data]'") from e

    out = Path(out_dir)
    tools: list[Tool] = []
    for cat in categories or TOOLRET_CATEGORIES:
        ds = load_dataset(TOOL_REPO, cat)["tools"]
        for r in ds:
            documentation = str(r.get("documentation") or "")
            doc = r.get("doc")
            if isinstance(doc, str):
                doc = parse_doc(doc)
            if not isinstance(doc, dict) or not doc:
                doc = parse_doc(documentation)
            tools.append(Tool(id=str(r["id"]), doc=doc, documentation=documentation, category=cat))
    n_tools = write_tools(out / "tools.jsonl", tools)

    queries: list[Query] = []
    for task in tasks or TOOLRET_TASKS:
        ds = load_dataset(QUERY_REPO, task)["queries"]
        for r in ds:
            queries.append(
                Query(
                    id=str(r["id"]),
                    text=str(r["query"]),
                    qrels=_labels(r.get("labels")),
                    instruction=str(r.get("instruction") or ""),
                    task=task,
                    category=str(r.get("category") or TASK_TO_CATEGORY.get(task, "")),
                )
            )
    n_q = write_queries(out / "queries.jsonl", queries)
    (out / "SOURCE.md").write_text(
        "ToolRet - https://huggingface.co/collections/mangopy/tool-retrieval-67becd739af04daa71c88db0\n"
        "Paper: https://arxiv.org/abs/2503.01763 (ACL 2025 Findings). Pulled with `toolrank data pull toolret`.\n"
    )
    return n_tools, n_q
