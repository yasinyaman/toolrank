"""The one on-disk format every benchmark and every customer tool set is converted to.

``tools.jsonl``   - one object per line: ``{"id", "doc"?, "documentation"?, "category"?}``
``queries.jsonl`` - one object per line: ``{"id", "text", "qrels": {tool_id: relevance}, "instruction"?, "task"?, "category"?}``
``pairs.jsonl``   - training pairs: ``{"id", "text", "instruction"?, "positives": [doc], "negatives"?: [doc]}``

Keeping evaluation independent of the Hugging Face ``datasets`` library means the Hub is needed
exactly once (``toolrank data pull``), on any machine that can reach it, and never at eval time.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

from toolrank.domain import Query, Tool, TrainPair
from toolrank.formats import parse_doc


def load_tools(path: str | Path) -> list[Tool]:
    with open(path, encoding="utf-8") as f:
        return tools_from_lines(f)


def tools_from_lines(lines: Iterable[str]) -> list[Tool]:
    out: list[Tool] = []
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        doc = r.get("doc")
        documentation = r.get("documentation") or ""
        if not isinstance(doc, dict) or not doc:
            doc = parse_doc(documentation)
        out.append(
            Tool(
                id=str(r["id"]),
                doc=doc,
                documentation=documentation,
                category=str(r.get("category") or ""),
            )
        )
    return out


def load_queries(path: str | Path, tasks: Iterable[str] | None = None) -> list[Query]:
    want = {t.lower() for t in tasks} if tasks else None
    out: list[Query] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            task = str(r.get("task") or "")
            if want and task.lower() not in want:
                continue
            qrels = {str(k): int(v) for k, v in (r.get("qrels") or {}).items()}
            out.append(
                Query(
                    id=str(r["id"]),
                    text=str(r["text"]),
                    qrels=qrels,
                    instruction=str(r.get("instruction") or ""),
                    task=task,
                    category=str(r.get("category") or ""),
                )
            )
    return out


def write_tools(path: str | Path, tools: Iterable[Tool]) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tools:
            f.write(
                json.dumps(
                    {
                        "id": t.id,
                        "doc": t.doc or None,
                        "documentation": t.documentation,
                        "category": t.category,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def with_server_names(src: str | Path, dst: str | Path) -> int:
    """``src``'s tools with ``"server": <category>`` first in each tool's JSON (``doc`` and
    ``documentation``), its queries copied as they are: Phase 0's ``_server`` sets
    (``docs/reports/faz0-week4.md``), byte for byte. -> the number of tools."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    with (
        open(src / "tools.jsonl", encoding="utf-8") as f,
        open(dst / "tools.jsonl", "w", encoding="utf-8") as g,
    ):
        for line in f:
            record = json.loads(line)
            doc = {"server": record["category"], **record["doc"]}
            record["doc"], record["documentation"] = doc, json.dumps(doc, ensure_ascii=False)
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    shutil.copy(src / "queries.jsonl", dst / "queries.jsonl")
    return n


def write_queries(path: str | Path, queries: Iterable[Query]) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(
                json.dumps(
                    {
                        "id": q.id,
                        "text": q.text,
                        "instruction": q.instruction,
                        "qrels": q.qrels,
                        "task": q.task,
                        "category": q.category,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def load_pairs(path: str | Path, limit: int = 0) -> list[TrainPair]:
    out: list[TrainPair] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out.append(
                TrainPair(
                    id=str(r["id"]),
                    text=str(r["text"]),
                    positives=tuple(str(x) for x in r.get("positives") or ()),
                    negatives=tuple(str(x) for x in r.get("negatives") or ()),
                    instruction=str(r.get("instruction") or ""),
                )
            )
            if limit and len(out) >= limit:
                break
    return out


def write_pairs(path: str | Path, pairs: Iterable[TrainPair]) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            row = {
                "id": p.id,
                "text": p.text,
                "instruction": p.instruction,
                "positives": list(p.positives),
                "negatives": list(p.negatives),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
