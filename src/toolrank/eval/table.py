"""The README's results table, rendered from curated ``toolrank eval`` reports.

``docs/results.toml`` lists the rows. A measured row names one report per benchmark (files in
``docs/results/``); a reported row carries published numbers, in the columns whose setting they
share. Before a number is printed its report is checked against the protocol: the benchmark and its
query count, no ``--limit`` / ``--tasks`` / ``--instruction``, top-100, no hybrid, the row's
instruction setting and, when the row names one, its heads file by sha256. Within a row every
report must agree on the tool and query format, the embedding model, the truncation and the heads.
``scripts/readme_table.py`` writes the table into the README between the markers; a test keeps
the two equal.

``leaderboard`` lays the same kind of reports out as the ToolRet leaderboard's sheets
(``scripts/leaderboard_table.py``), for a results submission.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

START, END = "<!-- results:start -->", "<!-- results:end -->"
SAME_IN_ROW = ("tool_format", "query_format", "emb_model", "truncate", "heads_sha256")


class TableError(ValueError):
    """A report that does not belong where the table puts it."""


@dataclass(frozen=True)
class Column:
    id: str  # what a reported row names its number by
    title: str
    report: str  # the row key naming the report (both ToolRet columns read one report)
    dataset: str
    n: int
    metric: str
    aggregate: str = "overall"  # or "category_macro", the ToolRet paper's aggregation


COLUMNS = (
    Column("toolret", "ToolRet NDCG@10", "toolret", "toolret", 7961, "NDCG@10"),
    Column(
        "toolret_cat", "ToolRet NDCG@10 cat-macro", "toolret", "toolret", 7961, "NDCG@10", "category_macro"
    ),
    Column("livemcpbench", "LiveMCPBench Recall@5", "livemcpbench", "livemcpbench_server", 94, "Recall@5"),
    Column("mcp_zero", "MCP-Zero top-1", "mcp_zero", "mcp_zero_server", 2792, "Precision@1"),
)


def _problems(report: dict[str, Any], col: Column, row: dict[str, Any]) -> list[str]:
    cfg = report.get("config") or {}
    out = []
    if report.get("dataset") != col.dataset:
        out.append(f"dataset {report.get('dataset')!r}, the column's is {col.dataset!r}")
    if report.get("n_queries") != col.n:
        out.append(f"{report.get('n_queries')} queries, the benchmark has {col.n}")
    if cfg.get("limit") or cfg.get("tasks"):
        out.append("a subset of the queries (--limit / --tasks)")
    if cfg.get("instruction"):
        out.append("one instruction for every query (--instruction)")
    if cfg.get("k") != 100:
        out.append(f"top-{cfg.get('k')}, the protocol ranks the top 100")
    if cfg.get("hybrid"):
        out.append("a hybrid run")
    if cfg.get("with_inst") != row["with_inst"]:
        out.append(f"with_inst {cfg.get('with_inst')}, the row says {row['with_inst']}")
    if "heads" in row and cfg.get("heads_sha256") != row["heads"]:
        out.append(f"heads {str(cfg.get('heads_sha256'))[:12]}…, the row names {row['heads'][:12]}…")
    if col.metric not in (report.get(col.aggregate) or {}):
        out.append(f"no {col.aggregate} {col.metric}")
    return out


def row_values(row: dict[str, Any], reports: Path) -> list[float | None]:
    """A row's numbers (percent), one per column; ``TableError`` names every problem found."""
    if "reported" in row:
        unknown = set(row["reported"]) - {c.id for c in COLUMNS}
        if unknown:
            raise TableError(f"{row['name']}: no column {sorted(unknown)}")
        return [row["reported"].get(c.id) for c in COLUMNS]
    loaded = {c.report: json.loads((reports / row[c.report]).read_text()) for c in COLUMNS if c.report in row}
    problems = [
        f"{row['name']} / {c.title} ({row[c.report]}): {p}"
        for c in COLUMNS
        if c.report in row
        for p in _problems(loaded[c.report], c, row)
    ]
    for key in SAME_IN_ROW:
        seen = {(r.get("config") or {}).get(key) for r in loaded.values()}
        if len(seen) > 1:
            problems.append(f"{row['name']}: its reports differ in {key}: {sorted(map(str, seen))}")
    if problems:
        raise TableError("\n".join(problems))
    return [
        round(100.0 * loaded[c.report][c.aggregate][c.metric], 2) if c.report in row else None
        for c in COLUMNS
    ]


def render(spec_path: str | Path) -> str:
    """The table and its notes as markdown, from ``docs/results.toml``."""
    spec_path = Path(spec_path)
    spec = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    reports = spec_path.parent / spec.get("reports", "results")
    rows = [(row["name"], row_values(row, reports)) for row in spec["rows"]]
    best = [max((v[i] for _, v in rows if v[i] is not None), default=None) for i in range(len(COLUMNS))]

    def cell(i: int, v: float | None) -> str:
        if v is None:
            return "—"
        return f"**{v:.2f}**" if v == best[i] else f"{v:.2f}"

    lines = [
        "| Retriever | " + " | ".join(c.title for c in COLUMNS) + " |",
        "| --- |" + " ---: |" * len(COLUMNS),
        *(
            "| " + name + " | " + " | ".join(cell(i, v) for i, v in enumerate(values)) + " |"
            for name, values in rows
        ),
    ]
    notes = spec.get("notes") or []
    return "\n".join(lines) + ("\n\n" + "\n".join(f"- {n}" for n in notes) if notes else "") + "\n"


def splice(text: str, block: str) -> str:
    """``text`` with ``block`` between the results markers (the markers stay)."""
    start, end = text.find(START), text.find(END)
    if start < 0 or end < start:
        raise TableError(f"the README needs {START} and {END} around the table")
    return text[: start + len(START)] + "\n" + block + text[end:]


# -- the ToolRet leaderboard: the Hugging Face Space mangopy/ToolRet-leaderboard, one sheet per setting
# and type; results are submitted as issues on mangopy/tool-retrieval-benchmark. "w/ meta" is the full
# tool documentation (toolrank's ``documentation`` text), "Avg" the plain mean of the three categories
# (``category_macro``), "API" ToolRet's web category; scores are on a 0-1 scale.
LEADERBOARD_SETTINGS = (("w/ meta w/ inst", True), ("w/ meta w/o inst", False))
LEADERBOARD_TYPES = (("Avg", None), ("Code", "code"), ("API", "web"), ("Customized", "customized"))
LEADERBOARD_METRICS = (
    ("Comp@10", "Comprehensiveness@10"),
    ("Recall@10", "Recall@10"),
    ("Prec@10", "Precision@10"),
    ("NDCG@10", "NDCG@10"),
)


@dataclass(frozen=True)
class LeaderboardRow:
    name: str
    with_inst: dict[str, Any]  # the report of the w/ inst run
    without_inst: dict[str, Any]
    params: str  # as the sheets write it, e.g. "7.6B"
    model_type: str  # e.g. "embedding model"


def _leaderboard_problems(report: dict[str, Any], with_inst: bool) -> list[str]:
    out = _problems(report, COLUMNS[1], {"with_inst": with_inst})
    tool_format = (report.get("config") or {}).get("tool_format")
    if tool_format != "documentation":
        out.append(f"tool format {tool_format!r}; the leaderboard's w/ meta is the full documentation")
    missing = sorted({c for _, c in LEADERBOARD_TYPES if c} - set(report.get("per_category") or {}))
    if missing:
        out.append(f"no per-category numbers for {missing}")
    return out


def _leaderboard_scores(report: dict[str, Any], category: str | None) -> list[float]:
    scores = report["category_macro"] if category is None else report["per_category"][category]
    return [scores[m] for _, m in LEADERBOARD_METRICS]


def _checked(rows: list[LeaderboardRow]) -> None:
    problems = [
        f"{r.name} ({setting}): {p}"
        for r in rows
        for (setting, w), rep in zip(LEADERBOARD_SETTINGS, (r.with_inst, r.without_inst), strict=True)
        for p in _leaderboard_problems(rep, w)
    ]
    if problems:
        raise TableError("\n".join(problems))


def leaderboard(rows: list[LeaderboardRow]) -> str:
    """One markdown table per leaderboard sheet (setting x type) with the sheet's columns."""
    _checked(rows)
    out = []
    for i, (setting, _) in enumerate(LEADERBOARD_SETTINGS):
        for kind, category in LEADERBOARD_TYPES:
            out += [
                f"**{setting}, {kind}**",
                "",
                "| Model | "
                + " | ".join(m for m, _ in LEADERBOARD_METRICS)
                + " | Number of Parameters | Model Type |",
                "| --- |" + " ---: |" * len(LEADERBOARD_METRICS) + " --- | --- |",
            ]
            for r in rows:
                scores = _leaderboard_scores((r.with_inst, r.without_inst)[i], category)
                out.append(
                    f"| {r.name} | "
                    + " | ".join(f"{v:.4f}" for v in scores)
                    + f" | {r.params} | {r.model_type} |"
                )
            out.append("")
    return "\n".join(out)


def leaderboard_latex(rows: list[LeaderboardRow]) -> str:
    """LaTeX rows in percent, per setting: N@10 P@10 R@10 C@10 for each category, then Avg N@10 C@10."""
    _checked(rows)
    cats = [(k, c) for k, c in LEADERBOARD_TYPES if c]
    head = " & ".join(f"{k} N@10 & P@10 & R@10 & C@10" for k, _ in cats) + " & Avg N@10 & Avg C@10"
    out = []
    for i, (setting, _) in enumerate(LEADERBOARD_SETTINGS):
        out += [f"% {setting}", f"% Model & {head}"]
        for r in rows:
            rep = (r.with_inst, r.without_inst)[i]
            cells = []
            for _, c in cats:
                s = rep["per_category"][c]
                cells += [s["NDCG@10"], s["Precision@10"], s["Recall@10"], s["Comprehensiveness@10"]]
            cells += [rep["category_macro"]["NDCG@10"], rep["category_macro"]["Comprehensiveness@10"]]
            out.append(f"{r.name} & " + " & ".join(f"{100 * v:.2f}" for v in cells) + " \\\\")
    return "\n".join(out) + "\n"
