"""Per-task difference between two results files: where run A gains and loses most against B.

    uv run python scripts/per_task_diff.py results/toolret_clm_name_desc_inst.json \
        results/toolret_qwen3emb_name_desc_inst.json --top 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="per-task metric difference, run A minus run B")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--metric", default="NDCG@10")
    p.add_argument("--top", type=int, default=3, help="tasks to show at each end (0 = all)")
    args = p.parse_args()

    a, b = (json.loads(Path(f).read_text()) for f in (args.a, args.b))
    counts = a.get("config", {}).get("task_counts", {})
    m = args.metric
    rows = sorted(
        (
            (t, 100 * a["per_task"][t][m], 100 * b["per_task"][t][m])
            for t in a["per_task"]
            if t in b["per_task"]
        ),
        key=lambda r: r[1] - r[2],
        reverse=True,
    )
    if args.top and len(rows) > 2 * args.top:
        rows = rows[: args.top] + rows[-args.top :]
    print(f"A = {a['scorer']}\nB = {b['scorer']}\n")
    print(f"| Task | n | A {m} | B {m} | A − B |\n| --- | ---: | ---: | ---: | ---: |")
    for task, x, y in rows:
        print(f"| {task} | {counts.get(task, '')} | {x:.2f} | {y:.2f} | {x - y:+.2f} |")


if __name__ == "__main__":
    main()
