"""Rows for the ToolRet leaderboard, from ``toolrank eval`` reports on the whole of ToolRet.

The leaderboard (the Hugging Face Space ``mangopy/ToolRet-leaderboard``) keeps one sheet per setting
and type: "w/ meta w/ inst" and "w/ meta w/o inst" (w/ meta = the full tool documentation, the
``documentation`` format), and Avg (the mean of the three categories), Code, API (the web category)
and Customized. Results are submitted as an issue on ``mangopy/tool-retrieval-benchmark``. Each
report is checked against the protocol first (``toolrank.eval.table``).

    uv run python scripts/leaderboard_table.py \\
      --row "toolrank heads v0.1 + Qwen3-Embedding-8B" docs/results/readme_toolret_heads.json \\
            docs/results/readme_toolret_heads_noinst.json "7.6B + 29.9M" "embedding model" \\
      --row "Qwen/Qwen3-Embedding-8B" docs/results/readme_toolret_qwen3emb.json \\
            docs/results/readme_toolret_qwen3emb_noinst.json 7.6B "embedding model" [--latex]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from toolrank.eval.table import LeaderboardRow, TableError, leaderboard, leaderboard_latex


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--row",
        nargs=5,
        action="append",
        required=True,
        metavar=("NAME", "W_INST", "WO_INST", "PARAMS", "TYPE"),
        help="a model: its name, the w/ inst and w/o inst reports, its size and its type",
    )
    p.add_argument("--latex", action="store_true", help="LaTeX rows in percent instead of the sheets")
    a = p.parse_args()
    rows = [
        LeaderboardRow(name, json.loads(Path(w).read_text()), json.loads(Path(wo).read_text()), params, kind)
        for name, w, wo, params, kind in a.row
    ]
    try:
        print((leaderboard_latex if a.latex else leaderboard)(rows), end="")
    except TableError as e:
        sys.exit(f"not the leaderboard's protocol:\n{e}")


if __name__ == "__main__":
    main()
