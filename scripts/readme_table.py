"""Write the results table (``docs/results.toml`` over the reports in ``docs/results/``) into the
README and the docs site's benchmarks page, between their markers, or check that both show it.

    uv run python scripts/readme_table.py --write     # after changing the rows or the reports
    uv run python scripts/readme_table.py --check     # what the tests do
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from toolrank.eval.table import TableError, render, splice

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (ROOT / "README.md", ROOT / "docs" / "benchmarks.md")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--spec", default=str(ROOT / "docs" / "results.toml"))
    p.add_argument(
        "--target",
        action="append",
        default=None,
        help="a file with the markers (default: README, benchmarks page)",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    a = p.parse_args()
    try:
        table = render(a.spec)
    except TableError as e:
        sys.exit(f"results table:\n{e}")
    stale = []
    for target in [Path(t) for t in a.target] if a.target else TARGETS:
        text = target.read_text(encoding="utf-8")
        try:
            new = splice(text, table)
        except TableError as e:
            sys.exit(f"{target.name}: {e}")
        if a.check:
            stale += [target.name] if new != text else []
            continue
        target.write_text(new, encoding="utf-8")
        print(f"wrote the results table into {target}")
    if stale:
        sys.exit(f"stale results table in {', '.join(stale)}: uv run python scripts/readme_table.py --write")
    if a.check:
        print("the results tables are current")


if __name__ == "__main__":
    main()
