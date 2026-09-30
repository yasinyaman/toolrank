"""The documentation stays true to the code: the CLI reference is the parser's, and no public page
speaks the project's internal planning language."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "docs" / "reference").exists():  # the sdist ships no docs
    pytest.skip("the docs are not here", allow_module_level=True)


def test_the_cli_reference_is_the_parsers():
    spec = importlib.util.spec_from_file_location("cli_reference", ROOT / "scripts" / "cli_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (ROOT / "docs" / "reference" / "cli.md").read_text() == module.render(), (
        "stale: uv run python scripts/cli_reference.py --write"
    )


def test_public_pages_do_not_name_internal_plan_weeks():
    public = [ROOT / name for name in ("README.md", "CONTRIBUTING.md", "CHANGELOG.md", "SECURITY.md")]
    public += [
        p
        for p in (ROOT / "docs").rglob("*.md")
        if p.relative_to(ROOT / "docs").parts[0] not in ("plan", "reports")
    ]
    hits = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in public
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"\b(Faz|Hafta)\b", line)
    ]
    assert hits == []
