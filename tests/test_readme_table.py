import json
import shutil
from pathlib import Path

import pytest

from toolrank.eval.table import END, START, TableError, render, splice

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "docs" / "results.toml").exists():  # the sdist ships no docs
    pytest.skip("the results table's reports are not here", allow_module_level=True)


def test_the_readme_and_the_docs_show_the_table_the_curated_reports_make():
    table = render(ROOT / "docs" / "results.toml")  # every cell passes the protocol checks
    for page in (ROOT / "README.md", ROOT / "docs" / "benchmarks.md"):
        text = page.read_text(encoding="utf-8")
        assert splice(text, table) == text, (
            f"stale {page.name}: uv run python scripts/readme_table.py --write"
        )
    assert "| Qwen3-Embedding-8B + toolrank heads v0.1 | **54.03** | 47.13 |" in table


def _spec(tmp_path, **row_extra):
    """docs/results.toml's heads row alone, over copies of its reports."""
    shutil.copytree(ROOT / "docs" / "results", tmp_path / "results")
    row = {
        "name": "heads",
        "with_inst": True,
        "heads": "f3c101251b9c23925e2925bc02c4492e6c3dea79bcfa1b56715affb20f9f72f0",
        "toolret": "readme_toolret_heads.json",
        "livemcpbench": "readme_livemcpbench_server_heads.json",
        "mcp_zero": "readme_mcp_zero_server_heads.json",
        **row_extra,
    }
    lines = ['reports = "results"', "[[rows]]"]
    lines += [f"{k} = {json.dumps(v)}" for k, v in row.items()]
    (tmp_path / "t.toml").write_text("\n".join(lines) + "\n")
    return tmp_path / "t.toml"


def _edit(tmp_path, name, **config):
    path = tmp_path / "results" / name
    report = json.loads(path.read_text())
    report["config"].update(config)
    path.write_text(json.dumps(report))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: _edit(p, "readme_toolret_heads.json", limit=50), "a subset of the queries"),
        (lambda p: _edit(p, "readme_toolret_heads.json", k=10), "top-10"),
        (lambda p: _edit(p, "readme_mcp_zero_server_heads.json", hybrid={"k_rrf": 60}), "a hybrid run"),
        (lambda p: _edit(p, "readme_mcp_zero_server_heads.json", instruction="x"), "one instruction"),
        (
            lambda p: _edit(p, "readme_livemcpbench_server_heads.json", heads_sha256="0" * 64),
            "heads 000000000000",
        ),
        (
            lambda p: _edit(p, "readme_livemcpbench_server_heads.json", tool_format="name_desc"),
            "differ in tool_format",
        ),
    ],
)
def test_a_cell_off_the_protocol_is_refused(tmp_path, change, message):
    spec = _spec(tmp_path)
    render(spec)
    change(tmp_path)
    with pytest.raises(TableError, match=message):
        render(spec)


def test_rows_and_columns_must_match_their_reports(tmp_path):
    with pytest.raises(TableError, match="with_inst True, the row says False"):
        render(_spec(tmp_path, with_inst=False))
    shutil.rmtree(tmp_path / "results")
    with pytest.raises(TableError, match="dataset 'livemcpbench_server', the column's is 'mcp_zero_server'"):
        render(_spec(tmp_path, mcp_zero="readme_livemcpbench_server_heads.json"))
    with pytest.raises(TableError, match="needs"):
        splice("no markers here", "table")
    assert splice(f"a\n{START}\nold\n{END}\nb", "new\n") == f"a\n{START}\nnew\n{END}\nb"


def test_the_leaderboard_sheets_come_from_checked_reports():
    from toolrank.eval.table import LeaderboardRow, leaderboard, leaderboard_latex

    def load(name):
        return json.loads((ROOT / "docs" / "results" / name).read_text())

    heads = LeaderboardRow(
        "heads",
        load("readme_toolret_heads.json"),
        load("readme_toolret_heads_noinst.json"),
        "7.6B",
        "embedding model",
    )
    sheets = leaderboard([heads])
    assert sheets.count("| Model | Comp@10 | Recall@10 | Prec@10 | NDCG@10 |") == 8  # 2 settings x 4 types
    avg = sheets.split("**w/ meta w/ inst, Avg**")[1].split("**")[0]
    assert "| heads | 0.4750 | 0.5763 | 0.0869 | 0.4713 | 7.6B | embedding model |" in avg  # cat-macro 47.13
    assert "heads & 54.40 & 7.97 & 70.03 & 67.64 & " in leaderboard_latex([heads])  # code, w/ inst
    swapped = LeaderboardRow("swapped", heads.without_inst, heads.with_inst, "7.6B", "embedding model")
    with pytest.raises(TableError, match="with_inst False, the row says True"):
        leaderboard([swapped])
