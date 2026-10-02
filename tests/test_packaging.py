"""What a fresh `pip install toolrank` user meets: one version, install hints instead of tracebacks,
and the packaged heads fetched into the cache."""

import hashlib
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

import toolrank
from toolrank.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_the_version_has_one_source(capsys):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" in project["project"]["dynamic"] and "version" not in project["project"]
    assert project["tool"]["hatch"]["version"]["path"] == "src/toolrank/__init__.py"
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"toolrank {toolrank.__version__}"


@pytest.mark.parametrize(
    "argv", [["serve", "--data", "x"], ["ingest", "mcp", "--server", "t=uvx mcp-server-time", "--out", "x"]]
)
def test_commands_that_need_the_mcp_extra_say_how_to_install_it(monkeypatch, argv):
    monkeypatch.setitem(sys.modules, "mcp", None)  # import mcp -> ImportError
    with pytest.raises(SystemExit, match=r"pip install 'toolrank\[mcp\]'"):
        main(argv)


def test_a_heads_variable_pointing_nowhere_is_an_error_not_a_quiet_fallback(tmp_path, monkeypatch):
    main(["data", "synth", "--out", str(tmp_path / "syn"), "--n-tools", "10", "--n-queries", "2"])
    monkeypatch.setenv("TOOLRANK_HEADS", str(tmp_path / "missing.npz"))
    with pytest.raises(SystemExit, match="TOOLRANK_HEADS=.*missing.npz: no such file"):
        main(["search", "--data", str(tmp_path / "syn"), "--emb-url", "http://unused/v1", "find a tool"])


def test_heads_pull_fetches_checks_and_caches_the_file(tmp_path, monkeypatch, capsys):
    from toolrank.adapters import heads_np

    src = tmp_path / "mirror.npz"
    src.write_bytes(b"heads")
    monkeypatch.setenv("TOOLRANK_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("TOOLRANK_HEADS", raising=False)
    monkeypatch.setattr(heads_np, "HEADS_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="sha256"):  # a mirror must serve the released file
        main(["heads", "pull", "--url", src.as_uri()])
    monkeypatch.setattr(heads_np, "HEADS_SHA256", hashlib.sha256(b"heads").hexdigest())
    assert main(["heads", "pull", "--url", src.as_uri()]) == 0
    cached = tmp_path / "cache" / "heads" / heads_np.HEADS_FILE
    assert cached.read_bytes() == b"heads" and str(cached) in capsys.readouterr().out
    monkeypatch.setattr(heads_np, "HEADS_URL", "")
    assert main(["heads", "pull"]) == 0  # cached: nothing to download


def test_search_and_serve_take_the_endpoint_from_the_environment_unless_flagged(monkeypatch):
    from toolrank.build import DEFAULT_EMB_MODEL, search_defaults
    from toolrank.cli import _env_list, build_parser

    monkeypatch.delenv("TOOLRANK_HEADS", raising=False)
    monkeypatch.setenv("TOOLRANK_CACHE", "/nonexistent")  # no cached heads: plain dense defaults
    monkeypatch.setenv("TOOLRANK_EMB_URL", "http://embedding:8000/v1")
    a = build_parser().parse_args(["search", "--data", "x", "q"])
    search_defaults(a)
    assert (a.emb_url, a.emb_model) == ("http://embedding:8000/v1", DEFAULT_EMB_MODEL)
    a = build_parser().parse_args(["search", "--data", "x", "--emb-url", "http://flag/v1", "q"])
    search_defaults(a)
    assert a.emb_url == "http://flag/v1"
    monkeypatch.setenv("TOOLRANK_ALLOWED_HOSTS", " toolrank, tools.example.com ,,")
    assert _env_list("TOOLRANK_ALLOWED_HOSTS") == ["toolrank", "tools.example.com"]


def _script(name):
    import importlib.util

    if not (ROOT / "scripts" / f"{name}.py").exists():  # the sdist ships no scripts
        pytest.skip("scripts/ is not here")

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to read uv.lock")
def test_third_party_notices_list_what_the_lock_installs():
    text, n = _script("third_party").render()
    assert (ROOT / "THIRD_PARTY_NOTICES.md").read_text() == text, "stale: scripts/third_party.py --write"
    assert n > 20 and "| mcp |" in text and "| numpy |" in text


def test_workflow_actions_are_pinned_to_commits():
    """A tag can be moved to other code; the release job publishes to PyPI, which cannot be undone.
    It also catches a tag that does not exist: setup-uv publishes full versions only, no ``v10``."""
    workflows = ROOT / ".github" / "workflows"
    if not workflows.is_dir():  # an sdist has no workflows
        pytest.skip("no .github/workflows here")
    uses = [
        line.strip()
        for f in sorted(workflows.glob("*.yml"))
        for line in f.read_text().splitlines()
        if line.strip().startswith(("uses:", "- uses:"))
    ]
    pinned = re.compile(r"(- )?uses: [\w.-]+/[\w./-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+")
    assert uses and [u for u in uses if not pinned.fullmatch(u)] == []


def test_the_release_check_finds_every_placeholder(tmp_path):
    check = _script("release_check").problems
    (tmp_path / "SECURITY.md").write_text("write to us at TODO(launch): an address\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("fine\n")
    files = ["SECURITY.md", "docs/guide.md", "docs/reports/w6.md", "missing.md"]
    found = check(tmp_path, files, heads_url="", version="0.1.0.dev0", tag="v0.1.0")
    assert found[0].startswith("SECURITY.md:1: write to us at TODO(launch)") and len(found) == 5
    assert "HEADS_URL is empty" in found[1] and "development version" in found[2]
    assert "no dated entry for 0.1.0.dev0" in found[3] and "does not match" in found[4]
    (tmp_path / "SECURITY.md").write_text("write to us at security@example.org\n")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [0.1.0] - 2026-12-14\n")
    assert check(tmp_path, files, heads_url="https://x/heads.npz", version="0.1.0", tag="v0.1.0") == []
    (tmp_path / "docs" / "heads").mkdir()
    card = tmp_path / "docs" / "heads" / "MODEL_CARD.md"
    card.write_text("The download address is empty until the file is hosted.\n")
    assert "still says the heads are not hosted" in check(tmp_path, files, "https://x", "0.1.0")[0]


def test_publish_heads_writes_the_address_into_the_code_and_the_card(tmp_path):
    ph, check = _script("publish_heads"), _script("release_check")
    assert ph.NOT_HOSTED == check.NOT_HOSTED  # the check looks for the sentence the script replaces
    assert ph.VERSION == "v0.1"  # the Hub tag the address names, from HEADS_FILE
    url = f"https://huggingface.co/someone/heads/resolve/v0.1/{ph.HEADS_FILE}"
    source = (ROOT / "src" / "toolrank" / "adapters" / "heads_np.py").read_text()
    for wrong in ("https://example.org/" + ph.HEADS_FILE, url.replace("v0.1/", "v0.1/other-"), url + '"'):
        with pytest.raises(ValueError, match="not toolrank-heads"):
            ph.with_heads_url(source, wrong)
    with pytest.raises(ValueError, match="found 2"):
        ph.with_heads_url(source + 'HEADS_URL = ""\n', url)
    written = ph.with_heads_url(source, url)
    compile(written, "heads_np.py", "exec")
    assert f'HEADS_URL = "{url}"' in written and ph.with_heads_url(written, url) == written

    def rest(s):
        return [line for line in s.splitlines() if not line.startswith("HEADS_URL = ")]

    assert rest(written) == rest(source)  # only that line changes
    card = f"# heads\n\nDownloads are checked. {ph.NOT_HOSTED} Mirrors work too.\n"
    hosted = ph.hosted_card(card, url)
    assert (
        hosted == f"# heads\n\nDownloads are checked. The file is downloaded from {url}. Mirrors work too.\n"
    )
    assert ph.hosted_card(hosted, url) == hosted  # a re-run changes nothing
    assert ph.readme(card, url).startswith("---\nlicense: apache-2.0\n") and ph.readme(card, url).endswith(
        hosted
    )
    with pytest.raises(ValueError, match="says neither"):
        ph.hosted_card("# a card that moved on\n", url)
    (tmp_path / "docs" / "heads").mkdir(parents=True)
    (tmp_path / "docs" / "heads" / "MODEL_CARD.md").write_text(hosted)
    (tmp_path / "CHANGELOG.md").write_text("## [0.1.0] - 2026-10-06\n")
    assert check.problems(tmp_path, [], heads_url=url, version="0.1.0") == []
    unpublished = check.problems(
        tmp_path, [], heads_url=url, version="0.1.0", backbone=("me/emb", "v0.2", False)
    )
    assert unpublished == [
        "src/toolrank/build.py: the default backbone me/emb@v0.2 is not on the Hub yet "
        "(scripts/publish_backbone.py --upload, then BACKBONE_PUBLISHED = True)"
    ]
    assert (
        check.problems(tmp_path, [], heads_url=url, version="0.1.0", backbone=("me/emb", "v0.2", True)) == []
    )


def test_the_readme_links_work_on_pypi_too():
    import re

    relative = re.findall(r"\]\((?!https?://|#)[^)]+\)", (ROOT / "README.md").read_text())
    assert relative == [], "PyPI shows the README without the repo: use https://github.com/... links"


@pytest.mark.parametrize(
    ("missing", "flags", "extra"),
    [
        ("faiss", ["--index", "faiss"], "faiss"),
        ("psycopg", ["--index", "pgvector", "--pg-dsn", "x"], "pgvector"),
    ],
)
def test_an_index_without_its_extra_says_how_to_install_it(tmp_path, monkeypatch, missing, flags, extra):
    main(["data", "synth", "--out", str(tmp_path / "syn"), "--n-tools", "10", "--n-queries", "2"])
    monkeypatch.setitem(sys.modules, missing, None)
    argv = ["eval", "--data", str(tmp_path / "syn"), "--scorer", "dense", "--emb-url", "http://unused/v1"]
    with pytest.raises(SystemExit, match=rf"pip install 'toolrank\[{extra}\]'"):
        main([*argv, "--emb-model", "m", *flags])


def test_torch_heads_without_torch_point_at_the_npz(tmp_path, monkeypatch):
    from toolrank.adapters.heads_np import load_heads

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=r"pip install 'toolrank\[clm\]' \(the packaged .npz heads do not\)"):
        load_heads(tmp_path / "heads.pt")


def test_publish_backbone_checks_the_merged_directory_and_flips_the_flag(tmp_path):
    pb = _script("publish_backbone")
    model = tmp_path / "merged"
    (model / "1_Pooling").mkdir(parents=True)
    for name in pb.REQUIRED:
        (model / name).write_text("{}")
    (model / "config.json").write_text(
        json.dumps({"hidden_size": 4096, "rope_parameters": {"rope_theta": 1e6}})
    )
    assert pb.check_model(model) == []
    (model / "config.json").write_text(
        json.dumps({"hidden_size": 4096, "rope_parameters": [{"rope_theta": 1}]})
    )
    (model / "tokenizer.json").unlink()
    assert pb.check_model(model) == [
        "tokenizer.json is missing",
        "config.json: rope_parameters is a list (a transformers 5 rewrite; restore the original)",
    ]
    card = "weights at {repo}@{revision}, sha256 {sha256}"
    assert pb.readme(card, "me/emb", "v0.2", "abc").endswith("weights at me/emb@v0.2, sha256 abc")
    source = (ROOT / "src" / "toolrank" / "build.py").read_text()
    assert "BACKBONE_PUBLISHED = True" in pb.with_published(source)
    with pytest.raises(ValueError, match="one BACKBONE_PUBLISHED"):
        pb.with_published("nothing here")
