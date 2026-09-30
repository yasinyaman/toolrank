import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    path = ROOT / "scripts" / f"{name}.py"
    if not path.exists():  # the sdist ships no scripts
        pytest.skip("scripts/ is not here")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _issue(login, day, *, pr=False, comments=0, labels=()):
    issue = {
        "user": {"login": login},
        "created_at": f"2026-10-{day:02d}T20:00:00Z",
        "comments": comments,
        "labels": [{"name": n} for n in labels],
    }
    if pr:
        issue["pull_request"] = {"url": "https://api.github.com/repos/o/r/pulls/1"}
    return issue


def test_the_gate_counts_outside_people_after_the_launch_only():
    m = _script("launch_metrics")
    issues = [
        _issue("yasinyaman", 7, comments=3),  # the maintainer
        _issue("dependabot[bot]", 8, pr=True),  # a bot
        _issue("ada", 5, labels=["feedback"]),  # before the launch
        _issue("ada", 9, labels=["feedback"], comments=1),
        _issue("bob", 10, labels=["feedback"]),
        _issue("carol", 11, pr=True),
        _issue("dan", 12),
    ]
    talks = [
        {"createdAt": "2026-10-13T09:00:00Z", "author": {"login": "erin"}, "comments": {"totalCount": 2}},
        {
            "createdAt": "2026-10-14T09:00:00Z",
            "author": {"login": "yasinyaman"},
            "comments": {"totalCount": 5},
        },
    ]
    got = m.summarize(
        {"stargazers_count": 312, "forks_count": 9, "subscribers_count": 4},
        issues,
        maintainers={"yasinyaman"},
        since="2026-10-06",
        discussions=talks,
        pypi={"data": {"last_week": 40, "last_month": 120}},
        hub={"downloads": 55, "likes": 3},
    )
    assert (got["outside_issues"], got["outside_prs"], got["feedback_issues"]) == (3, 1, 2)
    assert got["outside_threads"] == 3  # ada's answered issue, carol's PR, erin's discussion
    assert got["outside_discussions"] == 1 and got["outside_authors"] == 5  # ada, bob, carol, dan, erin
    assert got["gate"]["stars"] == "312 / 300"
    assert (got["pypi_last_week"], got["heads_downloads_30d"]) == (40, 55)
    assert "| stars | 312 |" in m.table(got)
    quiet = m.summarize(
        {"stargazers_count": 0, "forks_count": 0}, [], maintainers={"yasinyaman"}, since="2026-10-06"
    )
    assert quiet["outside_discussions"] is None and quiet["pypi_last_week"] is None  # not asked, not zero
