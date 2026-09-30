"""Launch metrics for the Faz 1 gate: stars, forks, what outside people opened, package and heads
downloads. Public endpoints only, no token: the GitHub REST API (60 requests an hour without one),
pypistats.org and the Hugging Face Hub. Discussions need GraphQL, so ``--discussions`` asks the ``gh``
CLI, logged in as you; the token stays with gh.

The gate (measured six weeks after the launch): 300 stars, written feedback from 3 outside users, and
at least one outside pull request or issue discussion. "Outside" means neither the maintainer nor a
bot; written feedback is an outside issue labelled ``feedback`` here, plus interview summaries, which
the gate report counts by hand.

    uv run python scripts/launch_metrics.py --since 2026-10-06 [--discussions] [--out results/launch_metrics.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

GATE = {"stars": 300, "feedback": 3, "outside_threads": 1}
DISCUSSIONS = """query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { createdAt author { login } category { name } comments { totalCount } }
    }
  }
}"""


def _get(url: str) -> Any:
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "toolrank-launch-metrics"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def outside(login: str | None, maintainers: set[str]) -> bool:
    return bool(login) and login not in maintainers and not login.endswith("[bot]")  # type: ignore[union-attr]


def summarize(
    repo: dict[str, Any],
    issues: list[dict[str, Any]],
    *,
    maintainers: set[str],
    since: str,
    discussions: list[dict[str, Any]] | None = None,
    pypi: dict[str, Any] | None = None,
    hub: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The gate's numbers from the raw API answers (issues as GitHub lists them, PRs included)."""
    mine = [i for i in issues if i["created_at"][:10] >= since and outside(i["user"]["login"], maintainers)]
    prs = [i for i in mine if "pull_request" in i]
    plain = [i for i in mine if "pull_request" not in i]
    feedback = [i for i in plain if "feedback" in {lb["name"] for lb in i.get("labels", [])}]
    threads = [i for i in mine if i.get("comments", 0) > 0 or "pull_request" in i]
    talks = [
        d
        for d in discussions or []
        if d["createdAt"][:10] >= since and outside((d.get("author") or {}).get("login"), maintainers)
    ]
    out = {
        "stars": repo["stargazers_count"],
        "forks": repo["forks_count"],
        "watchers": repo.get("subscribers_count"),
        "outside_issues": len(plain),
        "outside_prs": len(prs),
        "outside_threads": len(threads) + sum(1 for d in talks if d["comments"]["totalCount"] > 0),
        "feedback_issues": len(feedback),
        "outside_discussions": None if discussions is None else len(talks),
        "outside_authors": len({i["user"]["login"] for i in mine} | {d["author"]["login"] for d in talks}),
        "pypi_last_week": (pypi or {}).get("data", {}).get("last_week"),
        "pypi_last_month": (pypi or {}).get("data", {}).get("last_month"),
        "heads_downloads_30d": (hub or {}).get("downloads"),
        "heads_likes": (hub or {}).get("likes"),
    }
    out["gate"] = {
        "stars": f"{out['stars']} / {GATE['stars']}",
        "feedback": f"{out['feedback_issues']} issues (+ interviews, counted by hand) / {GATE['feedback']}",
        "outside_threads": f"{out['outside_threads']} / {GATE['outside_threads']}",
    }
    return out


def table(m: dict[str, Any]) -> str:
    rows = [(k, v) for k, v in m.items() if k not in ("gate", "measured_at", "since", "repo")]
    lines = ["| | |", "| --- | ---: |", *(f"| {k} | {'—' if v is None else v} |" for k, v in rows)]
    lines += ["", "| Gate | |", "| --- | --- |", *(f"| {k} | {v} |" for k, v in m["gate"].items())]
    return "\n".join(lines) + "\n"


def fetch_issues(repo: str, since: str) -> list[dict[str, Any]]:
    out, page = [], 1
    while True:
        q = urllib.parse.urlencode(
            {"state": "all", "since": f"{since}T00:00:00Z", "per_page": 100, "page": page}
        )
        batch = _get(f"https://api.github.com/repos/{repo}/issues?{q}")
        out += batch
        if len(batch) < 100:
            return out
        page += 1


def fetch_discussions(repo: str) -> list[dict[str, Any]]:
    owner, name = repo.split("/")
    out: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        cmd = [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={DISCUSSIONS}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
        ]
        if after:
            cmd += ["-F", f"after={after}"]
        j = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        page = j["data"]["repository"]["discussions"]
        out += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return out
        after = page["pageInfo"]["endCursor"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", default="yasinyaman/toolrank")
    p.add_argument("--maintainer", action="append", default=None, help="GitHub logins that are not 'outside'")
    p.add_argument("--pypi", default="toolrank")
    p.add_argument("--hub", default="yasinyaman/toolrank-heads-qwen3-emb-8b", help="the heads' model repo")
    p.add_argument("--since", required=True, help="the launch date, YYYY-MM-DD")
    p.add_argument("--discussions", action="store_true", help="count Discussions too (through the gh CLI)")
    p.add_argument("--out", default=None, help="also write the numbers as JSON")
    a = p.parse_args()
    dt.date.fromisoformat(a.since)
    maintainers = set(a.maintainer or [a.repo.split("/")[0]])

    def optional(url: str) -> Any:
        try:
            return _get(url)
        except (OSError, ValueError) as e:  # a package or model not published yet is not an error here
            print(f"skipped {url}: {e}", file=sys.stderr)
            return None

    try:
        repo = _get(f"https://api.github.com/repos/{a.repo}")
    except urllib.error.HTTPError as e:
        sys.exit(f"{a.repo}: {e} (not public yet?)")
    m = summarize(
        repo,
        fetch_issues(a.repo, a.since),
        maintainers=maintainers,
        since=a.since,
        discussions=fetch_discussions(a.repo) if a.discussions else None,
        pypi=optional(f"https://pypistats.org/api/packages/{a.pypi}/recent"),
        hub=optional(f"https://huggingface.co/api/models/{a.hub}"),
    )
    m = {
        "repo": a.repo,
        "since": a.since,
        "measured_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        **m,
    }
    print(table(m), end="")
    if a.out:
        Path(a.out).write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    main()
