"""Publish the packaged heads to the Hugging Face Hub (the last step of Faz 1 Hafta 2's heads box).

Stages the model repo — the ``.npz`` and a README that is ``docs/heads/MODEL_CARD.md`` under the
Hub's metadata header — after checking the file against ``HEADS_SHA256``. With ``--upload`` it
creates the repo and uploads both, tags the file's commit with the heads version (``v0.1``, so the
address never serves another file), downloads the file back through that address and checks the
sha256 again; then it writes the address into ``src/toolrank/adapters/heads_np.py`` (``HEADS_URL``)
and into the model card, for you to commit. Without ``--upload`` nothing leaves the machine and
nothing is written outside the stage directory.

The upload needs ``huggingface_hub`` and a login with write access, done by the maintainer
(``uvx --from huggingface_hub hf auth login``); the token never passes through this script.

    uv run python scripts/publish_heads.py --repo USER/toolrank-heads-qwen3-emb-8b            # dry run
    uv run --with huggingface_hub python scripts/publish_heads.py --repo USER/toolrank-heads-qwen3-emb-8b --upload
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from toolrank.adapters.heads_np import HEADS_FILE, HEADS_SHA256, download, sha256_file

FRONT_MATTER = """---
license: apache-2.0
base_model: Qwen/Qwen3-Embedding-8B
base_model_relation: adapter
library_name: toolrank
pipeline_tag: sentence-similarity
datasets:
- mangopy/ToolRet-Training-20w
tags:
- tool-retrieval
- mcp
- agents
- embeddings
---

"""
NOT_HOSTED = (
    "The download address is empty until the file is hosted."  # scripts/release_check.py looks for it
)
HEADS_URL_LINE = re.compile(r'^HEADS_URL = "[^"]*".*$', re.M)
VERSION = re.fullmatch(r"toolrank-heads-.+-(v[\d.]+)\.npz", HEADS_FILE).group(1)  # the Hub tag
HUB_URL = re.compile(r"https://huggingface\.co/[\w.-]+/[\w.-]+/resolve/[\w.-]+/" + re.escape(HEADS_FILE))
ROOT = Path(__file__).resolve().parents[1]


def hosted_card(card: str, url: str) -> str:
    """The model card once the file is served from ``url`` (unchanged if it already says so)."""
    hosted = f"The file is downloaded from {url}."
    if hosted in card:
        return card
    if NOT_HOSTED not in card:
        raise ValueError(f"the model card says neither {NOT_HOSTED!r} nor {hosted!r}")
    return card.replace(NOT_HOSTED, hosted)


def with_heads_url(source: str, url: str) -> str:
    """``heads_np.py``'s source with ``HEADS_URL`` set to ``url``, a file on the Hub."""
    if not HUB_URL.fullmatch(url):
        raise ValueError(f"{url!r} is not {HEADS_FILE} on the Hugging Face Hub")
    line = f'HEADS_URL = "{url}"  # the hosted file (scripts/publish_heads.py)'
    new, n = HEADS_URL_LINE.subn(lambda _: line, source)
    if n != 1:
        raise ValueError(f"expected one HEADS_URL line in heads_np.py, found {n}")
    return new


def readme(card: str, url: str) -> str:
    return FRONT_MATTER + hosted_card(card, url)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", required=True, help="Hub model repo id, e.g. USER/toolrank-heads-qwen3-emb-8b")
    p.add_argument("--file", default=f"dist/heads/{HEADS_FILE}")
    p.add_argument("--card", default="docs/heads/MODEL_CARD.md")
    p.add_argument("--stage", default="dist/heads/hub", help="where the repo's files are written first")
    p.add_argument("--private", action="store_true", help="a private repo (the default download then fails)")
    p.add_argument("--upload", action="store_true", help="without it: stage and check only")
    a = p.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", a.repo):
        sys.exit(f"--repo {a.repo!r}: expected USER/NAME")
    src = Path(a.file)
    got = sha256_file(src)
    if got != HEADS_SHA256:
        sys.exit(f"{src}: sha256 {got}, but heads_np.HEADS_SHA256 is {HEADS_SHA256}")
    url = f"https://huggingface.co/{a.repo}/resolve/{VERSION}/{HEADS_FILE}"
    stage = Path(a.stage)
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "README.md").write_text(readme(Path(a.card).read_text(), url))
    size = src.stat().st_size / 1e6
    print(f"{src} ({size:.1f} MB, sha256 {got[:12]}…) + {stage / 'README.md'} -> {a.repo}")
    if not a.upload:
        print(f"dry run: nothing uploaded. The file would be served from {url}")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"as {api.whoami()['name']}")  # fails here without a login
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    commit = api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=HEADS_FILE,
        repo_id=a.repo,
        commit_message=f"{HEADS_FILE} (sha256 {got})",
    )
    api.create_tag(a.repo, tag=VERSION, revision=commit.oid, exist_ok=True)  # what the address names
    api.upload_file(
        path_or_fileobj=str(stage / "README.md"),
        path_in_repo="README.md",
        repo_id=a.repo,
        commit_message="model card",
    )
    if a.private:
        print("private repo: the public download was not checked, and nothing was written")
        return
    download(url, stage / "check" / HEADS_FILE, HEADS_SHA256)
    print(f"downloaded back and checked: {url}")
    module = ROOT / "src" / "toolrank" / "adapters" / "heads_np.py"
    module.write_text(with_heads_url(module.read_text(), url))
    card = Path(a.card)
    card.write_text(hosted_card(card.read_text(), url))
    print(f"wrote HEADS_URL into {module.relative_to(ROOT)} and the address into {card}: commit both")


if __name__ == "__main__":
    main()
