"""Refuse a release that still carries placeholders: a ``TODO(launch)`` marker in a tracked file, an
empty ``HEADS_URL`` (pip users could not fetch the heads) or a model card that still says so, a
default backbone whose weights are not on the Hub yet, a
development version, no dated CHANGELOG entry for it, or a tag that is not ``v<version>``. The
release workflow runs it before building anything; run it before tagging.

    uv run python scripts/release_check.py [--tag v0.1.0]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
MARKER = "TODO(launch)"
SKIP = ("docs/plan/", "docs/reports/", "tests/", "scripts/release_check.py")  # plans may name the marker
NOT_HOSTED = "The download address is empty until the file is hosted."  # scripts/publish_heads.py replaces it


def problems(
    root: Path,
    files: list[str],
    heads_url: str,
    version: str,
    tag: str | None = None,
    backbone: tuple[str, str, bool] | None = None,
) -> list[str]:
    """``backbone``: the default backbone's (repo, revision, published) from ``toolrank.build``."""
    out = []
    for name in files:
        if name.startswith(SKIP):
            continue
        try:
            text = (root / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or gone
        out += [
            f"{name}:{n}: {line.strip()[:120]}"
            for n, line in enumerate(text.splitlines(), 1)
            if MARKER in line
        ]
    if not heads_url:
        out.append("src/toolrank/adapters/heads_np.py: HEADS_URL is empty; host the heads and fill it in")
    card = root / "docs" / "heads" / "MODEL_CARD.md"
    if card.exists() and NOT_HOSTED in card.read_text(encoding="utf-8"):
        out.append(
            "docs/heads/MODEL_CARD.md: still says the heads are not hosted (publish_heads.py fixes it)"
        )
    if backbone is not None and not backbone[2]:
        out.append(
            f"src/toolrank/build.py: the default backbone {backbone[0]}@{backbone[1]} is not on the Hub yet "
            "(scripts/publish_backbone.py --upload, then BACKBONE_PUBLISHED = True)"
        )
    if Version(version).is_devrelease:
        out.append(f"src/toolrank/__init__.py: {version} is a development version")
    changelog = root / "CHANGELOG.md"
    dated = rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$"
    if not changelog.exists() or not re.search(dated, changelog.read_text(encoding="utf-8"), re.M):
        out.append(f"CHANGELOG.md: no dated entry for {version} (## [{version}] - YYYY-MM-DD)")
    if tag is not None and tag != f"v{version}":
        out.append(f"the tag {tag} does not match the version {version}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", default=None, help="the release tag, e.g. v0.1.0")
    a = p.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    import toolrank
    from toolrank.adapters.heads_np import HEADS_URL
    from toolrank.build import BACKBONE_PUBLISHED, BACKBONE_REPO, BACKBONE_REVISION

    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    backbone = (BACKBONE_REPO, BACKBONE_REVISION, BACKBONE_PUBLISHED)
    found = problems(ROOT, files.splitlines(), HEADS_URL, toolrank.__version__, a.tag, backbone)
    if found:
        sys.exit("not ready to release:\n" + "\n".join(f"  {f}" for f in found))
    print(f"ready to release {toolrank.__version__}")


if __name__ == "__main__":
    main()
