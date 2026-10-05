"""Write THIRD_PARTY_NOTICES.md: the Python packages the Docker images install
(``toolrank[mcp,openapi,stem]`` as locked in uv.lock, for Linux) with the license each package's
own metadata states, then the models, the other extras and the images' base layers.

    uv run python scripts/third_party.py --write     # after a dependency change
    uv run python scripts/third_party.py --check     # what the tests do

Needs ``uv`` on PATH and the locked packages installed (``uv sync`` has them).
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import subprocess
import sys
from pathlib import Path

from packaging.markers import Marker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "THIRD_PARTY_NOTICES.md"
EXTRAS = ("mcp", "openapi", "stem")
LINUX = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "implementation_name": "cpython",
    "python_version": "3.12",
    "python_full_version": "3.12.0",
}

HEAD = """# Third-party notices

toolrank is licensed under the Apache License 2.0 (`LICENSE`); `NOTICE` credits the projects it
takes code or text from. This file lists what toolrank installs and runs alongside. It is generated
by `scripts/third_party.py`; each license is the one the package's own metadata states, so check a
package's distribution for its full terms.

## Python packages in the Docker images

`toolrank[mcp,openapi,stem]`, the versions locked in `uv.lock` (Linux):

| Package | Version | License |
| --- | --- | --- |
"""

TAIL = """
## Other optional extras

Installed only when asked for (`pip install "toolrank[<extra>]"`), never in the images:

| Extra | Packages | Licenses |
| --- | --- | --- |
| `clm` | torch | BSD-3-Clause and others (see PyTorch's `LICENSE`) |
| `data` | datasets, huggingface_hub | Apache-2.0 |
| `faiss` | faiss-cpu | MIT |
| `pgvector` | psycopg[binary], pgvector | LGPL-3.0-only (psycopg; used as a separate, replaceable library), MIT |
| `anthropic`, `openai` | anthropic, openai | MIT, Apache-2.0 |
| `langgraph`, `llamaindex` | langchain-core, llama-index-core | MIT, MIT |
| `langchain` | langchain | MIT |

## Models

- **yasinyaman/toolrank-emb-8b** v0.2 (Apache-2.0): the default embedding backbone, Qwen3-Embedding-8B
  with a merged LoRA trained on ToolRet-Training-20w, whose dataset card states no license; see
  `docs/backbone/MODEL_CARD.md`. vLLM downloads it at run time; no image or package includes its
  weights.
- **Qwen/Qwen3-Embedding-8B** (Apache-2.0): its base model, and the backbone of the v0.1 heads.
  Downloaded the same way.
- **toolrank heads v0.1** (`toolrank-heads-qwen3-emb-8b-v0.1.npz`, Apache-2.0): trained on
  ToolRet-Training-20w, whose dataset card states no license; see `docs/heads/MODEL_CARD.md`.
  Images built with the heads (`--build-context heads=...`) include the file.

## Container images

- **toolrank** (`deploy/docker/Dockerfile`): based on `python:3.12-slim-bookworm`, whose Debian
  packages carry their own licenses (`/usr/share/doc/*/copyright` in the image). It adds Node.js
  (MIT; npm is Artistic-2.0) from the official `node` image, for MCP servers started with `npx`,
  and uv (Apache-2.0 or MIT), for those started with `uvx`.
- **toolrank-vllm** (`deploy/docker/Dockerfile.vllm`): based on `vllm/vllm-openai`: vLLM
  (Apache-2.0), PyTorch (BSD-3-Clause) and NVIDIA CUDA libraries under NVIDIA's CUDA license
  terms, on Ubuntu. Builds on NVIDIA's NGC vLLM image are for local use only and are not published.
"""


def locked() -> list[tuple[str, str]]:
    """(name, version) of every package the images install, from uv.lock."""
    cmd = ["uv", "export", "--frozen", "--no-emit-project", "--no-hashes", "--no-default-groups"]
    cmd += [f"--extra={e}" for e in EXTRAS]
    text = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    out = []
    for line in text.splitlines():
        if not line or line[0] in "# ":
            continue
        spec, _, marker = line.partition(";")
        name, _, version = spec.strip().partition("==")
        if marker.strip() and not Marker(marker.strip()).evaluate(LINUX):
            continue
        out.append((name, version))
    return sorted(out, key=lambda nv: nv[0].lower())


def license_of(name: str) -> str:
    try:
        meta = md.metadata(name)
    except md.PackageNotFoundError:
        return "(not installed here: see the package)"
    expr = meta.get("License-Expression")
    if expr:
        return expr
    classifiers = [
        c.split(" :: ")[-1] for c in meta.get_all("Classifier") or [] if c.startswith("License ::")
    ]
    text = (meta.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 60:
        return text
    return ", ".join(classifiers) or "(see the package)"


def render() -> tuple[str, int]:
    rows = [f"| {name} | {version} | {license_of(name)} |" for name, version in locked()]
    return HEAD + "\n".join(rows) + "\n" + TAIL, len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    a = p.parse_args()
    text, n = render()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            sys.exit(f"{OUT.name} is stale: uv run python scripts/third_party.py --write")
        print(f"{OUT.name} is current")
        return
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.name}: {n} packages in the images")


if __name__ == "__main__":
    main()
