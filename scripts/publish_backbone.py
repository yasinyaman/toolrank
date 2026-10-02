"""Publish the default embedding backbone to the Hugging Face Hub: Qwen3-Embedding-8B with the LoRA
of ``scripts/lora_train.py`` merged in (``build.BACKBONE_REPO`` at ``build.BACKBONE_REVISION``).

Checks the merged directory (the files vLLM and sentence-transformers need, the original config and
tokenizer, not transformers 5's rewrites), hashes the weights and stages a README that is
``docs/backbone/MODEL_CARD.md`` under the Hub's metadata header. With ``--upload`` it creates the
repo, uploads the directory and the card, tags the commit with the revision (so the served weights
never change under a name), lists the tagged files back, and sets ``BACKBONE_PUBLISHED = True`` in
``src/toolrank/build.py`` for you to commit. Without ``--upload`` nothing leaves the machine.

Run it where the weights are (the GB10; 16 GB); the upload needs ``huggingface_hub`` and a login
with write access done by the maintainer (``uvx --from huggingface_hub hf auth login``).

    uv run python scripts/publish_backbone.py --model data/lora/qwen3-emb-lora-20k/merged          # dry run
    uv run --with huggingface_hub python scripts/publish_backbone.py --model data/lora/qwen3-emb-lora-20k/merged --upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "modules.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
)
SKIP = (".saved_by_transformers5/*", "train.json", "train.log", "*.pt")  # never uploaded
FRONT_MATTER = """---
license: apache-2.0
base_model: Qwen/Qwen3-Embedding-8B
base_model_relation: finetune
library_name: sentence-transformers
pipeline_tag: sentence-similarity
datasets:
- mangopy/ToolRet-Training-20w
tags:
- tool-retrieval
- mcp
- agents
- embeddings
- lora
---

"""
PUBLISHED_LINE = re.compile(r"^BACKBONE_PUBLISHED = (True|False)(.*)$", re.M)


def check_model(model: Path) -> list[str]:
    """What is wrong with the merged directory (empty when it is ready to publish)."""
    out = [f"{name} is missing" for name in REQUIRED if not (model / name).is_file()]
    config = model / "config.json"
    if config.is_file():
        cfg = json.loads(config.read_text())
        if cfg.get("hidden_size") != 4096:
            out.append(f"config.json: hidden_size {cfg.get('hidden_size')}, Qwen3-Embedding-8B has 4096")
        # vLLM's transformers 4.x misreads 5.x's list-valued fields (docs/reports/faz2-jev.md)
        for key in ("rope_parameters", "extra_special_tokens"):
            if isinstance(cfg.get(key), list):
                out.append(f"config.json: {key} is a list (a transformers 5 rewrite; restore the original)")
    tok = model / "tokenizer_config.json"
    if tok.is_file() and isinstance(json.loads(tok.read_text()).get("extra_special_tokens"), list):
        out.append("tokenizer_config.json: extra_special_tokens is a list (a transformers 5 rewrite)")
    return out


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def readme(card: str, repo: str, revision: str, digest: str) -> str:
    return FRONT_MATTER + card.replace("{repo}", repo).replace("{revision}", revision).replace(
        "{sha256}", digest
    )


def with_published(source: str) -> str:
    """``build.py``'s source with ``BACKBONE_PUBLISHED = True``."""
    new, n = PUBLISHED_LINE.subn(lambda m: f"BACKBONE_PUBLISHED = True{m.group(2)}", source)
    if n != 1:
        raise ValueError(f"expected one BACKBONE_PUBLISHED line in build.py, found {n}")
    return new


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from toolrank.build import BACKBONE_REPO, BACKBONE_REVISION

    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--model", required=True, help="the merged directory (scripts/lora_train.py's merged/)")
    p.add_argument("--repo", default=BACKBONE_REPO)
    p.add_argument("--revision", default=BACKBONE_REVISION, help="the Hub tag the weights are served under")
    p.add_argument("--card", default=str(ROOT / "docs" / "backbone" / "MODEL_CARD.md"))
    p.add_argument("--stage", default="dist/backbone/hub", help="where the README is written first")
    p.add_argument("--upload", action="store_true", help="without it: check and stage only")
    a = p.parse_args()
    model = Path(a.model)
    wrong = check_model(model)
    if wrong:
        sys.exit(f"{model}: not ready to publish:\n" + "\n".join(f"  {w}" for w in wrong))
    weights = model / "model.safetensors"
    print(f"hashing {weights} ({weights.stat().st_size / 1e9:.1f} GB) ...", flush=True)
    digest = sha256_file(weights)
    stage = Path(a.stage)
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "README.md").write_text(readme(Path(a.card).read_text(), a.repo, a.revision, digest))
    files = sorted(str(f.relative_to(model)) for f in model.rglob("*") if f.is_file())
    files = [
        f
        for f in files
        if not f.startswith(".saved_by_transformers5") and f not in ("train.json", "train.log")
    ]
    print(f"{len(files)} files ({', '.join(files)}), sha256 {digest[:12]}… -> {a.repo}@{a.revision}")
    if not a.upload:
        print(f"dry run: nothing uploaded; the card is staged in {stage / 'README.md'}")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"as {api.whoami()['name']}")  # fails here without a login
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(model),
        repo_id=a.repo,
        ignore_patterns=list(SKIP),
        commit_message=f"Qwen3-Embedding-8B + toolrank LoRA {a.revision} (model.safetensors sha256 {digest})",
    )
    commit = api.upload_file(
        path_or_fileobj=str(stage / "README.md"),
        path_in_repo="README.md",
        repo_id=a.repo,
        commit_message="model card",
    )
    api.create_tag(a.repo, tag=a.revision, revision=commit.oid, exist_ok=True)
    listed = {
        s.rfilename: s for s in api.model_info(a.repo, revision=a.revision, files_metadata=True).siblings
    }
    missing = [f for f in files if f not in listed]
    if missing:
        sys.exit(f"{a.repo}@{a.revision} lacks {missing}: nothing was written")
    lfs = getattr(listed["model.safetensors"], "lfs", None)
    if lfs is not None and getattr(lfs, "sha256", digest) != digest:
        sys.exit(f"{a.repo}@{a.revision}: the Hub's sha256 of model.safetensors differs: nothing was written")
    module = ROOT / "src" / "toolrank" / "build.py"
    module.write_text(with_published(module.read_text()))
    print(
        f"published {a.repo}@{a.revision}; set BACKBONE_PUBLISHED = True in {module.relative_to(ROOT)}: commit it"
    )


if __name__ == "__main__":
    main()
