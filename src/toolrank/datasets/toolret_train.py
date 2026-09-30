"""ToolRet's training set (``mangopy/ToolRet-Training-20w``): 208,826 requests, each with 1-6
positive tools and 15 hard negatives. Tools are documentation strings in the same JSON shapes as
the ToolRet corpus; ``prompt`` is a ToolRet-style instruction. ``pull_toolret_train`` downloads it
once (needs ``toolrank[data]`` and Hub access) and writes ``pairs.jsonl``.
"""

from __future__ import annotations

from pathlib import Path

from toolrank.datasets.jsonl import write_pairs
from toolrank.domain import TrainPair

TRAIN_REPO = "mangopy/ToolRet-Training-20w"


def pull_toolret_train(out_dir: str | Path) -> int:
    """Download the training set and write ``<out_dir>/pairs.jsonl``; returns the number of pairs."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "ToolRet-train download needs the 'datasets' library: pip install 'toolrank[data]'"
        ) from e

    rows = load_dataset(TRAIN_REPO, split="train")
    pairs = (
        TrainPair(
            id=str(r["id"]),
            text=str(r["query"]),
            positives=tuple(r["positive"] or ()),
            negatives=tuple(r["negative"] or ()),
            instruction=str(r.get("prompt") or ""),
        )
        for r in rows
    )
    return write_pairs(Path(out_dir) / "pairs.jsonl", pairs)
