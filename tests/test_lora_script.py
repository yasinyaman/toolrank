"""The pure parts of scripts/lora_train.py: the InfoNCE masking (torch), the step plan and the
learning-rate schedule. The model itself is never loaded here."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("lora_train", ROOT / "scripts" / "lora_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_and_schedule():
    m = _script()
    assert m.plan_steps(100, 16, 2, 1.0) == (7, 3) and m.plan_steps(64, 16, 2, 0.5) == (2, 1)
    assert m.lr_at(0, 100, 10, 1e-4) == pytest.approx(1e-5) and m.lr_at(9, 100, 10, 1e-4) == pytest.approx(
        1e-4
    )
    assert m.lr_at(10, 100, 10, 1e-4) == pytest.approx(1e-4)
    assert m.lr_at(100, 100, 10, 1e-4) == pytest.approx(0.0, abs=1e-12)
    assert math.isclose(m.lr_at(55, 100, 10, 1e-4), 0.5e-4, rel_tol=1e-6)  # halfway down the cosine


def test_infonce_masks_duplicate_positives():
    torch = pytest.importorskip("torch")
    m = _script()
    sim = torch.tensor([[0.9, 0.9, 0.1], [0.9, 0.9, 0.1], [0.0, 0.0, 0.8]])
    plain = m.infonce(sim, ["a", "b", "c"], tau=0.1)
    masked = m.infonce(sim, ["a", "a", "c"], tau=0.1)  # rows 0 and 1 share a positive text
    assert masked < plain  # the duplicate is no longer a competing negative
    assert torch.isfinite(masked)


def test_the_data_seed_is_apart_from_the_training_seed():
    """A replica run changes --seed (initialisation, dropout, batch order) and keeps the pairs."""
    m = _script()
    base = ["--pairs", "p.jsonl", "--dev", "d", "--out", "o"]
    assert m.parse([*base, "--seed", "3"]).data_seed == 3  # default: --seed, as before
    a = m.parse([*base, "--seed", "3", "--data-seed", "0"])
    assert (a.seed, a.data_seed) == (3, 0)


def test_only_the_listed_files_of_a_full_snapshot_are_copied(tmp_path):
    """A cache that also holds the base weights must not put them next to the merged ones."""
    m = _script()
    src, merged = tmp_path / "snapshot", tmp_path / "merged"
    names = [
        "config.json", "tokenizer.json", "vocab.json", "1_Pooling/config.json", "modules.json",
        "model-00001-of-00004.safetensors", "model.safetensors.index.json", "README.md", "LICENSE",
    ]  # fmt: skip
    for name in names:
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_text(name)
    merged.mkdir()
    (merged / "model.safetensors").write_text("merged weights")
    copied = m.copy_listed(src, merged)
    assert copied == ["1_Pooling/config.json", "config.json", "modules.json", "tokenizer.json", "vocab.json"]
    assert sorted(p.relative_to(merged).as_posix() for p in merged.rglob("*") if p.is_file()) == sorted(
        [*copied, "model.safetensors"]
    )
