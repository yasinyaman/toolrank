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
