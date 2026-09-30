"""Regenerate tests/fixtures/heads_golden.npz (needs torch): tiny heads of several shapes, random
inputs and torch's outputs, so CI checks the numpy forward pass without torch.

uv run python tests/fixtures/make_heads_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from toolrank.adapters.clm import make_head

CASES = {
    "gelu_layernorm_skip": dict(width=12, depth=3, activation="gelu", layernorm=True, skip=True),
    "relu_residual_deep": dict(width=10, depth=4, activation="relu", residual=True, proj=6),
    "silu_shallow": dict(width=8, depth=2, activation="silu", proj=5),
}
HIDDEN = 16


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(7, HIDDEN)
    out: dict[str, np.ndarray] = {"x": x.numpy()}
    for name, kw in CASES.items():
        proj = kw.pop("proj", HIDDEN)
        cfg = {**kw, "projection_dim": proj, "hidden_size": HIDDEN}
        heads = {h: make_head(proj=proj, hidden=HIDDEN, **kw) for h in ("state_head", "action_head")}
        for h, mod in heads.items():
            for k, v in mod.state_dict().items():
                if k == "out.weight" and kw.get("skip"):
                    v = v + 0.05 * torch.randn_like(v)  # a trained skip head is not the identity
                out[f"{name}:{h}/{k}"] = v.numpy()
            mod.load_state_dict({k: torch.as_tensor(out[f"{name}:{h}/{k}"]) for k in mod.state_dict()})
            with torch.no_grad():
                y = torch.nn.functional.normalize(mod.eval()(x), dim=-1)
            out[f"{name}:{h}:y"] = y.numpy()
        out[f"{name}:cfg"] = np.array(json.dumps(cfg, sort_keys=True))
    path = Path(__file__).with_name("heads_golden.npz")
    with open(path, "wb") as f:
        np.savez(f, **out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
