"""CLM-format heads in plain numpy, so the packaged checkpoint runs without torch.

A ``.npz`` holds each head's state-dict tensors as ``state_head/<key>`` and ``action_head/<key>``
(float16 or float32; float32 in memory), ``logit_scale`` and ``cfg``, a JSON string. The forward
pass is ``make_head``'s (``adapters/clm.py``): Linear and activation, then per hidden layer Linear
→ LayerNorm (eps 1e-5) → activation (added to its input when ``residual``), a last Linear, plus
the head's input when ``skip``; rows come out L2-normalised. GELU is the exact (erf) form, erf from
Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7). Checkpoints load with ``allow_pickle=False``, so a
downloaded file cannot run code.

A packaged ``cfg`` also says how the heads are served (``backbone``, ``tool_format``,
``query_format``, ``truncate``, ``instruction``); ``toolrank.build`` takes those as defaults.
``export_npz`` (torch needed) writes one from a ``.pt``; ``default_heads`` finds or downloads the
packaged file (from ``HEADS_URL``, which ``scripts/publish_heads.py`` writes when it uploads the
file; ``TOOLRANK_HEADS_URL`` points the download at another copy, still checked against
``HEADS_SHA256``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

HEADS_FILE = "toolrank-heads-qwen3-emb-8b-v0.1.npz"
HEADS_URL = "https://huggingface.co/yasinyaman/toolrank-heads-qwen3-emb-8b/resolve/v0.1/toolrank-heads-qwen3-emb-8b-v0.1.npz"  # the hosted file (scripts/publish_heads.py)
HEADS_SHA256 = "f3c101251b9c23925e2925bc02c4492e6c3dea79bcfa1b56715affb20f9f72f0"
_EPS = 1e-5
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _erf(x: np.ndarray) -> np.ndarray:
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (_A[0] + t * (_A[1] + t * (_A[2] + t * (_A[3] + t * _A[4]))))
    return s * (1.0 - poly * np.exp(-a * a))


def _act(name: str, x: np.ndarray) -> np.ndarray:
    if name == "gelu":
        return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "silu":
        return x / (1.0 + np.exp(-x))
    raise ValueError(f"unknown activation {name!r}")


def _expected_keys(cfg: dict[str, Any]) -> list[str]:
    keys = ["inp.weight", "inp.bias", "out.weight", "out.bias"]
    for i in range(int(cfg["depth"]) - 2):
        keys += [f"hidden.{i}.weight", f"hidden.{i}.bias"]
        if cfg.get("layernorm"):
            keys += [f"norms.{i}.weight", f"norms.{i}.bias"]
    return keys


class NumpyHeads:
    """State + action heads from a ``.npz``; same interface as ``CLMHeads``."""

    def __init__(self, path: str | Path, batch: int = 4096):
        self.path, self.batch = Path(path), batch
        with np.load(self.path, allow_pickle=False) as z:
            self.cfg: dict[str, Any] = json.loads(str(z["cfg"]))
            arrays = {k: np.asarray(z[k], dtype=np.float32) for k in z.files if k != "cfg"}
        self.heads: dict[str, dict[str, np.ndarray]] = {}
        for head in ("state_head", "action_head"):
            params = {k.split("/", 1)[1]: v for k, v in arrays.items() if k.startswith(head + "/")}
            want = _expected_keys(self.cfg)
            if sorted(params) != sorted(want):
                raise ValueError(f"{self.path}: {head} has {sorted(params)}, expected {sorted(want)}")
            hidden, width = int(self.cfg.get("hidden_size", 4096)), int(self.cfg["width"])
            if params["inp.weight"].shape != (width, hidden):
                raise ValueError(f"{self.path}: {head} inp.weight is {params['inp.weight'].shape}")
            self.heads[head] = params
        self.proj_dim = int(self.heads["state_head"]["out.weight"].shape[0])
        self.hidden = int(self.cfg.get("hidden_size", 4096))
        if self.cfg.get("skip") and self.proj_dim != int(self.cfg.get("hidden_size", 4096)):
            raise ValueError(f"{self.path}: a skip head must keep the input width")
        self.scale = float(min(math.exp(float(arrays.get("logit_scale", np.zeros(1)).reshape(-1)[0])), 100.0))
        self.n_params = sum(v.size for p in self.heads.values() for v in p.values())

    def _forward(self, p: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        cfg, act = self.cfg, self.cfg.get("activation", "gelu")
        h = _act(act, x @ p["inp.weight"].T + p["inp.bias"])
        for i in range(int(cfg["depth"]) - 2):
            z = h @ p[f"hidden.{i}.weight"].T + p[f"hidden.{i}.bias"]
            if cfg.get("layernorm"):
                mu = z.mean(axis=-1, keepdims=True)
                var = ((z - mu) ** 2).mean(axis=-1, keepdims=True)
                z = (z - mu) / np.sqrt(var + _EPS) * p[f"norms.{i}.weight"] + p[f"norms.{i}.bias"]
            z = _act(act, z)
            h = h + z if cfg.get("residual") else z
        out = h @ p["out.weight"].T + p["out.bias"]
        return x + out if cfg.get("skip") else out

    def _project(self, head: str, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if len(x) == 0:
            return np.zeros((0, self.proj_dim), dtype=np.float32)
        check_width(self.path, self.cfg, self.hidden, x)
        outs = []
        for s in range(0, len(x), self.batch):
            y = self._forward(self.heads[head], x[s : s + self.batch]).astype(np.float32)
            outs.append(y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-12))
        return np.concatenate(outs)

    def project_states(self, x: np.ndarray) -> np.ndarray:
        return self._project("state_head", x)

    def project_actions(self, x: np.ndarray) -> np.ndarray:
        return self._project("action_head", x)


def check_width(path: Path, cfg: dict[str, Any], hidden: int, x: np.ndarray) -> None:
    """Refuse vectors of another width than the heads were trained on, naming both: otherwise the
    first matrix product fails deep inside the index build with shapes only."""
    if x.ndim != 2 or x.shape[1] != hidden:
        width = x.shape[1] if x.ndim == 2 else x.shape
        backbone = cfg.get("backbone") or "its backbone"
        raise ValueError(
            f"{path.name} expects {hidden}-dimensional vectors from {backbone}, the embedding endpoint "
            f"returns {width}: point --emb-url / --emb-model at that model, or use heads trained on this one"
        )


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """A packaged ``.npz`` in the checkpoint shape training uses — ``state_head`` and
    ``action_head`` ({parameter: float32 array}), ``logit_scale`` (the log, as stored) and ``cfg``
    — so ``toolrank finetune`` can continue from the released heads. Shapes are checked as
    ``NumpyHeads`` checks them."""
    heads = NumpyHeads(path)
    with np.load(path, allow_pickle=False) as z:
        stored = (
            np.asarray(z["logit_scale"], dtype=np.float32).reshape(-1) if "logit_scale" in z.files else None
        )
    return {
        "state_head": dict(heads.heads["state_head"]),
        "action_head": dict(heads.heads["action_head"]),
        "logit_scale": float(stored[0]) if stored is not None and stored.size else math.log(20.0),
        "cfg": dict(heads.cfg),
    }


def load_heads(path: str | Path, device: str | None = None) -> Any:
    """``.npz`` -> ``NumpyHeads``; anything else is a torch checkpoint -> ``CLMHeads``."""
    if str(path).endswith(".npz"):
        return NumpyHeads(path)
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise ValueError(
            f"{path}: .pt heads need torch: pip install 'toolrank[clm]' (the packaged .npz heads do not)"
        )
    from toolrank.adapters.clm import CLMHeads

    return CLMHeads(path, device=device)


def export_npz(
    src: str | Path, dst: str | Path, *, dtype: str = "float16", serving: dict[str, Any] | None = None
) -> str:
    """A torch checkpoint (``.pt``) -> the ``.npz`` ``NumpyHeads`` reads; -> its sha256."""
    import torch

    ck = torch.load(str(src), map_location="cpu", weights_only=True)
    cfg = {**dict(ck["cfg"]), **(serving or {})}
    arrays: dict[str, np.ndarray] = {"cfg": np.array(json.dumps(cfg, sort_keys=True))}
    for head in ("state_head", "action_head"):
        for k, v in ck[head].items():
            arrays[f"{head}/{k}"] = v.detach().float().numpy().astype(dtype)
    arrays["logit_scale"] = np.asarray(
        torch.as_tensor(ck["logit_scale"]).float().numpy(), dtype=np.float32
    ).reshape(1)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        np.savez(f, **arrays)
    return sha256_file(dst)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str | Path, sha256: str) -> Path:
    """``url`` -> ``dest`` through a temp file in the same directory, kept only if the sha256 matches."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".part{os.getpid()}")
    try:
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            for block in iter(lambda: r.read(1 << 20), b""):
                f.write(block)
        got = sha256_file(tmp)
        if sha256 and got != sha256:
            raise ValueError(f"{url}: sha256 {got}, expected {sha256}")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def default_heads(*, url: str | None = None, sha256: str | None = None) -> Path:
    """The packaged heads: ``TOOLRANK_HEADS``, else the cached file, else a download from
    ``TOOLRANK_HEADS_URL`` or ``HEADS_URL``."""
    env = os.environ.get("TOOLRANK_HEADS")
    if env:
        if not Path(env).is_file():  # a typo must not quietly serve without the heads
            raise ValueError(f"TOOLRANK_HEADS={env}: no such file")
        return Path(env)
    cached = (
        Path(os.environ.get("TOOLRANK_CACHE", Path.home() / ".cache" / "toolrank")) / "heads" / HEADS_FILE
    )
    if cached.exists():
        return cached
    url = (os.environ.get("TOOLRANK_HEADS_URL") or HEADS_URL) if url is None else url
    if not url:
        raise FileNotFoundError(
            "the packaged heads are not hosted yet: pass --clm-ckpt PATH, set TOOLRANK_HEADS, or "
            "download from a mirror (toolrank heads pull --url URL)"
        )
    return download(url, cached, HEADS_SHA256 if sha256 is None else sha256)
