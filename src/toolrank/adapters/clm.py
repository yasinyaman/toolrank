"""CLM scorer: frozen backbone embeddings -> state head / action head -> cosine.

Mirrors ``clm.heads`` from https://github.com/Contrastive-LM/CLM (Apache-2.0): a checkpoint is a
``torch.save`` dict with ``state_head`` / ``action_head`` state dicts, ``logit_scale`` and ``cfg``
(``width``, ``depth``, optional ``projection_dim``, ``activation``, ``layernorm``, ``residual``).
Each head is a ``hidden -> width -> ... -> proj`` MLP; the score of a (state, candidate) pair is
``exp(logit_scale) * cos(state_head(s), action_head(c))``. The scale does not change the ranking,
so it is ignored here.

The backbone (Qwen3-8B, last-token pooling, L2-normalised) comes from any OpenAI-compatible
embeddings endpoint - ``OpenAIEmbeddings`` with ``truncate_prompt_tokens=2048`` is the reference
setup. Only the ~20M-parameter heads run in this process, so a laptop CPU is enough.

Text conventions (``clm.schema``): the state is ``context + "\\n\\n" + question`` (our ``clm`` query
format when an instruction exists), candidates are passed verbatim (our tool formats).

toolrank extension: ``cfg["skip"]`` makes each head ``x + MLP(x)`` (so ``projection_dim`` equals the
input width). With the last layer zeroed it starts as the identity, which is how heads are trained
on top of an embedding model without starting below its zero-shot quality. Reference checkpoints
have no ``skip`` and load unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from toolrank.adapters.dense import DenseScorer
from toolrank.formats import NamedFormatter
from toolrank.ports import TextEncoder, VectorIndex

HIDDEN = 4096
PROJ_DIM = 512
HF_REPO = "Contrastive-LM/CLM-v0.1-8B"
HF_FILE = "CLM_v0.1-8B.pt"


def default_checkpoint() -> Path | None:
    p = os.environ.get("CLM_CKPT")
    if p and Path(p).exists():
        return Path(p)
    p2 = Path(os.environ.get("CLM_CKPT_DIR", Path.home() / ".cache" / "clm")) / HF_FILE
    return p2 if p2.exists() else None


def make_head(
    width: int,
    depth: int = 2,
    proj: int = PROJ_DIM,
    activation: str = "gelu",
    layernorm: bool = False,
    residual: bool = False,
    hidden: int = HIDDEN,
    skip: bool = False,
):
    """Same module layout as ``clm.heads.make_head`` so reference checkpoints load verbatim;
    ``skip`` adds the input to the output (toolrank extension, see the module docstring)."""
    import torch.nn as nn

    if skip and proj != hidden:
        raise ValueError(f"a skip head keeps the input width: projection_dim {proj} != hidden {hidden}")
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList(
                (nn.LayerNorm(width) if layernorm else nn.Identity()) for _ in range(depth - 2)
            )
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual
            self.skip = skip

        def forward(self, x):
            x0 = x
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms, strict=True):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return x0 + self.out(x) if self.skip else self.out(x)

    return Head()


class CLMHeads:
    """State + action projection heads loaded from one checkpoint file."""

    def __init__(self, path: str | Path, device: str | None = None, batch: int = 4096):
        import torch

        self.path = Path(path)
        self.device = (
            device or os.environ.get("CLM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch = batch
        # weights_only: torch before 2.6 unpickles anything by default, and a checkpoint is a download
        ck = torch.load(str(self.path), map_location="cpu", weights_only=True)
        cfg: dict[str, Any] = dict(ck["cfg"])
        kw = dict(
            width=cfg["width"],
            depth=cfg["depth"],
            proj=ck.get("projection_dim", cfg.get("projection_dim", PROJ_DIM)),
            activation=cfg.get("activation", "gelu"),
            layernorm=cfg.get("layernorm", False),
            residual=cfg.get("residual", False),
            hidden=cfg.get("hidden_size", HIDDEN),
            skip=cfg.get("skip", False),
        )
        self.state_head, self.action_head = make_head(**kw), make_head(**kw)
        self.state_head.load_state_dict(ck["state_head"])
        self.action_head.load_state_dict(ck["action_head"])
        self.state_head.eval().to(self.device)
        self.action_head.eval().to(self.device)
        self.cfg, self.proj_dim, self.hidden = cfg, int(kw["proj"]), int(kw["hidden"])
        self.scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))
        self.n_params = sum(p.numel() for h in (self.state_head, self.action_head) for p in h.parameters())

    def _project(self, head, x: np.ndarray) -> np.ndarray:
        import torch

        from toolrank.adapters.heads_np import check_width

        if len(x):
            check_width(self.path, self.cfg, self.hidden, np.asarray(x))
        outs = []
        with torch.no_grad():
            for s in range(0, len(x), self.batch):
                t = torch.from_numpy(np.ascontiguousarray(x[s : s + self.batch], dtype=np.float32)).to(
                    self.device
                )
                outs.append(torch.nn.functional.normalize(head(t), dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.proj_dim), dtype=np.float32)

    def project_states(self, x: np.ndarray) -> np.ndarray:
        return self._project(self.state_head, x)

    def project_actions(self, x: np.ndarray) -> np.ndarray:
        return self._project(self.action_head, x)


class CLMScorer(DenseScorer):
    """``DenseScorer`` with CLM heads on both sides. Backbone rows are cached by the encoder, so
    swapping heads (e.g. after a fine-tune) re-projects 43k tools in seconds without re-encoding."""

    name = "clm"

    def __init__(
        self,
        encoder: TextEncoder,
        heads: Any,  # CLMHeads or NumpyHeads: project_states / project_actions / path
        tool_format: NamedFormatter | str = "name_desc",
        query_format: NamedFormatter | str = "clm",
        *,
        index: VectorIndex | None = None,
        fingerprint: str = "",
        server_weight: float = 0.0,
    ):
        super().__init__(
            encoder,
            tool_format,
            query_format,
            project_tools=heads.project_actions,
            project_queries=heads.project_states,
            label=f"clm[{heads.path.stem}]",
            index=index,
            fingerprint=fingerprint,
            server_weight=server_weight,
        )
        self.heads = heads
