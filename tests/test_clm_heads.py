import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toolrank.adapters.clm import CLMHeads, CLMScorer, make_head  # noqa: E402
from toolrank.domain import Query, Tool  # noqa: E402


def _fake_checkpoint(path, width=64, depth=3, proj=16, hidden=32):
    sh, ah = make_head(width, depth, proj, hidden=hidden), make_head(width, depth, proj, hidden=hidden)
    torch.save(
        {
            "state_head": sh.state_dict(),
            "action_head": ah.state_dict(),
            "logit_scale": torch.tensor(2.0),
            "cfg": {
                "width": width,
                "depth": depth,
                "projection_dim": proj,
                "hidden_size": hidden,
                "activation": "gelu",
            },
        },
        path,
    )


class _RandEncoder:
    name = "rand"

    def __init__(self, dim):
        self.dim, self.rng = dim, np.random.default_rng(0)

    def encode(self, texts, *, kind="document"):
        v = self.rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_heads_load_and_project(tmp_path):
    ck = tmp_path / "heads.pt"
    _fake_checkpoint(ck)
    h = CLMHeads(ck, device="cpu")
    assert h.proj_dim == 16 and h.n_params > 0
    z = h.project_states(np.random.default_rng(1).standard_normal((5, 32)).astype(np.float32))
    assert z.shape == (5, 16)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)
    assert abs(h.scale - float(np.exp(2.0))) < 1e-4


def test_checkpoints_load_weights_only(tmp_path, monkeypatch):
    """A .pt is a download: said explicitly, since torch before 2.6 unpickles anything by default."""
    from toolrank.finetune import load_checkpoint

    ck = tmp_path / "heads.pt"
    _fake_checkpoint(ck)
    seen, real = [], torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **kw: seen.append(kw.get("weights_only")) or real(*a, **kw))
    CLMHeads(ck, device="cpu")
    load_checkpoint(str(ck))
    assert seen == [True, True]


def test_clm_scorer_end_to_end(tmp_path):
    ck = tmp_path / "heads.pt"
    _fake_checkpoint(ck)
    s = CLMScorer(_RandEncoder(32), CLMHeads(ck, device="cpu"), "name_desc", "clm")
    tools = [Tool(id=str(i), doc={"name": f"t{i}", "description": "d"}) for i in range(10)]
    s.index(tools)
    r = s.rank([Query(id="q", text="x", qrels={"1": 1}, instruction="i")], k=4)
    assert len(r[0].tool_ids) == 4 and s.name.startswith("clm[heads]")
