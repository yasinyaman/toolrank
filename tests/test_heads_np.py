import json
from pathlib import Path

import numpy as np
import pytest

from toolrank.adapters.heads_np import NumpyHeads, default_heads, download, load_heads, sha256_file
from toolrank.build import build_scorer
from toolrank.cli import build_parser

GOLDEN = Path(__file__).parent / "fixtures" / "heads_golden.npz"
CASES = ("gelu_layernorm_skip", "relu_residual_deep", "silu_shallow")


def _case_npz(tmp_path, case, dtype="float32", serving=None) -> Path:
    """One golden case in the .npz layout NumpyHeads reads."""
    with np.load(GOLDEN, allow_pickle=False) as z:
        cfg = {**json.loads(str(z[f"{case}:cfg"])), **(serving or {})}
        arrays = {
            k.split(":", 1)[1]: z[k].astype(dtype) for k in z.files if k.startswith(case + ":") and "/" in k
        }
    path = tmp_path / f"{case}.npz"
    with open(path, "wb") as f:
        np.savez(f, cfg=np.array(json.dumps(cfg)), logit_scale=np.array([2.0], np.float32), **arrays)
    return path


def test_numpy_forward_matches_torch_on_every_head_shape(tmp_path):
    with np.load(GOLDEN, allow_pickle=False) as z:
        x = z["x"]
        for case in CASES:
            heads = NumpyHeads(_case_npz(tmp_path, case))
            assert np.allclose(heads.project_states(x), z[f"{case}:state_head:y"], atol=1e-5), case
            assert np.allclose(heads.project_actions(x), z[f"{case}:action_head:y"], atol=1e-5), case
    assert heads.scale == pytest.approx(np.exp(2.0)) and heads.project_states(x[:0]).shape == (0, 5)


def test_float16_storage_and_strict_keys(tmp_path):
    with np.load(GOLDEN, allow_pickle=False) as z:
        x, want = z["x"], z["gelu_layernorm_skip:state_head:y"]
    got = NumpyHeads(_case_npz(tmp_path, "gelu_layernorm_skip", "float16")).project_states(x)
    assert (got * want).sum(axis=1).min() > 0.999  # cosine to the float32 output
    path = _case_npz(tmp_path, "silu_shallow")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "action_head/out.bias"}
    with open(path, "wb") as f:
        np.savez(f, **arrays)
    with pytest.raises(ValueError, match="action_head"):
        NumpyHeads(path)


def test_export_from_torch_round_trips(tmp_path):
    pytest.importorskip("torch")
    from test_clm_heads import _fake_checkpoint
    from toolrank.adapters.clm import CLMHeads
    from toolrank.adapters.heads_np import export_npz

    _fake_checkpoint(tmp_path / "h.pt")
    digest = export_npz(tmp_path / "h.pt", tmp_path / "h.npz", dtype="float32", serving={"truncate": 8192})
    assert digest == sha256_file(tmp_path / "h.npz")
    x = np.random.default_rng(3).standard_normal((6, 32)).astype(np.float32)
    npz, pt = load_heads(tmp_path / "h.npz"), CLMHeads(tmp_path / "h.pt", device="cpu")
    assert isinstance(npz, NumpyHeads) and npz.cfg["truncate"] == 8192
    assert np.allclose(npz.project_actions(x), pt.project_actions(x), atol=1e-5)


def test_download_checks_sha256_and_default_heads(tmp_path, monkeypatch):
    src = _case_npz(tmp_path, "silu_shallow")
    digest = sha256_file(src)
    dest = download(src.as_uri(), tmp_path / "cache" / "h.npz", digest)
    assert sha256_file(dest) == digest
    with pytest.raises(ValueError, match="sha256"):
        download(src.as_uri(), tmp_path / "cache" / "bad.npz", "0" * 64)
    assert not (tmp_path / "cache" / "bad.npz").exists() and len(list((tmp_path / "cache").iterdir())) == 1
    monkeypatch.setenv("TOOLRANK_CACHE", str(tmp_path / "empty"))
    monkeypatch.delenv("TOOLRANK_HEADS", raising=False)
    with pytest.raises(FileNotFoundError, match="not hosted yet"):
        default_heads(url="")
    got = default_heads(url=src.as_uri(), sha256=digest)
    assert got.parent == tmp_path / "empty" / "heads" and sha256_file(got) == digest
    got.unlink()
    monkeypatch.setenv("TOOLRANK_HEADS_URL", src.as_uri())  # a mirror; the sha256 is still checked
    assert sha256_file(default_heads(sha256=digest)) == digest
    got.unlink()
    with pytest.raises(ValueError, match="sha256"):
        default_heads()  # the real file's sha256, not this one's
    monkeypatch.setenv("TOOLRANK_HEADS", str(src))
    assert default_heads() == src


def test_packaged_heads_bring_their_serving_defaults(tmp_path):
    serving = {"tool_format": "documentation", "query_format": "instruct_query", "truncate": 8192}
    npz = _case_npz(tmp_path, "gelu_layernorm_skip", serving=serving)
    a = build_parser().parse_args(
        ["eval", "--data", "x", "--scorer", "clm", "--clm-ckpt", str(npz), "--with-inst"]
    )
    s = build_scorer(a)
    assert (s.tool_format.name, s.query_format.name, a.truncate) == ("documentation", "instruct_query", 8192)
    assert s.name.startswith("clm[gelu_layernorm_skip]/")


def test_eval_records_the_heads_file_it_ran_with(tmp_path, monkeypatch):
    from toolrank.adapters.embeddings_api import OpenAIEmbeddings
    from toolrank.cli import main

    def fake_post(self, texts, **kw):  # 16-d vectors, the golden heads' input width
        rng = np.random.default_rng(len(texts))
        return [rng.standard_normal(16).astype(np.float32) for _ in texts], 0

    monkeypatch.setattr(OpenAIEmbeddings, "_post", fake_post)
    npz = _case_npz(tmp_path, "gelu_layernorm_skip")
    syn = tmp_path / "syn"
    assert main(["data", "synth", "--out", str(syn), "--n-tools", "20", "--n-queries", "5"]) == 0
    runs = {
        "heads": [
            "--scorer",
            "clm",
            "--clm-ckpt",
            str(npz),
            "--emb-url",
            "http://unused/v1",
            "--emb-model",
            "m",
        ],
        "bm25": ["--scorer", "bm25"],
    }
    for name, flags in runs.items():
        out = tmp_path / f"{name}.json"
        assert (
            main(["eval", "--data", str(syn), *flags, "--cache-dir", "", "--with-inst", "--out", str(out)])
            == 0
        )
        runs[name] = json.loads(out.read_text())["config"]["heads_sha256"]
    assert runs == {"heads": sha256_file(npz), "bm25": None}


def test_vectors_of_another_width_are_refused_with_both_widths_named(tmp_path):
    heads = NumpyHeads(_case_npz(tmp_path, "silu_shallow", serving={"backbone": "Qwen/Qwen3-Embedding-8B"}))
    with np.load(GOLDEN, allow_pickle=False) as z:
        x = z["x"]
    wrong = np.zeros((2, x.shape[1] + 3), np.float32)
    with pytest.raises(
        ValueError,
        match=rf"expects {x.shape[1]}-dimensional vectors from Qwen/Qwen3-Embedding-8B.*returns {x.shape[1] + 3}",
    ):
        heads.project_actions(wrong)
    with pytest.raises(ValueError, match="--emb-url"):
        heads.project_states(wrong)
    assert heads.project_states(x).shape[0] == len(x)  # the right width still works
