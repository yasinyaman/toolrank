"""LoRA fine-tuning of the embedding backbone itself on ToolRet-train pairs (Faz 2).

``toolrank finetune`` trains heads on cached vectors; this trains Qwen3-Embedding-8B's own weights
(LoRA adapters) with the same data, texts and selection rule, then merges them into full weights
that vLLM serves like the original (compose profile ``lora``, port 8097, ``qwen3-emb-lora``). GPU,
``[lora]`` extra. On the GB10, from ~/toolrank, under nohup:

    uv run --extra lora python scripts/lora_train.py --pairs data/toolret_train/pairs.jsonl \\
      --dev data/mcp_zero_server --eval data/toolret --eval data/livemcpbench_server \\
      --n-train 20000 --out data/lora/qwen3-emb-lora-20k

Texts exactly as served: queries in the ``instruct_query`` format with the pair's instruction (the
serving instruction when it has none), tools as ``documentation``; a pair whose request equals a
dev or eval query is dropped (``finetune.split_pairs``). Loss: InfoNCE over the micro-batch's
positives (in-batch negatives; the mined negatives stay out, Faz 0 found they cost 10 points), one
sampled positive per pair, duplicate tool texts masked. Last-token pooling and L2 normalisation,
the way vLLM pools this model; ``--check-parity`` compares the untouched model with the served
vectors first, so a tokenisation mismatch shows up before training. The dev set is scored
in-process every ``--eval-every`` steps (``finetune.EvalSet``: exact top-k, the benchmark's
metrics) and picks the adapter that is kept; the dev set is never reported, ``toolrank eval`` on
the served model is.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toolrank.build import DEFAULT_SERVING  # noqa: E402
from toolrank.datasets.jsonl import load_pairs, load_queries, load_tools  # noqa: E402
from toolrank.finetune import EvalSet, leaks, pair_texts, split_pairs, with_instruction  # noqa: E402
from toolrank.formats import query_format, tool_format  # noqa: E402

MODEL = "Qwen/Qwen3-Embedding-8B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pairs", required=True)
    p.add_argument("--dev", required=True, help="picks the checkpoint; never reported")
    p.add_argument("--eval", action="append", default=[], help="its queries are only dropped from training")
    p.add_argument("--out", required=True, help="adapter/, merged/ and train.json go here")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--micro-batch", type=int, default=16, help="pairs per forward: its in-batch negatives")
    p.add_argument("--accumulate", type=int, default=2, help="micro-batches per optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--tau", type=float, default=0.05, help="InfoNCE temperature")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--query-tokens", type=int, default=256)
    p.add_argument("--doc-tokens", type=int, default=768)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument(
        "--keep-all",
        action="store_true",
        help="keep the adapter of every evaluated step (checkpoints/step-N, 170 MB each), so another "
        "dev set can pick again without training again",
    )
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0, help="initialisation, dropout and batch order")
    p.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="which training pairs are drawn, so the same pairs train across --seed replicas (default: --seed)",
    )
    p.add_argument("--instruction", default=DEFAULT_SERVING["instruction"])
    p.add_argument("--check-parity", action="store_true", help="compare 8 vectors with --emb-url first")
    p.add_argument("--emb-url", default="http://127.0.0.1:8091/v1")
    p.add_argument("--emb-model", default="qwen3-emb")
    p.add_argument("--no-merge", action="store_true")
    p.add_argument("--limit-dev", type=int, default=0, help="first N dev queries (smoke tests)")
    a = p.parse_args(argv)
    if a.data_seed is None:
        a.data_seed = a.seed
    return a


# -- pure parts (tested without a model) --------------------------------------------------------
def infonce(sim: Any, doc_texts: list[str], tau: float) -> Any:
    """Cross-entropy of each query against the batch's positives at ``sim / tau``; a duplicate of
    the query's own positive elsewhere in the batch is masked out, not treated as a negative."""
    import torch

    n = sim.shape[0]
    logits = sim / tau
    same = torch.tensor(
        [[doc_texts[i] == doc_texts[j] and i != j for j in range(n)] for i in range(n)], device=sim.device
    )
    logits = logits.masked_fill(same, float("-inf"))
    return torch.nn.functional.cross_entropy(logits, torch.arange(n, device=sim.device))


def plan_steps(n_pairs: int, micro: int, accumulate: int, epochs: float) -> tuple[int, int]:
    """-> (micro-batches, optimizer steps) for the run."""
    micro_batches = int(math.ceil(n_pairs / micro) * epochs)
    return micro_batches, max(1, micro_batches // accumulate)


def lr_at(step: int, total: int, warmup: int, lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


# -- the model ----------------------------------------------------------------------------------
class Encoder:
    """The backbone with LoRA adapters: last-token pooled, L2-normalised vectors."""

    def __init__(self, name: str, rank: int, alpha: int, dropout: float, device: str):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS, bias="none")
        self.model = get_peft_model(base, cfg).to(device)
        self.model.print_trainable_parameters()

    def _inner(self) -> Any:  # the decoder stack, below the lm_head (which the pooling never uses)
        return self.model.base_model.model.model

    def encode(self, texts: list[str], max_tokens: int, train: bool) -> Any:
        torch = self.torch
        batch = self.tok(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(
            self.device
        )
        with torch.set_grad_enabled(train):
            h = self._inner()(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).last_hidden_state
            v = h[:, -1].float()  # left padding: the last position is the last real token
            return torch.nn.functional.normalize(v, dim=-1)

    def encode_many(self, texts: list[str], max_tokens: int, batch: int) -> Any:
        import numpy as np

        self.model.eval()
        out = []
        for s in range(0, len(texts), batch):
            out.append(self.encode(texts[s : s + batch], max_tokens, train=False).cpu().numpy())
        self.model.train()
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


def parity(enc: Encoder, texts: list[str], url: str, model: str, max_tokens: int) -> float:
    """Mean cosine between this process's vectors and the served model's: ~1.0 when pooling and
    tokenisation agree, visibly lower when they do not."""
    import numpy as np

    from toolrank.adapters.embeddings_api import OpenAIEmbeddings

    served = OpenAIEmbeddings(model, url, truncate_prompt_tokens=8192, cache_dir=".cache/toolrank").encode(
        texts
    )
    mine = enc.encode_many(texts, max_tokens, 8)
    served = served / np.linalg.norm(served, axis=1, keepdims=True)
    return float(np.mean(np.sum(served * mine, axis=1)))


ORIGINAL_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "modules.json",
    "config_sentence_transformers.json",
    "1_Pooling/*",
]


def copy_original_files(model: str, merged: Path) -> None:
    """The merged directory gets the original repository's config, tokenizer and pooling files, not
    the ones this process's transformers would write: a serving stack with an older transformers
    reads neither a list-valued ``extra_special_tokens`` (it crashes) nor ``rope_parameters`` (it
    falls back to the default ``rope_theta`` and the vectors are silently wrong). Only the weights
    are new."""
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(model, allow_patterns=ORIGINAL_FILES))
    for stale in ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        (merged / stale).unlink(missing_ok=True)
    copy_listed(src, merged)


def copy_listed(src: Path, merged: Path) -> list[str]:
    """Copy the files of ``src`` that ``ORIGINAL_FILES`` names into ``merged`` -> their paths. The
    snapshot directory holds whatever the cache has, not only what ``allow_patterns`` fetched: once the
    base weights were in the cache, copying it whole put the base model's shards and their
    ``model.safetensors.index.json`` next to the merged weights, and vLLM loads the shards an index
    lists (7 Oct 2026, the 60k run)."""
    import fnmatch
    import shutil

    copied = []
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src).as_posix()
        if f.is_file() and any(fnmatch.fnmatchcase(rel, pattern) for pattern in ORIGINAL_FILES):
            dest = merged / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            copied.append(rel)
    return copied


def main(argv: list[str] | None = None) -> int:
    a = parse(argv)
    import torch

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    tf, qf = tool_format("documentation"), query_format("instruct_query")
    dev_dir = Path(a.dev)
    dev_tools, dev_queries = load_tools(dev_dir / "tools.jsonl"), load_queries(dev_dir / "queries.jsonl")
    if a.limit_dev:
        dev_queries = dev_queries[: a.limit_dev]
    eval_queries = {Path(e).name: load_queries(Path(e) / "queries.jsonl") for e in a.eval}
    loaded = load_pairs(Path(a.pairs))
    pairs = [p for p in loaded if p.positives]
    dropped = {
        "no_positives": len(loaded) - len(pairs),
        **leaks(pairs, {dev_dir.name: dev_queries, **eval_queries}),
    }
    everything = list(dev_queries) + [q for qs in eval_queries.values() for q in qs]
    train, _, _ = split_pairs(pairs, everything, a.n_train, 0, a.data_seed)
    train, filled = with_instruction(train, a.instruction)
    states, positives, _ = pair_texts(train, tf, qf)
    log(
        f"{len(loaded):,} pairs -> train {len(train):,} (data seed {a.data_seed}, seed {a.seed}); "
        f"dropped {dropped}; instruction filled {filled:,}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoder(a.model, a.rank, a.alpha, a.dropout, device)
    if a.check_parity:
        sample = [tf(t) for t in dev_tools[:8]]
        cos = parity(enc, sample, a.emb_url, a.emb_model, a.doc_tokens)
        log(f"parity with {a.emb_url} ({a.emb_model}) on 8 tool texts: mean cosine {cos:.4f}")
        if cos < 0.98:
            log("pooling or tokenisation differs from the served model: stopping")
            return 2

    dev = EvalSet(
        dev_dir.name,
        list(dev_queries),
        [t.id for t in dev_tools],
        enc.encode_many([tf(t) for t in dev_tools], a.doc_tokens, a.eval_batch),
        enc.encode_many([qf(q) for q in dev_queries], a.query_tokens, a.eval_batch),
    )
    ident = lambda x: x  # noqa: E731

    def score_dev() -> dict[str, float]:
        dev.tool_vecs = enc.encode_many([tf(t) for t in dev_tools], a.doc_tokens, a.eval_batch)
        dev.query_vecs = enc.encode_many([qf(q) for q in dev_queries], a.query_tokens, a.eval_batch)
        s = dev.score(ident, ident, ks=(1, 10))
        return {"NDCG@10": s.overall["NDCG@10"], "Precision@1": s.overall["Precision@1"]}

    best = score_dev()
    log(f"dev {dev.name} before training: {best}")
    history = [{"step": 0, **best}]
    enc.model.save_pretrained(out / "adapter")

    params = [p for p in enc.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    micro_batches, steps = plan_steps(len(train), a.micro_batch, a.accumulate, a.epochs)
    log(f"{micro_batches:,} micro-batches of {a.micro_batch} pairs, {steps:,} optimizer steps, lr {a.lr}")
    order: list[int] = []
    rng = random.Random(a.seed)
    step = 0
    t0 = time.time()
    enc.model.train()
    for done, mb in enumerate(range(micro_batches), 1):
        if len(order) < a.micro_batch:
            fresh = list(range(len(train)))
            rng.shuffle(fresh)
            order += fresh
        idx, order = order[: a.micro_batch], order[a.micro_batch :]
        q_texts = [states[i] for i in idx]
        d_texts = [rng.choice(positives[i]) for i in idx]
        zq = enc.encode(q_texts, a.query_tokens, train=True)
        zd = enc.encode(d_texts, a.doc_tokens, train=True)
        loss = infonce(zq @ zd.T, d_texts, a.tau) / a.accumulate
        loss.backward()
        if done % a.accumulate == 0 or mb == micro_batches - 1:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps, a.warmup, a.lr)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                log(
                    f"step {step}/{steps} loss {loss.item() * a.accumulate:.4f} lr {lr_at(step, steps, a.warmup, a.lr):.2e} {el / 60:.1f} min, eta {el / step * (steps - step) / 60:.0f} min"
                )
            if step % a.eval_every == 0 or step == steps:
                cur = score_dev()
                history.append({"step": step, **cur})
                better = cur["NDCG@10"] > best["NDCG@10"]
                log(f"dev at step {step}: {cur}{' (best, kept)' if better else ''}")
                if a.keep_all:
                    enc.model.save_pretrained(out / "checkpoints" / f"step-{step}")
                if better:
                    best = cur
                    enc.model.save_pretrained(out / "adapter")
    report = {
        "model": a.model,
        "pairs": a.pairs,
        "n_train": len(train),
        "dropped": dropped,
        "dev": dev.name,
        "best": best,
        "history": history,
        "config": {k: v for k, v in vars(a).items() if k not in ("pairs", "dev", "out")},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "train.json").write_text(json.dumps(report, indent=2))
    log(f"best dev {best}; adapter in {out / 'adapter'}")
    if a.no_merge:
        return 0
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    log("merging the best adapter into full weights")
    enc.model.cpu()  # make room: the merge loads a second copy of the base weights
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    copy_original_files(a.model, out / "merged")
    log(f"merged weights in {out / 'merged'}: serve them with the compose profile lora")
    return 0


if __name__ == "__main__":
    sys.exit(main())
