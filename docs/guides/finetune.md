# Fine-tune the heads

The packaged heads were trained on ToolRet's public request-to-tool pairs. With pairs from your own
agents, `toolrank finetune` trains heads for your catalogue; for what `toolrank serve` logged, use
[`toolrank learn`](learn.md), which needs no pairs file.

```bash
pip install "toolrank[clm]"           # torch, for training only; the result runs in numpy
toolrank finetune --data pairs.jsonl --dev dev/ --eval test/ --out heads.pt --npz heads.npz
```

It embeds with the base Qwen3-Embedding-8B served as `qwen3-emb` on port 8091 (`--emb-url`,
`--emb-model`), and the heads it trains belong on that backbone; their cfg records it.

- `pairs.jsonl` holds requests with the tools that serve them (`toolrank data pull toolret-train`
  writes one; the format is in `toolrank.datasets.jsonl`).
- `--dev` is a benchmark-format set (`tools.jsonl` + `queries.jsonl`) that picks the epoch and is
  reported nowhere else. Recall on held-out training pairs is no guide: it kept rising in our runs
  while the benchmarks fell.
- `--eval` sets (repeatable) are scored with `toolrank eval` on the saved heads at the end. A dev
  set that is also an eval set, or shares queries with one, is refused.

## What it does

1. Drops pairs without a positive, and pairs whose request matches a dev or eval query (counted per
   set).
2. Embeds only what the cache lacks. `--embed-only` stops here, without torch, and resumes where it
   left off.
3. Trains on the frozen backbone's vectors with the settings of the released heads: skip heads,
   lr 1e-5, batch 512, 5 epochs, in-batch negatives only.
4. Scores the dev set after every epoch, exactly as `toolrank eval` would. The starting heads
   compete too: a later epoch wins only if it is strictly better, so the command never returns
   heads worse than it started with.
5. Saves `heads.pt` (and `--npz heads.npz`) with their serving settings (backbone, text formats,
   truncation, instruction) and the dev result that chose them. `toolrank search --clm-ckpt
   heads.npz --emb-model qwen3-emb` and `serve` with the same flags take the rest from the file. Heads
   put on a backbone their cfg does not name stop search and serve when `--emb-model` was left to its
   default, and get a warning otherwise.

`--init-ckpt default` continues from the packaged heads instead of the identity.

## Mined negatives

`--neg 15` adds the dataset's mined negatives per pair. On ToolRet's pairs they raised recall on
held-out pairs from 89.5 to 96.9 while the dev set fell from 87.2 to 78.7 NDCG@10; the command kept
the starting heads. Try them only with a dev set you trust.

## A dev set of your own

A model picked on a benchmark cannot be reported on it. `toolrank data gen-queries` makes a
selection set from any catalogue instead: it samples tools evenly over the catalogue's sources and
has a chat model write one request per tool, in three styles (a person's task, an agent's note for
its next step, a goal in everyday words; `--styles situation` writes a problem without the
operation, the hardest to match). The model is told not to use the tool's name, and a request that
names it anyway is dropped.

```bash
toolrank data gen-queries --data tools/ --out dev/ --n 1000 \
  --exclude data/toolret --exclude data/livemcpbench_server     # queries it must not repeat
toolrank eval --data dev/ --scorer dense --with-inst ...       # rows per source and per style
```

`--gen-url` and `--gen-model` name an OpenAI-compatible chat endpoint (by default a local
`qwen3-8b-chat` on port 8093); answers are cached in `dev/generations.jsonl`, so a rerun writes only
what is new. The result is a benchmark-format directory for `finetune --dev`, `learn --dev`,
`toolrank eval` and `toolrank calibrate` ([confidence](serve.md)).

It is a selection set, not a benchmark: one tool is the answer to each request, so catalogues with
near-identical tools make some requests ambiguous, and the requests come from one model. One-tool
requests are also easy: on a catalogue of GitHub's and Stripe's APIs every backbone we have finds
the tool in its top five 95–99% of the time, so they catch a model that got worse (the v0.1 heads
cost the LoRA backbone 1–6 points there) and will not rank models a point apart.

`--tools-per-request 2` (or 3) writes tasks instead: each sampled tool gets related tools of its
source as partners, the model writes one task that needs all of them (or declines when they do not
belong together), and all of them are gold. These are much harder (the base model gets every tool
of a three-tool task into its top ten a third of the time) and they are where models differ: on
the same catalogue the LoRA backbone beats the base model by 12 NDCG@10 points on two-tool tasks
and by 20 on three-tool tasks. Make both kinds, and report on sets they share nothing with.
