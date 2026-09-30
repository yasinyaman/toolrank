# Fine-tune the heads

The packaged heads were trained on ToolRet's public request-to-tool pairs. With pairs from your own
agents, `toolrank finetune` trains heads for your catalogue; for what `toolrank serve` logged, use
[`toolrank learn`](learn.md), which needs no pairs file.

```bash
pip install "toolrank[clm]"           # torch, for training only; the result runs in numpy
toolrank finetune --data pairs.jsonl --dev dev/ --eval test/ --out heads.pt --npz heads.npz
```

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
   heads.npz` and `serve` need no other flag.

`--init-ckpt default` continues from the packaged heads instead of the identity.

## Mined negatives

`--neg 15` adds the dataset's mined negatives per pair. On ToolRet's pairs they raised recall on
held-out pairs from 89.5 to 96.9 while the dev set fell from 87.2 to 78.7 NDCG@10; the command kept
the starting heads. Try them only with a dev set you trust.
