# toolrank heads v0.1 for Qwen3-Embedding-8B

Two small learned heads that sit on top of [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
sentence embeddings for tool retrieval. The state head reads the request, the action head reads a
tool; each computes `x + MLP(x)` (a *skip* head: 4096 → 1536 → 1536 → 4096, GELU, LayerNorm), then
L2-normalises, and a tool's score is the cosine of the two. The heads start as the identity, so
training only adds a correction on top of the base model.

Code, documentation and benchmarks: [github.com/yasinyaman/toolrank](https://github.com/yasinyaman/toolrank),
[yaman.dev/toolrank](https://yaman.dev/toolrank/); the package:
[`pip install toolrank`](https://pypi.org/project/toolrank/).

| | |
| --- | --- |
| File | `toolrank-heads-qwen3-emb-8b-v0.1.npz`, float16, 59.8 MB |
| sha256 | `f3c101251b9c23925e2925bc02c4492e6c3dea79bcfa1b56715affb20f9f72f0` |
| Parameters | 29.9M (both heads) |
| Format | numpy `.npz`, loaded with `allow_pickle=False`; runs without torch (`toolrank.adapters.heads_np.NumpyHeads`) |
| Source checkpoint | `qwen_full_skip_neg0_e5.pt` (torch), exported with `toolrank heads export --dtype float16` |
| License | Apache-2.0, like toolrank; see *Training data* for the data it was trained on |

## Serving (stored in the checkpoint's `cfg`, used as defaults by toolrank)

- Backbone: `Qwen/Qwen3-Embedding-8B`, last-token pooling, 4096-d, rows L2-normalised,
  `truncate_prompt_tokens` 8192. bf16 or FP8: with the backbone in FP8 (vLLM's `--quantization fp8`)
  every benchmark stays within a query or two of bf16 (ToolRet 53.94 / 47.27, LiveMCPBench 53.48,
  MCP-Zero 79.51).
- Tools: toolrank's `documentation` text. For ingested MCP and OpenAPI tools that is
  `{"server", "name", "description", "inputSchema"}` as JSON.
- Requests: `Instruct: {instruction}\nQuery: {request}` (`instruct_query`). The default instruction,
  chosen among three on LiveMCPBench and MCP-Zero, is
  `Given an agent's request for a tool, retrieve the MCP tool that fulfills it.`

## Training

- Data: 206K request–tool pairs from ToolRet-Training-20w
  (`mangopy/ToolRet-Training-20w`). Pairs whose request equals a ToolRet benchmark request were
  dropped.
- Objective: InfoNCE with in-batch negatives only. The dataset's mined negatives cost up to 10
  points and were not used.
- Optimisation: lr 1e-5, batch 512, 5 epochs; epoch 4 was picked on held-out training pairs.
- The backbone stays frozen. Its vectors come from a cache, so training takes minutes.

## Training data license

ToolRet-Training-20w states no license: its dataset card has no license field and no license
text. The ToolRet code repository (`mangopy/tool-retrieval-benchmark`) is Apache-2.0, but that does
not cover the data, and the data itself is drawn from earlier benchmarks with their own terms. The
toolrank maintainers publish these heads accepting that risk.

If you need clean data provenance, use Qwen3-Embedding-8B without heads (Apache-2.0; 1–3 points
lower) or train your own heads on your data.

## Results

All numbers are w/ inst, and each set is scored under its own protocol
([benchmarks](https://yaman.dev/toolrank/benchmarks/)):

| | ToolRet NDCG@10 (micro / cat-macro) | LiveMCPBench Recall@5 | MCP-Zero top-1 |
| --- | ---: | ---: | ---: |
| Qwen3-Embedding-8B | 51.11 / 46.54 | 50.82 | 78.19 |
| + these heads (torch `.pt`) | 54.03 / 47.14 | 53.03 | 79.87 |
| + these heads (this `.npz`, numpy) | 54.03 / 47.13 | 53.03 | 79.87 |

- ToolRet has 44,453 tools and 7,961 queries.
- LiveMCPBench has 525 tools and 94 tasks; the server name is in the tool text.
- MCP-Zero has 2,792 tools and one LLM-written request per tool; the server name is in the tool
  text.
- float16 against the torch checkpoint: projection cosine ≥ 0.999999, top-10 overlap ≥ 99.89% on
  all three sets.

## Use

```bash
toolrank search --data data/mytools "create an invoice for this customer" --clm-ckpt default --emb-model qwen3-emb
toolrank eval --data data/toolret --scorer clm --clm-ckpt path/to/toolrank-heads-qwen3-emb-8b-v0.1.npz \
  --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --with-inst
```

`--emb-model` names the served Qwen3-Embedding-8B. Without it, search and serve would use the default
backbone (`toolrank-emb-v0.2`), which these heads do not fit: they stop and say so (a backbone named with
`--emb-model`, and eval, get a warning). `--clm-ckpt default` reads `TOOLRANK_HEADS`, else `~/.cache/toolrank/heads/`, else downloads the file
and checks its sha256. The file is downloaded from https://huggingface.co/yasinyaman/toolrank-heads-qwen3-emb-8b/resolve/v0.1/toolrank-heads-qwen3-emb-8b-v0.1.npz.
`TOOLRANK_HEADS_URL` points the download at a mirror; the sha256 is checked all the same.

## Limitations

- English requests and tool texts only.
- Trained on ToolRet's task mix. The gain is mostly on large ToolRet tasks and small elsewhere
  (out of domain +2.2 LiveMCPBench Recall@5, +1.7 MCP-Zero top-1, +0.2 LiveMCPBench NDCG@10).
- The ToolRet gate threshold (50 cat-macro) is not reached.
- The heads only fit Qwen3-Embedding-8B vectors. Another backbone needs its own heads.
