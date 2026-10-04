# On a laptop or a Mac

The default backbone, `toolrank-emb-v0.2`, is an 8B model, and [Docker](docker.md) serves it with vLLM
on a GPU of 16 GB (8 GB in FP8). Without such a GPU, [Ollama](https://ollama.com) runs GGUF builds of
the same model, at the same quality and more slowly, and of smaller Qwen3-Embedding models, which are
faster and score lower.

## Pick a model

Measured on a laptop with an RTX 3050 Ti (4 GB) and 14 GB of RAM, Ollama 0.35.1, `num_ctx` 8192. The
benchmarks run with instructions, as in the [README table](../benchmarks.md):

| Model (Ollama name) | GGUF | LiveMCPBench NDCG@10 | Recall@5 | ToolRet NDCG@10 (cat-macro) | One new search | 525 tools embedded in |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `toolrank-emb-v0.2-q4_k_m` | 5.0 GB | 55.75 | 52.22 | 59.50 (54.51) | 0.67 s | 5.4 min |
| `toolrank-emb-v0.2-q8_0` | 8.7 GB | 55.23 | 52.17 | | 0.78 s | 8.8 min |
| `qwen3-emb-4b-q4_k_m` | 2.5 GB | 48.71 | 45.79 | 50.41 (44.99) | 0.33 s | 2.8 min |
| `qwen3-emb-0.6b-q8_0` | 0.6 GB | 49.21 | 47.39 | 48.54 (43.14) | 0.05 s | 29 s |
| *vLLM, bf16, on a GPU* | | *55.74* | *52.06* | *58.90 (54.36)* | | |

- The LoRA backbone's GGUF builds score as vLLM does. Q4_K_M is level with it on LiveMCPBench and 0.6
  points above on ToolRet (7,961 requests). Q8_0 is half a point behind on LiveMCPBench, within that
  benchmark's noise (94 requests). Take Q4_K_M: it is smaller and faster.
- Qwen3-Embedding-0.6B and 4B are faster, and score about 7 points lower on LiveMCPBench and 9 to 11
  lower on ToolRet. They are still well ahead of BM25 (ToolRet 39.27, 36.41 cat-macro). The 0.6B's
  ToolRet row is the f16 build, which scores like Q8_0 on LiveMCPBench (49.12).
- **One new search** is the median time of a search for a request the server has not seen before:
  one request at a time, on LiveMCPBench, embedding included. A request seen before is answered from
  the embedding cache in a few milliseconds, whatever the model.
- On a 4 GB GPU the 8B builds run mostly on the CPU. A catalogue is embedded once: ingest stores the
  vectors and a restart reuses them. On this laptop the first ingest embeds about 100 tools a minute
  with the 8B, 200 with the 4B and 700 with the 0.6B, so 2,000 tools take about 20, 10 and 3 minutes.

## Set up

```bash
# the LoRA backbone as GGUF (TODO(launch): publish yasinyaman/toolrank-emb-8b-GGUF); hf: pip install huggingface_hub
hf download yasinyaman/toolrank-emb-8b-GGUF toolrank-emb-8b-v0.2-Q4_K_M.gguf --revision v0.2 --local-dir models
printf 'FROM ./models/toolrank-emb-8b-v0.2-Q4_K_M.gguf\nPARAMETER num_ctx 8192\n' > Modelfile
ollama create toolrank-emb-v0.2-q4_k_m -f Modelfile

export TOOLRANK_EMB_URL=http://127.0.0.1:11434/v1 TOOLRANK_EMB_MODEL=toolrank-emb-v0.2-q4_k_m
toolrank ingest mcp --server time="uvx mcp-server-time" --out tools/ --emb-url $TOOLRANK_EMB_URL
toolrank serve --data tools/
```

The small models come from Qwen's own GGUF repositories, in the same way:

```bash
hf download Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf --local-dir models
printf 'FROM ./models/Qwen3-Embedding-0.6B-Q8_0.gguf\nPARAMETER num_ctx 8192\n' > Modelfile
ollama create qwen3-emb-0.6b-q8_0 -f Modelfile
```

## Good to know

- **Context.** Ollama ignores vLLM's `truncate_prompt_tokens` (`--truncate`). It cuts a longer input
  at `num_ctx` instead, keeping the start, as vLLM does, so set `num_ctx` to 8192 as above. Ollama's
  own `qwen3-embedding:0.6b` and `:4b` work too, but with a context of 4096.
- **Heads.** The packaged heads take the vectors of Qwen3-Embedding-8B (4096 numbers). The names above
  skip them. For a model under another name, pass `--clm-ckpt none` to `search` and `serve`.
- **Names.** The embedding cache is keyed by the model's name, so give each build its own name (the
  quantization in it, as above). Two builds under one name would share vectors that differ.
- **llama.cpp's own server** (`llama-server --embeddings`) works for short texts. In embedding mode
  it keeps an output row for every input token, about 0.6 MB per token for Qwen3, so an 8192-token
  input needs about 5 GB of GPU memory. It also ignores truncation. In our test, a longer input than
  fit crashed the server instead of returning an error. Ollama plans its memory and offloads to the
  CPU instead.
- **Apple Silicon.** Ollama runs these models on the Mac's GPU. We have not measured them there.
