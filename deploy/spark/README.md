# DGX Spark — vLLM pooling servers

Embedding servers, all OpenAI-compatible (`/v1/embeddings`, `/v1/models`):

| Port | Model | Purpose | Notes |
| --- | --- | --- | --- |
| 8090 | `Qwen/Qwen3-8B` | CLM backbone (last-token pooling) | `--max-model-len 2048` is the CLM reference; raise to 8192 for long conversation states (the `deepswe-clm-heads-8k` head was trained that way) |
| 8091 | `Qwen/Qwen3-Embedding-8B` | the serving backbone | `--max-model-len 8192` (the model allows 32k); instruction on the query side (`instruct_query` format) |
| 8094 | `Qwen/Qwen3-Embedding-8B`, FP8 | the same, quantized at load (profile `fp8`) | served as `qwen3-emb-fp8` |

## Install

```bash
# on the Spark, as the user that runs vLLM
vllm --version                 # the units below use `--runner pooling`; on vLLM < 0.10 replace with `--task embed`
sudo cp deploy/spark/*.service /etc/systemd/system/
sudo sed -i "s#__VLLM__#$(which vllm)#; s#__USER__#$USER#" /etc/systemd/system/vllm-qwen3-8b-pooling.service /etc/systemd/system/vllm-qwen3-embedding-8b.service
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-qwen3-8b-pooling vllm-qwen3-embedding-8b
curl -s http://127.0.0.1:8090/v1/models | head -c 300
curl -s http://127.0.0.1:8091/v1/models | head -c 300
```

Memory: bf16 weights are ~16 GB per 8B model; `--gpu-memory-utilization` is a fraction of the
GB10's unified memory, so keep the two pooling servers plus your generation server under 1.0 in
total. Start with 0.2 each and raise only if vLLM reports too little KV space.

## Install without a host vLLM (the GB10: NGC container)

The GB10 (`ssh gb10`; Ubuntu 24.04 aarch64, CUDA 13) has no host `vllm`, so the units above have
nothing to point `__VLLM__` at. vLLM runs there from NVIDIA's NGC image
(`nvcr.io/nvidia/vllm:26.01-py3`, vLLM 0.13, which has `--runner pooling`). `compose.yaml` starts
the same two servers with the same flags; the embedding server waits until the backbone is
healthy so the two startup memory checks don't race. Check `docker ps` for other vLLM containers
first and keep the sum of their `--gpu-memory-utilization` under 1.0.

```bash
# on the GB10, from ~/toolrank; the first start downloads ~16 GB of weights per model
docker compose -f deploy/spark/compose.yaml up -d
docker compose -f deploy/spark/compose.yaml ps          # both "healthy" when ready
curl -s http://127.0.0.1:8090/v1/models | head -c 300
curl -s http://127.0.0.1:8091/v1/models | head -c 300
```

An FP8 copy of the backbone (weights quantized at load, 7.6 GiB instead of 14.1) is off by
default; `docker compose -f deploy/spark/compose.yaml --profile fp8 up -d qwen3-8b-fp8` serves it
on 8092 as `qwen3-8b-fp8`. In week 2 it kept the CLM heads' outputs at cosine ≥ 0.997 to bf16 and
cut batch-1 latency from 86 to 49 ms.

The same profile has an FP8 copy of the embedding model, `qwen3-embedding-8b-fp8`, on 8094 as
`qwen3-emb-fp8`. On all three benchmarks it is within a query or two of bf16 (with and without the
heads), at 7.6 GiB of weights and 55 ms batch-1 latency instead of 99; the Docker images serve
it by default. Keep the two served names apart: the embedding cache is keyed by the name, not the
dtype.

A chat copy of the backbone (the default generate runner, same weights) writes MCP-Zero's queries
and is off by default too: `docker compose -f deploy/spark/compose.yaml --profile gen up -d
qwen3-8b-chat` serves it on 8093 as `qwen3-8b-chat`; `docker compose -f deploy/spark/compose.yaml
stop qwen3-8b-chat` frees its memory afterwards.

Postgres 17 with pgvector for the pgvector index (tests and trials only; bound to 127.0.0.1) is the
`pg` profile: `docker compose -f deploy/spark/compose.yaml --profile pg up -d toolrank-pg`, then
`export TOOLRANK_PG_DSN=postgresql://toolrank:toolrank@127.0.0.1:55440/toolrank`. It keeps its data in
the `toolrank-pg` volume; another project's pgvector container on this host (port 55433) is separate.

For a faster first start, download both models in parallel before `up -d` (compose fetches them
one after the other):

```bash
cd ~/toolrank && .venv/bin/hf download Qwen/Qwen3-8B & .venv/bin/hf download Qwen/Qwen3-Embedding-8B & wait
```

What the first start on the GB10 showed (vLLM 0.13.0 in NGC 26.01):

- `--runner pooling` works; `--convert auto` resolves to `embed`, and pooling models override
  `cudagraph_mode` to `PIECEWISE` (a harmless warning). Both servers pool the LAST token and
  return unit-norm vectors.
- ~90 s to healthy from a warm Hugging Face cache: 14.1 GiB of weights and 5.5–6.3 GiB of KV
  cache at 0.2 utilization, ~21 GB per server in `nvidia-smi`.
- huggingface_hub 1.x keeps blobs in a shared `hub/blobs/` tree behind relative symlinks; they
  resolve inside the container because the whole `~/.cache/huggingface` is mounted.
- Chunked prefill hangs pooling on this vLLM: a request split across steps next to many short
  ones stays "Running" with no progress, the engine stops answering and clients wait out their
  timeout (reproduced with one 128-text ToolRet batch). Both servers therefore run with
  `--no-enable-chunked-prefill --max-num-batched-tokens 8192` (the budget must be at least
  `--max-model-len`); the larger budget also lifted throughput from ~3.7k to ~4.3k tokens/s.
- Vectors are not bit-reproducible: the same batch encoded twice differs by up to ~3e-3 per
  component (GPU nondeterminism, also batch composition), with prefix caching on or off.
  toolrank's embedding cache is what keeps numbers stable.
- Batch-1 embedding latency is ~67 ms p50 for either 8B model (memory-bandwidth bound in bf16).

## Smoke test from the Mac (Tailscale)

MagicDNS is off, so use the GB10's Tailscale IP (`export GB10=<that address>`; its `.local` name also
works on the LAN):

```bash
curl -s http://$GB10:8090/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b","input":["What causes tides on Earth?"],"encoding_format":"float"}' | head -c 200
```

## CLM heads and clm-serve (optional, for the playground)

```bash
pip install contrastive-lm
mkdir -p ~/.cache/clm && curl -L -o ~/.cache/clm/CLM_v0.1-8B.pt \
  https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/resolve/main/CLM_v0.1-8B.pt
CLM_EMB_URL=http://127.0.0.1:8090/v1/embeddings clm-serve --port 8700    # web playground at :8700
```

toolrank itself does not need clm-serve: `CLMScorer` loads the `.pt` directly and calls
`/v1/embeddings` on port 8090.
