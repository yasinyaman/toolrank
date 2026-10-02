# Docker

Two images, both for `linux/amd64` and `linux/arm64`:

| Image | What it runs | Needs |
| --- | --- | --- |
| `ghcr.io/yasinyaman/toolrank` | `toolrank` (serve, ingest, search) with the packaged heads, Node.js and uv for stdio MCP servers | an embedding endpoint serving the backbone |
| `toolrank-vllm`, built from `deploy/docker/Dockerfile.vllm` | the same, plus vLLM serving the backbone inside the container | an NVIDIA GPU (16 GB; NVIDIA driver 580 or newer) |

`deploy/docker/` has a compose file for each: `compose.yaml` runs the official vLLM image and the
published `toolrank` image as two services, and `compose.bundle.yaml` builds `toolrank-vllm` and runs
it alone. The bundled image is not published: it is the official vLLM image with toolrank on top,
too large to build for two platforms on the free CI runners.

## Two services: vLLM and toolrank

```bash
cd deploy/docker
cp .env.example .env                                   # set TOOLRANK_API_KEY
docker compose run --rm toolrank ingest mcp --config /config/toolrank.json --out /data
docker compose up -d
```

```yaml
--8<-- "deploy/docker/compose.yaml:12:"
```

- `toolrank.json` lists the MCP servers, as an MCP client's config file does; stdio servers run
  inside the toolrank container (`uvx` and `npx` are there), HTTP ones anywhere it can reach.
- OpenAPI specs: `docker compose run --rm -v $PWD/specs:/specs:ro toolrank ingest openapi
  /specs/billing.yaml --name billing --out /data`, and their base URLs and headers go in the
  `openapi` section of `toolrank.json`.
- Re-run the ingest line after changing the list: the running server picks the new catalogue up.
- The catalogue, the embedding cache, the index and the usage log live in the `data` volume; the
  model in `models`; uvx and npx caches in `home`.
- toolrank runs as uid 1000 in the image: what you mount for it (`toolrank.json`, specs) must be
  readable to that uid, and a `/data` you bind-mount writable to it.

## One container: toolrank-vllm

```bash
cd deploy/docker
cp .env.example .env
docker compose -f compose.bundle.yaml run --rm toolrank ingest mcp --config /config/toolrank.json --out /data
docker compose -f compose.bundle.yaml up -d
```

The first command builds the image from your checkout: it pulls the official vLLM image (about 10
GB) and fetches the released heads, checking their sha256. Every command that embeds (serve, ingest,
search) starts vLLM on `127.0.0.1` inside the container,
waits for it, then runs toolrank; if either exits, the container exits, and the restart policy
brings both back. Other commands (`--version`, `heads`) run toolrank alone.

Inside the container vLLM runs as root, for the GPU and the model volume. toolrank and the stdio MCP
servers it starts never do: they run as the image's unprivileged `toolrank` user or, when the data
directory is one you bind-mounted, as that directory's owner, so your files stay yours. The data
directory is `/data`, or the `--data` you give `serve` and `search`, or a command's `--out`; one that
Docker created for you (owned by root) is handed to the `toolrank` user only at `/data`. Whatever you
mount for toolrank (`toolrank.json`, specs) must be readable to that user; toolrank says which uid it
runs as when it cannot read a config. `docker compose exec toolrank toolrank …` runs as that user too.
With `--user`, vLLM and toolrank both run as the user you name, and the model volume must be
writable to it. Under a rootless engine, a directory of yours is used as it is.

| Variable | Default | |
| --- | --- | --- |
| `TOOLRANK_BACKBONE`, `TOOLRANK_BACKBONE_REVISION` | `yasinyaman/toolrank-emb-8b`, `v0.2` | the weights; `Qwen/Qwen3-Embedding-8B` for the base model (served as `qwen3-emb`, with the packaged heads) |
| `TOOLRANK_FP8` | `1` | FP8 weights, served as `toolrank-emb-v0.2-fp8`; `0` for bf16, served as `toolrank-emb-v0.2` |
| `VLLM_GPU_MEMORY_UTILIZATION` | vLLM's | the share of GPU memory vLLM may take |
| `VLLM_EXTRA_ARGS` | | more `vllm serve` flags |

## FP8 or bf16

Both setups serve the backbone in FP8 (the bf16 weights quantized as vLLM loads them): half the
weight memory, and within a point of bf16 in our runs (see [Benchmarks](../benchmarks.md)).
FP8 and bf16 are served under different names (`toolrank-emb-v0.2-fp8`, `toolrank-emb-v0.2`), which keeps their
vectors apart in the embedding cache: switching re-embeds the catalogue once.

## Settings

| Variable | Used by | |
| --- | --- | --- |
| `TOOLRANK_API_KEY` | serve | the bearer token; required on `0.0.0.0` |
| `TOOLRANK_EMB_URL`, `TOOLRANK_EMB_MODEL` | serve, ingest, search | the embedding endpoint and its served name |
| `TOOLRANK_EMB_API_KEY` | serve, ingest, search | a bearer token for the embedding endpoint, if it needs one (`OPENAI_API_KEY` is sent only to `api.openai.com`) |
| `TOOLRANK_ALLOWED_HOSTS` | serve | names the server answers to besides localhost (the compose service, a proxy's host) |
| `TOOLRANK_HEADS` | serve, search | a heads file other than the packaged one |

The compose files publish the port on `127.0.0.1` only, and the server speaks plain HTTP: to reach
it from another machine, put a reverse proxy that terminates TLS in front rather than publishing
the port. Add the proxy's host name to `TOOLRANK_ALLOWED_HOSTS`: a name without a port also matches
the requests a proxy on 80 or 443 forwards.

## GPU memory

The model takes about 8 GB in FP8 and 16 GB in bf16. vLLM reserves `VLLM_GPU_MEMORY_UTILIZATION` of
the GPU (0.9 by default) for the weights and its cache; on a GPU shared with other services, lower it
(0.2 on a 120 GB NVIDIA GB10).

## Building the images

From the repository root:

```bash
docker build -f deploy/docker/Dockerfile --build-context heads=dist/heads -t toolrank .
docker build -f deploy/docker/Dockerfile.vllm -t toolrank-vllm .
```

`--build-context heads=DIR` bakes in the released heads found in `DIR` (`toolrank-heads-*.npz`).
Without it, the `toolrank` image serves the backbone alone until `toolrank heads pull`, and
`toolrank-vllm` downloads the released heads while it builds (`--build-arg HEADS_PULL=0` skips that).
`scripts/container_smoke.py IMAGE` checks an image without a GPU, against a fake embedding server.
