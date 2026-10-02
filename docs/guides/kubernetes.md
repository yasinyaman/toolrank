# Kubernetes

The chart in `deploy/helm/toolrank` runs `toolrank serve` with its embedding backbone. It is not
published to a chart repository yet: install it from a checkout.

```bash
helm install toolrank deploy/helm/toolrank -n toolrank --create-namespace \
  --set auth.apiKey="$TOOLRANK_API_KEY"
kubectl -n toolrank port-forward svc/toolrank 8765:8765
```

## The backbone

`embedding.mode` picks where Qwen3-Embedding-8B runs:

| Mode | What runs | Needs |
| --- | --- | --- |
| `vllm` (default) | a vLLM pod next to toolrank, weights on their own volume | a GPU node (`nvidia.com/gpu`) |
| `external` | nothing: `embedding.url` and `embedding.model` name your endpoint | an OpenAI-compatible `/v1/embeddings` |
| `bundled` | one pod with the `toolrank-vllm` image | a GPU node, and the image built from `deploy/docker/Dockerfile.vllm` and pushed to your registry (`embedding.bundled.image`) |

`embedding.profile` sets the weights: `fp8` (the default, quantized at load, about 8 GB, served as
`qwen3-emb-fp8`) or `bf16` (about 16 GB, `qwen3-emb`). In our runs FP8 ranks like bf16 within a
query or two per benchmark. The embedding cache is keyed by the served name, so switching profiles
re-embeds the catalogue once. The first start of a vLLM pod downloads 16 GB; the probes allow 30
minutes for it.

## Your tools

`config` is the `toolrank.json` of [Serve them to agents](serve.md#the-config-file): MCP servers
and OpenAPI base URLs and headers. An init container runs `toolrank ingest mcp` on every start, so
`helm upgrade` with a new server rolls the pod and the server appears in the catalogue. Helm merges
your values into the chart's, so drop the example server explicitly (`mcpServers: {time: null}`).
Credentials go in a Secret listed under `envFrom` and are referred to as `${VAR}` in the config.
OpenAPI specs are ingested with `ingest.args`, or with `kubectl exec` into the pod: the server
re-reads the catalogue without a restart.

## Keys and tenants

At least one key is required, since the server listens on the pod network: `auth.apiKey` (or
`auth.apiKeySecret`) for one token, `auth.apiKeys` (or `auth.apiKeysSecret`, a Secret holding the
file) for [tenants](serve.md#tenants), each with its own sources and credentials. The Service's
names (`toolrank`, `toolrank.<namespace>.svc`, ...) are allowed as Host; add an ingress host to
`serve.allowedHosts`. Terminate TLS at the ingress.

## What it creates

One toolrank Deployment (one replica, `Recreate`: the index snapshot, the usage log and the learned
heads live on one `ReadWriteOnce` volume), its Service, a ConfigMap, a Secret when the keys are
given inline, the data volume (and a weights volume for `vllm` and `bundled`). The volumes are kept
on `helm uninstall`: they hold the embedding cache and the usage log. `serve.args` passes further
flags (`--server-weight 0.2`, `--co-use 2`, `--allow-write`). Prometheus scrapes `/v1/metrics` with
a key that reaches every source ([Metrics](serve.md#metrics)).

## Trying it

`scripts/helm_smoke.py` installs the chart on any cluster without a GPU: a fake embedding server in
the cluster, `external` mode, then a search, a call to a stdio MCP server inside the pod, a refused
token, the metrics, a key limited to one source, an upgrade that adds a server, and an uninstall
that keeps the data volume. On a laptop, [kind](https://kind.sigs.k8s.io/) gives a cluster in
Docker:

```bash
kind create cluster --name toolrank
uv run python scripts/helm_smoke.py                    # the published image
docker build -f deploy/docker/Dockerfile -t toolrank:dev . && kind load docker-image toolrank:dev --name toolrank
uv run python scripts/helm_smoke.py --image toolrank:dev --tenants
kind delete cluster --name toolrank
```

The GPU modes have been checked by rendering and by the API server's validation, not yet on a GPU
cluster.
