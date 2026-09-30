# syntax=docker/dockerfile:1
# toolrank without a model: the MCP proxy and REST API (`toolrank serve`) and ingestion, talking to
# any OpenAI-compatible endpoint that serves Qwen3-Embedding-8B (compose.yaml pairs it with vLLM).
# Node.js and uv come along, so MCP servers started with npx or uvx run inside the container.
#
# From the repo root; `--build-context heads=DIR` bakes in the released heads found in DIR
# (toolrank-heads-*.npz; without it the image serves the backbone alone until `toolrank heads pull`):
#   docker build -f deploy/docker/Dockerfile --build-context heads=dist/heads -t toolrank .

# The base images, each named once by tag and digest: a tag can be moved to other content, a digest
# cannot. They sit on FROM lines because Dependabot reads nothing else (it moves tag and digest).
# Each build argument defaults to its pinned stage and takes an image name instead, e.g. from a
# registry mirror:  --build-arg PYTHON_IMAGE=mirror.example.com/python:3.12-slim-bookworm
ARG PYTHON_IMAGE=python-pinned
ARG NODE_IMAGE=node-pinned
ARG UV_IMAGE=uv-pinned

FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS python-pinned
FROM node:24-bookworm-slim@sha256:0e0ff40c39bc087845bfb27465a0df4ea419520094bc35842ff83dd8cbe6f9b6 AS node-pinned
FROM ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 AS uv-pinned

FROM ${PYTHON_IMAGE} AS base
FROM ${NODE_IMAGE} AS node
FROM ${UV_IMAGE} AS uv

FROM scratch AS heads

# only released heads files, whatever else the heads context holds
FROM base AS released-heads
# a bind mount, not COPY: BuildKit hands a cached COPY from the empty stage to a build with the
# heads context and the other way round; a bind mount's cache follows what is mounted
RUN --mount=type=bind,from=heads,target=/context \
    mkdir /heads && find /context -maxdepth 1 -name 'toolrank-heads-*.npz' -exec cp {} /heads/ \;

FROM base AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
# the versions locked in uv.lock (the ones THIRD_PARTY_NOTICES.md lists), each checked against the
# lock's hashes, then toolrank itself;
# compiled here, since the non-root user cannot write __pycache__ next to them
RUN uv export --frozen --no-emit-project --extra mcp --extra openapi --extra stem -o /requirements.txt \
    && uv build --wheel -o /wheels \
    && uv venv --python /usr/local/bin/python3 /opt/toolrank/venv \
    && uv pip install --python /opt/toolrank/venv/bin/python --compile-bytecode --require-hashes -r /requirements.txt \
    && uv pip install --python /opt/toolrank/venv/bin/python --compile-bytecode --no-deps /wheels/toolrank-*.whl

FROM base
RUN apt-get update && apt-get install -y --no-install-recommends tini && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && useradd --create-home --uid 1000 toolrank \
    && mkdir -p /data /opt/toolrank/heads \
    && chown toolrank /data /opt/toolrank /opt/toolrank/heads
COPY --from=build /opt/toolrank/venv /opt/toolrank/venv
COPY --from=released-heads --chown=toolrank /heads/ /opt/toolrank/heads/
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /usr/share/doc/toolrank/

# TOOLRANK_CACHE: the heads' home (baked in, or `toolrank heads pull`); the data dir keeps the
# catalogue, embedding cache, index and usage log. Set TOOLRANK_API_KEY and TOOLRANK_EMB_URL, and
# TOOLRANK_ALLOWED_HOSTS for names other than localhost (a compose service, a proxy's host).
ENV PATH=/opt/toolrank/venv/bin:$PATH \
    TOOLRANK_CACHE=/opt/toolrank \
    PYTHONUNBUFFERED=1
USER toolrank
WORKDIR /data
VOLUME /data
EXPOSE 8765
# healthy once the semantic index is up (a BM25 stand-in answers before that)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --start-interval=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=4)"]
# tini reaps what stdio MCP servers (npx, uvx) leave behind
ENTRYPOINT ["tini", "--", "toolrank"]
CMD ["serve", "--data", "/data", "--host", "0.0.0.0"]
