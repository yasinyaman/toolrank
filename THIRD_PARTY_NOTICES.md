# Third-party notices

toolrank is licensed under the Apache License 2.0 (`LICENSE`); `NOTICE` credits the projects it
takes code or text from. This file lists what toolrank installs and runs alongside. It is generated
by `scripts/third_party.py`; each license is the one the package's own metadata states, so check a
package's distribution for its full terms.

## Python packages in the Docker images

`toolrank[mcp,openapi,stem]`, the versions locked in `uv.lock` (Linux):

| Package | Version | License |
| --- | --- | --- |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| attrs | 26.1.0 | MIT |
| bm25s | 0.3.11 | MIT License |
| cffi | 2.1.1 | MIT-0 |
| click | 8.5.0 | BSD-3-Clause |
| cryptography | 50.0.1 | Apache-2.0 OR BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| httpcore2 | 2.13.1 | BSD-3-Clause |
| httpx2 | 2.13.1 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| jsonschema | 4.26.0 | MIT |
| jsonschema-specifications | 2025.9.1 | MIT |
| mcp | 2.2.0 | MIT |
| mcp-types | 2.2.0 | MIT |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| opentelemetry-api | 1.45.0 | Apache-2.0 |
| pycparser | 3.0 | BSD-3-Clause |
| pydantic | 2.13.5 | MIT |
| pydantic-core | 2.46.5 | MIT |
| pyjwt | 2.15.1 | MIT |
| pystemmer | 3.1.0 | MIT, BSD |
| python-multipart | 0.0.32 | Apache-2.0 |
| pyyaml | 6.0.3 | MIT |
| referencing | 0.37.0 | MIT |
| rpds-py | 2026.6.3 | MIT |
| sse-starlette | 3.5.0 | BSD-3-Clause |
| starlette | 1.7.0 | BSD-3-Clause |
| truststore | 0.10.4 | MIT |
| typing-extensions | 4.16.0 | PSF-2.0 |
| typing-inspection | 0.4.4 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |

## Other optional extras

Installed only when asked for (`pip install "toolrank[<extra>]"`), never in the images:

| Extra | Packages | Licenses |
| --- | --- | --- |
| `clm` | torch | BSD-3-Clause and others (see PyTorch's `LICENSE`) |
| `data` | datasets, huggingface_hub | Apache-2.0 |
| `faiss` | faiss-cpu | MIT |
| `pgvector` | psycopg[binary], pgvector | LGPL-3.0-only (psycopg; used as a separate, replaceable library), MIT |
| `anthropic`, `openai` | anthropic, openai | MIT, Apache-2.0 |
| `langgraph`, `llamaindex` | langchain-core, llama-index-core | MIT, MIT |

## Models

- **Qwen/Qwen3-Embedding-8B** (Apache-2.0): the embedding backbone. vLLM downloads it at run time;
  no image or package includes its weights.
- **toolrank heads v0.1** (`toolrank-heads-qwen3-emb-8b-v0.1.npz`, Apache-2.0): trained on
  ToolRet-Training-20w, whose dataset card states no license; see `docs/heads/MODEL_CARD.md`.
  Images built with the heads (`--build-context heads=...`) include the file.

## Container images

- **toolrank** (`deploy/docker/Dockerfile`): based on `python:3.12-slim-bookworm`, whose Debian
  packages carry their own licenses (`/usr/share/doc/*/copyright` in the image). It adds Node.js
  (MIT; npm is Artistic-2.0) from the official `node` image, for MCP servers started with `npx`,
  and uv (Apache-2.0 or MIT), for those started with `uvx`.
- **toolrank-vllm** (`deploy/docker/Dockerfile.vllm`): based on `vllm/vllm-openai`: vLLM
  (Apache-2.0), PyTorch (BSD-3-Clause) and NVIDIA CUDA libraries under NVIDIA's CUDA license
  terms, on Ubuntu. Builds on NVIDIA's NGC vLLM image are for local use only and are not published.
