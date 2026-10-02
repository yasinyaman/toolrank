#!/usr/bin/env bash
# The toolrank-vllm image's entrypoint. For commands that embed (serve, ingest, search, eval,
# finetune) it starts vLLM with the embedding backbone on 127.0.0.1, waits until it answers, then
# runs toolrank with the container's arguments; when either process exits, the other is stopped and
# the container exits with that status, so the runtime's restart policy takes over. Other commands
# (--version, heads, formats) run toolrank alone.
#
# vLLM runs as the container's user (root by default: the GPU and /models). `toolrank` in this image
# is a wrapper that never runs it as root (as-toolrank.sh: the owner of /data, else the image's
# `toolrank` user), so the stdio MCP servers toolrank starts are not root either.
#
#   TOOLRANK_BACKBONE=REPO           the weights (default: toolrank's LoRA-trained Qwen3-Embedding-8B,
#   TOOLRANK_BACKBONE_REVISION=TAG   yasinyaman/toolrank-emb-8b at v0.2; Qwen/Qwen3-Embedding-8B for
#                                    the base model, which the packaged heads go with)
#   TOOLRANK_FP8=1|0                 FP8 weights (quantized at load, served as <name>-fp8) or bf16
#                                    (<name>): the names keep the two apart in the embedding cache;
#                                    <name> is toolrank-emb-v0.2, qwen3-emb for the base model, else
#                                    the repo id with / as -
#   VLLM_GPU_MEMORY_UTILIZATION=0.9  the share of GPU memory vLLM may take
#   VLLM_EXTRA_ARGS="..."            more `vllm serve` flags
#   TOOLRANK_API_KEY                 required by `serve` on 0.0.0.0
#   TOOLRANK_ENTRYPOINT_DRY_RUN=1    print the two command lines and exit
set -euo pipefail

case "${1:-}" in
  serve | ingest | search | eval | finetune) ;;
  *) exec toolrank "$@" ;;
esac

default_repo=yasinyaman/toolrank-emb-8b default_revision=v0.2 # build.BACKBONE_REPO / _REVISION
model=${TOOLRANK_BACKBONE:-$default_repo}
case "$model" in
  "$default_repo") name=toolrank-emb-$default_revision revision=${TOOLRANK_BACKBONE_REVISION:-$default_revision} ;;
  Qwen/Qwen3-Embedding-8B) name=qwen3-emb revision=${TOOLRANK_BACKBONE_REVISION:-} ;;
  *) name=${model//\//-} revision=${TOOLRANK_BACKBONE_REVISION:-} ;;
esac
port=${VLLM_PORT:-8091}
args=(serve "$model" --runner pooling --max-model-len 8192 --no-enable-chunked-prefill
  --max-num-batched-tokens 8192 --host 127.0.0.1 --port "$port")
[[ -n "$revision" ]] && args+=(--revision "$revision")
if [[ "${TOOLRANK_FP8:-1}" == 1 ]]; then
  served=${TOOLRANK_EMB_MODEL:-$name-fp8}
  args+=(--quantization fp8)
else
  served=${TOOLRANK_EMB_MODEL:-$name}
fi
args+=(--served-model-name "$served")
if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  args+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi
read -r -a extra <<<"${VLLM_EXTRA_ARGS:-}"
args+=(${extra[@]+"${extra[@]}"})
export TOOLRANK_EMB_URL="http://127.0.0.1:$port/v1" TOOLRANK_EMB_MODEL="$served"

if [[ -n "${TOOLRANK_ENTRYPOINT_DRY_RUN:-}" ]]; then
  echo "vllm ${args[*]}"
  echo "TOOLRANK_EMB_URL=$TOOLRANK_EMB_URL TOOLRANK_EMB_MODEL=$TOOLRANK_EMB_MODEL toolrank $*"
  exit 0
fi

vllm "${args[@]}" &
vllm_pid=$!
toolrank_pid=
trap 'kill -TERM $vllm_pid $toolrank_pid 2>/dev/null || true' TERM INT
echo "toolrank-vllm: waiting for vLLM ($model as $served) on 127.0.0.1:$port" >&2
until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$port/health', timeout=2)" 2>/dev/null; do
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    wait "$vllm_pid" || exit $?
    exit 1
  fi
  sleep 2
done

toolrank "$@" &
toolrank_pid=$!
set +e
wait -n "$vllm_pid" "$toolrank_pid"
status=$?
kill -TERM "$vllm_pid" "$toolrank_pid" 2>/dev/null
wait
exit "$status"
