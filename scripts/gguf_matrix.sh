#!/usr/bin/env bash
# Backlog D1.2 / D1.3: Qwen3-Embedding GGUFs (and the LoRA backbone's) served by Ollama, scored like the
# README's vLLM rows: dense, documentation + instruct_query, w/ inst, --truncate 8192. Restartable: a
# result that exists is skipped. From the repo root of the machine that runs the evals:
#
#   EMB_URL=http://$LAPTOP:11434/v1 UV=~/.local/bin/uv bash scripts/gguf_matrix.sh
#   SETS=toolret MODELS="qwen3-emb-0.6b-f16" EMB_URL=... bash scripts/gguf_matrix.sh
#
# MODELS are Ollama model names, each created with `PARAMETER num_ctx 8192` (Ollama ignores vLLM's
# truncate_prompt_tokens and cuts at num_ctx instead, keeping the head like vLLM). SETS are data dirs,
# TAG the results' prefix (results/<TAG>_<set>_<model>.json).
set -u
EMB_URL=${EMB_URL:?set EMB_URL to the Ollama endpoint, e.g. http://\$LAPTOP:11434/v1}
MODELS=${MODELS:-"qwen3-emb-0.6b-f16 qwen3-emb-0.6b-q8_0 qwen3-emb-4b-q4_k_m qwen3-emb-4b-q8_0 qwen3-emb-8b-q4_k_m qwen3-emb-8b-q8_0 toolrank-emb-v0.2-q4_k_m toolrank-emb-v0.2-q8_0"}
SETS=${SETS:-"livemcpbench_server"}
TAG=${TAG:-gguf}
UV=${UV:-uv}

for set in $SETS; do
  for m in $MODELS; do
    out=results/${TAG}_${set}_${m}.json
    [ -s "$out" ] && continue
    echo "$(date -Is) == $out"
    $UV run toolrank eval --data "data/$set" --scorer dense --emb-url "$EMB_URL" --emb-model "$m" \
      --truncate 8192 --tool-format documentation --query-format instruct_query --with-inst \
      --out "$out" 2>&1 | tail -3 || echo "FAILED $out"
  done
done
echo "$(date -Is) gguf matrix done"
