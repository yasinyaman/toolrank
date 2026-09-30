#!/usr/bin/env bash
# Phase 0 week 2: zero-shot matrix on ToolRet.
#
# For every tool format: CLM (plain, and the `clm` state w/ inst), clm-raw (the same backbone
# without heads, same text) and Qwen3-Embedding-8B (plain, `instruct_query` w/ inst). Runs are
# sequential and restartable: a run whose results file exists is skipped, a failed run is retried
# on the next start. Formats go shortest first; each format's first run encodes the 44k-tool
# corpus into the embedding cache, the runs after it reuse it. On the GB10, from ~/toolrank, with
# both pooling servers up (deploy/spark/compose.yaml):
#
#   tmux new -d -s matrix 'bash scripts/run_matrix.sh 2>&1 | tee -a data/logs/matrix.log'
set -uo pipefail
cd "$(dirname "$0")/.."

UV=${UV:-$HOME/.local/bin/uv}
DATA=${DATA:-data/toolret}
FORMATS=${FORMATS:-"name_desc schema example_call documentation"}
# CLM backbone: the reference 2048-token setup; Qwen3-Embedding-8B: its server's --max-model-len.
CLM="--emb-url http://127.0.0.1:8090/v1 --emb-model qwen3-8b --truncate 2048 --emb-batch 128"
EMB="--emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 --emb-batch 128"

run() { # run <name> <toolrank eval args...> -> results/toolret_<name>.json
  local out="results/toolret_$1.json"
  shift
  if [[ -s "$out" ]]; then
    echo "skip $out"
    return
  fi
  echo "== $(date -Is) $out"
  local start=$SECONDS
  if "$UV" run toolrank eval --data "$DATA" "$@" --out "$out" | grep -E '\*\*(Avg|Cat-macro)\*\*|latency|saved'; then
    echo "   took $((SECONDS - start)) s"
  else
    echo "   FAILED after $((SECONDS - start)) s: $out"
  fi
}

mkdir -p results data/logs
for f in $FORMATS; do
  run "clm_${f}_plain" --scorer clm $CLM --tool-format "$f"
  run "clm_${f}_inst" --scorer clm $CLM --tool-format "$f" --with-inst
  run "clmraw_${f}_plain" --scorer dense $CLM --tool-format "$f"
  run "clmraw_${f}_inst" --scorer dense $CLM --tool-format "$f" --with-inst --query-format clm
  run "qwen3emb_${f}_plain" --scorer dense $EMB --tool-format "$f"
  run "qwen3emb_${f}_inst" --scorer dense $EMB --tool-format "$f" --with-inst
done
echo "== $(date -Is) matrix done"
