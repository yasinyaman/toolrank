#!/usr/bin/env bash
# Cross-encoders in Jev's role (`docs/reports/faz2-jev.md`): Qwen3-Reranker-8B (8095) and
# bge-reranker-v2-gemma (8096), served by vLLM's score API (compose profile rerank), reorder the
# packaged heads' top 20 (documentation cut to 3000) and top 100 (name_desc cut to 1000) and BM25's
# top 30: exactly the texts the Jev rows read. On LiveMCPBench the cross-encoder also runs alone
# (every pair). Scores are cached in .cache/toolrank/scores.sqlite. Restartable. On the GB10:
#
#   docker compose -f deploy/spark/compose.yaml --profile rerank up -d qwen3-reranker bge-reranker
#   PYTHONUNBUFFERED=1 nohup bash scripts/cross_rerank.sh > data/logs/cross_rerank.log 2>&1 &
#   # then, on the Mac: scp 'gb10:toolrank/results/cross_*.json' results/
set -uo pipefail
cd "$(dirname "$0")/.."

UV=${UV:-$HOME/.local/bin/uv}
HEADS=${HEADS:-dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz}
EMB_URL=${EMB_URL:-http://127.0.0.1:8091/v1}
EMB_MODEL=${EMB_MODEL:-qwen3-emb}
TAG=${TAG:-cross}
LIMIT=${LIMIT:-0}
MODELS=${MODELS:-"qwen3 bge"}         # qwen3 = Qwen3-Reranker-8B on 8095, bge = bge-reranker-v2-gemma on 8096
ROWS=${ROWS:-"heads_x20doc bm25_x30 heads_x100 alone"}
SETS=${SETS:-"livemcpbench_server mcp_zero_server toolret"}
BM25="--scorer bm25 --no-stem --tool-format documentation --with-inst"
EMB="--emb-url $EMB_URL --emb-model $EMB_MODEL --truncate 8192 --emb-batch 128"
EMB="$EMB --tool-format documentation --query-format instruct_query --with-inst"
LIM=()
[[ $LIMIT -gt 0 ]] && LIM=(--limit "$LIMIT")

url() { # url <model> -> the score endpoint
  case $1 in
    qwen3) echo http://127.0.0.1:8095 ;;
    bge) echo http://127.0.0.1:8096 ;;
    *) echo "unknown model $1" >&2; exit 1 ;;
  esac
}
served() { [[ $1 == qwen3 ]] && echo qwen3-reranker || echo bge-reranker; }
endpoint() { echo "--rerank-emb-url $(url $1) --rerank-emb-model $(served $1) --rerank-template $1"; }

run() { # run <row> <set> <toolrank eval args...> -> results/<TAG>_<set>_<row>.json
  local row=$1 out="results/${TAG}_$2_$1.json"
  shift 2
  [[ " $ROWS " == *" ${row%%_*}_${row#*_} "* || " $ROWS " == *" ${row#*_} "* ]] || return 0
  if [[ -s "$out" ]]; then
    echo "skip $out"
    return
  fi
  echo "== $(date -Is) $out"
  "$UV" run toolrank eval "$@" "${LIM[@]}" --out "$out" || echo "FAILED $out"
}

for m in $MODELS; do
  X="--rerank cross $(endpoint $m) --rerank-workers ${WORKERS:-8}"
  for d in $SETS; do
    KS=()
    [[ $d == mcp_zero_server ]] && KS=(--ks 1,5,10,20)
    run "${m}_heads_x20doc" "$d" --data "data/$d" --scorer clm --clm-ckpt "$HEADS" $EMB "${KS[@]}" \
      $X --rerank-depth 20 --rerank-tool-format documentation --rerank-max-chars 3000
    run "${m}_bm25_x30" "$d" --data "data/$d" $BM25 "${KS[@]}" \
      $X --rerank-depth 30 --rerank-tool-format name_desc --rerank-max-chars 1000
    run "${m}_heads_x100" "$d" --data "data/$d" --scorer clm --clm-ckpt "$HEADS" $EMB "${KS[@]}" \
      $X --rerank-depth 100 --rerank-tool-format name_desc --rerank-max-chars 1000
    if [[ $d == livemcpbench_server ]]; then
      run "${m}_alone" "$d" --data "data/$d" --scorer cross --with-inst --tool-format name_desc \
        --emb-url "$(url $m)" --emb-model "$(served $m)" --cross-template "$m" --emb-batch 64
    fi
  done
done
echo "== $(date -Is) done"
