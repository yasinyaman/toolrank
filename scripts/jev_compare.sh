#!/usr/bin/env bash
# Jev (TypeSafe AI) next to toolrank's own scorers, same protocol, same sets, same tool text
# (`docs/reports/faz1-jev.md`). Per set: the packaged heads reranked by Jev over the top 100
# (name_desc option text) and over the top 20 (the full documentation, cut to 3000 characters),
# BM25 top 30 reranked by Jev (TypeSafe's own re-ranking recipe), zero-shot Qwen3-Embedding-8B
# reranked the same two ways (is the fine-tune still needed under Jev?), and on the MCP sets Jev
# alone (chunked Choice questions). The base rows (bm25, qwen3emb, heads) are `readme_results.sh`'s.
# Restartable: a run whose results file exists is skipped; Jev answers are cached in
# .cache/toolrank/jev.sqlite, so a rerun asks nothing. On the GB10, from ~/toolrank, 8091 up:
#
#   export TYPESAFE_API_KEY=...          # the key never leaves the shell; .env is not synced
#   LIMIT=50 TAG=jevsmoke bash scripts/jev_compare.sh      # the smoke test: a few cents
#   PYTHONUNBUFFERED=1 nohup bash scripts/jev_compare.sh > data/logs/jev_compare.log 2>&1 &
#   # then, on the Mac: scp 'gb10:toolrank/results/jev_*.json' results/
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -n ${TYPESAFE_API_KEY:-} ]] || { echo "set TYPESAFE_API_KEY" >&2; exit 1; }

UV=${UV:-$HOME/.local/bin/uv}
HEADS=${HEADS:-dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz}
EMB_URL=${EMB_URL:-http://127.0.0.1:8091/v1}
EMB_MODEL=${EMB_MODEL:-qwen3-emb}
TAG=${TAG:-jev}
LIMIT=${LIMIT:-0}
ROWS=${ROWS:-"heads_jev100 heads_jev20doc bm25_jev30 jev qwen3emb_jev100 qwen3emb_jev20doc"}
SETS=${SETS:-"toolret livemcpbench_server mcp_zero_server"}
JEV="--jev-model ${JEV_MODEL:-jev-1.13.0} --jev-workers ${JEV_WORKERS:-8}"
BM25="--scorer bm25 --no-stem --tool-format documentation --with-inst"
EMB="--emb-url $EMB_URL --emb-model $EMB_MODEL --truncate 8192 --emb-batch 128"
EMB="$EMB --tool-format documentation --query-format instruct_query --with-inst"
LIM=()
[[ $LIMIT -gt 0 ]] && LIM=(--limit "$LIMIT")

for d in livemcpbench mcp_zero; do
  [[ -s data/${d}_server/tools.jsonl ]] || "$UV" run toolrank data server-names "data/$d"
done

run() { # run <row> <set> <toolrank eval args...> -> results/<TAG>_<set>_<row>.json
  local row=$1 out="results/${TAG}_$2_$1.json"
  shift 2
  [[ " $ROWS " == *" $row "* ]] || return 0
  if [[ -s "$out" ]]; then
    echo "skip $out"
    return
  fi
  echo "== $(date -Is) $out"
  "$UV" run toolrank eval "$@" "${LIM[@]}" --out "$out" || echo "FAILED $out"
}

for d in $SETS; do
  KS=()
  [[ $d == mcp_zero_server ]] && KS=(--ks 1,5,10,20) # Precision@1 = MCP-Zero's top-1 accuracy
  run heads_jev100 "$d" --data "data/$d" --scorer clm --clm-ckpt "$HEADS" $EMB "${KS[@]}" \
    --rerank jev --rerank-depth 100 $JEV
  run heads_jev20doc "$d" --data "data/$d" --scorer clm --clm-ckpt "$HEADS" $EMB "${KS[@]}" \
    --rerank jev --rerank-depth 20 --jev-tool-format documentation --jev-max-chars 3000 $JEV
  run bm25_jev30 "$d" --data "data/$d" $BM25 "${KS[@]}" --rerank jev --rerank-depth 30 $JEV
  # zero-shot Qwen3-Embedding-8B under Jev: does the fine-tune still matter once Jev reorders the top?
  run qwen3emb_jev100 "$d" --data "data/$d" --scorer dense $EMB "${KS[@]}" --rerank jev --rerank-depth 100 $JEV
  run qwen3emb_jev20doc "$d" --data "data/$d" --scorer dense $EMB "${KS[@]}" \
    --rerank jev --rerank-depth 20 --jev-tool-format documentation --jev-max-chars 3000 $JEV
  [[ $d == toolret ]] && continue # 44k tools: Jev alone is not feasible there
  run jev "$d" --data "data/$d" --scorer jev --with-inst --tool-format name_desc "${KS[@]}" \
    --jev-chunk 200 --jev-per-chunk 20 $JEV
done
echo "== $(date -Is) done"
