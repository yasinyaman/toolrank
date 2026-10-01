#!/usr/bin/env bash
# CLM in Jev's role (`docs/reports/faz2-jev.md`): the packaged heads' top 100 and top 20 reordered by
# CLM_v0.1-8B (Qwen3-8B pooling on 8090, example_call text, the `clm` query format: the setting of
# Faz 0's best CLM row) and by Faz 0's fine-tuned CLM heads; BM25's top 30 the same way. Local, free,
# and all cache hits once the Faz 0 matrix has run. Restartable: a run whose results file exists is
# skipped. On the GB10, from ~/toolrank, 8090 and 8091 up:
#
#   PYTHONUNBUFFERED=1 nohup bash scripts/clm_rerank.sh > data/logs/clm_rerank.log 2>&1 &
#   # then, on the Mac: scp 'gb10:toolrank/results/clm_*.json' results/
set -uo pipefail
cd "$(dirname "$0")/.."

UV=${UV:-$HOME/.local/bin/uv}
HEADS=${HEADS:-dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz}
CLM_CKPT=${CLM_CKPT:-$HOME/.cache/clm/CLM_v0.1-8B.pt}
CLM_FT=${CLM_FT:-data/heads/clm_60k_lr1e-2.pt}
CLM_FORMAT=${CLM_FORMAT:-example_call}
EMB_URL=${EMB_URL:-http://127.0.0.1:8091/v1}
EMB_MODEL=${EMB_MODEL:-qwen3-emb}
TAG=${TAG:-clm}
LIMIT=${LIMIT:-0}
ROWS=${ROWS:-"heads_clm100 heads_clm20 heads_clmft100 bm25_clm30"}
SETS=${SETS:-"toolret livemcpbench_server mcp_zero_server"}
BM25="--scorer bm25 --no-stem --tool-format documentation --with-inst"
EMB="--emb-url $EMB_URL --emb-model $EMB_MODEL --truncate 8192 --emb-batch 128"
EMB="$EMB --tool-format documentation --query-format instruct_query --with-inst"
CLM="--rerank clm --rerank-emb-url http://127.0.0.1:8090/v1 --rerank-emb-model qwen3-8b --rerank-truncate 2048"
CLM="$CLM --rerank-tool-format $CLM_FORMAT --rerank-query-format clm"
LIM=()
[[ $LIMIT -gt 0 ]] && LIM=(--limit "$LIMIT")

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
  run heads_clm100 "$d" --data "data/$d" --scorer clm --clm-ckpt "$HEADS" $EMB "${KS[@]}" \
    $CLM --rerank-clm-ckpt "$CLM_CKPT" --rerank-depth 100
  run heads_clm20 "$d" --data "data/$d" --scorer clm --clm-ckpt "$HEADS" $EMB "${KS[@]}" \
    $CLM --rerank-clm-ckpt "$CLM_CKPT" --rerank-depth 20
  run heads_clmft100 "$d" --data "data/$d" --scorer clm --clm-ckpt "$HEADS" $EMB "${KS[@]}" \
    $CLM --rerank-clm-ckpt "$CLM_FT" --rerank-depth 100
  run bm25_clm30 "$d" --data "data/$d" $BM25 "${KS[@]}" $CLM --rerank-clm-ckpt "$CLM_CKPT" --rerank-depth 30
done
echo "== $(date -Is) done"
