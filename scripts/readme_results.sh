#!/usr/bin/env bash
# The runs behind the README's results table (docs/results.toml): BM25 without and with the
# instruction, Qwen3-Embedding-8B, and Qwen3-Embedding-8B + the released v0.1 heads, on ToolRet and
# on the MCP sets with server names in the tool text (`toolrank data server-names`). Restartable: a
# run whose results file exists is skipped. On the GB10, from ~/toolrank, with the 8091 server up
# (every text is in the embedding cache already, so no tokens go out):
#
#   PYTHONUNBUFFERED=1 nohup bash scripts/readme_results.sh > data/logs/readme_results.log 2>&1 &
#   # then, on the Mac: scp 'gb10:toolrank/results/readme_*.json' docs/results/
#
# Another backbone endpoint, e.g. the FP8 copy (compose profile fp8, port 8094), for chosen rows:
#   EMB_URL=http://127.0.0.1:8094/v1 EMB_MODEL=qwen3-emb-fp8 TAG=fp8 ROWS="qwen3emb heads" \
#     bash scripts/readme_results.sh        # -> results/readme_fp8_<set>_<row>.json
# (SETS="livemcpbench_server mcp_zero_server" skips ToolRet's 44k-tool encode)
set -uo pipefail
cd "$(dirname "$0")/.."

UV=${UV:-$HOME/.local/bin/uv}
HEADS=${HEADS:-dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz}
EMB_URL=${EMB_URL:-http://127.0.0.1:8091/v1}
EMB_MODEL=${EMB_MODEL:-qwen3-emb}
TAG=${TAG:-}
ROWS=${ROWS:-"bm25 bm25_inst qwen3emb heads"}
SETS=${SETS:-"toolret livemcpbench_server mcp_zero_server"}
BM25="--scorer bm25 --no-stem --tool-format documentation"
EMB="--emb-url $EMB_URL --emb-model $EMB_MODEL --truncate 8192 --emb-batch 128"
EMB="$EMB --tool-format documentation --query-format instruct_query --with-inst"

for d in livemcpbench mcp_zero; do
  [[ -s data/${d}_server/tools.jsonl ]] || "$UV" run toolrank data server-names "data/$d"
done

run() { # run <row> <set> <toolrank eval args...> -> results/readme_[<TAG>_]<set>_<row>.json
  local row=$1 out="results/readme_${TAG:+${TAG}_}$2_$1.json"
  shift 2
  [[ " $ROWS " == *" $row "* ]] || return 0
  if [[ -s "$out" ]]; then
    echo "skip $out"
    return
  fi
  echo "== $(date -Is) $out"
  "$UV" run toolrank eval "$@" --out "$out" || echo "FAILED $out"
}

for d in $SETS; do
  KS=()
  [[ $d == mcp_zero_server ]] && KS=(--ks 1,5,10,20) # Precision@1 = MCP-Zero's top-1 accuracy
  run bm25 "$d" --data "data/$d" $BM25 "${KS[@]}"
  run bm25_inst "$d" --data "data/$d" $BM25 --with-inst "${KS[@]}"
  run qwen3emb "$d" --data "data/$d" --scorer dense $EMB "${KS[@]}"
  run heads "$d" --data "data/$d" --scorer clm --clm-ckpt "$HEADS" $EMB "${KS[@]}"
done
echo "== $(date -Is) done"
