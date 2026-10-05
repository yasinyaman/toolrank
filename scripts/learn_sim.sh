#!/usr/bin/env bash
# The learning loop on simulated traffic (Faz 2 Hafta 4): a benchmark stands in for a served
# catalogue (scripts/learn_sim.py). For every traffic size in SIZES (0 = all of it): serve that many
# traffic queries with the starting heads, learn from the usage log, and score the starting and the
# learned heads on the held-out queries, on the held-out queries about tools the traffic never
# asked for, and on the EVALS sets (what the heads forgot).
#
#   bash scripts/learn_sim.sh                                   # ToolRet, guard MCP-Zero
#   BENCH=data/mcp_zero_server NAME=mcpzero GUARD=data/livemcpbench_server \
#     EVALS="data/toolret data/livemcpbench_server" SIZES="300 1000 0" bash scripts/learn_sim.sh
#   SIZES=0 TAG=e10 LEARN="--epochs 10" bash scripts/learn_sim.sh   # another recipe, the same logs
#   SIZES=0 NOISE=0.2 bash scripts/learn_sim.sh                 # an agent that calls a wrong tool 1 time in 5
#   SIZES=0 SEED=1 NAME=toolret_s1 bash scripts/learn_sim.sh    # another split of the queries
#   HEADS=none EMB_MODEL=toolrank-emb-v0.2 bash scripts/learn_sim.sh   # headless: dense scoring,
#                                                                      # learn from identity heads
#
# Restartable: a run dir that has a usage log is not served again, a report that exists is not
# recomputed. Reports: results/sim_<NAME>_*.json; then `toolrank compare results/sim_<NAME>_*heldout.json`.
set -u
cd "$(dirname "$0")/.."

UV=${UV:-$HOME/.local/bin/uv}
BENCH=${BENCH:-data/toolret}
NAME=${NAME:-toolret}
GUARD=${GUARD:-data/mcp_zero_server}
EVALS=${EVALS:-"data/livemcpbench_server data/mcp_zero_server"}
SIZES=${SIZES:-"100 300 1000 3000 0"}
NOISE=${NOISE:-0}
SEED=${SEED:-0}
TAG=${TAG:-default}
LEARN=${LEARN:-}
EMB_URL=${EMB_URL:-http://127.0.0.1:8091/v1}
EMB_MODEL=${EMB_MODEL:-qwen3-emb}
CACHE=${CACHE:-.cache/toolrank}
HEADS=${HEADS:-dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz}
ROOT=data/sim/$NAME
EMB="--emb-url $EMB_URL --emb-model $EMB_MODEL --truncate 8192 --cache-dir $CACHE"
if [ "$HEADS" = none ]; then
  # headless (the v0.2 backbone): serve and score the start with the embedding model alone, learn
  # from fresh identity heads (--init none); no packaged heads anywhere, cached ones included
  unset TOOLRANK_HEADS || true
  INIT=(--init none)
  SERVE=(--clm-ckpt none)
else
  export TOOLRANK_HEADS=$PWD/$HEADS
  INIT=()
  SERVE=()
fi
set -o pipefail # a failed eval or learn, not tail, decides "FAILED"

score() { # score <heads|none> <report prefix>: the held-out sets and EVALS with these heads
  local heads=$1 prefix=$2 dir out
  local ckpt=(--scorer clm --clm-ckpt "$heads")
  [ "$heads" = none ] && ckpt=(--scorer dense)
  for dir in "$ROOT/heldout" "$ROOT/heldout_new" $EVALS; do
    out=results/${prefix}_$(basename "$dir").json
    [ -s "$out" ] && continue
    # shellcheck disable=SC2086
    "$UV" run toolrank eval --data "$dir" "${ckpt[@]}" $EMB --tool-format documentation \
      --query-format instruct_query --with-inst --out "$out" | tail -n 4 || echo "FAILED $out"
  done
}

[ -s "$ROOT/heldout_new/queries.jsonl" ] || "$UV" run python scripts/learn_sim.py split --bench "$BENCH" --out "$ROOT" --seed "$SEED"
score "$HEADS" "sim_${NAME}_base"

for n in $SIZES; do
  run=n$n
  [ "$NOISE" != 0 ] && run=${run}_noise$NOISE
  dir=$ROOT/runs/$run
  if ! ls "$dir"/usage/usage-*.jsonl > /dev/null 2>&1; then
    mkdir -p "$dir" && ln -sf "$PWD/$BENCH/tools.jsonl" "$dir/tools.jsonl"
    # shellcheck disable=SC2086
    "$UV" run python scripts/learn_sim.py traffic --queries "$ROOT/traffic.jsonl" --limit "$n" --noise "$NOISE" \
      --report "results/sim_${NAME}_${run}_traffic.json" -- --data "$dir" $EMB ${SERVE[@]+"${SERVE[@]}"} \
      > /dev/null || echo "FAILED traffic $run"
  fi
  heads=$dir/learned/$TAG.npz
  if [ ! -s "results/learn_sim_${NAME}_${run}_${TAG}.json" ]; then
    # shellcheck disable=SC2086
    "$UV" run toolrank learn --data "$dir" --out "$heads" --dev "$GUARD" $EMB ${INIT[@]+"${INIT[@]}"} $LEARN \
      --name "sim_${NAME}_${run}_${TAG}" | tail -n 6
  fi
  [ -s "$heads" ] && score "$heads" "sim_${NAME}_${run}_${TAG}"
done
