#!/bin/bash
# Train every model, one at a time (8 GB RAM), then export demo/weights.js + demo/parity_fixture.json.
#
# Iteration plan, from measured speed on this M1 (batch 8, rollout 64-96 steps, avg 80):
#   MPS 1.05 s/iter (real 800-iter run: 14.0 min); CPU 2.5-3.3 s/iter -> MPS.
#   lizard regenerating 3000 + persistent 3000 + growing 3000 = 9000 iters * 1.05 s = ~158 min
#   + ~1 min startup/export  => ~159 min total.
# The three lizards share N=3000 (fair ablation; the minimum allowed). Even that is ~9 min over the
# 150-minute budget. Butterfly and mushroom run after the lizards, also at 3000 iters: at 2000 they stop
# right at the learning-rate drop and slowly drift apart over long runs. Total ~5 h awake.
set -e
cd "$(dirname "$0")"
mkdir -p logs checkpoints

run() {  # name target regime iters
  echo "== $1 ($3, $4 iters) $(date +%H:%M)"
  caffeinate -i python3 nca.py train --target "$2" --regime "$3" --iters "$4" \
    --device mps --out "checkpoints/$1.pt" > "logs/$1.log" 2>&1
}

run lizard            lizard    regenerating 3000
run lizard_persistent lizard    persistent   3000
run lizard_growing    lizard    growing      3000
python3 nca.py export
echo "== lizards done $(date +%H:%M)"
run butterfly         butterfly regenerating 3000
run mushroom          mushroom  regenerating 3000
python3 nca.py export
echo "== done $(date +%H:%M)"
