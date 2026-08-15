#!/usr/bin/env bash
# Shard the B-factor scoring run across the 8 GPUs on ribo.
set -euo pipefail
cd "$HOME/bayes_design/bfactor"
mkdir -p logs
N=${N:-8}
IN=${IN:-structures400.json}
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i nohup ../.venv/bin/python score_conditionals.py \
    --in "$IN" --out "residues_$i.csv" --src "$HOME/bayes_design/src" \
    --device cuda --batch 32 --shard "$i" --nshards "$N" --dump-probs "probs_$i.npz" \
    > "logs/shard_$i.log" 2>&1 &
done
wait
head -1 residues_0.csv > residues.csv
for i in $(seq 0 $((N - 1))); do tail -n +2 "residues_$i.csv" >> residues.csv; done
wc -l residues.csv
