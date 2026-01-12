#!/bin/bash

R_SCRIPT="run_GEMLI_MI_top1300_sample1500.R"

DATASETS=(
  "junker"
  "knoblich_day16_1"
  "knoblich_day16_2"
  "knoblich_day42_1"
  "knoblich_day42_2"
  "treutlein"
  "weinreb"
  "you1"
  "you2"
  "you3"
  "you4"
  "morris_rna" 
  "morris_atac"
)



N_RUNS=5

for ds in "${DATASETS[@]}"; do
  echo "================ Dataset: $ds ================"
  for iter in $(seq 1 $N_RUNS); do
    echo ">>> $ds, iteration $iter"
    Rscript "$R_SCRIPT" "$ds" "$iter"
  done
done
