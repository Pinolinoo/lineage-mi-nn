#!/bin/bash
set -e  # stop on first error

R_SCRIPT="run_GEMLI_both_same_subsample.R"

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

echo -e "\n===== Starting GEMLI runs (ORIGINAL + MI-top1300, shared subsample) =====\n"

for ds in "${DATASETS[@]}"; do
  echo -e "\n============================================"
  echo "  DATASET: $ds"
  echo "  MODE: ORIGINAL + MI-top1300 (shared cells)"
  echo "============================================"

  for iter in $(seq 1 $N_RUNS); do
    echo -e "\n--- Running $ds iteration $iter ---\n"
    Rscript "$R_SCRIPT" "$ds" "$iter"
    echo -e "\n--- Finished $ds iteration $iter ---\n"
  done
done

echo -e "\n===== All GEMLI runs finished successfully! =====\n"
