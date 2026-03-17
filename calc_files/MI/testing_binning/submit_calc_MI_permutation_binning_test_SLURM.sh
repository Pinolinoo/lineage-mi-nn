#!/bin/bash

# -----------------------------
# Job settings
# -----------------------------
MEM="128G"                  # total memory per job
TIME="20:00:00"             # hh:mm:ss
CPUS=10
JOB_NAME="MI_Permutation_Calc"

# -----------------------------
# Python / Conda
# -----------------------------
CONDA_ENV="/home/proth/miniconda3/envs/TOSICA"
PYTHON_BIN="$CONDA_ENV/bin/python"

SCRIPT="mi_label_entropy_all_genes_parallel_discretize.py"

# -----------------------------
# Datasets to process
# -----------------------------
DATASETS=(
    treutlein
    weinreb
    junker
)

# -----------------------------
# Discretization scenarios
# (label | arguments)
# -----------------------------
SCENARIOS=(
  "round_d1|--discretization round --round_decimals 1"
  "bin_w0p05|--discretization bin --bin_width 0.05"
  "bin_w0p10|--discretization bin --bin_width 0.10"
  "bin_w0p20|--discretization bin --bin_width 0.20"
  "none|--discretization none"
)

# -----------------------------
# Submit jobs
# -----------------------------
for dataset in "${DATASETS[@]}"; do
  for scenario in "${SCENARIOS[@]}"; do

    LABEL="${scenario%%|*}"
    ARGS="${scenario##*|}"

    sbatch \
      --job-name="${JOB_NAME}_${dataset}_${LABEL}" \
      --time="$TIME" \
      --mem="$MEM" \
      --cpus-per-task="$CPUS" \
      --output="logs_${JOB_NAME}_${dataset}_${LABEL}.out" \
      --error="logs_${JOB_NAME}_${dataset}_${LABEL}.err" \
      --wrap="$PYTHON_BIN $SCRIPT $dataset $ARGS"

  done
done
