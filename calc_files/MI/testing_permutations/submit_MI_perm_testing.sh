#!/bin/bash

# -----------------------------
# Job settings
# -----------------------------
MEM="128G"
TIME="20:00:00"
CPUS=10
JOB_NAME="MI_Permutation_Subset"

# -----------------------------
# Python / Conda
# -----------------------------
CONDA_ENV="/home/proth/miniconda3/envs/TOSICA"
PYTHON_BIN="$CONDA_ENV/bin/python"

SCRIPT="calc_MI_with_permutations_subset.py"

# -----------------------------
# READ reference MI results from here (existing all-genes results)
# -----------------------------
REFERENCE_BASE_DIR="/fast/AG_Metzger/philipp/python_files/gene_signature/mutual_information_approach/November_25_ReDo/calc_files/MI/testing_binning/mi_analysis_results_MI_label_entropy_all_genes_parallel"

# -----------------------------
# WRITE new subset permutation outputs here
# -----------------------------
OUTPUT_BASE_DIR="/fast/AG_Metzger/philipp/python_files/gene_signature/mutual_information_approach/November_25_ReDo/calc_files/MI/testing_permutations/mi_analysis_results_MI_label_entropy_all_genes_parallel"

# -----------------------------
# Datasets + permutation grid
# -----------------------------
DATASETS=(treutlein weinreb junker)
PERMS=(50 100 200 500 1000)

# -----------------------------
# Discretization: round d1
# -----------------------------
DISCRETIZATION="round"
ROUND_DECIMALS=1
BIN_WIDTH=0.05   # unused for round; ok to pass

# -----------------------------
# Subset selection (fixed across perm runs)
# -----------------------------
TOP_N=200
RANDOM_N=200
RANDOM_SEED=0
REFERENCE_NPERM=50

# -----------------------------
# Submit jobs
# -----------------------------
for dataset in "${DATASETS[@]}"; do
  for nperm in "${PERMS[@]}"; do
    sbatch \
      --job-name="${JOB_NAME}_${dataset}_${nperm}_roundd1" \
      --time="$TIME" \
      --mem="$MEM" \
      --cpus-per-task="$CPUS" \
      --output="logs_${JOB_NAME}_${dataset}_${nperm}_roundd1.out" \
      --error="logs_${JOB_NAME}_${dataset}_${nperm}_roundd1.err" \
      --wrap="$PYTHON_BIN $SCRIPT $dataset \
        --num_permutations $nperm \
        --discretization $DISCRETIZATION \
        --round_decimals $ROUND_DECIMALS \
        --bin_width $BIN_WIDTH \
        --reference_base_dir $REFERENCE_BASE_DIR \
        --output_base_dir $OUTPUT_BASE_DIR \
        --top_n $TOP_N \
        --random_n $RANDOM_N \
        --random_seed $RANDOM_SEED \
        --reference_nperm $REFERENCE_NPERM"
  done
done
