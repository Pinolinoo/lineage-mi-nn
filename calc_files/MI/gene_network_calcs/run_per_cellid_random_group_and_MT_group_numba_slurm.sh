#!/bin/bash
set -euo pipefail

# -----------------------------
# Job settings
# -----------------------------
MEM="100G"
TIME="120:00:00"
CORES=10
JOB_NAME="cellid_gene_sets"

CONDA_ENV="/home/proth/miniconda3/envs/TOSICA"
PYTHON_BIN="$CONDA_ENV/bin/python"

# Your new script
PY_SCRIPT="mi_cellid_vs_gene_group_numba.py"

N_COMBOS=150

# All datasets
DATASETS=(
  "you1"
  "you2"
  "you3"
  "you4"
  "knoblich_day16_1"
  "knoblich_day16_2"
  "knoblich_day42_1"
  "knoblich_day42_2"
  "treutlein"
  "weinreb"
  "junker"
)

mkdir -p logs

for dataset in "${DATASETS[@]}"; do
  sbatch \
    --job-name="${JOB_NAME}_${dataset}" \
    --time="$TIME" \
    --mem="$MEM" \
    --cpus-per-task="$CORES" \
    --output="logs/${JOB_NAME}_${dataset}_%j.out" \
    --error="logs/${JOB_NAME}_${dataset}_%j.err" \
    --export=ALL,DATASET="$dataset",N_COMBOS="$N_COMBOS" \
    --wrap "
      set -euo pipefail

      # -----------------------------
      # HARD RULE: no nested threading
      # -----------------------------
      export OMP_NUM_THREADS=1
      export MKL_NUM_THREADS=1
      export OPENBLAS_NUM_THREADS=1
      export NUMEXPR_NUM_THREADS=1
      export VECLIB_MAXIMUM_THREADS=1
      export NUMBA_NUM_THREADS=1

      # Optional: make NumPy/BLAS behave more nicely on HPC
      export KMP_INIT_AT_FORK=FALSE

      # Tell the Python script how many workers it may use
      export N_WORKERS=\${SLURM_CPUS_PER_TASK}

      # Optional: control discretization (matches your Python default = 1)
      # export ROUND_DECIMALS=1

      echo \"SLURM job \$SLURM_JOB_ID on \$SLURMD_NODENAME\"
      echo \"dataset=\$DATASET  combos=\$N_COMBOS  cpus=\$SLURM_CPUS_PER_TASK  workers=\$N_WORKERS\"
      echo \"Threads: OMP=\$OMP_NUM_THREADS MKL=\$MKL_NUM_THREADS OPENBLAS=\$OPENBLAS_NUM_THREADS NUMBA=\$NUMBA_NUM_THREADS\"

      $PYTHON_BIN $PY_SCRIPT \"\$DATASET\" \"\$N_COMBOS\"
    "
done
