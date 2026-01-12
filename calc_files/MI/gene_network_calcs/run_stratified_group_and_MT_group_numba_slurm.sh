#!/bin/bash
set -euo pipefail

# -----------------------------
# Job settings
# -----------------------------
MEM="180G"
TIME="120:00:00"
CORES=10
JOB_NAME="cellid_strat_rand"

CONDA_ENV="/home/proth/miniconda3/envs/TOSICA"
PYTHON_BIN="$CONDA_ENV/bin/python"

PY_SCRIPT="mi_group_stratified_random.py"

N_COMBOS=150

# Optional knobs (override here if desired)
ROUND_DECIMALS=1      # discretization for expression vectors
N_STRATA=10           # entropy strata for stratified random sets
INCLUDE_PLAIN_RANDOM=1

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

      # Make BLAS fork-safe
      export KMP_INIT_AT_FORK=FALSE

      # Tell Python how many workers it may use
      export N_WORKERS=\${SLURM_CPUS_PER_TASK}

      # Analysis knobs
      export ROUND_DECIMALS=${ROUND_DECIMALS}
      export N_STRATA=${N_STRATA}
      export INCLUDE_PLAIN_RANDOM=${INCLUDE_PLAIN_RANDOM}

      echo \"SLURM job \$SLURM_JOB_ID on \$SLURMD_NODENAME\"
      echo \"dataset=\$DATASET  combos=\$N_COMBOS\"
      echo \"cpus=\$SLURM_CPUS_PER_TASK  workers=\$N_WORKERS\"
      echo \"ROUND_DECIMALS=\$ROUND_DECIMALS  N_STRATA=\$N_STRATA\"
      echo \"INCLUDE_PLAIN_RANDOM=\$INCLUDE_PLAIN_RANDOM\"

      $PYTHON_BIN $PY_SCRIPT \"\$DATASET\" \"\$N_COMBOS\"
    "
done
