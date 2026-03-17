#!/bin/bash

# Job settings
MEM="180G"
TIME="120:00:00"
CORES=10                   # bump back up if you want many workers
JOB_NAME="range_gene_sets"

CONDA_ENV="/home/proth/miniconda3/envs/TOSICA"
PYTHON_BIN="$CONDA_ENV/bin/python"
N_COMBOS=1000

#DATASETS=("you1" "you2" "you3" "you4" "knoblich_day16_1" "knoblich_day16_2" "knoblich_day42_1" "knoblich_day42_2" "treutlein" "weinreb" "junker")

DATASETS=("treutlein")

mkdir -p logs

for dataset in "${DATASETS[@]}"; do
  sbatch \
    --job-name="rand_${dataset}" \
    --time="$TIME" \
    --mem="$MEM" \
    --cpus-per-task="$CORES" \
    --output="logs/rand_${dataset}_%j.out" \
    --error="logs/rand_${dataset}_%j.err" \
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

      # Tell your script how many workers it may use
      export N_WORKERS=\${SLURM_CPUS_PER_TASK}

      echo \"SLURM job \$SLURM_JOB_ID on \$SLURMD_NODENAME\"
      echo \"dataset=\$DATASET  combos=\$N_COMBOS  cpus=\$SLURM_CPUS_PER_TASK  workers=\$N_WORKERS\"
      echo \"Threads: OMP=\$OMP_NUM_THREADS MKL=\$MKL_NUM_THREADS OPENBLAS=\$OPENBLAS_NUM_THREADS NUMBA=\$NUMBA_NUM_THREADS\"

      $PYTHON_BIN mi_random_gene_sets_numba.py \"\$DATASET\" \"\$N_COMBOS\"
    "
done
