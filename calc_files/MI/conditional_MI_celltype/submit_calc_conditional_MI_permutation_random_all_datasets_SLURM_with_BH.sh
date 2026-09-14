#!/bin/bash

# ------------------ JOB SETTINGS ------------------

MEM="128G"                       # Total memory per job
TIME="24:00:00"                  # Maximum runtime: hh:mm:ss
CPUS=20                          # CPUs per dataset
JOB_NAME="MI_Permutation_BH_10000"

# ------------------ PYTHON ENVIRONMENT ------------------

CONDA_ENV="/home/proth/miniconda3/envs/TOSICA"
PYTHON_BIN="$CONDA_ENV/bin/python"

# New Python script containing empirical p-values and BH correction
PYTHON_SCRIPT="calc_conditional_MI_with_permutations_all_genes_with_BH.py"

# ------------------ DATASETS ------------------


DATASETS=(
     "treutlein"
     "weinreb"
     "junker"
     "you1"
     "you2"
     "you3"
     "you4"
 )

# ------------------ LOG DIRECTORY ------------------

LOG_DIR="logs_${JOB_NAME}"
mkdir -p "$LOG_DIR"

# ------------------ SUBMIT JOBS ------------------

for dataset in "${DATASETS[@]}"; do
    echo "Submitting dataset: $dataset"

    job_id=$(
        sbatch --parsable \
            --job-name="${JOB_NAME}_${dataset}" \
            --time="$TIME" \
            --mem="$MEM" \
            --cpus-per-task="$CPUS" \
            --chdir="$WORK_DIR" \
            --output="${LOG_DIR}/${dataset}_%j.out" \
            --error="${LOG_DIR}/${dataset}_%j.err" \
            --wrap="export NSLOTS=\${SLURM_CPUS_PER_TASK}; \
                    \"$PYTHON_BIN\" \"$PYTHON_SCRIPT\" \"$dataset\""
    )

    status=$?

    if [[ $status -eq 0 ]]; then
        echo "Submitted $dataset as job $job_id"
    else
        echo "ERROR: sbatch submission failed for $dataset"
    fi
done
