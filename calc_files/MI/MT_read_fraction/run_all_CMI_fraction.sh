#!/bin/bash

# ------------------ JOB SETTINGS ------------------

MEM="128G"                       # Total memory per job
TIME="120:00:00"                 # Maximum runtime: hh:mm:ss
CPUS=100                         # CPUs per dataset
JOB_NAME="CMI_MT_LOO_10000"

# ------------------ WORKING DIRECTORY ------------------
# Directory containing this bash file and the Python script

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------ PYTHON ENVIRONMENT ------------------

CONDA_ENV="/home/proth/miniconda3/envs/TOSICA"
PYTHON_BIN="$CONDA_ENV/bin/python"

# Leave-one-MT-gene-out mitochondrial fraction CMI analysis
PYTHON_SCRIPT="calc_conditional_MI_MT_fraction_all_genes_no_permutations.py"

# ------------------ DATASETS ------------------

DATASETS=(
    "treutlein"
    "weinreb"
    "junker"
    "you1"
    "you2"
    "you3"
    "you4"
    "knoblich_day16_1"
    "knoblich_day16_2"
    "knoblich_day42_1"
    "knoblich_day42_2"
)

# ------------------ LOG DIRECTORY ------------------

LOG_DIR="${WORK_DIR}/logs_${JOB_NAME}"
mkdir -p "$LOG_DIR"

# ------------------ SAFETY CHECKS ------------------

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found or not executable:"
    echo "  $PYTHON_BIN"
    exit 1
fi

if [[ ! -f "${WORK_DIR}/${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script not found:"
    echo "  ${WORK_DIR}/${PYTHON_SCRIPT}"
    exit 1
fi

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
