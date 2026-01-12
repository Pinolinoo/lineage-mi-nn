#!/bin/bash

# Job settings
MEM="128G"                  # total memory per job
TIME="20:00:00"             # hh:mm:ss
JOB_NAME="MI_Permutation_Calc"

# Path to Conda env Python
CONDA_ENV="/home/proth/miniconda3/envs/TOSICA"
PYTHON_BIN="$CONDA_ENV/bin/python"

# List of dataset names
#DATASETS=("treutlein" "weinreb" "junker" "you1" "you2" "you3" "you4" "knoblich_day16_1" "knoblich_day16_2" "knoblich_day42_1" "knoblich_day42_2")

DATASETS=("you2")

# Submit a job for each dataset
for dataset in "${DATASETS[@]}"; do
    sbatch \
        --job-name="${JOB_NAME}_${dataset}" \
        --time="$TIME" \
        --mem="$MEM" \
        --output="logs_${JOB_NAME}_${dataset}.out" \
        --error="logs_${JOB_NAME}_${dataset}.err" \
        --cpus-per-task=10\
        --wrap="$PYTHON_BIN calc_MI_with_permutations_all_genes.py $dataset"
done
