#!/bin/bash

# -------------------------------------------------
# DATASET + MI FILE (edit these)
# -------------------------------------------------

DATASET="/fast/AG_Metzger/philipp/python_files/gene_signature/mutual_information_approach/November_25_ReDo/preprocessing/pkl_files_filtered/treutlein_all_genes.pkl"

MI_FILE="/fast/AG_Metzger/philipp/python_files/gene_signature/mutual_information_approach/November_25_ReDo/calc_files/MI/mi_analysis_results_MI_label_entropy_all_genes_parallel/MI_label_entropy_treutlein_50_permutations.csv"

OUTPUT_PREFIX="treutlein_random_convergence"

# -------------------------------------------------
# Random test settings
# -------------------------------------------------

N_RANDOM=200
N_GENES=200
SEED=0

# -------------------------------------------------
# Submit to SLURM
# -------------------------------------------------

sbatch \
  --export=ALL,DATASET="$DATASET",MI_FILE="$MI_FILE",OUTPUT_PREFIX="$OUTPUT_PREFIX",N_RANDOM="$N_RANDOM",N_GENES="$N_GENES",SEED="$SEED" \
  run_one_dataset.sh
