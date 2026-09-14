#!/bin/bash
#SBATCH --job-name=gene_clustering
#SBATCH --time=120:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --output=gene_clustering_%j.out
#SBATCH --error=gene_clustering_%j.err
#SBATCH --chdir=.

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TOSICA

# Run the Python script with parameters provided via --export from sbatch
python run_MT_clustering_for_batch_fast.py \
    --dataset "$DATASET" \
    --mi "$MI_FILE" \
    --output_prefix "$OUTPUT_PREFIX"