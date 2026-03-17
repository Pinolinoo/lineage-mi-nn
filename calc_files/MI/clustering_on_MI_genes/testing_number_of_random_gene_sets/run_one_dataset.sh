#!/bin/bash
#SBATCH --job-name=gene_clustering
#SBATCH --time=60:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --output=gene_clustering_%j.out
#SBATCH --error=gene_clustering_%j.err
#SBATCH --chdir=.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate TOSICA

python run_MT_clustering_for_batch_fast_testing_np_sets.py \
  --dataset "$DATASET" \
  --mi "$MI_FILE" \
  --output_prefix "$OUTPUT_PREFIX" \
  --n_random 1000 \
  --n_genes 200 \
  --seed 0 \
  --outdir "/fast/AG_Metzger/philipp/python_files/gene_signature/mutual_information_approach/November_25_ReDo/calc_files/MI/clustering_on_MI_genes/testing_number_of_random_gene_sets/"
