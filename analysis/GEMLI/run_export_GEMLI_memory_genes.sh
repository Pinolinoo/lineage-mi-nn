#!/bin/bash
set -e

R_SCRIPT="export_GEMLI_memory_genes.R"

DATASETS=(
  "junker"
  "knoblich_day16_1"
  "knoblich_day16_2"
  "knoblich_day42_1"
  "knoblich_day42_2"
  "treutlein"
  "weinreb"
  "you1"
  "you2"
  "you3"
  "you4"
)

echo
echo "===== Exporting GEMLI memory genes for all datasets ====="
echo

for ds in "${DATASETS[@]}"; do
    echo "---------------------------------------------"
    echo ">>> Dataset: $ds"
    echo "---------------------------------------------"
    Rscript "$R_SCRIPT" "$ds"
    echo
done

echo "===== Finished exporting memory genes ====="
