#!/bin/bash

R_SCRIPT="check_GEMLI_inputs.R"

# dataset keys = MI file prefixes
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

for ds in "${DATASETS[@]}"; do
  echo "--------------------------------------------------------"
  echo "Checking dataset key: $ds"
  echo "--------------------------------------------------------"
  Rscript "$R_SCRIPT" "$ds" || {
    echo "ERROR while checking $ds – aborting."
    exit 1
  }
  echo
done

echo "All checks finished."
