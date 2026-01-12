#!/bin/bash

# Define paths
GSEA_JAR="/Users/proth/Downloads/GSEA_4.4.0/gsea-cli.sh"  # Adjust this path if needed
GMT_FILE_HUMAN="/Users/proth/Downloads/GSEA_4.4.0/c5.all.v2024.1.Hs.symbols.gmt"  # Adjust this path if needed
GMT_FILE_MOUSE="/Users/proth/Downloads/GSEA_4.4.0/m5.all.v2024.1.Mm.symbols.gmt"  # Adjust this path if needed
RANKFILES_DIR="rnk_files_IG"  # Path to your folder containing .rnk files
OUTPUT_DIR="gsea_results_IG_all_above_zero"  # Where to save the GSEA results

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all .rnk files in the rankfiles directory
for rnk_file in "$RANKFILES_DIR"/*.rnk; do
    # Get the base name of the .rnk file (e.g., "dataset1" from "dataset1.rnk")
    base_name=$(basename "$rnk_file" .rnk)
    result_dir="$OUTPUT_DIR/$base_name"

    # Create result directory for this file
    mkdir -p "$result_dir"

    # Determine the appropriate GMT file
    if [[ "$base_name" == *"weinreb"* ]]; then
        gmt_file="$GMT_FILE_MOUSE"
    else
        gmt_file="$GMT_FILE_HUMAN"
    fi

    # Run GSEA PreRanked
    echo "Running GSEA for $base_name..."
    sh "$GSEA_JAR" GseaPreranked -rnk "$rnk_file" -gmx "$gmt_file" -out "$result_dir" -plot_top_x 20 

done

echo "All GSEA analyses completed successfully."

