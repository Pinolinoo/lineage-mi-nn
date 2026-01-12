#!/usr/bin/env Rscript

## ------------------------------------------------------------
## Libraries / library path
## ------------------------------------------------------------
.libPaths("/home/proth/R/x86_64-pc-linux-gnu-library/4.3")
library(GEMLI)
library(reticulate)

## ------------------------------------------------------------
## Args: <dataset_key> <iteration_id>
## ------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_GEMLI_original_sample1500.R <dataset_key> <iteration_id>\n")
}
dataset_key <- args[1]
iteration_id <- as.numeric(args[2])

cat("Running ORIGINAL GEMLI for dataset key:", dataset_key,
    " iteration:", iteration_id, "\n")

## ------------------------------------------------------------
## Mapping: dataset_key -> .pkl filename (from your screenshot)
## ------------------------------------------------------------
dataset_file_map <- c(
  "junker"             = "junker_z5_all_genes_nolog.pkl",
  "knoblich_day16_1"   = "knoblich_day16_1_nolog.pkl",
  "knoblich_day16_2"   = "knoblich_day16_2_nolog.pkl",
  "knoblich_day42_1"   = "knoblich_day42_1_nolog.pkl",
  "knoblich_day42_2"   = "knoblich_day42_2_nolog.pkl",
  "treutlein"          = "treutlein_all_genes_nolog.pkl",
  "weinreb"            = "weinreb_all_genes_nolog.pkl",
  "you1"               = "you_stage1_barcode_all_genes_nolog.pkl",
  "you2"               = "you_stage2_barcode_all_genes_nolog.pkl",
  "you3"               = "you_stage3_barcode_all_genes_nolog.pkl",
  "you4"               = "you_stage4_barcode_all_genes_nolog.pkl",
  "morris_rna"         = "morris_RNA_nolog.pkl",
  "morris_atac"        = "morris_ATAC.pkl"
)

if (!dataset_key %in% names(dataset_file_map)) {
  stop(paste0("Unknown dataset_key: ", dataset_key,
              "\nKnown keys: ", paste(names(dataset_file_map), collapse = ", "), "\n"))
}

dataset_filename <- dataset_file_map[[dataset_key]]
dataset_path <- file.path(getwd(), "Datasets", dataset_filename)

cat("Dataset path:", dataset_path, "\n")

if (!file.exists(dataset_path)) {
  stop(paste("Dataset file not found:", dataset_path))
}

## ------------------------------------------------------------
## Load dataset (.pkl) and sample 1500 cells
## ------------------------------------------------------------
pandas <- import("pandas")

df <- pandas$read_pickle(dataset_path)
df_r <- as.data.frame(df)

cat("Original dataset dimensions:",
    nrow(df_r), "cells x", ncol(df_r), "columns\n")

n_cells  <- nrow(df_r)
n_sample <- min(1500, n_cells)

set.seed(as.numeric(Sys.time()) + iteration_id)
idx <- sample(n_cells, n_sample)
sampled_df_r <- df_r[idx, , drop = FALSE]

cat("Subsampled dataset dimensions:",
    nrow(sampled_df_r), "cells x", ncol(sampled_df_r), "columns\n")

## ------------------------------------------------------------
## Build GEMLI_items (all genes, original GEMLI)
## ------------------------------------------------------------
barcodes  <- as.vector(sampled_df_r[, ncol(sampled_df_r)])
cellIDs   <- rownames(sampled_df_r)
genenames <- colnames(sampled_df_r)[1:(ncol(sampled_df_r) - 1)]
counts_matrix <- as.matrix(sampled_df_r[, 1:(ncol(sampled_df_r) - 1)])

expr_matrix <- t(counts_matrix)
colnames(expr_matrix) <- cellIDs
rownames(expr_matrix) <- genenames

GEMLI_items <- list()
GEMLI_items[["gene_expression"]] <- expr_matrix
names(barcodes) <- cellIDs
GEMLI_items[["barcodes"]] <- barcodes

cat("Starting ORIGINAL GEMLI calculation (all genes)...\n")

## ------------------------------------------------------------
## Run original GEMLI (no MI filtering)
## ------------------------------------------------------------
set.seed(as.numeric(Sys.time()) + 1000 + iteration_id)

GEMLI_items <- predict_lineages(
  GEMLI_items,
  repetitions = 100
)

GEMLI_items <- test_lineages(
  GEMLI_items,
  valid_fam_sizes = (1:50),
  plot_results = TRUE
)

## ------------------------------------------------------------
## Save results
## ------------------------------------------------------------
dir.create("results_original_GEMLI", showWarnings = FALSE, recursive = TRUE)

output_filename <- file.path(
  "results_original_GEMLI",
  paste0(
    "GEMLI_testing_results_original_GEMLI_",
    dataset_key,
    "_1500cells_iter", iteration_id,
    ".csv"
  )
)

write.csv(GEMLI_items$testing_results,
          output_filename,
          row.names = FALSE)

cat("Saved ORIGINAL GEMLI testing results to:\n",
    output_filename, "\n")
