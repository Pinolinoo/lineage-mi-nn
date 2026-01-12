#!/usr/bin/env Rscript

## ------------------------------------------------------------
## Library path + packages
## ------------------------------------------------------------
.libPaths("/home/proth/R/x86_64-pc-linux-gnu-library/4.3")
library(GEMLI)
library(reticulate)

## ------------------------------------------------------------
## Args
## ------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript export_GEMLI_memory_genes_significant.R <dataset_key>")
}
dataset_key <- args[1]

cat(">>> [R] Extracting SIGNIFICANT GEMLI memory genes for dataset:",
    dataset_key, "\n")

## ------------------------------------------------------------
## Mapping dataset_key → actual dataset filename
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
    stop(paste("Unknown dataset:", dataset_key))
}

dataset_filename <- dataset_file_map[[dataset_key]]
dataset_path <- file.path(getwd(), "Datasets", dataset_filename)

cat(">>> [R] Dataset path:", dataset_path, "\n")

if (!file.exists(dataset_path)) {
    stop(paste("Dataset file not found:", dataset_path))
}

## ------------------------------------------------------------
## Load dataset
## ------------------------------------------------------------
pandas <- import("pandas")

cat(">>> [R] Reading dataset...\n")
df <- pandas$read_pickle(dataset_path)
df_r <- as.data.frame(df)

cat(">>> [R] Dimensions:",
    nrow(df_r), "cells x", ncol(df_r), "columns\n")

## ------------------------------------------------------------
## Convert to GEMLI format
## ------------------------------------------------------------
barcodes  <- as.vector(df_r[, ncol(df_r)])
cellIDs   <- rownames(df_r)
genenames <- colnames(df_r)[1:(ncol(df_r) - 1)]
counts_matrix <- as.matrix(df_r[, 1:(ncol(df_r) - 1)])

expr_matrix <- t(counts_matrix)
colnames(expr_matrix) <- cellIDs
rownames(expr_matrix) <- genenames

GEMLI_items <- list(
  gene_expression = as.matrix(expr_matrix),
  barcodes        = setNames(barcodes, cellIDs)
)

## ------------------------------------------------------------
## Run memory gene discovery
## ------------------------------------------------------------
cat(">>> [R] Running GEMLI memory gene discovery...\n")

memory_res <- memory_gene_calling(
    GEMLI_items,
    ground_truth = TRUE
)

memory_df <- memory_res$memory_genes

## ------------------------------------------------------------
## Filter SIGNIFICANT memory genes (p < 0.05)
## ------------------------------------------------------------
memory_df_sig <- memory_df[memory_df$p < 0.05, , drop = FALSE]
memory_df_sig <- memory_df_sig[order(memory_df_sig$var, decreasing = TRUE), ]

cat(">>> [R] Genes tested:", nrow(memory_df), "\n")
cat(">>> [R] Significant memory genes (p < 0.05):",
    nrow(memory_df_sig), "\n")

## ------------------------------------------------------------
## Export result
## ------------------------------------------------------------
output_filename <- paste0(
    "GEMLI_memory_genes_significant_",
    dataset_key,
    ".csv"
)

write.csv(
    rownames(memory_df_sig),
    output_filename,
    row.names = FALSE
)

cat(">>> [R] Memory gene list saved to:\n")
cat("    ", output_filename, "\n")
cat(">>> [R] Done.\n")
