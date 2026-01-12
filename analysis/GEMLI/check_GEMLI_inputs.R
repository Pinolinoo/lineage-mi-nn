#!/usr/bin/env Rscript

.libPaths("/home/proth/R/x86_64-pc-linux-gnu-library/4.3")
suppressMessages(library(reticulate))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript check_GEMLI_inputs.R <dataset_key>\n")
}
dataset_key <- args[1]
cat("=== Checking dataset key:", dataset_key, "===\n")

# ------------------ mapping: key -> .pkl filename ------------------
dataset_file_map <- c(
  "junker"             = "junker_z5_all_genes.pkl",
  "knoblich_day16_1"   = "knoblich_day16_1.pkl",
  "knoblich_day16_2"   = "knoblich_day16_2.pkl",
  "knoblich_day42_1"   = "knoblich_day42_1.pkl",
  "knoblich_day42_2"   = "knoblich_day42_2.pkl",
  "treutlein"          = "treutlein_all_genes.pkl",
  "weinreb"            = "weinreb_all_genes.pkl",
  "you1"               = "you_stage1_barcode_all_genes.pkl",
  "you2"               = "you_stage2_barcode_all_genes.pkl",
  "you3"               = "you_stage3_barcode_all_genes.pkl",
  "you4"               = "you_stage4_barcode_all_genes.pkl"
)

if (!dataset_key %in% names(dataset_file_map)) {
  stop(paste0("Unknown dataset_key: ", dataset_key,
              "\nKnown keys: ", paste(names(dataset_file_map), collapse = ", "), "\n"))
}

dataset_filename <- dataset_file_map[[dataset_key]]

dataset_path <- file.path(getwd(), "Datasets", dataset_filename)
mi_path      <- file.path(getwd(), "MIscores",
                          paste0("MI_label_entropy_", dataset_key, "_50_permutations.csv"))

cat("Dataset path:", dataset_path, "\n")
cat("MI path     :", mi_path, "\n")

if (!file.exists(dataset_path)) {
  stop(paste("ERROR: Dataset file not found:", dataset_path))
}
if (!file.exists(mi_path)) {
  stop(paste("ERROR: MI file not found:", mi_path))
}

pandas <- import("pandas")

cat("Reading dataset pickle...\n")
df <- pandas$read_pickle(dataset_path)
df_r <- as.data.frame(df)

cat("  -> Dataset loaded successfully.\n")
cat("  -> Dimensions:", nrow(df_r), "cells x", ncol(df_r), "columns\n")
cat("  -> First 5 column names:\n")
print(colnames(df_r)[1:min(5, ncol(df_r))])
cat("\n")

cat("Reading MI CSV...\n")
mi_df <- read.csv(mi_path, stringsAsFactors = FALSE)

cat("  -> MI file loaded successfully.\n")
cat("  -> Dimensions:", nrow(mi_df), "rows x", ncol(mi_df), "columns\n")
cat("  -> Column names:\n")
print(colnames(mi_df))
cat("\n")

cat("=== OK:", dataset_key, "looks consistent. ===\n")
