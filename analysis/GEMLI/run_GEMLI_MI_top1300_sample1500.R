#!/usr/bin/env Rscript

## ------------------------------------------------------------
## Libraries
## ------------------------------------------------------------
.libPaths("/home/proth/R/x86_64-pc-linux-gnu-library/4.3")
library(GEMLI)
library(reticulate)

## ------------------------------------------------------------
## Args: <dataset_key> <iteration_id>
## ------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_GEMLI_MI_top1300_sample1500.R <dataset_key> <iteration_id>\n")
}
dataset_key <- args[1]
iteration_id <- as.numeric(args[2])

cat("Running GEMLI for dataset key:", dataset_key, " iteration:", iteration_id, "\n")

## ------------------------------------------------------------
## Mapping: dataset_key -> dataset filename (from your screenshot)
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
mi_path      <- file.path(getwd(), "MIscores",
                          paste0("MI_label_entropy_", dataset_key, "_50_permutations.csv"))

cat("Dataset path:", dataset_path, "\n")
cat("MI path     :", mi_path, "\n")

if (!file.exists(dataset_path)) stop(paste("Dataset file not found:", dataset_path))
if (!file.exists(mi_path))      stop(paste("MI file not found:", mi_path))

## ------------------------------------------------------------
## Custom GEMLI wrapper (same as before)
## ------------------------------------------------------------
predict_lineages_with_custom_list_of_genes <- function(
  GEMLI_items,
  custom_genes,
  repetitions = 100,
  sample_size = (2/3),
  desired_cluster_size = c(2, 3),
  fast = FALSE
) {
  data_matrix <- GEMLI_items[["gene_expression"]]
  marker_genes <- custom_genes
  
  results <- Matrix::Matrix(0,
                            nrow = ncol(data_matrix),
                            ncol = ncol(data_matrix),
                            sparse = TRUE)
  rownames(results) <- colnames(data_matrix)
  colnames(results) <- colnames(data_matrix)
  
  for (i in seq_len(repetitions)) {
    marker_genes_sample <- sample(
      intersect(marker_genes, rownames(data_matrix)),
      round(length(intersect(marker_genes, rownames(data_matrix))) * sample_size, 0)
    )
    
    tryCatch({
      cell_clusters <- quantify_clusters_iterative(
        data_matrix,
        marker_genes_sample,
        N = 2,
        fast = fast
      )
      cell_clusters_unique_name <- cell_clusters
      
      for (colname in seq_len(ncol(cell_clusters))) {
        cell_clusters_unique_name[!is.na(cell_clusters_unique_name[, colname]), colname] <-
          paste0(colname, "_", cell_clusters_unique_name[!is.na(cell_clusters_unique_name[, colname]), colname])
      }
      
      clustersize_dict <- table(cell_clusters_unique_name)
      smallest_clusters <- names(clustersize_dict)[clustersize_dict %in% desired_cluster_size]
      
      best_prediction <- Matrix::Matrix(FALSE,
                                        nrow = ncol(data_matrix),
                                        ncol = ncol(data_matrix),
                                        sparse = TRUE)
      rownames(best_prediction) <- colnames(data_matrix)
      colnames(best_prediction) <- colnames(data_matrix)
      
      for (cluster in smallest_clusters) {
        cells_in_cluster <- rownames(best_prediction)[
          rowSums(cell_clusters_unique_name == cluster, na.rm = TRUE) > 0
        ]
        best_prediction[cells_in_cluster, cells_in_cluster] <- TRUE
      }
      
      diag(best_prediction) <- FALSE
      results <- results + best_prediction
    }, error = function(e) {
      message("Error during iteration ", i, ": ", e$message)
    })
  }
  
  GEMLI_items[["prediction"]] <- results
  return(GEMLI_items)
}

## ------------------------------------------------------------
## Load dataset (.pkl) and sample 1500 cells
## ------------------------------------------------------------
pandas <- import("pandas")

df <- pandas$read_pickle(dataset_path)
df_r <- as.data.frame(df)

cat("Original dataset dimensions:", nrow(df_r), "cells x", ncol(df_r), "columns\n")

n_cells  <- nrow(df_r)
n_sample <- min(5000, n_cells)

set.seed(as.numeric(Sys.time()) + iteration_id)
idx <- sample(n_cells, n_sample)
df_r_sub <- df_r[idx, , drop = FALSE]


cat("Subsampled dataset dimensions:", nrow(df_r_sub), "cells x", ncol(df_r_sub), "columns\n")

# Last column = barcodes, others = genes
barcodes  <- as.vector(df_r_sub[, ncol(df_r_sub)])
cellIDs   <- rownames(df_r_sub)
genenames <- colnames(df_r_sub)[1:(ncol(df_r_sub) - 1)]
counts_matrix <- as.matrix(df_r_sub[, 1:(ncol(df_r_sub) - 1)])

expr_matrix <- t(counts_matrix)
colnames(expr_matrix) <- cellIDs
rownames(expr_matrix) <- genenames

GEMLI_items <- list()
GEMLI_items[["gene_expression"]] <- expr_matrix
names(barcodes) <- cellIDs
GEMLI_items[["barcodes"]] <- barcodes

## ------------------------------------------------------------
## Load MI file, filter, sort, top 1300 genes
## ------------------------------------------------------------
mi_df <- read.csv(mi_path, stringsAsFactors = FALSE)

mi_filtered <- mi_df[mi_df$true_mi > mi_df$X95_perm_mi, ]
mi_sorted   <- mi_filtered[order(mi_filtered$normalized_MI_label_entropy,
                                 decreasing = TRUE), ]
top_genes   <- head(mi_sorted$gene, 1300)

cat("Significant genes after filtering:", nrow(mi_filtered), "\n")
cat("Using top", length(top_genes), "MI genes.\n")

## ------------------------------------------------------------
## Run GEMLI
## ------------------------------------------------------------
set.seed(as.numeric(Sys.time()) + 1000 + iteration_id)

GEMLI_items_MI_genes <- predict_lineages_with_custom_list_of_genes(
  GEMLI_items,
  custom_genes = top_genes,
  repetitions = 100
)

GEMLI_items_MI_genes <- test_lineages(
  GEMLI_items_MI_genes,
  valid_fam_sizes = (1:50),
  plot_results = TRUE
)

## ------------------------------------------------------------
## Save results
## ------------------------------------------------------------
dir.create("results_sign_genes", showWarnings = FALSE, recursive = TRUE)

output_filename <- file.path(
  "results_sign_genes",
  paste0("GEMLI_MI_top1300_", dataset_key,
         "_1500cells_iter", iteration_id, ".csv")
)

write.csv(GEMLI_items_MI_genes$testing_results,
          output_filename,
          row.names = FALSE)

cat("Saved GEMLI testing results to:\n", output_filename, "\n")
