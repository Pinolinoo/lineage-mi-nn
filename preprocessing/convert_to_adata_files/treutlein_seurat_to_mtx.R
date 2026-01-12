library(Seurat)
library(SeuratDisk)
library(Matrix)

srt <- readRDS("~/Desktop/treutlein_seurat.rds")

srt

counts_mat <- GetAssayData(srt, assay = "RNA", slot = "counts")

colnames <- colnames(counts_mat)
rownames <- rownames(counts_mat)

is_integer_like <- all(counts_mat@x == round(counts_mat@x))
if (!is_integer_like) {
  warning("Values in RNA@counts are not all integers – this may not be raw counts.")
} else {
  message("RNA@counts looks like raw integer counts.")
}
#export lineage barcodes: 
write.csv(
  srt@meta.data["GeneBarcodeMerge"],
  file = "~/Desktop/exporting treutlein/treutlein_GeneBarcodeMerge.csv",
  row.names = TRUE
)

writeMM(counts_mat, file = "~/Desktop/exporting treutlein/treutlein.mtx")
write.csv(colnames, file = "~/Desktop/exporting treutlein/treutlein_cellnames.csv", row.names = TRUE)
write.csv(rownames, file = "~/Desktop/exporting treutlein/treutlein_genenames.csv", row.names = TRUE)


