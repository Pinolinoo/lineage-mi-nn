library(Seurat)
library(SeuratDisk)
library(Matrix)
library(qs)


stage1 <- qread("/November_25_ReDo/raw_datasets/you/seurat_files_from_authors/stemdata_stageI.qs")
stage2 <- qread("/November_25_ReDo/raw_datasets/you/seurat_files_from_authors/You et al Data/stemdata_stageII.qs")
stage3 <- qread("/November_25_ReDo/raw_datasets/you/seurat_files_from_authors/You et al Data/stemdata_stageIII.qs")
stage4 <- qread("/November_25_ReDo/raw_datasets/you/seurat_files_from_authors/You et al Data/stemdata_stageIV.qs")

### only stage 4 ###
stage4 <- JoinLayers(stage4) 
#######################

matrix <- GetAssayData(stage1, assay = "RNA", slot = "counts")

dim(matrix)
sparse_matrix <- as(matrix, "TsparseMatrix")

gene_names <- rownames(matrix)
cell_names <- colnames(matrix)

# exporting files for python one by one of the above 

writeMM(sparse_matrix, file = "/gene_signature/mutual_information_approach/November_25_ReDo/raw_datasets/youYou et al Data/you_stageI_raw.mtx")
write.csv(gene_names, file = "/gene_signature/mutual_information_approach/November_25_ReDo/raw_datasets/youyou_stageI_raw_genenames.csv", row.names = TRUE)
write.csv(cell_names, file = "/gene_signature/mutual_information_approach/November_25_ReDo/raw_datasets/youyou_stageI_raw_cellnames.csv", row.names = TRUE)

