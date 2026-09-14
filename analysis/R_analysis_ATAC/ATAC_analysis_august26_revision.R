suppressPackageStartupMessages({
  library(ChIPseeker)
  library(TxDb.Mmusculus.UCSC.mm10.knownGene)
  library(dplyr)
  library(readr)
  library(ggplot2)
})

# ----------------------------
# INPUT
# ----------------------------
master_path <- "ATAC_master_peaks_MI_sign.tsv"

peaks <- read_tsv(master_path, col_types = cols(
  chrom = col_character(),
  start = col_integer(),
  end = col_integer(),
  mi = col_double(),
  is_significant = col_logical()
)) %>%
  mutate(peak_id = paste0(chrom, ":", start, "-", end))

# restrict to canonical chromosomes (recommended)
canonical <- c(paste0("chr", 1:19), "chrX", "chrY", "chrM")
peaks <- peaks %>% dplyr::filter(chrom %in% canonical)

# ----------------------------
# Annotate 
# ----------------------------
tmp_bed3 <- tempfile(fileext = ".bed")
write.table(peaks %>% dplyr::select(chrom, start, end),
            file = tmp_bed3, sep = "\t", quote = FALSE, col.names = FALSE, row.names = FALSE)

txdb <- TxDb.Mmusculus.UCSC.mm10.knownGene
bed_gr <- readPeakFile(tmp_bed3)
peakAnno <- annotatePeak(bed_gr, TxDb = txdb, tssRegion = c(-3000, 3000))

anno_df <- as.data.frame(peakAnno) %>%
  dplyr::transmute(
    chrom = as.character(seqnames),
    start = as.integer(start) - 1L,  # IMPORTANT: GRanges 1-based -> BED 0-based
    end   = as.integer(end),
    annotation = as.character(annotation),
    distanceToTSS = as.integer(distanceToTSS),
    geneId = as.character(geneId),   # <-- ADD THIS
    peak_id = paste0(chrom, ":", start, "-", end)
  )

df <- peaks %>%
  dplyr::inner_join(anno_df, by = "peak_id") %>%
  dplyr::mutate(
    promoter_prox = (annotation == "Promoter (<=1kb)"),
    abs_dist = abs(distanceToTSS)
  )

# ----------------------------
# Split: significant vs background
# ----------------------------
df_sign <- df %>% dplyr::filter(is_significant)
df_bg   <- df %>% dplyr::filter(!is_significant)

# Background baselines
bg_prom <- mean(df_bg$promoter_prox, na.rm = TRUE)
bg_med_dist <- median(df_bg$abs_dist, na.rm = TRUE)

# ----------------------------
# 1) Promoter fraction vs MI deciles (sign only)
# ----------------------------
n_bins <- 10
df_sign <- df_sign %>% mutate(mi_bin = ntile(mi, n_bins))

summ_prom <- df_sign %>%
  group_by(mi_bin) %>%
  summarise(
    n = n(),
    frac_prom = mean(promoter_prox, na.rm = TRUE),
    # binomial SE for CI
    se = sqrt(frac_prom * (1 - frac_prom) / n),
    ci_low = pmax(0, frac_prom - 1.96 * se),
    ci_high = pmin(1, frac_prom + 1.96 * se),
    mi_min = min(mi, na.rm = TRUE),
    mi_max = max(mi, na.rm = TRUE),
    .groups="drop"
  ) %>%
  mutate(mi_bin_label = paste0("Q", mi_bin, "\n[", signif(mi_min,3), ", ", signif(mi_max,3), "]"))

p_prom <- ggplot(summ_prom, aes(x = mi_bin, y = frac_prom)) +
  geom_point(size = 2.5) +
  geom_line() +
  geom_hline(yintercept = bg_prom, linetype = "dashed") +
  scale_x_continuous(
    breaks = summ_prom$mi_bin,
    labels = summ_prom$mi_bin_label
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    x = "ATAC MI decile (significant peaks only)",
    y = "Fraction promoter-proximal (≤1 kb)",
    title = "Promoter proximity increases with chromatin lineage information",
    subtitle = "Dashed line indicates background (non-significant peaks)"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 8),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

out_prom <- sub("\\.tsv$", "_promoter_fraction_sign_deciles_with_background.pdf", master_path)
ggsave(out_prom, p_prom, width = 10, height = 4)
message("Wrote: ", out_prom)

# ----------------------------
# 2) Distance-to-TSS vs MI deciles (sign only)
# ----------------------------
summ_dist <- df_sign %>%
  group_by(mi_bin) %>%
  summarise(
    n = n(),
    med = median(abs_dist, na.rm = TRUE),
    q25 = quantile(abs_dist, 0.25, na.rm = TRUE),
    q75 = quantile(abs_dist, 0.75, na.rm = TRUE),
    mi_min = min(mi, na.rm = TRUE),
    mi_max = max(mi, na.rm = TRUE),
    .groups="drop"
  ) %>%
  mutate(mi_bin_label = paste0("Q", mi_bin, "\n[", signif(mi_min,3), ", ", signif(mi_max,3), "]"))

p_dist <- ggplot(summ_dist, aes(x = mi_bin, y = med)) +
  geom_point(size = 2) +
  geom_line() +
  geom_errorbar(aes(ymin = q25, ymax = q75), width = 0.15) +
  geom_hline(yintercept = bg_med_dist, linetype = "dashed") +
  scale_x_continuous(breaks = summ_dist$mi_bin, labels = summ_dist$mi_bin_label) +
  scale_y_log10() +
  labs(
    x = "MI decile (significant peaks only; label shows MI range)",
    y = "Median |distance to TSS| (bp, log scale)",
    title = "Higher MI regions are closer to TSS",
    subtitle = "Dashed line: background (non-significant peaks)"
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(size = 8))

out_dist <- sub("\\.tsv$", "_dist_to_TSS_sign_deciles_with_background.pdf", master_path)
ggsave(out_dist, p_dist, width = 10, height = 4)
message("Wrote: ", out_dist)

plot_df <- df %>%
  dplyr::filter(!is.na(mi), !is.na(abs_dist)) %>%
  dplyr::mutate(
    significance = ifelse(is_significant, "Significant", "Non-significant")
  )

p_scatter <- ggplot(plot_df, aes(x = mi, y = abs_dist, color = significance)) +
  geom_point(alpha = 0.25, size = 0.6) +
  scale_color_manual(
    values = c(
      "Significant" = "#D55E00",      # orange/red
      "Non-significant" = "grey70"
    )
  ) +
  labs(
    x = "Mutual information (MI)",
    y = "|distance to TSS| (bp, log scale)",
    title = "Lineage-informative ATAC regions are closer to promoters",
    color = NULL
  ) +
  theme_bw() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )

out_scatter <- sub(
  "\\.tsv$",
  "_scatter_MI_vs_dist_to_TSS_sign_vs_background.pdf",
  master_path
)

ggsave(out_scatter, p_scatter, width = 6.5, height = 4.5)
message("Wrote: ", out_scatter)

# ============================================================
# EXPORT genes linked to ATAC peaks, grouped by MI decile
# ============================================================


suppressPackageStartupMessages({
  library(org.Mm.eg.db)
  library(AnnotationDbi)
  library(dplyr)
})

# ----------------------------
# 1) Base table: one row per PEAK with geneId, MI, significance
# ----------------------------
gene_df <- df %>%
  dplyr::select(geneId, mi, is_significant) %>%
  dplyr::filter(!is.na(geneId), geneId != "") %>%
  dplyr::mutate(geneId = as.character(geneId))

# ----------------------------
# 2) Map Entrez -> SYMBOL (add gene_symbol to gene_df)
# ----------------------------
gene_symbols <- AnnotationDbi::mapIds(
  org.Mm.eg.db,
  keys = unique(gene_df$geneId),
  column = "SYMBOL",
  keytype = "ENTREZID",
  multiVals = "first"
)

gene_df <- gene_df %>%
  dplyr::mutate(gene_symbol = unname(gene_symbols[geneId])) %>%
  dplyr::filter(!is.na(gene_symbol), gene_symbol != "")

# ----------------------------
# 3) Significant peaks: compute MI deciles ONLY here
# ----------------------------
n_bins <- 10
gene_df_sign <- gene_df %>%
  dplyr::filter(is_significant) %>%
  dplyr::mutate(mi_bin = dplyr::ntile(mi, n_bins))

# ----------------------------
# 4) Assign each gene to ONE decile:
#    decile of its highest-MI significant peak
# ----------------------------
gene_deciles <- gene_df_sign %>%
  dplyr::group_by(gene_symbol) %>%
  dplyr::summarise(
    best_atac_mi = max(mi, na.rm = TRUE),
    mi_bin = mi_bin[which.max(mi)][1],
    .groups = "drop"
  ) %>%
  dplyr::mutate(group = paste0("Q", mi_bin)) %>%
  dplyr::select(gene_symbol, group, best_atac_mi)

# ----------------------------
# 5) Background genes:
#    genes that appear in non-sign peaks BUT never in significant peaks
# ----------------------------
bg_genes <- gene_df %>%
  dplyr::filter(!is_significant) %>%
  dplyr::distinct(gene_symbol) %>%
  dplyr::anti_join(gene_deciles %>% dplyr::select(gene_symbol), by = "gene_symbol") %>%
  dplyr::mutate(
    group = "Background_non_sign",
    best_atac_mi = NA_real_
  ) %>%
  dplyr::select(gene_symbol, group, best_atac_mi)

# ----------------------------
# 6) Combine + export
# ----------------------------
gene_groups <- dplyr::bind_rows(gene_deciles, bg_genes)

out_file <- sub("\\.tsv$", "_ATAC_gene_groups_by_MI_decile.tsv", master_path)

write.table(gene_groups, out_file, sep = "\t", quote = FALSE, row.names = FALSE)
message("Wrote: ", out_file)

# sanity checks
print(table(gene_groups$group))
message("Total genes exported: ", nrow(gene_groups))


gene_df %>%
  dplyr::filter(gene_symbol == "Mex3c") %>%
  summarise(
    n_peaks = n(),
    n_sign  = sum(is_significant, na.rm = TRUE),
    max_mi_all = max(mi, na.rm = TRUE),
    max_mi_sign = ifelse(sum(is_significant, na.rm = TRUE) > 0,
                         max(mi[is_significant], na.rm = TRUE),
                         NA_real_)
  )

