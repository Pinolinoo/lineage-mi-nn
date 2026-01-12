import argparse
import scanpy as sc
import pickle
import pandas as pd
import numpy as np
from typing import List


# ------------------------------------------------------------------
# Helper: get top MI genes from CSV
# ------------------------------------------------------------------
def get_top_sign_genes_from_csv(
    csv_path: str,
    top_n: int = 200
) -> List[str]:
    """
    Reads one MI CSV file and returns top_n gene names.
      - filters: true_mi > 95_perm_mi
      - sorts by normalized_MI_label_entropy (descending)
      - returns list of strings (gene names)
    """
    df = pd.read_csv(csv_path)

    # Filter MI > perm threshold
    df = df.loc[df["true_mi"] > df["95_perm_mi"]]

    # Sort by normalized MI
    df_sorted = df[["gene", "normalized_MI_label_entropy"]].sort_values(
        by="normalized_MI_label_entropy",
        ascending=False,
    )

    # Extract gene names
    top_genes = (
        df_sorted["gene"]
        .head(top_n)
        .dropna()
        .astype(str)
        .tolist()
    )

    return top_genes


# ------------------------------------------------------------------
# New helper: compute accuracy & Jaccard from label vectors
# ------------------------------------------------------------------
def compute_pairwise_scores_from_labels(lineage_labels, cluster_labels):
    """
    lineage_labels: 1D array-like of lineage barcodes (original labels)
    cluster_labels: 1D array-like of cluster labels (e.g. Leiden)

    Computes:
      - Accuracy over all unordered cell pairs
      - Jaccard index between the pairwise "same lineage" and
        "same cluster" boolean vectors, without constructing them explicitly.
    """
    lineage_labels = np.asarray(lineage_labels)
    cluster_labels = np.asarray(cluster_labels)

    n = lineage_labels.size
    if n < 2:
        return np.nan, np.nan

    # total number of unordered pairs
    T = n * (n - 1) // 2

    df_lab = pd.DataFrame({
        "lineage": lineage_labels,
        "cluster": cluster_labels
    })

    # same-lineage pairs: sum over lineages of n_k choose 2
    lineage_counts = df_lab["lineage"].value_counts().to_numpy()
    S_L = ((lineage_counts * (lineage_counts - 1)) // 2).sum()

    # same-cluster pairs: sum over clusters of m_c choose 2
    cluster_counts = df_lab["cluster"].value_counts().to_numpy()
    S_C = ((cluster_counts * (cluster_counts - 1)) // 2).sum()

    # intersection: same lineage AND same cluster
    joint_counts = df_lab.groupby(["lineage", "cluster"]).size().to_numpy()
    TP = ((joint_counts * (joint_counts - 1)) // 2).sum()

    # TN: different in both lineage and cluster
    TN = T - S_L - S_C + TP

    accuracy = (TP + TN) / T

    # Jaccard = intersection / union
    union = S_L + S_C - TP
    jaccard = TP / union if union > 0 else np.nan

    return accuracy, jaccard


# ------------------------------------------------------------------
# Clustering + lineage comparison on a given gene set (fast version)
# ------------------------------------------------------------------
def clustering_and_barcode_comparison_on_set_of_genes(adata, geneset, lineage_labels):
    # Subset the data to the geneset (copy small matrix: n_cells x ~200 genes)
    adata_sub = adata[:, geneset].copy()

    sc.pp.neighbors(adata_sub, n_neighbors=10, use_rep="X")
    sc.tl.leiden(adata_sub, resolution=3)

    accuracy, jaccard = compute_pairwise_scores_from_labels(
        lineage_labels,
        adata_sub.obs["leiden"].to_numpy()
    )

    print(adata_sub)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Jaccard index: {jaccard:.4f}")

    return accuracy, jaccard, adata_sub


def main():
    # ------------------------------------------------------------------
    # Args for batch launcher
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mi",
        type=str,
        required=True,
        help="Path to MI CSV file (MI_label_entropy_*_50_permutations.csv)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset pkl file (e.g. treutlein_all_genes.pkl)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Prefix for the output CSV file",
    )
    args = parser.parse_args()

    MI_path = args.mi
    dataset_path = args.dataset

    # ------------------------------------------------------------------
    # Load MI genes (top 200)
    # ------------------------------------------------------------------
    MI_genes = get_top_sign_genes_from_csv(MI_path, top_n=200)

    # ------------------------------------------------------------------
    # Load dataset and convert to AnnData
    # ------------------------------------------------------------------
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)

    # DataFrame -> AnnData
    adata = sc.AnnData(dataset.iloc[:, :-1].values)
    adata.var_names = dataset.columns[:-1]
    adata.var_names_make_unique()
    adata.obs["lineage barcode"] = dataset.iloc[:, -1].values

    # store lineage labels once and reuse
    lineage_labels = adata.obs["lineage barcode"].to_numpy()

    # ------------------------------------------------------------------
    # HVGs (top 200)
    # ------------------------------------------------------------------
    sc.pp.highly_variable_genes(adata, n_top_genes=200)
    hvg_200_genes = adata.var[adata.var["highly_variable"]].index.tolist()

    # ------------------------------------------------------------------
    # Top 200 MI genes (intersect with var_names just in case)
    # ------------------------------------------------------------------
    top_200_mi_genes = [g for g in MI_genes if g in adata.var_names]

    # ------------------------------------------------------------------
    # Random 200-gene sets (20 iterations)
    # ------------------------------------------------------------------
    accuracy_random = []
    jaccard_random = []
    for i in range(20):
        random_genes = (
            adata.var_names.to_series().sample(n=200, replace=False).index
        )
        accuracy, jaccard, _ = clustering_and_barcode_comparison_on_set_of_genes(
            adata, random_genes, lineage_labels
        )
        accuracy_random.append(accuracy)
        jaccard_random.append(jaccard)

    # ------------------------------------------------------------------
    # HVG genes
    # ------------------------------------------------------------------
    accuracy_hvg, jaccard_hvg, adata_hvg = clustering_and_barcode_comparison_on_set_of_genes(
        adata, hvg_200_genes, lineage_labels
    )

    # ------------------------------------------------------------------
    # Top 200 MI genes
    # ------------------------------------------------------------------
    accuracy_top_mi, jaccard_top_mi, adata_top_mi = clustering_and_barcode_comparison_on_set_of_genes(
        adata, top_200_mi_genes, lineage_labels
    )

    # ------------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------------
    data = {
        "accuracy": accuracy_random + [accuracy_hvg, accuracy_top_mi],
        "jaccard": jaccard_random + [jaccard_hvg, jaccard_top_mi],
    }

    df = pd.DataFrame(data)

    # Label gene sets
    gene_set = ["200 random genes"] * 20 + ["HVG genes", "Top 200 MI genes"]
    df["gene_set"] = gene_set

    print(df)

    # Save results
    out_path = (
        "/fast/AG_Metzger/philipp/python_files/gene_signature/"
        "mutual_information_approach/November_25_ReDo/calc_files/MI/"
        f"clustering_on_MI_genes/{args.output_prefix}_"
        "accuracy_jaccard_scores_20iterations_200genes_fast.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
