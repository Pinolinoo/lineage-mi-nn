import argparse
import scanpy as sc
import pickle
import pandas as pd
import numpy as np
from typing import List


# ------------------------------------------------------------------
# Helper: get top MI genes from CSV
# ------------------------------------------------------------------
def get_top_mi_genes_from_csv(
    csv_path: str,
    top_n: int = 200
) -> List[str]:
    """
    Reads one MI CSV file and returns the top_n genes ranked by
    normalized_MI_label_entropy.

    IMPORTANT:
    No significance filtering is performed.
    The top 200 genes are selected purely based on their normalized MI rank.
    """
    df = pd.read_csv(csv_path)

    # Sort all genes by normalized MI
    df_sorted = df[
        ["gene", "normalized_MI_label_entropy"]
    ].sort_values(
        by="normalized_MI_label_entropy",
        ascending=False,
    )

    # Extract top gene names
    top_genes = (
        df_sorted["gene"]
        .dropna()
        .astype(str)
        .head(top_n)
        .tolist()
    )

    return top_genes


# ------------------------------------------------------------------
# Compute accuracy & Jaccard from label vectors
# ------------------------------------------------------------------
def compute_pairwise_scores_from_labels(lineage_labels, cluster_labels):
    """
    lineage_labels: 1D array-like of lineage barcodes
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

    # Total number of unordered cell pairs
    T = n * (n - 1) // 2

    df_lab = pd.DataFrame({
        "lineage": lineage_labels,
        "cluster": cluster_labels
    })

    # Same-lineage pairs
    lineage_counts = df_lab["lineage"].value_counts().to_numpy()
    S_L = ((lineage_counts * (lineage_counts - 1)) // 2).sum()

    # Same-cluster pairs
    cluster_counts = df_lab["cluster"].value_counts().to_numpy()
    S_C = ((cluster_counts * (cluster_counts - 1)) // 2).sum()

    # Same lineage AND same cluster
    joint_counts = (
        df_lab
        .groupby(["lineage", "cluster"])
        .size()
        .to_numpy()
    )

    TP = ((joint_counts * (joint_counts - 1)) // 2).sum()

    # Different lineage AND different cluster
    TN = T - S_L - S_C + TP

    # Pairwise accuracy
    accuracy = (TP + TN) / T

    # Jaccard = intersection / union
    union = S_L + S_C - TP
    jaccard = TP / union if union > 0 else np.nan

    return accuracy, jaccard


# ------------------------------------------------------------------
# Clustering + lineage comparison on a given gene set
# ------------------------------------------------------------------
def clustering_and_barcode_comparison_on_set_of_genes(
    adata,
    geneset,
    lineage_labels
):
    """
    Subsets AnnData to a given gene set, performs Leiden clustering,
    and compares cluster membership to lineage barcode membership.
    """

    adata_sub = adata[:, geneset].copy()

    sc.pp.neighbors(
        adata_sub,
        n_neighbors=10,
        use_rep="X"
    )

    sc.tl.leiden(
        adata_sub,
        resolution=3
    )

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
    # Arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mi",
        type=str,
        required=True,
        help="Path to MI CSV file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset pkl file",
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
    # Load top 200 MI genes
    #
    # NOTE:
    # No significance filtering.
    # Simply take the 200 highest normalized-MI genes.
    # ------------------------------------------------------------------
    MI_genes = get_top_mi_genes_from_csv(
        MI_path,
        top_n=200
    )

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)

    # DataFrame -> AnnData
    adata = sc.AnnData(
        dataset.iloc[:, :-1].values
    )

    adata.var_names = dataset.columns[:-1]
    adata.var_names_make_unique()

    adata.obs["lineage barcode"] = (
        dataset.iloc[:, -1].values
    )

    # Store lineage labels once
    lineage_labels = (
        adata.obs["lineage barcode"].to_numpy()
    )

    # ------------------------------------------------------------------
    # HVGs: top 200
    # ------------------------------------------------------------------
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=200
    )

    hvg_200_genes = (
        adata.var[
            adata.var["highly_variable"]
        ]
        .index
        .tolist()
    )

    # ------------------------------------------------------------------
    # Top 200 MI genes
    #
    # Intersect with var_names in case the MI CSV contains a gene
    # that is absent from the loaded dataset.
    # ------------------------------------------------------------------
    top_200_mi_genes = [
        g for g in MI_genes
        if g in adata.var_names
    ]

    print("\n--------------------------------")
    print("Gene-set sizes")
    print("--------------------------------")
    print(f"Top MI genes requested: 200")
    print(f"Top MI genes present:   {len(top_200_mi_genes)}")
    print(f"HVG genes:              {len(hvg_200_genes)}")
    print("--------------------------------\n")

    if len(top_200_mi_genes) < 200:
        print(
            "WARNING: Fewer than 200 of the top MI genes "
            "were found in adata.var_names."
        )

    # ------------------------------------------------------------------
    # Random 200-gene sets
    #
    # 200 iterations.
    # random_state=i makes every iteration reproducible.
    # ------------------------------------------------------------------
    accuracy_random = []
    jaccard_random = []

    for i in range(200):

        print(
            f"\nRandom gene set {i + 1}/200"
        )

        random_genes = (
            adata.var_names
            .to_series()
            .sample(
                n=200,
                replace=False,
                random_state=i
            )
            .index
        )

        accuracy, jaccard, _ = (
            clustering_and_barcode_comparison_on_set_of_genes(
                adata,
                random_genes,
                lineage_labels
            )
        )

        accuracy_random.append(accuracy)
        jaccard_random.append(jaccard)

    # ------------------------------------------------------------------
    # HVG genes
    # ------------------------------------------------------------------
    print("\nRunning HVG clustering...")

    accuracy_hvg, jaccard_hvg, adata_hvg = (
        clustering_and_barcode_comparison_on_set_of_genes(
            adata,
            hvg_200_genes,
            lineage_labels
        )
    )

    # ------------------------------------------------------------------
    # Top 200 MI genes
    # ------------------------------------------------------------------
    print("\nRunning top-200 MI clustering...")

    accuracy_top_mi, jaccard_top_mi, adata_top_mi = (
        clustering_and_barcode_comparison_on_set_of_genes(
            adata,
            top_200_mi_genes,
            lineage_labels
        )
    )

    # ------------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------------
    data = {
        "accuracy":
            accuracy_random
            + [accuracy_hvg, accuracy_top_mi],

        "jaccard":
            jaccard_random
            + [jaccard_hvg, jaccard_top_mi],
    }

    df = pd.DataFrame(data)

    # Gene-set labels
    df["gene_set"] = (
        ["200 random genes"] * 200
        + ["HVG genes", "Top 200 MI genes"]
    )

    print("\nFinal results:")
    print(df)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = (
        "/fast/AG_Metzger/philipp/python_files/gene_signature/"
        "mutual_information_approach/November_25_ReDo/calc_files/MI/"
        f"clustering_on_MI_genes/{args.output_prefix}_"
        "accuracy_jaccard_scores_200iterations_200genes_"
        "fast_revision.csv"
    )

    df.to_csv(
        out_path,
        index=False
    )

    print(f"\nSaved results to:\n{out_path}")


if __name__ == "__main__":
    main()