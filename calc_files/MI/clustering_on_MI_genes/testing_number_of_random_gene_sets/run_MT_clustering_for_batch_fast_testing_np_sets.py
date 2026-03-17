import argparse
import os
import scanpy as sc
import pickle
import pandas as pd
import numpy as np
from typing import List, Tuple


# ------------------------------------------------------------------
# Helper: get top MI genes from CSV
# ------------------------------------------------------------------
def get_top_sign_genes_from_csv(csv_path: str, top_n: int = 200) -> List[str]:
    df = pd.read_csv(csv_path)

    # Filter MI > perm threshold
    df = df.loc[df["true_mi"] > df["95_perm_mi"]]

    # Sort by normalized MI
    df_sorted = df[["gene", "normalized_MI_label_entropy"]].sort_values(
        by="normalized_MI_label_entropy",
        ascending=False,
    )

    top_genes = (
        df_sorted["gene"]
        .head(top_n)
        .dropna()
        .astype(str)
        .tolist()
    )
    return top_genes


# ------------------------------------------------------------------
# Pairwise accuracy & Jaccard from label vectors (no explicit pair vec)
# ------------------------------------------------------------------
def compute_pairwise_scores_from_labels(lineage_labels, cluster_labels) -> Tuple[float, float]:
    lineage_labels = np.asarray(lineage_labels)
    cluster_labels = np.asarray(cluster_labels)

    n = lineage_labels.size
    if n < 2:
        return np.nan, np.nan

    T = n * (n - 1) // 2

    df_lab = pd.DataFrame({
        "lineage": lineage_labels,
        "cluster": cluster_labels
    })

    lineage_counts = df_lab["lineage"].value_counts().to_numpy()
    S_L = ((lineage_counts * (lineage_counts - 1)) // 2).sum()

    cluster_counts = df_lab["cluster"].value_counts().to_numpy()
    S_C = ((cluster_counts * (cluster_counts - 1)) // 2).sum()

    joint_counts = df_lab.groupby(["lineage", "cluster"]).size().to_numpy()
    TP = ((joint_counts * (joint_counts - 1)) // 2).sum()

    TN = T - S_L - S_C + TP

    accuracy = (TP + TN) / T

    union = S_L + S_C - TP
    jaccard = TP / union if union > 0 else np.nan

    return float(accuracy), float(jaccard)


# ------------------------------------------------------------------
# Clustering + lineage comparison on a given gene set
# ------------------------------------------------------------------
def clustering_and_barcode_comparison_on_set_of_genes(
    adata: sc.AnnData,
    geneset: List[str],
    lineage_labels,
    n_neighbors: int = 10,
    leiden_resolution: float = 3.0,
) -> Tuple[float, float]:
    adata_sub = adata[:, geneset].copy()

    sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.leiden(adata_sub, resolution=leiden_resolution)

    accuracy, jaccard = compute_pairwise_scores_from_labels(
        lineage_labels,
        adata_sub.obs["leiden"].to_numpy()
    )
    return accuracy, jaccard


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mi", type=str, required=True,
                        help="Path to MI CSV file (MI_label_entropy_*_50_permutations.csv)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset pkl file (e.g. treutlein_all_genes.pkl)")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="Prefix for the output CSV file")

    # NEW: controls for the random test
    parser.add_argument("--n_random", type=int, default=20,
                        help="Number of random gene sets to sample (default: 20)")
    parser.add_argument("--n_genes", type=int, default=200,
                        help="Number of genes per set (default: 200)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducibility (default: 0)")

    # NEW: output directory
    parser.add_argument(
        "--outdir",
        type=str,
        default="/fast/AG_Metzger/philipp/python_files/gene_signature/"
                "mutual_information_approach/November_25_ReDo/calc_files/MI/"
                "clustering_on_MI_genes_random_convergence/",
        help="Directory to write results into (will be created if missing)."
    )

    # Optional: clustering params (nice if you want to keep constant across runs)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--leiden_resolution", type=float, default=3.0)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # --------------------------
    # Load MI genes
    # --------------------------
    MI_genes = get_top_sign_genes_from_csv(args.mi, top_n=args.n_genes)

    # --------------------------
    # Load dataset and create AnnData
    # --------------------------
    with open(args.dataset, "rb") as file:
        dataset = pickle.load(file)

    adata = sc.AnnData(dataset.iloc[:, :-1].values)
    adata.var_names = dataset.columns[:-1]
    adata.var_names_make_unique()
    adata.obs["lineage barcode"] = dataset.iloc[:, -1].values

    lineage_labels = adata.obs["lineage barcode"].to_numpy()

    # --------------------------
    # HVGs
    # --------------------------
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_genes)
    hvg_genes = adata.var[adata.var["highly_variable"]].index.tolist()

    # --------------------------
    # Top MI genes (intersect just in case)
    # --------------------------
    top_mi_genes = [g for g in MI_genes if g in adata.var_names]
    if len(top_mi_genes) < args.n_genes:
        print(f"[WARN] Only {len(top_mi_genes)} / {args.n_genes} MI genes found in var_names.")

    # --------------------------
    # Random sets
    # --------------------------
    all_genes = adata.var_names.to_numpy()
    if all_genes.size < args.n_genes:
        raise ValueError(f"Dataset has only {all_genes.size} genes, but --n_genes={args.n_genes} requested.")

    rows = []

    for i in range(args.n_random):
        random_genes = rng.choice(all_genes, size=args.n_genes, replace=False)
        acc, jac = clustering_and_barcode_comparison_on_set_of_genes(
            adata,
            list(random_genes),
            lineage_labels,
            n_neighbors=args.n_neighbors,
            leiden_resolution=args.leiden_resolution,
        )
        rows.append({
            "gene_set": "random",
            "iteration": i,
            "n_genes": args.n_genes,
            "seed": args.seed,
            "accuracy": acc,
            "jaccard": jac,
        })

    # HVG baseline
    acc_hvg, jac_hvg = clustering_and_barcode_comparison_on_set_of_genes(
        adata,
        hvg_genes,
        lineage_labels,
        n_neighbors=args.n_neighbors,
        leiden_resolution=args.leiden_resolution,
    )
    rows.append({
        "gene_set": "HVG",
        "iteration": np.nan,
        "n_genes": args.n_genes,
        "seed": args.seed,
        "accuracy": acc_hvg,
        "jaccard": jac_hvg,
    })

    # MI baseline
    acc_mi, jac_mi = clustering_and_barcode_comparison_on_set_of_genes(
        adata,
        top_mi_genes,
        lineage_labels,
        n_neighbors=args.n_neighbors,
        leiden_resolution=args.leiden_resolution,
    )
    rows.append({
        "gene_set": "Top_MI",
        "iteration": np.nan,
        "n_genes": args.n_genes,
        "seed": args.seed,
        "accuracy": acc_mi,
        "jaccard": jac_mi,
    })

    df = pd.DataFrame(rows)

    # --------------------------
    # Save
    # --------------------------
    os.makedirs(args.outdir, exist_ok=True)

    out_path = os.path.join(
        args.outdir,
        f"{args.output_prefix}_nRandom{args.n_random}_nGenes{args.n_genes}_seed{args.seed}_acc_jacc.csv"
    )
    df.to_csv(out_path, index=False)
    print(df)
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
