import sys
import os
import pickle
import hashlib
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from numba import njit
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests


# ------------------ CONFIGURATION ------------------

dataset_name = sys.argv[1]          # e.g. "weinreb"
num_permutations = 10000

# Fixed seed for reproducibility
random_seed = 12345

# FDR level for Benjamini-Hochberg correction
fdr_alpha = 0.05

SAVE_DIR = "./mi_analysis_results_MI_label_entropy_all_genes_parallel_with_BH"
os.makedirs(SAVE_DIR, exist_ok=True)

EPSILON = 1e-9
n_cores = int(os.environ.get("NSLOTS", os.cpu_count()))

base_dir = Path(
    "/fast/AG_Metzger/philipp/python_files/gene_signature/"
    "mutual_information_approach/November_25_ReDo/"
    "preprocessing/pkl_files_filtered"
)

file_dict = {
    "treutlein": base_dir / "treutlein_all_genes.pkl",
    "weinreb": base_dir / "weinreb_all_genes.pkl",
    "junker": base_dir / "junker_z5_all_genes.pkl",

    "you1": base_dir / "you_stage1_barcode_all_genes.pkl",
    "you2": base_dir / "you_stage2_barcode_all_genes.pkl",
    "you3": base_dir / "you_stage3_barcode_all_genes.pkl",
    "you4": base_dir / "you_stage4_barcode_all_genes.pkl",

    "knoblich_day16_1": base_dir / "knoblich_day16_1.pkl",
    "knoblich_day16_2": base_dir / "knoblich_day16_2.pkl",
    "knoblich_day42_1": base_dir / "knoblich_day42_1.pkl",
    "knoblich_day42_2": base_dir / "knoblich_day42_2.pkl",

    "morris_rna": base_dir / "morris_RNA.pkl",
    "morris_atac": base_dir / "morris_ATAC.pkl",
}


# ------------------ HELPERS ------------------

def calculate_entropy(labels):
    """Entropy of discrete barcode labels H(Y)."""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -(probs * np.log2(probs + 1e-12)).sum()


def make_gene_seed(dataset, gene, base_seed):
    """
    Produce a stable, reproducible seed for each dataset/gene combination.

    Unlike Python's built-in hash(), hashlib gives the same result across
    sessions and independently of multiprocessing order.
    """
    key = f"{dataset}|{gene}|{base_seed}".encode("utf-8")
    digest = hashlib.sha256(key).digest()

    # Convert first 8 bytes to an integer accepted by NumPy
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


# ------------------ MUTUAL INFORMATION CORE (NUMBA) ------------------

@njit
def mutual_information(p_g_loc, p_gene, p_location):
    MI = 0.0

    for i in range(p_g_loc.shape[0]):
        for j in range(p_g_loc.shape[1]):
            if p_g_loc[i, j] > 0.0:
                denom = p_gene[i] * p_location[j]

                if denom > 0.0:
                    MI += (
                        p_g_loc[i, j]
                        * np.log2(p_g_loc[i, j] / denom)
                    )

    return MI


def calc_mi_from_df(df):
    """
    Calculate MI between discretized gene expression and barcode labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns:
        - gene
        - barcode_cont
    """
    df = df.sort_values("barcode_cont")

    barcodes_local = np.asarray(df["barcode_cont"].values)

    values = np.round(
        df["gene"].values[:, None],
        decimals=1,
    ).astype(np.float32)

    _, inv = np.unique(
        values,
        axis=0,
        return_inverse=True,
    )

    expression_indices = inv
    max_index = expression_indices.max() + 1

    change_indices = (
        np.where(np.diff(barcodes_local) != 0)[0] + 1
    )

    split_arrays = np.split(
        expression_indices,
        change_indices,
    )

    bincounts_all = np.stack(
        [
            np.bincount(arr, minlength=max_index)
            for arr in split_arrays
        ]
    )

    nonempty_expression_states = ~np.all(
        bincounts_all == 0,
        axis=0,
    )

    comb_array = bincounts_all[
        :,
        nonempty_expression_states,
    ].T

    p_g_loc = comb_array / np.sum(comb_array)

    p_gene = np.sum(p_g_loc, axis=1)
    p_location = np.sum(p_g_loc, axis=0)

    return mutual_information(
        p_g_loc,
        p_gene,
        p_location,
    )


def calc_noise_mi(
    X_col,
    barcodes_local,
    true_mi,
    num_permutations,
    rng,
):
    """
    Generate the permutation null distribution and calculate an empirical
    one-sided p-value.

    The alternative hypothesis is:

        observed MI > MI expected under shuffled barcode labels

    The empirical p-value is:

        p = (number of permuted MI >= observed MI + 1)
            / (number of permutations + 1)
    """
    x_flat = X_col.flatten()

    if len(x_flat) != len(barcodes_local):
        return {
            "mean_perm_mi": np.nan,
            "std_perm_mi": np.nan,
            "95_perm_mi": np.nan,
            "n_perm_ge_true_mi": np.nan,
            "empirical_p_value": np.nan,
        }

    shuffled_mis = np.empty(
        num_permutations,
        dtype=np.float64,
    )

    try:
        for i in range(num_permutations):
            permuted_barcodes = rng.permutation(barcodes_local)

            temp_df = pd.DataFrame(
                {
                    "gene": x_flat,
                    "barcode_cont": permuted_barcodes,
                }
            )

            shuffled_mis[i] = calc_mi_from_df(temp_df)

    except Exception as e:
        print(f"[ERROR in calc_noise_mi] issue: {e}")

        return {
            "mean_perm_mi": np.nan,
            "std_perm_mi": np.nan,
            "95_perm_mi": np.nan,
            "n_perm_ge_true_mi": np.nan,
            "empirical_p_value": np.nan,
        }

    mean_perm = np.mean(shuffled_mis)
    std_perm = np.std(shuffled_mis)
    perc95 = np.percentile(shuffled_mis, 95)

    # One-sided permutation test:
    # How many shuffled MI values are at least as large as the true MI?
    n_exceedances = np.sum(shuffled_mis >= true_mi)

    # Phipson-Smyth style correction: never report p = 0
    empirical_p_value = (
        n_exceedances + 1
    ) / (
        num_permutations + 1
    )

    return {
        "mean_perm_mi": mean_perm,
        "std_perm_mi": std_perm,
        "95_perm_mi": perc95,
        "n_perm_ge_true_mi": int(n_exceedances),
        "empirical_p_value": empirical_p_value,
    }


# ------------------ PARALLEL WORKER ------------------

def initializer(
    _gene_data,
    _barcodes,
    _H_barcode,
):
    global gene_data, barcodes, H_barcode

    gene_data = _gene_data
    barcodes = _barcodes
    H_barcode = _H_barcode


def empty_result(gene):
    """Return a consistent failed-result row."""
    return {
        "gene": gene,
        "true_mi": np.nan,
        "mean_perm_mi": np.nan,
        "95_perm_mi": np.nan,
        "std_perm_mi": np.nan,
        "n_perm_ge_true_mi": np.nan,
        "empirical_p_value": np.nan,
        "significant_percentile_95": False,
        "significant_uncorrected_p_0.05": False,
        "normalized_MI_label_entropy": np.nan,
    }


def process_gene(gene):
    try:
        X = gene_data[gene].values[:, None]

        if X.shape[0] != barcodes.shape[0]:
            return empty_result(gene)

        # True MI for this gene
        df_true = pd.DataFrame(
            {
                "gene": X.flatten(),
                "barcode_cont": barcodes,
            }
        )

        true_mi = calc_mi_from_df(df_true)

        # Reproducible random-number generator for this gene
        gene_seed = make_gene_seed(
            dataset=dataset_name,
            gene=gene,
            base_seed=random_seed,
        )

        rng = np.random.default_rng(gene_seed)

        # Permutation-based null MI and empirical p-value
        permutation_results = calc_noise_mi(
            X_col=X,
            barcodes_local=barcodes,
            true_mi=true_mi,
            num_permutations=num_permutations,
            rng=rng,
        )

        mean_perm = permutation_results["mean_perm_mi"]
        std_perm = permutation_results["std_perm_mi"]
        perc95 = permutation_results["95_perm_mi"]
        n_exceedances = permutation_results["n_perm_ge_true_mi"]
        empirical_p = permutation_results["empirical_p_value"]

        # Normalize only by barcode entropy H(Y)
        norm_mi_label_entropy = (
            true_mi / (H_barcode + EPSILON)
        )

        # Original significance criterion
        significant_percentile = (
            np.isfinite(perc95)
            and true_mi > perc95
        )

        # Uncorrected empirical p-value criterion
        significant_uncorrected = (
            np.isfinite(empirical_p)
            and empirical_p < 0.05
        )

        return {
            "gene": gene,
            "true_mi": true_mi,
            "mean_perm_mi": mean_perm,
            "95_perm_mi": perc95,
            "std_perm_mi": std_perm,
            "n_perm_ge_true_mi": n_exceedances,
            "empirical_p_value": empirical_p,
            "significant_percentile_95": significant_percentile,
            "significant_uncorrected_p_0.05": significant_uncorrected,
            "normalized_MI_label_entropy": norm_mi_label_entropy,
        }

    except Exception as e:
        print(f"[PROCESS_GENE ERROR] {gene}: {e}")
        return empty_result(gene)


# ------------------ MAIN ------------------

if __name__ == "__main__":
    if dataset_name not in file_dict:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {list(file_dict.keys())}"
        )

    print(
        f"Processing dataset: {dataset_name} "
        f"with {num_permutations} permutations..."
    )

    # Load data
    with open(file_dict[dataset_name], "rb") as f:
        full_df = pickle.load(f)

    # Sort and split genes / barcodes
    full_df = full_df.sort_values("barcode_cont")

    barcodes = full_df["barcode_cont"].values.copy()
    gene_data = full_df.iloc[:, :-1]

    genes = gene_data.columns.tolist()

    # Entropy of barcodes H(Y)
    H_barcode = calculate_entropy(barcodes)

    print(
        f"Entropy of barcodes H(Y) for {dataset_name}: "
        f"{H_barcode:.4f}"
    )

    print(f"Number of cells: {len(barcodes):,}")
    print(f"Number of tested genes/features: {len(genes):,}")
    print(f"Number of CPU cores: {n_cores}")

    # Parallel processing
    with mp.Pool(
        processes=n_cores,
        initializer=initializer,
        initargs=(
            gene_data,
            barcodes,
            H_barcode,
        ),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_gene, genes),
                total=len(genes),
                desc="Processing genes",
            )
        )

    mi_df = pd.DataFrame(results)

    # ------------------ BH CORRECTION ------------------

    # Only finite p-values can be supplied to multipletests.
    valid_p_mask = np.isfinite(
        mi_df["empirical_p_value"].to_numpy()
    )

    # Initialise columns for all rows
    mi_df["bh_adjusted_p_value"] = np.nan
    mi_df["significant_BH_FDR_0.05"] = False

    if valid_p_mask.sum() > 0:
        reject_bh, p_values_bh, _, _ = multipletests(
            mi_df.loc[
                valid_p_mask,
                "empirical_p_value",
            ].to_numpy(),
            alpha=fdr_alpha,
            method="fdr_bh",
        )

        mi_df.loc[
            valid_p_mask,
            "bh_adjusted_p_value",
        ] = p_values_bh

        mi_df.loc[
            valid_p_mask,
            "significant_BH_FDR_0.05",
        ] = reject_bh

    # Sort by observed MI, matching the likely downstream use
    mi_df = mi_df.sort_values(
        "true_mi",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    # ------------------ SUMMARY ------------------

    n_valid = valid_p_mask.sum()

    n_sig_percentile = int(
        mi_df["significant_percentile_95"].sum()
    )

    n_sig_uncorrected = int(
        mi_df["significant_uncorrected_p_0.05"].sum()
    )

    n_sig_bh = int(
        mi_df["significant_BH_FDR_0.05"].sum()
    )

    print("\n------------------ SIGNIFICANCE SUMMARY ------------------")
    print(f"Tested genes/features with valid p-values: {n_valid:,}")

    if n_valid > 0:
        print(
            "Significant using original 95th-percentile rule: "
            f"{n_sig_percentile:,} "
            f"({100 * n_sig_percentile / n_valid:.2f}%)"
        )

        print(
            "Significant using uncorrected empirical p < 0.05: "
            f"{n_sig_uncorrected:,} "
            f"({100 * n_sig_uncorrected / n_valid:.2f}%)"
        )

        print(
            f"Significant after BH correction at FDR {fdr_alpha}: "
            f"{n_sig_bh:,} "
            f"({100 * n_sig_bh / n_valid:.2f}%)"
        )

        print("\nPermutation p-value counts:")

        p_value_counts = (
            mi_df.loc[valid_p_mask, "empirical_p_value"]
            .value_counts()
            .sort_index()
        )

        for p_value, count in p_value_counts.items():
            print(
                f"  p = {p_value:.6f}: "
                f"{count:,} genes/features"
            )

    # Save result table
    output_path = os.path.join(
        SAVE_DIR,
        (
            f"MI_label_entropy_{dataset_name}_"
            f"{num_permutations}_permutations_with_BH.csv"
        ),
    )

    mi_df.to_csv(output_path, index=False)

    print(f"\nDone. Results saved to {output_path}")