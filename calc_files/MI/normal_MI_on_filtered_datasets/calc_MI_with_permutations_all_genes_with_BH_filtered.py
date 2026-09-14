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


# ============================================================
# CONFIGURATION
# ============================================================

dataset_name = sys.argv[1]          # e.g. "weinreb"

# Number of permutations per gene
num_permutations = 10000

# Keep only lineages containing MORE THAN 3 cells
# i.e. minimum lineage size = 4 cells
MIN_CELLS_PER_LINEAGE = 4

# Fixed seed for reproducibility
random_seed = 12345

# FDR level for Benjamini-Hochberg correction
fdr_alpha = 0.05

# Separate output directory so the original analysis is not overwritten
SAVE_DIR = (
    "./mi_analysis_results_MI_label_entropy_all_genes_parallel_with_BH_"
    "lineages_gt3"
)

os.makedirs(SAVE_DIR, exist_ok=True)

EPSILON = 1e-9

# Number of available CPU cores
n_cores = int(
    os.environ.get(
        "NSLOTS",
        os.cpu_count(),
    )
)


# ============================================================
# INPUT FILES
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def calculate_entropy(labels):
    """
    Entropy of discrete barcode / lineage labels H(Y).
    """
    _, counts = np.unique(
        labels,
        return_counts=True,
    )

    probs = counts / counts.sum()

    return -(
        probs * np.log2(probs + 1e-12)
    ).sum()


def make_gene_seed(
    dataset,
    gene,
    base_seed,
):
    """
    Produce a stable, reproducible seed for each dataset/gene
    combination.

    hashlib is used instead of Python's built-in hash() so that
    results are reproducible across sessions and independently
    of multiprocessing order.
    """

    key = (
        f"{dataset}|{gene}|{base_seed}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        key
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


# ============================================================
# MUTUAL INFORMATION CORE
# ============================================================

@njit
def mutual_information(
    p_g_loc,
    p_gene,
    p_location,
):
    """
    Calculate mutual information from the joint and marginal
    probability distributions.
    """

    MI = 0.0

    for i in range(
        p_g_loc.shape[0]
    ):
        for j in range(
            p_g_loc.shape[1]
        ):

            if p_g_loc[i, j] > 0.0:

                denom = (
                    p_gene[i]
                    * p_location[j]
                )

                if denom > 0.0:

                    MI += (
                        p_g_loc[i, j]
                        * np.log2(
                            p_g_loc[i, j]
                            / denom
                        )
                    )

    return MI


def calc_mi_from_df(df):
    """
    Calculate MI between discretized gene expression and
    lineage/barcode labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns:
        - gene
        - barcode_cont
    """

    # Sort cells by lineage/barcode
    df = df.sort_values(
        "barcode_cont"
    )

    barcodes_local = np.asarray(
        df["barcode_cont"].values
    )

    # Discretize expression to 0.1
    values = np.round(
        df["gene"].values[:, None],
        decimals=1,
    ).astype(
        np.float32
    )

    # Convert each unique expression state to an integer index
    _, inv = np.unique(
        values,
        axis=0,
        return_inverse=True,
    )

    expression_indices = inv

    max_index = (
        expression_indices.max()
        + 1
    )

    # Find boundaries between lineages
    change_indices = (
        np.where(
            np.diff(
                barcodes_local
            ) != 0
        )[0]
        + 1
    )

    # Expression states for each lineage
    split_arrays = np.split(
        expression_indices,
        change_indices,
    )

    # Counts of each expression state for every lineage
    bincounts_all = np.stack(
        [
            np.bincount(
                arr,
                minlength=max_index,
            )
            for arr in split_arrays
        ]
    )

    # Remove expression states that never occur
    nonempty_expression_states = (
        ~np.all(
            bincounts_all == 0,
            axis=0,
        )
    )

    # Rows = expression states
    # Columns = lineages
    comb_array = bincounts_all[
        :,
        nonempty_expression_states,
    ].T

    # Joint probability P(expression, lineage)
    p_g_loc = (
        comb_array
        / np.sum(comb_array)
    )

    # Marginal probabilities
    p_gene = np.sum(
        p_g_loc,
        axis=1,
    )

    p_location = np.sum(
        p_g_loc,
        axis=0,
    )

    return mutual_information(
        p_g_loc,
        p_gene,
        p_location,
    )


# ============================================================
# PERMUTATION TEST
# ============================================================

def calc_noise_mi(
    X_col,
    barcodes_local,
    true_mi,
    num_permutations,
    rng,
):
    """
    Generate the permutation null distribution and calculate
    an empirical one-sided p-value.

    Alternative hypothesis:

        observed MI > MI expected under shuffled lineage labels

    Empirical p-value:

        p = (number of permuted MI >= observed MI + 1)
            / (number of permutations + 1)
    """

    x_flat = X_col.flatten()

    if (
        len(x_flat)
        != len(barcodes_local)
    ):

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

        for i in range(
            num_permutations
        ):

            # Shuffle lineage labels over the FILTERED cells
            permuted_barcodes = (
                rng.permutation(
                    barcodes_local
                )
            )

            temp_df = pd.DataFrame(
                {
                    "gene": x_flat,
                    "barcode_cont": permuted_barcodes,
                }
            )

            shuffled_mis[i] = (
                calc_mi_from_df(
                    temp_df
                )
            )

    except Exception as e:

        print(
            f"[ERROR in calc_noise_mi] issue: {e}"
        )

        return {
            "mean_perm_mi": np.nan,
            "std_perm_mi": np.nan,
            "95_perm_mi": np.nan,
            "n_perm_ge_true_mi": np.nan,
            "empirical_p_value": np.nan,
        }

    mean_perm = np.mean(
        shuffled_mis
    )

    std_perm = np.std(
        shuffled_mis
    )

    perc95 = np.percentile(
        shuffled_mis,
        95,
    )

    # Number of null MI values >= observed MI
    n_exceedances = np.sum(
        shuffled_mis >= true_mi
    )

    # Phipson-Smyth-style correction:
    # empirical p-values are never exactly zero
    empirical_p_value = (
        n_exceedances + 1
    ) / (
        num_permutations + 1
    )

    return {
        "mean_perm_mi": mean_perm,
        "std_perm_mi": std_perm,
        "95_perm_mi": perc95,
        "n_perm_ge_true_mi": int(
            n_exceedances
        ),
        "empirical_p_value": empirical_p_value,
    }


# ============================================================
# PARALLEL WORKER INITIALIZER
# ============================================================

def initializer(
    _gene_data,
    _barcodes,
    _H_barcode,
):
    """
    Make the filtered expression matrix, lineage labels and
    lineage entropy available to each multiprocessing worker.
    """

    global gene_data
    global barcodes
    global H_barcode

    gene_data = _gene_data
    barcodes = _barcodes
    H_barcode = _H_barcode


# ============================================================
# EMPTY / FAILED RESULT
# ============================================================

def empty_result(gene):
    """
    Return a consistent failed-result row.
    """

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


# ============================================================
# PROCESS SINGLE GENE
# ============================================================

def process_gene(gene):

    try:

        X = gene_data[
            gene
        ].values[:, None]

        if (
            X.shape[0]
            != barcodes.shape[0]
        ):
            return empty_result(
                gene
            )

        # ----------------------------------------------------
        # True MI
        # ----------------------------------------------------

        df_true = pd.DataFrame(
            {
                "gene": X.flatten(),
                "barcode_cont": barcodes,
            }
        )

        true_mi = calc_mi_from_df(
            df_true
        )

        # ----------------------------------------------------
        # Reproducible RNG for this gene
        # ----------------------------------------------------

        gene_seed = make_gene_seed(
            dataset=dataset_name,
            gene=gene,
            base_seed=random_seed,
        )

        rng = np.random.default_rng(
            gene_seed
        )

        # ----------------------------------------------------
        # Permutation null
        # ----------------------------------------------------

        permutation_results = (
            calc_noise_mi(
                X_col=X,
                barcodes_local=barcodes,
                true_mi=true_mi,
                num_permutations=num_permutations,
                rng=rng,
            )
        )

        mean_perm = (
            permutation_results[
                "mean_perm_mi"
            ]
        )

        std_perm = (
            permutation_results[
                "std_perm_mi"
            ]
        )

        perc95 = (
            permutation_results[
                "95_perm_mi"
            ]
        )

        n_exceedances = (
            permutation_results[
                "n_perm_ge_true_mi"
            ]
        )

        empirical_p = (
            permutation_results[
                "empirical_p_value"
            ]
        )

        # ----------------------------------------------------
        # Normalize by lineage entropy H(Y)
        # ----------------------------------------------------

        norm_mi_label_entropy = (
            true_mi
            / (
                H_barcode
                + EPSILON
            )
        )

        # ----------------------------------------------------
        # Original percentile significance
        # ----------------------------------------------------

        significant_percentile = (
            np.isfinite(
                perc95
            )
            and true_mi > perc95
        )

        # ----------------------------------------------------
        # Uncorrected empirical p < 0.05
        # ----------------------------------------------------

        significant_uncorrected = (
            np.isfinite(
                empirical_p
            )
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

        print(
            f"[PROCESS_GENE ERROR] "
            f"{gene}: {e}"
        )

        return empty_result(
            gene
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Check dataset name
    # --------------------------------------------------------

    if dataset_name not in file_dict:

        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: "
            f"{list(file_dict.keys())}"
        )

    print(
        "\n============================================================"
    )

    print(
        f"Processing dataset: {dataset_name}"
    )

    print(
        f"Permutations per gene: {num_permutations:,}"
    )

    print(
        "Lineage filter: "
        f">{MIN_CELLS_PER_LINEAGE - 1} cells "
        f"(minimum {MIN_CELLS_PER_LINEAGE})"
    )

    print(
        "============================================================\n"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    with open(
        file_dict[dataset_name],
        "rb",
    ) as f:

        full_df = pickle.load(
            f
        )

    # --------------------------------------------------------
    # LINEAGE SIZE FILTER
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Everything below this point is calculated ONLY using
    # lineages containing at least 4 cells.
    #
    # Therefore this affects:
    #
    #   - true MI
    #   - H(lineage)
    #   - normalized MI
    #   - permutation null distributions
    #   - empirical p-values
    #   - BH correction
    #
    # --------------------------------------------------------

    cells_before = len(
        full_df
    )

    lineages_before = (
        full_df[
            "barcode_cont"
        ].nunique()
    )

    # Count number of cells belonging to each lineage
    lineage_counts = (
        full_df[
            "barcode_cont"
        ].value_counts()
    )

    # Keep only lineages with MORE THAN 3 cells
    valid_lineages = (
        lineage_counts[
            lineage_counts
            >= MIN_CELLS_PER_LINEAGE
        ].index
    )

    # Filter dataframe
    full_df = (
        full_df[
            full_df[
                "barcode_cont"
            ].isin(
                valid_lineages
            )
        ]
        .copy()
    )

    cells_after = len(
        full_df
    )

    lineages_after = (
        full_df[
            "barcode_cont"
        ].nunique()
    )

    removed_cells = (
        cells_before
        - cells_after
    )

    removed_lineages = (
        lineages_before
        - lineages_after
    )

    print(
        "---------------- LINEAGE FILTER ----------------"
    )

    print(
        f"Cells before filtering:     "
        f"{cells_before:,}"
    )

    print(
        f"Cells after filtering:      "
        f"{cells_after:,}"
    )

    print(
        f"Cells removed:              "
        f"{removed_cells:,}"
    )

    print()

    print(
        f"Lineages before filtering:  "
        f"{lineages_before:,}"
    )

    print(
        f"Lineages after filtering:   "
        f"{lineages_after:,}"
    )

    print(
        f"Lineages removed:           "
        f"{removed_lineages:,}"
    )

    print(
        "------------------------------------------------\n"
    )

    # Safety check
    if cells_after == 0:

        raise ValueError(
            f"No cells remain in dataset "
            f"'{dataset_name}' after requiring "
            f"> {MIN_CELLS_PER_LINEAGE - 1} "
            f"cells per lineage."
        )

    if lineages_after < 2:

        raise ValueError(
            f"Only {lineages_after} lineage remains "
            f"after filtering dataset '{dataset_name}'. "
            f"MI between expression and lineage cannot "
            f"be meaningfully calculated."
        )

    # --------------------------------------------------------
    # Sort filtered dataset
    # --------------------------------------------------------

    full_df = full_df.sort_values(
        "barcode_cont"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Split expression matrix and barcode labels
    # --------------------------------------------------------

    barcodes = (
        full_df[
            "barcode_cont"
        ]
        .values
        .copy()
    )

    # Same structure as original script:
    # barcode_cont is assumed to be the final column
    gene_data = full_df.iloc[
        :,
        :-1
    ]

    genes = (
        gene_data
        .columns
        .tolist()
    )

    # --------------------------------------------------------
    # Entropy AFTER lineage filtering
    # --------------------------------------------------------

    H_barcode = calculate_entropy(
        barcodes
    )

    print(
        f"Entropy of filtered barcodes H(Y) "
        f"for {dataset_name}: "
        f"{H_barcode:.4f}"
    )

    print(
        f"Number of filtered cells: "
        f"{len(barcodes):,}"
    )

    print(
        f"Number of retained lineages: "
        f"{len(np.unique(barcodes)):,}"
    )

    print(
        f"Number of tested genes/features: "
        f"{len(genes):,}"
    )

    print(
        f"Number of CPU cores: "
        f"{n_cores}"
    )

    # --------------------------------------------------------
    # Parallel gene processing
    # --------------------------------------------------------

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
                pool.imap(
                    process_gene,
                    genes,
                ),
                total=len(
                    genes
                ),
                desc="Processing genes",
            )
        )

    mi_df = pd.DataFrame(
        results
    )

    # ========================================================
    # BENJAMINI-HOCHBERG CORRECTION
    # ========================================================

    valid_p_mask = np.isfinite(
        mi_df[
            "empirical_p_value"
        ].to_numpy()
    )

    # Initialise result columns
    mi_df[
        "bh_adjusted_p_value"
    ] = np.nan

    mi_df[
        "significant_BH_FDR_0.05"
    ] = False

    if (
        valid_p_mask.sum()
        > 0
    ):

        reject_bh, p_values_bh, _, _ = (
            multipletests(
                mi_df.loc[
                    valid_p_mask,
                    "empirical_p_value",
                ].to_numpy(),
                alpha=fdr_alpha,
                method="fdr_bh",
            )
        )

        mi_df.loc[
            valid_p_mask,
            "bh_adjusted_p_value",
        ] = p_values_bh

        mi_df.loc[
            valid_p_mask,
            "significant_BH_FDR_0.05",
        ] = reject_bh

    # --------------------------------------------------------
    # Sort output by observed MI
    # --------------------------------------------------------

    mi_df = (
        mi_df
        .sort_values(
            "true_mi",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    # Recalculate valid mask after sorting
    valid_p_mask_sorted = np.isfinite(
        mi_df[
            "empirical_p_value"
        ].to_numpy()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    n_valid = int(
        valid_p_mask_sorted.sum()
    )

    n_sig_percentile = int(
        mi_df[
            "significant_percentile_95"
        ].sum()
    )

    n_sig_uncorrected = int(
        mi_df[
            "significant_uncorrected_p_0.05"
        ].sum()
    )

    n_sig_bh = int(
        mi_df[
            "significant_BH_FDR_0.05"
        ].sum()
    )

    print(
        "\n"
        "------------------ SIGNIFICANCE SUMMARY "
        "------------------"
    )

    print(
        "Tested genes/features with valid p-values: "
        f"{n_valid:,}"
    )

    if n_valid > 0:

        print(
            "Significant using original "
            "95th-percentile rule: "
            f"{n_sig_percentile:,} "
            f"({100 * n_sig_percentile / n_valid:.2f}%)"
        )

        print(
            "Significant using uncorrected "
            "empirical p < 0.05: "
            f"{n_sig_uncorrected:,} "
            f"({100 * n_sig_uncorrected / n_valid:.2f}%)"
        )

        print(
            f"Significant after BH correction "
            f"at FDR {fdr_alpha}: "
            f"{n_sig_bh:,} "
            f"({100 * n_sig_bh / n_valid:.2f}%)"
        )

        print(
            "\nPermutation p-value counts:"
        )

        p_value_counts = (
            mi_df.loc[
                valid_p_mask_sorted,
                "empirical_p_value",
            ]
            .value_counts()
            .sort_index()
        )

        for (
            p_value,
            count,
        ) in p_value_counts.items():

            print(
                f"  p = {p_value:.6f}: "
                f"{count:,} genes/features"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_path = os.path.join(
        SAVE_DIR,
        (
            f"MI_label_entropy_{dataset_name}_"
            f"lineages_gt3_"
            f"{num_permutations}_permutations_"
            f"with_BH.csv"
        ),
    )

    mi_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nDone. Results saved to:\n"
        f"{output_path}"
    )