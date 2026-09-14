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

dataset_name = sys.argv[1]
num_permutations = 10000

random_seed = 12345
fdr_alpha = 0.05

# Keep only lineages represented by at least this many cells
MIN_CELLS_PER_LINEAGE = 3

SAVE_DIR = (
    "./conditional_mi_analysis_results_"
    "MI_label_entropy_all_genes_parallel_with_BH_"
    f"min{MIN_CELLS_PER_LINEAGE}cells_per_lineage"
)

os.makedirs(SAVE_DIR, exist_ok=True)

EPSILON = 1e-9

n_cores = int(
    os.environ.get("NSLOTS", os.cpu_count())
)


base_dir = Path(
    "/fast/AG_Metzger/philipp/python_files/gene_signature/"
    "mutual_information_approach/November_25_ReDo/"
    "preprocessing/pkl_files_filtered_with_celltype"
)


file_dict = {

    "treutlein":
        base_dir / "treutlein_all_genes.pkl",

    "weinreb":
        base_dir / "weinreb_all_genes.pkl",

    "junker":
        base_dir / "junker_z5_all_genes.pkl",

    "you1":
        base_dir / "you_stage1_barcode_all_genes.pkl",

    "you2":
        base_dir / "you_stage2_barcode_all_genes.pkl",

    "you3":
        base_dir / "you_stage3_barcode_all_genes.pkl",

    "you4":
        base_dir / "you_stage4_barcode_all_genes.pkl",

}


# ============================================================
# BASIC ENTROPY FUNCTIONS
# ============================================================

def calculate_entropy(labels):
    """
    Shannon entropy of a discrete variable.
    """
    _, counts = np.unique(
        labels,
        return_counts=True
    )

    probs = counts / counts.sum()

    return -(
        probs * np.log2(probs + 1e-12)
    ).sum()


@njit
def entropy_from_integer_labels(labels):
    """
    Fast entropy calculation for arbitrary non-negative
    integer labels.

    Uses sorting rather than constructing a potentially huge
    dense contingency matrix.
    """

    n = len(labels)

    if n == 0:
        return 0.0

    sorted_labels = np.sort(labels)

    entropy = 0.0

    count = 1

    for i in range(1, n):

        if sorted_labels[i] == sorted_labels[i - 1]:
            count += 1

        else:

            p = count / n

            entropy -= p * np.log2(p)

            count = 1

    # final category
    p = count / n

    entropy -= p * np.log2(p)

    return entropy


# ============================================================
# REPRODUCIBLE GENE SEEDS
# ============================================================

def make_gene_seed(dataset, gene, base_seed):

    key = (
        f"{dataset}|{gene}|{base_seed}"
    ).encode("utf-8")

    digest = hashlib.sha256(key).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


# ============================================================
# JOINT ENTROPY HELPERS
# ============================================================

def joint_entropy_2(
    variable1,
    variable2,
    n_variable2,
):
    """
    H(variable1, variable2)

    Both variables must already be integer encoded.
    """

    joint_codes = (
        variable1.astype(np.int64)
        * np.int64(n_variable2)
        + variable2.astype(np.int64)
    )

    return entropy_from_integer_labels(
        joint_codes
    )


def joint_entropy_3(
    variable1,
    variable2,
    variable3,
    n_variable2,
    n_variable3,
):
    """
    H(variable1, variable2, variable3)

    Unique integer encoding:

        ((v1 * n_v2) + v2) * n_v3 + v3
    """

    joint_codes = (
        (
            variable1.astype(np.int64)
            * np.int64(n_variable2)
            + variable2.astype(np.int64)
        )
        * np.int64(n_variable3)
        + variable3.astype(np.int64)
    )

    return entropy_from_integer_labels(
        joint_codes
    )


# ============================================================
# CONDITIONAL MI
# ============================================================

def calculate_conditional_mi_from_codes(
    expression_codes,
    barcode_codes,
    celltype_codes,
    H_XC,
    H_C,
    n_barcodes,
    n_expression_states,
):
    """
    Calculate:

        I(X;Y|C)
        = H(X,C) + H(Y,C)
          - H(C) - H(X,Y,C)

    where

        X = lineage barcode
        Y = gene expression
        C = cell type
    """

    n_celltypes = (
        celltype_codes.max() + 1
    )

    # H(Y,C)
    H_YC = joint_entropy_2(
        expression_codes,
        celltype_codes,
        n_celltypes,
    )

    # H(X,Y,C)
    H_XYC = joint_entropy_3(
        barcode_codes,
        expression_codes,
        celltype_codes,
        n_expression_states,
        n_celltypes,
    )

    conditional_mi = (
        H_XC
        + H_YC
        - H_C
        - H_XYC
    )

    # Only protect against tiny floating-point negatives
    if conditional_mi < 0 and conditional_mi > -1e-10:
        conditional_mi = 0.0

    return conditional_mi, H_YC


# ============================================================
# PERMUTATION NULL
# ============================================================

def calc_noise_mi(
    expression_codes,
    barcodes_local,
    cell_types_local,
    celltype_indices,
    true_mi,
    H_XC,
    H_YC,
    H_C,
    n_expression_states,
    n_celltypes,
    num_permutations,
    rng,
):
    """
    Permutation null for conditional MI.

    Lineage labels are shuffled WITHIN CELL TYPE ONLY.

    Since the shuffle preserves:

        H(X,C)
        H(Y,C)
        H(C)

    only H(X,Y,C) needs to be recalculated for each permutation.
    """

    shuffled_mis = np.empty(
        num_permutations,
        dtype=np.float64,
    )

    try:

        for i in range(num_permutations):

            # Copy original barcode codes
            permuted_barcodes = (
                barcodes_local.copy()
            )

            # Shuffle only within each cell type
            for indices in celltype_indices:

                permuted_barcodes[indices] = (
                    rng.permutation(
                        barcodes_local[indices]
                    )
                )

            # Only this entropy changes
            H_XYC_perm = joint_entropy_3(
                permuted_barcodes,
                expression_codes,
                cell_types_local,
                n_expression_states,
                n_celltypes,
            )

            shuffled_mis[i] = (
                H_XC
                + H_YC
                - H_C
                - H_XYC_perm
            )

            if (
                shuffled_mis[i] < 0
                and shuffled_mis[i] > -1e-10
            ):
                shuffled_mis[i] = 0.0

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


    mean_perm = np.mean(shuffled_mis)

    std_perm = np.std(shuffled_mis)

    perc95 = np.percentile(
        shuffled_mis,
        95
    )


    n_exceedances = np.sum(
        shuffled_mis >= true_mi
    )


    empirical_p_value = (
        n_exceedances + 1
    ) / (
        num_permutations + 1
    )


    return {

        "mean_perm_mi":
            mean_perm,

        "std_perm_mi":
            std_perm,

        "95_perm_mi":
            perc95,

        "n_perm_ge_true_mi":
            int(n_exceedances),

        "empirical_p_value":
            empirical_p_value,
    }


# ============================================================
# PARALLEL WORKER
# ============================================================

def initializer(
    _gene_data,
    _barcodes,
    _cell_types,
    _celltype_indices,
    _H_barcode,
    _H_XC,
    _H_C,
    _n_barcodes,
    _n_celltypes,
):

    global gene_data
    global barcodes
    global cell_types
    global celltype_indices

    global H_barcode
    global H_XC
    global H_C

    global n_barcodes
    global n_celltypes


    gene_data = _gene_data

    barcodes = _barcodes

    cell_types = _cell_types

    celltype_indices = (
        _celltype_indices
    )


    H_barcode = _H_barcode

    H_XC = _H_XC

    H_C = _H_C


    n_barcodes = _n_barcodes

    n_celltypes = _n_celltypes


def empty_result(gene):

    return {

        "gene":
            gene,

        "true_mi":
            np.nan,

        "mean_perm_mi":
            np.nan,

        "95_perm_mi":
            np.nan,

        "std_perm_mi":
            np.nan,

        "n_perm_ge_true_mi":
            np.nan,

        "empirical_p_value":
            np.nan,

        "significant_percentile_95":
            False,

        "significant_uncorrected_p_0.05":
            False,

        "normalized_MI_label_entropy":
            np.nan,
    }


# ============================================================
# PROCESS ONE GENE
# ============================================================

def process_gene(gene):

    try:

        # ----------------------------------------------------
        # Expression values
        # ----------------------------------------------------

        expression = (
            gene_data[gene]
            .to_numpy()
        )


        if (
            len(expression)
            != len(barcodes)
        ):
            return empty_result(gene)


        # ----------------------------------------------------
        # SAME DISCRETIZATION AS ORIGINAL SCRIPT
        #
        # Original:
        #
        # np.round(..., decimals=1).astype(np.float32)
        # ----------------------------------------------------

        expression_rounded = np.round(
            expression,
            decimals=1,
        ).astype(np.float32)


        # Integer encode expression states
        _, expression_codes = np.unique(
            expression_rounded,
            return_inverse=True,
        )


        expression_codes = (
            expression_codes.astype(
                np.int64
            )
        )


        n_expression_states = (
            expression_codes.max() + 1
        )


        # ----------------------------------------------------
        # H(Y,C)
        # ----------------------------------------------------

        H_YC = joint_entropy_2(
            expression_codes,
            cell_types,
            n_celltypes,
        )


        # ----------------------------------------------------
        # TRUE H(X,Y,C)
        # ----------------------------------------------------

        H_XYC = joint_entropy_3(
            barcodes,
            expression_codes,
            cell_types,
            n_expression_states,
            n_celltypes,
        )


        # ----------------------------------------------------
        # TRUE CONDITIONAL MI
        #
        # I(X;Y|C)
        # =
        # H(X,C) + H(Y,C) - H(C) - H(X,Y,C)
        # ----------------------------------------------------

        true_mi = (
            H_XC
            + H_YC
            - H_C
            - H_XYC
        )


        if (
            true_mi < 0
            and true_mi > -1e-10
        ):
            true_mi = 0.0


        # ----------------------------------------------------
        # Reproducible RNG
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
        # PERMUTATION NULL
        # ----------------------------------------------------

        permutation_results = (
            calc_noise_mi(

                expression_codes=
                    expression_codes,

                barcodes_local=
                    barcodes,

                cell_types_local=
                    cell_types,

                celltype_indices=
                    celltype_indices,

                true_mi=
                    true_mi,

                H_XC=
                    H_XC,

                H_YC=
                    H_YC,

                H_C=
                    H_C,

                n_expression_states=
                    n_expression_states,

                n_celltypes=
                    n_celltypes,

                num_permutations=
                    num_permutations,

                rng=
                    rng,
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
        # SAME NORMALIZATION AS ORIGINAL SCRIPT
        # ----------------------------------------------------

        norm_mi_label_entropy = (
            true_mi
            / (
                H_barcode
                + EPSILON
            )
        )


        # ----------------------------------------------------
        # SIGNIFICANCE
        # ----------------------------------------------------

        significant_percentile = (
            np.isfinite(perc95)
            and true_mi > perc95
        )


        significant_uncorrected = (
            np.isfinite(empirical_p)
            and empirical_p < 0.05
        )


        return {

            "gene":
                gene,

            "true_mi":
                true_mi,

            "mean_perm_mi":
                mean_perm,

            "95_perm_mi":
                perc95,

            "std_perm_mi":
                std_perm,

            "n_perm_ge_true_mi":
                n_exceedances,

            "empirical_p_value":
                empirical_p,

            "significant_percentile_95":
                significant_percentile,

            "significant_uncorrected_p_0.05":
                significant_uncorrected,

            "normalized_MI_label_entropy":
                norm_mi_label_entropy,
        }


    except Exception as e:

        print(
            f"[PROCESS_GENE ERROR] "
            f"{gene}: {e}"
        )

        return empty_result(gene)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    if dataset_name not in file_dict:

        raise ValueError(
            f"Unknown dataset "
            f"'{dataset_name}'. "
            f"Available datasets: "
            f"{list(file_dict.keys())}"
        )


    print(
        f"Processing dataset: "
        f"{dataset_name} "
        f"with "
        f"{num_permutations} "
        f"permutations..."
    )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    with open(
        file_dict[dataset_name],
        "rb"
    ) as f:

        full_df = pickle.load(f)


    # --------------------------------------------------------
    # COLUMN STRUCTURE
    #
    # [... genes ... | cell type | barcode]
    # --------------------------------------------------------

    cell_type_column = (
        full_df.columns[-2]
    )

    barcode_column = (
        full_df.columns[-1]
    )


    # --------------------------------------------------------
    # FILTER SMALL LINEAGES
    #
    # Keep only lineages with at least MIN_CELLS_PER_LINEAGE
    # cells. This filtering is applied identically to every
    # dataset BEFORE gene extraction, integer encoding,
    # entropy calculations, and permutations.
    # --------------------------------------------------------

    n_cells_before_filter = len(full_df)

    lineage_counts_before = (
        full_df[barcode_column]
        .value_counts(dropna=False)
    )

    n_lineages_before_filter = len(
        lineage_counts_before
    )

    lineages_to_keep = (
        lineage_counts_before[
            lineage_counts_before
            >= MIN_CELLS_PER_LINEAGE
        ]
        .index
    )

    full_df = (
        full_df[
            full_df[barcode_column]
            .isin(lineages_to_keep)
        ]
        .copy()
    )

    n_cells_after_filter = len(full_df)

    lineage_counts_after = (
        full_df[barcode_column]
        .value_counts(dropna=False)
    )

    n_lineages_after_filter = len(
        lineage_counts_after
    )

    print(
        "\n------------------ "
        "LINEAGE FILTER "
        "------------------"
    )

    print(
        f"Minimum cells per lineage: "
        f"{MIN_CELLS_PER_LINEAGE}"
    )

    print(
        f"Cells before filtering: "
        f"{n_cells_before_filter:,}"
    )

    print(
        f"Cells after filtering: "
        f"{n_cells_after_filter:,}"
    )

    print(
        f"Cells removed: "
        f"{n_cells_before_filter - n_cells_after_filter:,}"
    )

    print(
        f"Lineages before filtering: "
        f"{n_lineages_before_filter:,}"
    )

    print(
        f"Lineages after filtering: "
        f"{n_lineages_after_filter:,}"
    )

    print(
        f"Lineages removed: "
        f"{n_lineages_before_filter - n_lineages_after_filter:,}"
    )


    if n_cells_after_filter == 0:

        raise ValueError(
            f"No cells remain after filtering "
            f"for lineages with >= "
            f"{MIN_CELLS_PER_LINEAGE} cells."
        )


    print(
        f"Cell-type column: "
        f"{cell_type_column}"
    )

    print(
        f"Barcode column: "
        f"{barcode_column}"
    )


    # --------------------------------------------------------
    # GENES
    # --------------------------------------------------------

    gene_data = (
        full_df.iloc[:, :-2]
        .copy()
    )

    genes = (
        gene_data.columns.tolist()
    )


    # --------------------------------------------------------
    # INTEGER ENCODE BARCODE + CELL TYPE
    #
    # Only category identity matters for MI.
    # --------------------------------------------------------

    barcode_values = (
        full_df.iloc[:, -1]
        .to_numpy()
    )

    celltype_values = (
        full_df.iloc[:, -2]
        .to_numpy()
    )


    _, barcodes = np.unique(
        barcode_values,
        return_inverse=True,
    )


    _, cell_types = np.unique(
        celltype_values,
        return_inverse=True,
    )


    barcodes = (
        barcodes.astype(np.int64)
    )

    cell_types = (
        cell_types.astype(np.int64)
    )


    n_barcodes = (
        barcodes.max() + 1
    )

    n_celltypes = (
        cell_types.max() + 1
    )


    # --------------------------------------------------------
    # PRECOMPUTE CELL-TYPE INDICES ONCE
    #
    # Used for within-cell-type permutations.
    # --------------------------------------------------------

    celltype_indices = [

        np.where(
            cell_types == c
        )[0]

        for c in range(
            n_celltypes
        )
    ]


    # --------------------------------------------------------
    # CONSTANT ENTROPIES
    # --------------------------------------------------------

    H_barcode = (
        calculate_entropy(
            barcodes
        )
    )


    H_C = (
        calculate_entropy(
            cell_types
        )
    )


    # H(X,C)
    H_XC = joint_entropy_2(
        barcodes,
        cell_types,
        n_celltypes,
    )


    print(
        f"Entropy of barcodes H(X): "
        f"{H_barcode:.4f}"
    )

    print(
        f"Entropy of cell type H(C): "
        f"{H_C:.4f}"
    )

    print(
        f"Joint entropy H(X,C): "
        f"{H_XC:.4f}"
    )


    print(
        f"Number of cells: "
        f"{len(barcodes):,}"
    )

    print(
        f"Number of lineages: "
        f"{n_barcodes:,}"
    )

    print(
        f"Number of cell types: "
        f"{n_celltypes:,}"
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
    # WARM UP NUMBA
    #
    # Avoid first worker paying compilation cost repeatedly.
    # --------------------------------------------------------

    _ = entropy_from_integer_labels(
        np.array(
            [0, 0, 1, 1, 2],
            dtype=np.int64,
        )
    )


    # --------------------------------------------------------
    # PARALLEL PROCESSING
    # --------------------------------------------------------

    with mp.Pool(

        processes=n_cores,

        initializer=initializer,

        initargs=(

            gene_data,

            barcodes,

            cell_types,

            celltype_indices,

            H_barcode,

            H_XC,

            H_C,

            n_barcodes,

            n_celltypes,
        ),

    ) as pool:


        results = list(

            tqdm(

                pool.imap(
                    process_gene,
                    genes
                ),

                total=len(genes),

                desc="Processing genes",
            )
        )


    mi_df = pd.DataFrame(
        results
    )


    # ========================================================
    # BH CORRECTION
    # ========================================================

    valid_p_mask = np.isfinite(
        mi_df[
            "empirical_p_value"
        ].to_numpy()
    )


    mi_df[
        "bh_adjusted_p_value"
    ] = np.nan


    mi_df[
        "significant_BH_FDR_0.05"
    ] = False


    if valid_p_mask.sum() > 0:


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
    # ADD FILTER METADATA TO OUTPUT
    # --------------------------------------------------------

    mi_df[
        "min_cells_per_lineage"
    ] = MIN_CELLS_PER_LINEAGE

    mi_df[
        "n_cells_before_lineage_filter"
    ] = n_cells_before_filter

    mi_df[
        "n_cells_after_lineage_filter"
    ] = n_cells_after_filter

    mi_df[
        "n_lineages_before_filter"
    ] = n_lineages_before_filter

    mi_df[
        "n_lineages_after_filter"
    ] = n_lineages_after_filter


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    mi_df = (
        mi_df
        .sort_values(
            "true_mi",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    n_valid = (
        valid_p_mask.sum()
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
        "\n------------------ "
        "SIGNIFICANCE SUMMARY "
        "------------------"
    )


    print(
        "Tested genes/features "
        "with valid p-values: "
        f"{n_valid:,}"
    )


    if n_valid > 0:


        print(

            "Significant using original "
            "95th-percentile rule: "

            f"{n_sig_percentile:,} "

            f"("
            f"{100 * n_sig_percentile / n_valid:.2f}%"
            f")"
        )


        print(

            "Significant using "
            "uncorrected empirical "
            "p < 0.05: "

            f"{n_sig_uncorrected:,} "

            f"("
            f"{100 * n_sig_uncorrected / n_valid:.2f}%"
            f")"
        )


        print(

            f"Significant after BH "
            f"correction at FDR "
            f"{fdr_alpha}: "

            f"{n_sig_bh:,} "

            f"("
            f"{100 * n_sig_bh / n_valid:.2f}%"
            f")"
        )


        print(
            "\nPermutation p-value counts:"
        )


        p_value_counts = (

            mi_df.loc[
                valid_p_mask,
                "empirical_p_value"
            ]

            .value_counts()

            .sort_index()
        )


        for p_value, count in (
            p_value_counts.items()
        ):

            print(

                f"  p = "
                f"{p_value:.6f}: "

                f"{count:,} "
                f"genes/features"
            )


    # ========================================================
    # SAVE
    # ========================================================

    output_path = os.path.join(

        SAVE_DIR,

        (
            f"conditional_MI_label_entropy_"
            f"{dataset_name}_"
            f"min{MIN_CELLS_PER_LINEAGE}cells_per_lineage_"
            f"{num_permutations}_"
            f"permutations_with_BH.csv"
        ),
    )


    mi_df.to_csv(
        output_path,
        index=False
    )


    print(
        f"\nDone. Results saved to "
        f"{output_path}"
    )