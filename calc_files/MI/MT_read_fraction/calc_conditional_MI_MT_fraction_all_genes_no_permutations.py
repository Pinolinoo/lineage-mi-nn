import argparse
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm


# ============================================================
# CONFIGURATION / CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Calculate ordinary MI and conditional MI for ALL genes while "
        "conditioning on a leave-one-gene-out mitochondrial transcript fraction. "
        "No permutations are performed."
    )
)

parser.add_argument(
    "dataset",
    help="Dataset key, e.g. treutlein, weinreb, junker, you1, knoblich_day16_1",
)

parser.add_argument(
    "--quantile-bins",
    type=int,
    nargs="+",
    default=[5],
    help=(
        "Requested numbers of quantile bins for the mitochondrial fraction. "
        "Default: 5. You can optionally run robustness settings, e.g. "
        "--quantile-bins 4 5 10."
    ),
)

parser.add_argument(
    "--min-cells-per-lineage",
    type=int,
    default=1,
    help=(
        "Minimum number of cells required per lineage before analysis. "
        "Default 1 keeps the same cells as the original MI analysis."
    ),
)

parser.add_argument(
    "--sum-chunk-size",
    type=int,
    default=500,
    help=(
        "Number of genes per chunk when reconstructing total normalized counts "
        "from the log1p matrix (default: 500)."
    ),
)

args = parser.parse_args()

dataset_name = args.dataset
QUANTILE_BINS = sorted(set(args.quantile_bins))
MIN_CELLS_PER_LINEAGE = args.min_cells_per_lineage
SUM_CHUNK_SIZE = args.sum_chunk_size

if any(q < 2 for q in QUANTILE_BINS):
    raise ValueError("Every --quantile-bins value must be >= 2")

if MIN_CELLS_PER_LINEAGE < 1:
    raise ValueError("--min-cells-per-lineage must be >= 1")

if SUM_CHUNK_SIZE < 1:
    raise ValueError("--sum-chunk-size must be >= 1")


SAVE_DIR = Path("./conditional_mi_MT_fraction_all_genes_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-12
n_cores = int(os.environ.get("NSLOTS", os.cpu_count() or 1))

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
}


# ============================================================
# ENTROPY HELPERS
# ============================================================

@njit
def entropy_from_integer_labels(labels):
    """Fast Shannon entropy for non-negative integer labels, in bits."""
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

    p = count / n
    entropy -= p * np.log2(p)

    return entropy


def joint_entropy_2(variable1, variable2, n_variable2):
    """H(variable1, variable2) for integer-encoded variables."""
    joint_codes = (
        variable1.astype(np.int64) * np.int64(n_variable2)
        + variable2.astype(np.int64)
    )
    return entropy_from_integer_labels(joint_codes)


def joint_entropy_3(
    variable1,
    variable2,
    variable3,
    n_variable2,
    n_variable3,
):
    """H(variable1, variable2, variable3) for integer-encoded variables."""
    joint_codes = (
        (
            variable1.astype(np.int64) * np.int64(n_variable2)
            + variable2.astype(np.int64)
        )
        * np.int64(n_variable3)
        + variable3.astype(np.int64)
    )
    return entropy_from_integer_labels(joint_codes)


# ============================================================
# WORKER GLOBALS
# ============================================================

def initializer(
    _gene_data,
    _mt_gene_set,
    _mt_total_linear,
    _total_linear,
    _barcode_values,
):
    global gene_data
    global mt_gene_set
    global mt_total_linear
    global total_linear
    global barcode_values

    gene_data = _gene_data
    mt_gene_set = _mt_gene_set
    mt_total_linear = _mt_total_linear
    total_linear = _total_linear
    barcode_values = _barcode_values


# ============================================================
# ONE GENE / ONE BINNING CHOICE
# ============================================================

def empty_result(gene, requested_q, is_mt_gene, error_message=""):
    return {
        "gene": gene,
        "is_mt_gene": is_mt_gene,
        "requested_quantile_bins": requested_q,
        "actual_quantile_bins": np.nan,
        "n_cells_used": np.nan,
        "n_lineages_used": np.nan,
        "quantile_edges": "",
        "unconditional_mi": np.nan,
        "conditional_mi_mt_fraction_LOO": np.nan,
        "delta_cmi_minus_mi": np.nan,
        "H_lineage": np.nan,
        "H_mt_fraction_bin": np.nan,
        "H_lineage_given_mt_fraction": np.nan,
        "mean_mt_fraction_LOO": np.nan,
        "median_mt_fraction_LOO": np.nan,
        "error": error_message,
    }


def process_task(task):
    gene, requested_q = task
    is_mt_gene = gene in mt_gene_set

    try:
        # ----------------------------------------------------
        # TESTED GENE EXPRESSION
        # Same 0.1 discretization as the ordinary MI analysis.
        # ----------------------------------------------------
        expression_log = gene_data[gene].to_numpy(dtype=np.float64, copy=False)
        expression_rounded = np.round(
            expression_log,
            decimals=1,
        ).astype(np.float32)

        # Reverse log1p to normalized-count scale for the fraction.
        gene_linear = np.expm1(expression_log)

        # ----------------------------------------------------
        # LEAVE-ONE-GENE-OUT MITOCHONDRIAL FRACTION
        #
        # For an MT gene:
        #   numerator   = all MT counts except tested MT gene
        #   denominator = all counts except tested gene
        #
        # For a non-MT gene:
        #   numerator   = all MT counts
        #   denominator = all counts except tested gene
        #
        # Thus the conditioning variable never directly contains
        # the tested gene itself.
        # ----------------------------------------------------
        if is_mt_gene:
            numerator = mt_total_linear - gene_linear
        else:
            numerator = mt_total_linear

        denominator = total_linear - gene_linear

        valid_mask = (
            np.isfinite(expression_rounded)
            & np.isfinite(numerator)
            & np.isfinite(denominator)
            & (denominator > 0)
            & (numerator >= -1e-10)
        )

        if valid_mask.sum() < 2:
            return empty_result(
                gene,
                requested_q,
                is_mt_gene,
                "Fewer than 2 valid cells after LOO fraction calculation.",
            )

        expression_valid = expression_rounded[valid_mask]
        numerator_valid = np.maximum(numerator[valid_mask], 0.0)
        denominator_valid = denominator[valid_mask]
        r_mt_loo = numerator_valid / denominator_valid
        barcodes_valid_raw = barcode_values[valid_mask]

        # ----------------------------------------------------
        # QUANTILE-BIN MITOCHONDRIAL FRACTION
        # ----------------------------------------------------
        try:
            r_codes, bin_edges = pd.qcut(
                r_mt_loo,
                q=requested_q,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
        except ValueError as exc:
            return empty_result(
                gene,
                requested_q,
                is_mt_gene,
                f"qcut failed: {exc}",
            )

        r_codes = np.asarray(r_codes)
        qcut_valid = pd.notna(r_codes)

        if not np.all(qcut_valid):
            expression_valid = expression_valid[qcut_valid]
            r_mt_loo = r_mt_loo[qcut_valid]
            barcodes_valid_raw = barcodes_valid_raw[qcut_valid]
            r_codes = r_codes[qcut_valid]

        if len(r_codes) < 2:
            return empty_result(
                gene,
                requested_q,
                is_mt_gene,
                "Fewer than 2 cells remain after quantile binning.",
            )

        r_codes = r_codes.astype(np.int64)
        actual_q = int(np.unique(r_codes).size)

        if actual_q < 2:
            return empty_result(
                gene,
                requested_q,
                is_mt_gene,
                "MT fraction has fewer than 2 distinct quantile bins.",
            )

        # Re-encode bins in case duplicate quantile edges were dropped.
        _, r_codes = np.unique(r_codes, return_inverse=True)
        r_codes = r_codes.astype(np.int64)
        n_r_states = int(r_codes.max() + 1)

        # Integer encode gene-expression states and lineage labels.
        _, expression_codes = np.unique(
            expression_valid,
            return_inverse=True,
        )
        expression_codes = expression_codes.astype(np.int64)
        n_expression_states = int(expression_codes.max() + 1)

        _, barcodes = np.unique(
            barcodes_valid_raw,
            return_inverse=True,
        )
        barcodes = barcodes.astype(np.int64)
        n_lineages = int(barcodes.max() + 1)

        # ----------------------------------------------------
        # ORDINARY MI
        # I(L;X) = H(L) + H(X) - H(L,X)
        # ----------------------------------------------------
        H_L = entropy_from_integer_labels(barcodes)
        H_X = entropy_from_integer_labels(expression_codes)

        H_LX = joint_entropy_2(
            barcodes,
            expression_codes,
            n_expression_states,
        )

        unconditional_mi = H_L + H_X - H_LX

        if unconditional_mi < 0 and unconditional_mi > -1e-10:
            unconditional_mi = 0.0

        # ----------------------------------------------------
        # CONDITIONAL MI
        # I(L;X|R) = H(L,R) + H(X,R) - H(R) - H(L,X,R)
        # ----------------------------------------------------
        H_R = entropy_from_integer_labels(r_codes)

        H_LR = joint_entropy_2(
            barcodes,
            r_codes,
            n_r_states,
        )

        H_XR = joint_entropy_2(
            expression_codes,
            r_codes,
            n_r_states,
        )

        H_LXR = joint_entropy_3(
            barcodes,
            expression_codes,
            r_codes,
            n_expression_states,
            n_r_states,
        )

        conditional_mi = H_LR + H_XR - H_R - H_LXR

        if conditional_mi < 0 and conditional_mi > -1e-10:
            conditional_mi = 0.0

        H_L_given_R = H_LR - H_R

        edge_string = ";".join(f"{edge:.10g}" for edge in bin_edges)

        return {
            "gene": gene,
            "is_mt_gene": is_mt_gene,
            "requested_quantile_bins": requested_q,
            "actual_quantile_bins": actual_q,
            "n_cells_used": len(barcodes),
            "n_lineages_used": n_lineages,
            "quantile_edges": edge_string,
            "unconditional_mi": unconditional_mi,
            "conditional_mi_mt_fraction_LOO": conditional_mi,
            "delta_cmi_minus_mi": conditional_mi - unconditional_mi,
            "H_lineage": H_L,
            "H_mt_fraction_bin": H_R,
            "H_lineage_given_mt_fraction": H_L_given_R,
            "mean_mt_fraction_LOO": float(np.mean(r_mt_loo)),
            "median_mt_fraction_LOO": float(np.median(r_mt_loo)),
            "error": "",
        }

    except Exception as exc:
        return empty_result(
            gene,
            requested_q,
            is_mt_gene,
            f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if dataset_name not in file_dict:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {list(file_dict.keys())}"
        )

    print(f"Processing dataset: {dataset_name}")
    print("Permutations: NONE")
    print(f"Requested quantile-bin choices: {QUANTILE_BINS}")
    print(f"Minimum cells per lineage: {MIN_CELLS_PER_LINEAGE}")
    print(f"CPU cores: {n_cores}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    with open(file_dict[dataset_name], "rb") as f:
        full_df = pickle.load(f)

    if "barcode_cont" in full_df.columns:
        barcode_column = "barcode_cont"
    else:
        barcode_column = full_df.columns[-1]

    gene_columns = [
        col for col in full_df.columns
        if col != barcode_column
    ]

    # --------------------------------------------------------
    # OPTIONAL LINEAGE FILTER
    # --------------------------------------------------------
    n_cells_before_filter = len(full_df)
    lineage_counts_before = full_df[barcode_column].value_counts(dropna=False)
    n_lineages_before_filter = len(lineage_counts_before)

    if MIN_CELLS_PER_LINEAGE > 1:
        lineages_to_keep = lineage_counts_before[
            lineage_counts_before >= MIN_CELLS_PER_LINEAGE
        ].index

        full_df = full_df[
            full_df[barcode_column].isin(lineages_to_keep)
        ].copy()

    n_cells_after_filter = len(full_df)
    n_lineages_after_filter = full_df[barcode_column].nunique(dropna=False)

    if n_cells_after_filter == 0:
        raise ValueError("No cells remain after lineage filtering.")

    print("\n------------------ LINEAGE FILTER ------------------")
    print(f"Cells before:    {n_cells_before_filter:,}")
    print(f"Cells after:     {n_cells_after_filter:,}")
    print(f"Lineages before: {n_lineages_before_filter:,}")
    print(f"Lineages after:  {n_lineages_after_filter:,}")

    gene_data_main = full_df[gene_columns].copy()
    barcode_values_main = full_df[barcode_column].to_numpy(copy=True)

    # --------------------------------------------------------
    # IDENTIFY MITOCHONDRIAL GENES
    # --------------------------------------------------------
    mt_genes_main = [
        gene
        for gene in gene_columns
        if str(gene).lower().startswith("mt-")
    ]
    mt_gene_set_main = set(mt_genes_main)

    if len(mt_genes_main) < 2:
        raise ValueError(
            f"Only {len(mt_genes_main)} mitochondrial gene(s) found. "
            "At least 2 are required for the leave-one-out MT fraction."
        )

    print(f"\nNumber of tested genes: {len(gene_columns):,}")
    print(f"Detected MT genes: {len(mt_genes_main)}")
    print(mt_genes_main)

    # --------------------------------------------------------
    # RECONSTRUCT NORMALIZED-COUNT SCALE
    #
    # Stored expression is log1p(total-count-normalized expression).
    # expm1 reverses the log1p transformation. The within-cell
    # MT/total ratio is invariant to the total-count scaling factor.
    # --------------------------------------------------------
    n_cells = len(gene_data_main)
    total_linear_main = np.zeros(n_cells, dtype=np.float64)

    print("\nReconstructing total normalized counts from log1p values...")

    for start in tqdm(
        range(0, len(gene_columns), SUM_CHUNK_SIZE),
        desc="Summing genes",
    ):
        end = min(start + SUM_CHUNK_SIZE, len(gene_columns))

        block = gene_data_main.iloc[:, start:end].to_numpy(
            dtype=np.float64,
            copy=False,
        )

        if np.nanmin(block) < -1e-8:
            raise ValueError(
                "Negative expression values found. This script assumes "
                "non-negative log1p-transformed normalized expression."
            )

        total_linear_main += np.expm1(block).sum(axis=1)

    mt_expression_log = gene_data_main[mt_genes_main].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    mt_total_linear_main = np.expm1(mt_expression_log).sum(axis=1)

    if np.any(~np.isfinite(total_linear_main)):
        raise ValueError("Non-finite total reconstructed counts encountered.")

    if np.any(total_linear_main <= 0):
        raise ValueError(
            "At least one cell has non-positive reconstructed total counts."
        )

    # --------------------------------------------------------
    # WARM UP NUMBA
    # --------------------------------------------------------
    _ = entropy_from_integer_labels(
        np.array([0, 0, 1, 1, 2], dtype=np.int64)
    )

    # --------------------------------------------------------
    # PARALLEL PROCESSING: ALL GENES x QUANTILE SETTINGS
    # --------------------------------------------------------
    tasks = [
        (gene, q)
        for q in QUANTILE_BINS
        for gene in gene_columns
    ]

    processes = min(n_cores, len(tasks))

    print(f"\nIndependent gene/bin tasks: {len(tasks):,}")
    print(f"Worker processes: {processes}")

    with mp.Pool(
        processes=processes,
        initializer=initializer,
        initargs=(
            gene_data_main,
            mt_gene_set_main,
            mt_total_linear_main,
            total_linear_main,
            barcode_values_main,
        ),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_task, tasks),
                total=len(tasks),
                desc="All genes x quantile setting",
            )
        )

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # ADD METADATA
    # --------------------------------------------------------
    result_df["dataset"] = dataset_name
    result_df["min_cells_per_lineage"] = MIN_CELLS_PER_LINEAGE
    result_df["n_cells_before_lineage_filter"] = n_cells_before_filter
    result_df["n_cells_after_lineage_filter"] = n_cells_after_filter
    result_df["n_lineages_before_filter"] = n_lineages_before_filter
    result_df["n_lineages_after_lineage_filter"] = n_lineages_after_filter
    result_df["mt_fraction_definition"] = (
        "MT numerator excludes tested gene if MT; total denominator excludes tested gene for all genes"
    )
    result_df["expression_discretization"] = "round(log1p expression, 1 decimal)"
    result_df["permutations"] = "none"

    result_df = result_df.sort_values(
        ["requested_quantile_bins", "unconditional_mi"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n================== SUMMARY ==================")

    for q in QUANTILE_BINS:
        sub = result_df[result_df["requested_quantile_bins"] == q]
        valid = (
            np.isfinite(sub["unconditional_mi"])
            & np.isfinite(sub["conditional_mi_mt_fraction_LOO"])
        )

        print(f"\nRequested quantile bins: {q}")
        print(f"  Valid genes: {int(valid.sum()):,}/{len(sub):,}")
        print(
            "  MT genes among valid genes: "
            f"{int(sub.loc[valid, 'is_mt_gene'].sum())}"
        )

        if valid.any():
            print(
                "  Mean raw MI:  "
                f"{sub.loc[valid, 'unconditional_mi'].mean():.6f}"
            )
            print(
                "  Mean raw CMI: "
                f"{sub.loc[valid, 'conditional_mi_mt_fraction_LOO'].mean():.6f}"
            )

        errors = sub.loc[
            sub["error"].astype(str) != "",
            ["gene", "error"],
        ]
        print(f"  Errors: {len(errors):,}")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    q_label = "-".join(str(q) for q in QUANTILE_BINS)
    output_path = SAVE_DIR / (
        f"conditional_MI_MT_fraction_all_genes_{dataset_name}_"
        f"q{q_label}_no_permutations.csv"
    )

    result_df.to_csv(output_path, index=False)

    print(f"\nDone. Results saved to: {output_path}")
