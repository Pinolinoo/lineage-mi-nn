import argparse
import hashlib
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


# ============================================================
# CONFIGURATION / CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Calculate leave-one-mitochondrial-gene-out conditional mutual "
        "information I(lineage; MT-gene expression | MT fraction excluding gene)."
    )
)

parser.add_argument(
    "dataset",
    help="Dataset key, e.g. treutlein, weinreb, junker, you1, knoblich_day16_1",
)

parser.add_argument(
    "--num-permutations",
    type=int,
    default=10000,
    help="Number of within-MT-fraction-bin lineage permutations (default: 10000).",
)

parser.add_argument(
    "--quantile-bins",
    type=int,
    nargs="+",
    default=[4, 5, 10],
    help=(
        "Requested numbers of quantile bins for the leave-one-out MT fraction. "
        "Default: 4 5 10. Duplicate quantile edges are dropped automatically."
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
    "--random-seed",
    type=int,
    default=12345,
    help="Base random seed (default: 12345).",
)

parser.add_argument(
    "--fdr-alpha",
    type=float,
    default=0.05,
    help="BH FDR level, applied separately for each quantile-bin setting.",
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
num_permutations = args.num_permutations
QUANTILE_BINS = sorted(set(args.quantile_bins))
MIN_CELLS_PER_LINEAGE = args.min_cells_per_lineage
random_seed = args.random_seed
fdr_alpha = args.fdr_alpha
SUM_CHUNK_SIZE = args.sum_chunk_size

if num_permutations < 1:
    raise ValueError("--num-permutations must be >= 1")

if any(q < 2 for q in QUANTILE_BINS):
    raise ValueError("Every --quantile-bins value must be >= 2")

if MIN_CELLS_PER_LINEAGE < 1:
    raise ValueError("--min-cells-per-lineage must be >= 1")

if SUM_CHUNK_SIZE < 1:
    raise ValueError("--sum-chunk-size must be >= 1")


SAVE_DIR = Path("./conditional_mi_MT_fraction_LOO_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-12
n_cores = int(os.environ.get("NSLOTS", os.cpu_count() or 1))

base_dir = Path(
    "/fast/AG_Metzger/philipp/python_files/gene_signature/"
    "mutual_information_approach/November_25_ReDo/"
    "preprocessing/pkl_files_filtered"
)

# Main scRNA-seq datasets used in the lineage-MI analysis.
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
# BASIC ENTROPY FUNCTIONS
# ============================================================


def calculate_entropy(labels):
    """Shannon entropy of a discrete variable in bits."""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -(probs * np.log2(probs + EPSILON)).sum()


@njit
def entropy_from_integer_labels(labels):
    """Fast entropy for non-negative integer labels, in bits."""
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


# ============================================================
# REPRODUCIBLE SEEDS
# ============================================================


def make_task_seed(dataset, gene, requested_q, base_seed):
    """Stable seed independent of process order and Python hash randomization."""
    key = f"{dataset}|{gene}|q{requested_q}|{base_seed}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


# ============================================================
# JOINT ENTROPY HELPERS
# ============================================================


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
    _mt_genes,
    _mt_expression_log,
    _mt_linear,
    _mt_total_linear,
    _total_linear,
    _barcode_values,
):
    global mt_genes
    global mt_expression_log
    global mt_linear
    global mt_total_linear
    global total_linear
    global barcode_values
    global mt_gene_to_index

    mt_genes = _mt_genes
    mt_expression_log = _mt_expression_log
    mt_linear = _mt_linear
    mt_total_linear = _mt_total_linear
    total_linear = _total_linear
    barcode_values = _barcode_values
    mt_gene_to_index = {gene: i for i, gene in enumerate(mt_genes)}


# ============================================================
# HELPERS FOR ONE GENE / ONE BINNING CHOICE
# ============================================================


def empty_result(gene, requested_q, error_message=""):
    return {
        "gene": gene,
        "requested_quantile_bins": requested_q,
        "actual_quantile_bins": np.nan,
        "n_cells_used": np.nan,
        "n_lineages_used": np.nan,
        "quantile_edges": "",
        "unconditional_mi": np.nan,
        "conditional_mi_mt_fraction_LOO": np.nan,
        "mean_perm_cmi": np.nan,
        "std_perm_cmi": np.nan,
        "95_perm_cmi": np.nan,
        "n_perm_ge_true_cmi": np.nan,
        "empirical_p_value": np.nan,
        "significant_percentile_95": False,
        "significant_uncorrected_p_0.05": False,
        "H_lineage": np.nan,
        "H_mt_fraction_bin": np.nan,
        "H_lineage_given_mt_fraction": np.nan,
        "normalized_CMI_by_H_lineage": np.nan,
        "normalized_CMI_by_H_lineage_given_mt_fraction": np.nan,
        "fraction_unconditional_MI_remaining": np.nan,
        "mean_mt_fraction_LOO": np.nan,
        "median_mt_fraction_LOO": np.nan,
        "error": error_message,
    }



def process_task(task):
    gene, requested_q = task

    try:
        gene_idx = mt_gene_to_index[gene]

        # ----------------------------------------------------
        # GENE EXPRESSION, SAME DISCRETIZATION AS MI PIPELINE
        # ----------------------------------------------------
        expression = mt_expression_log[:, gene_idx]
        expression_rounded = np.round(expression, decimals=1).astype(np.float32)

        # ----------------------------------------------------
        # LEAVE-ONE-GENE-OUT MT FRACTION
        #
        # Crucially, gene g is removed from BOTH numerator and
        # denominator, so the conditioning variable does not
        # directly contain the tested gene.
        #
        # R_MT,-g = sum(other MT normalized counts)
        #           / sum(all other normalized counts)
        # ----------------------------------------------------
        gene_linear = mt_linear[:, gene_idx]
        numerator = mt_total_linear - gene_linear
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
                "Fewer than 2 valid cells after LOO fraction calculation.",
            )

        expression_valid = expression_rounded[valid_mask]
        numerator_valid = np.maximum(numerator[valid_mask], 0.0)
        denominator_valid = denominator[valid_mask]
        r_mt_loo = numerator_valid / denominator_valid
        barcodes_valid_raw = barcode_values[valid_mask]

        # ----------------------------------------------------
        # QUANTILE-BIN CONTINUOUS MT FRACTION
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
            return empty_result(gene, requested_q, f"qcut failed: {exc}")

        r_codes = np.asarray(r_codes)

        # qcut can theoretically yield NaN labels in pathological cases.
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
                "Fewer than 2 cells remain after quantile binning.",
            )

        r_codes = r_codes.astype(np.int64)
        actual_q = int(np.unique(r_codes).size)

        if actual_q < 2:
            return empty_result(
                gene,
                requested_q,
                "Leave-one-out MT fraction has fewer than 2 distinct quantile bins.",
            )

        # Re-encode in case duplicate bins were dropped.
        _, r_codes = np.unique(r_codes, return_inverse=True)
        r_codes = r_codes.astype(np.int64)
        n_r_states = int(r_codes.max() + 1)

        # Integer encode expression states and lineage labels.
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
        # ENTROPIES FOR TRUE CONDITIONAL MI
        #
        # I(L;X|R) = H(L,R) + H(X,R) - H(R) - H(L,X,R)
        # ----------------------------------------------------
        H_L = calculate_entropy(barcodes)
        H_R = calculate_entropy(r_codes)
        H_X = calculate_entropy(expression_codes)

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

        true_cmi = H_LR + H_XR - H_R - H_LXR

        if true_cmi < 0 and true_cmi > -1e-10:
            true_cmi = 0.0

        # Also calculate ordinary MI on exactly the same cells.
        H_LX = joint_entropy_2(
            barcodes,
            expression_codes,
            n_expression_states,
        )
        unconditional_mi = H_L + H_X - H_LX

        if unconditional_mi < 0 and unconditional_mi > -1e-10:
            unconditional_mi = 0.0

        # The natural conditional ceiling for information about lineage is H(L|R).
        H_L_given_R = H_LR - H_R

        normalized_by_H_L = true_cmi / (H_L + EPSILON)
        normalized_by_H_L_given_R = true_cmi / (H_L_given_R + EPSILON)

        fraction_mi_remaining = (
            true_cmi / unconditional_mi
            if unconditional_mi > EPSILON
            else np.nan
        )

        # ----------------------------------------------------
        # CONDITIONAL PERMUTATION NULL
        #
        # Shuffle lineage labels WITHIN R_MT,-g bins only.
        # This preserves H(L,R), H(X,R), and H(R), so only
        # H(L,X,R) must be recalculated for each permutation.
        # ----------------------------------------------------
        bin_indices = [
            np.where(r_codes == r)[0]
            for r in range(n_r_states)
        ]

        rng = np.random.default_rng(
            make_task_seed(
                dataset=dataset_name,
                gene=gene,
                requested_q=requested_q,
                base_seed=random_seed,
            )
        )

        shuffled_cmis = np.empty(num_permutations, dtype=np.float64)

        constant_term = H_LR + H_XR - H_R

        for i in range(num_permutations):
            permuted_barcodes = barcodes.copy()

            for indices in bin_indices:
                permuted_barcodes[indices] = rng.permutation(
                    barcodes[indices]
                )

            H_LXR_perm = joint_entropy_3(
                permuted_barcodes,
                expression_codes,
                r_codes,
                n_expression_states,
                n_r_states,
            )

            shuffled_cmis[i] = constant_term - H_LXR_perm

            if shuffled_cmis[i] < 0 and shuffled_cmis[i] > -1e-10:
                shuffled_cmis[i] = 0.0

        mean_perm = float(np.mean(shuffled_cmis))
        std_perm = float(np.std(shuffled_cmis))
        perc95 = float(np.percentile(shuffled_cmis, 95))

        n_exceedances = int(np.sum(shuffled_cmis >= true_cmi))
        empirical_p = (n_exceedances + 1) / (num_permutations + 1)

        significant_percentile = bool(true_cmi > perc95)
        significant_uncorrected = bool(empirical_p < 0.05)

        edge_string = ";".join(f"{edge:.10g}" for edge in bin_edges)

        return {
            "gene": gene,
            "requested_quantile_bins": requested_q,
            "actual_quantile_bins": actual_q,
            "n_cells_used": len(barcodes),
            "n_lineages_used": n_lineages,
            "quantile_edges": edge_string,
            "unconditional_mi": unconditional_mi,
            "conditional_mi_mt_fraction_LOO": true_cmi,
            "mean_perm_cmi": mean_perm,
            "std_perm_cmi": std_perm,
            "95_perm_cmi": perc95,
            "n_perm_ge_true_cmi": n_exceedances,
            "empirical_p_value": empirical_p,
            "significant_percentile_95": significant_percentile,
            "significant_uncorrected_p_0.05": significant_uncorrected,
            "H_lineage": H_L,
            "H_mt_fraction_bin": H_R,
            "H_lineage_given_mt_fraction": H_L_given_R,
            "normalized_CMI_by_H_lineage": normalized_by_H_L,
            "normalized_CMI_by_H_lineage_given_mt_fraction": normalized_by_H_L_given_R,
            "fraction_unconditional_MI_remaining": fraction_mi_remaining,
            "mean_mt_fraction_LOO": float(np.mean(r_mt_loo)),
            "median_mt_fraction_LOO": float(np.median(r_mt_loo)),
            "error": "",
        }

    except Exception as exc:
        return empty_result(gene, requested_q, f"{type(exc).__name__}: {exc}")


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
    print(f"Permutations per MT gene / binning choice: {num_permutations:,}")
    print(f"Requested quantile-bin choices: {QUANTILE_BINS}")
    print(f"Minimum cells per lineage: {MIN_CELLS_PER_LINEAGE}")
    print(f"CPU cores: {n_cores}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    with open(file_dict[dataset_name], "rb") as f:
        full_df = pickle.load(f)

    # The original filtered pkl files have genes followed by barcode_cont.
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

    gene_data = full_df[gene_columns]
    barcode_values_main = full_df[barcode_column].to_numpy(copy=True)

    # --------------------------------------------------------
    # IDENTIFY MITOCHONDRIAL GENES
    # Same definition used in the expression-matching analysis.
    # --------------------------------------------------------
    mt_genes_main = [
        gene
        for gene in gene_columns
        if str(gene).lower().startswith("mt-")
    ]

    if len(mt_genes_main) < 2:
        raise ValueError(
            f"Only {len(mt_genes_main)} mitochondrial gene(s) found. "
            "At least 2 are required for leave-one-gene-out MT fraction."
        )

    print(f"\nDetected {len(mt_genes_main)} MT genes:")
    print(mt_genes_main)

    # --------------------------------------------------------
    # RECONSTRUCT NORMALIZED COUNT SCALE
    #
    # Stored expression is log1p(total-count-normalized expression).
    # expm1 therefore returns the normalized-count scale. Because
    # total-count normalization multiplies all genes in a cell by
    # the same factor, the MT/total ratio is invariant to that factor.
    #
    # We sum in chunks to avoid constructing a second full dense
    # cell x gene matrix in memory.
    # --------------------------------------------------------
    n_cells = len(gene_data)
    total_linear_main = np.zeros(n_cells, dtype=np.float64)

    print("\nReconstructing total normalized counts from log1p values...")

    for start in tqdm(
        range(0, len(gene_columns), SUM_CHUNK_SIZE),
        desc="Summing genes",
    ):
        end = min(start + SUM_CHUNK_SIZE, len(gene_columns))

        block = gene_data.iloc[:, start:end].to_numpy(
            dtype=np.float64,
            copy=False,
        )

        # Numerical guard: tiny negative log-values, if any, are harmless,
        # but true negative expression values would indicate an incompatible
        # preprocessing format.
        if np.nanmin(block) < -1e-8:
            raise ValueError(
                "Negative expression values found. This script assumes "
                "non-negative log1p-transformed normalized expression."
            )

        total_linear_main += np.expm1(block).sum(axis=1)

    mt_expression_log_main = gene_data[mt_genes_main].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    mt_linear_main = np.expm1(mt_expression_log_main)
    mt_total_linear_main = mt_linear_main.sum(axis=1)

    if np.any(~np.isfinite(total_linear_main)):
        raise ValueError("Non-finite total reconstructed counts encountered.")

    if np.any(total_linear_main <= 0):
        raise ValueError("At least one cell has non-positive reconstructed total counts.")

    # --------------------------------------------------------
    # WARM UP NUMBA
    # --------------------------------------------------------
    _ = entropy_from_integer_labels(
        np.array([0, 0, 1, 1, 2], dtype=np.int64)
    )

    # --------------------------------------------------------
    # PARALLEL PROCESSING: MT GENE x QUANTILE CHOICE
    # --------------------------------------------------------
    tasks = [
        (gene, q)
        for q in QUANTILE_BINS
        for gene in mt_genes_main
    ]

    # No reason to launch more processes than independent tasks.
    processes = min(n_cores, len(tasks))

    with mp.Pool(
        processes=processes,
        initializer=initializer,
        initargs=(
            mt_genes_main,
            mt_expression_log_main,
            mt_linear_main,
            mt_total_linear_main,
            total_linear_main,
            barcode_values_main,
        ),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_task, tasks),
                total=len(tasks),
                desc="MT gene x quantile setting",
            )
        )

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # BH CORRECTION
    #
    # Each requested quantile setting is a separate sensitivity
    # analysis, so BH is applied across the MT genes within each
    # setting, not across the robustness settings pooled together.
    # --------------------------------------------------------
    result_df["bh_adjusted_p_value"] = np.nan
    result_df["significant_BH_FDR_0.05"] = False

    for q in QUANTILE_BINS:
        q_mask = (
            (result_df["requested_quantile_bins"] == q)
            & np.isfinite(result_df["empirical_p_value"])
        )

        if q_mask.sum() == 0:
            continue

        reject_bh, p_values_bh, _, _ = multipletests(
            result_df.loc[q_mask, "empirical_p_value"].to_numpy(),
            alpha=fdr_alpha,
            method="fdr_bh",
        )

        result_df.loc[q_mask, "bh_adjusted_p_value"] = p_values_bh
        result_df.loc[q_mask, "significant_BH_FDR_0.05"] = reject_bh

    # --------------------------------------------------------
    # ADD ANALYSIS METADATA
    # --------------------------------------------------------
    result_df["dataset"] = dataset_name
    result_df["num_permutations"] = num_permutations
    result_df["min_cells_per_lineage"] = MIN_CELLS_PER_LINEAGE
    result_df["n_cells_before_lineage_filter"] = n_cells_before_filter
    result_df["n_cells_after_lineage_filter"] = n_cells_after_filter
    result_df["n_lineages_before_filter"] = n_lineages_before_filter
    result_df["n_lineages_after_lineage_filter"] = n_lineages_after_filter
    result_df["mt_fraction_definition"] = (
        "sum(expm1(other_MT_genes)) / sum(expm1(all_genes_except_tested_gene))"
    )
    result_df["expression_discretization"] = "round(log1p expression, 1 decimal)"
    result_df["permutation_scheme"] = "shuffle lineage labels within LOO MT-fraction bins"

    result_df = result_df.sort_values(
        ["requested_quantile_bins", "conditional_mi_mt_fraction_LOO"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n================== SUMMARY ==================")

    for q in QUANTILE_BINS:
        sub = result_df[result_df["requested_quantile_bins"] == q]
        valid = np.isfinite(sub["empirical_p_value"])

        print(f"\nRequested quantile bins: {q}")
        print(f"  MT genes with valid result: {int(valid.sum())}/{len(sub)}")

        if valid.any():
            print(
                "  Median fraction of ordinary MI remaining after conditioning: "
                f"{sub.loc[valid, 'fraction_unconditional_MI_remaining'].median():.3f}"
            )
            print(
                "  BH-significant MT genes: "
                f"{int(sub['significant_BH_FDR_0.05'].sum())}/{int(valid.sum())}"
            )

        errors = sub.loc[sub["error"].astype(str) != "", ["gene", "error"]]
        if len(errors) > 0:
            print("  Errors:")
            for _, row in errors.iterrows():
                print(f"    {row['gene']}: {row['error']}")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    q_label = "-".join(str(q) for q in QUANTILE_BINS)
    output_path = SAVE_DIR / (
        f"conditional_MI_MT_fraction_LOO_{dataset_name}_"
        f"q{q_label}_{num_permutations}_permutations_with_BH.csv"
    )

    result_df.to_csv(output_path, index=False)

    print(f"\nDone. Results saved to: {output_path}")
