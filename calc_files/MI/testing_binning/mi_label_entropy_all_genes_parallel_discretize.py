import sys
import numpy as np
import pandas as pd
import pickle
import os
import multiprocessing as mp
from numba import njit
from tqdm import tqdm
from pathlib import Path
import argparse

# ------------------ ARGPARSE ------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-gene MI vs barcode labels with configurable discretization."
    )
    p.add_argument("dataset_name", type=str, help="e.g. treutlein / weinreb / ...")
    p.add_argument("--num_permutations", type=int, default=50)

    # discretization scenarios
    p.add_argument(
        "--discretization",
        type=str,
        default="round",
        choices=["round", "bin", "none"],
        help="How to discretize gene values before MI.",
    )
    p.add_argument("--round_decimals", type=int, default=1, help="Used if discretization=round.")
    p.add_argument("--bin_width", type=float, default=0.05, help="Used if discretization=bin.")

    p.add_argument(
        "--save_dir",
        type=str,
        default="./mi_analysis_results_MI_label_entropy_all_genes_parallel",
        help="Base output directory (a subfolder per discretization is created).",
    )
    return p.parse_args()

# ------------------ CONFIGURATION ------------------

EPSILON = 1e-9
n_cores = int(os.environ.get("NSLOTS", os.cpu_count()))

base_dir = Path(
    "/fast/AG_Metzger/philipp/python_files/gene_signature/"
    "mutual_information_approach/November_25_ReDo/preprocessing/pkl_files_filtered"
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

def make_save_subdir(save_dir_base, discretization, round_decimals, bin_width):
    if discretization == "round":
        sub = f"discret_round_d{round_decimals}"
    elif discretization == "bin":
        # keep a filesystem-friendly representation
        bw = str(bin_width).replace(".", "p")
        sub = f"discret_bin_w{bw}"
    else:
        sub = "discret_none"
    out_dir = os.path.join(save_dir_base, sub)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, sub

# ------------------ MUTUAL INFORMATION CORE (NUMBA) ------------------

@njit
def mutual_information(p_g_loc, p_gene, p_location):
    MI = 0.0
    for i in range(p_g_loc.shape[0]):
        for j in range(p_g_loc.shape[1]):
            if p_g_loc[i, j] > 0.0:
                denom = p_gene[i] * p_location[j]
                if denom > 0.0:
                    MI += p_g_loc[i, j] * np.log2(p_g_loc[i, j] / denom)
    return MI

def discretize_values(x, discretization, round_decimals=1, bin_width=0.05):
    """
    x: (n_cells,) float array
    Returns integer-coded discrete states (n_cells,) int64.
    """
    # ensure finite
    x = np.asarray(x)
    mask = np.isfinite(x)
    if not np.all(mask):
        # set non-finite to a dedicated value by replacing with min finite
        finite_vals = x[mask]
        if finite_vals.size == 0:
            return np.zeros_like(x, dtype=np.int64)
        x = x.copy()
        x[~mask] = finite_vals.min()

    if discretization == "none":
        # exact float states -> unique mapping (can be many states)
        # using float32 reduces accidental tiny diffs + memory
        xq = x.astype(np.float32)

    elif discretization == "round":
        xq = np.round(x.astype(np.float32), decimals=round_decimals)

    elif discretization == "bin":
        # stable binning even if values could be negative:
        # shift by per-gene min so bins start at 0
        x32 = x.astype(np.float32)
        xmin = np.min(x32)
        # bin index: floor((x - xmin)/bin_width)
        xq = np.floor((x32 - xmin) / np.float32(bin_width)).astype(np.int32)
        return xq.astype(np.int64)

    else:
        raise ValueError(f"Unknown discretization: {discretization}")

    # map quantized floats to contiguous ints
    _, inv = np.unique(xq, return_inverse=True)
    return inv.astype(np.int64)

def calc_mi_from_arrays(x, barcodes, discretization, round_decimals=1, bin_width=0.05):
    """
    x: (n_cells,) float
    barcodes: (n_cells,) labels
    """
    # assumes barcodes already sorted outside for speed, but works regardless
    order = np.argsort(barcodes, kind="mergesort")
    b = barcodes[order]
    x = x[order]

    A = discretize_values(x, discretization, round_decimals, bin_width)
    max_index = int(A.max()) + 1 if A.size > 0 else 1

    change_indices = np.where(np.diff(b) != 0)[0] + 1
    split_arrays = np.split(A, change_indices)

    bincounts_all = np.stack(
        [np.bincount(arr, minlength=max_index) for arr in split_arrays]
    )

    # drop empty categories
    comb_array = bincounts_all[:, ~np.all(bincounts_all == 0, axis=0)].T
    p_g_loc = comb_array / np.sum(comb_array)

    p_gene = np.sum(p_g_loc, axis=1)
    p_location = np.sum(p_g_loc, axis=0)

    return mutual_information(p_g_loc, p_gene, p_location)

def calc_noise_mi(x, barcodes, num_permutations, discretization, round_decimals=1, bin_width=0.05):
    """
    Permutation-based null MI via barcode shuffling.
    x: (n_cells,) float array
    barcodes: (n_cells,) array of labels
    """
    if len(x) != len(barcodes):
        return np.nan, np.nan, np.nan

    shuffled_mis = np.empty(num_permutations, dtype=np.float64)
    try:
        for i in range(num_permutations):
            permuted = np.random.permutation(barcodes)
            shuffled_mis[i] = calc_mi_from_arrays(
                x, permuted, discretization, round_decimals, bin_width
            )
    except Exception as e:
        print(f"[ERROR in calc_noise_mi] issue: {e}")
        return np.nan, np.nan, np.nan

    mean_perm = np.mean(shuffled_mis)
    std_perm = np.std(shuffled_mis)
    perc95 = np.percentile(shuffled_mis, 95)
    return mean_perm, std_perm, perc95

# ------------------ PARALLEL WORKER ------------------
def count_states(x, discretization, round_decimals=1, bin_width=0.05):
    """
    Returns (n_states, state_ids) where state_ids are integer-coded states per cell.
    """
    state_ids = discretize_values(x, discretization, round_decimals, bin_width)
    n_states = int(state_ids.max()) + 1 if state_ids.size > 0 else 0
    return n_states, state_ids

def initializer(_gene_data, _barcodes, _H_barcode, _num_permutations, _disc, _rd, _bw):
    global gene_data, barcodes, H_barcode, num_permutations, discretization, round_decimals, bin_width
    gene_data = _gene_data
    barcodes = _barcodes
    H_barcode = _H_barcode
    num_permutations = _num_permutations
    discretization = _disc
    round_decimals = _rd
    bin_width = _bw

def process_gene(gene):
    try:
        x = gene_data[gene].values.astype(np.float32)
        if x.shape[0] != barcodes.shape[0]:
            return {
                "gene": gene,
                "n_states": np.nan,
                "true_mi": np.nan,
                "mean_perm_mi": np.nan,
                "95_perm_mi": np.nan,
                "std_perm_mi": np.nan,
                "normalized_MI_label_entropy": np.nan,
            }

        # NEW: diagnostic state count
        n_states, _ = count_states(x, discretization, round_decimals, bin_width)

        true_mi = calc_mi_from_arrays(
            x, barcodes, discretization, round_decimals, bin_width
        )

        mean_perm, std_perm, perc95 = calc_noise_mi(
            x, barcodes, num_permutations, discretization, round_decimals, bin_width
        )

        norm_mi_label_entropy = true_mi / (H_barcode + EPSILON)

        return {
            "gene": gene,
            "n_states": n_states,  # NEW
            "true_mi": true_mi,
            "mean_perm_mi": mean_perm,
            "95_perm_mi": perc95,
            "std_perm_mi": std_perm,
            "normalized_MI_label_entropy": norm_mi_label_entropy,
        }

    except Exception as e:
        print(f"[PROCESS_GENE ERROR] {gene}: {e}")
        return {
            "gene": gene,
            "n_states": np.nan,
            "true_mi": np.nan,
            "mean_perm_mi": np.nan,
            "95_perm_mi": np.nan,
            "std_perm_mi": np.nan,
            "normalized_MI_label_entropy": np.nan,
        }


# ------------------ MAIN ------------------

if __name__ == "__main__":
    args = parse_args()

    dataset_name = args.dataset_name
    num_permutations = args.num_permutations
    discretization = args.discretization
    round_decimals = args.round_decimals
    bin_width = args.bin_width

    SAVE_DIR, save_tag = make_save_subdir(args.save_dir, discretization, round_decimals, bin_width)

    print(f"🔍 Processing dataset: {dataset_name}")
    print(f"    permutations: {num_permutations}")
    print(f"    discretization: {discretization} (tag: {save_tag})")
    if discretization == "round":
        print(f"    round_decimals: {round_decimals}")
    if discretization == "bin":
        print(f"    bin_width: {bin_width}")
    print(f"    cores: {n_cores}")
    print(f"    output_dir: {SAVE_DIR}")

    if dataset_name not in file_dict:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Options: {list(file_dict.keys())}")

    # Load data
    with open(file_dict[dataset_name], "rb") as f:
        full_df = pickle.load(f)

    # Sort and split genes / barcodes
    full_df = full_df.sort_values("barcode_cont", kind="mergesort")
    barcodes = full_df["barcode_cont"].values.copy()
    gene_data = full_df.iloc[:, :-1]  # all gene columns
    genes = gene_data.columns.tolist()

    # Entropy of barcodes H(Y)
    H_barcode = calculate_entropy(barcodes)
    print(f"📊 Entropy of barcodes H(Y) for {dataset_name}: {H_barcode:.6f}")

    # Parallel processing
    with mp.Pool(
        processes=n_cores,
        initializer=initializer,
        initargs=(gene_data, barcodes, H_barcode, num_permutations, discretization, round_decimals, bin_width),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_gene, genes),
                total=len(genes),
                desc="⏳ Processing genes",
            )
        )

    mi_df = pd.DataFrame(results)

    # encode settings in filename for easy bash + bookkeeping
    output_path = os.path.join(
        SAVE_DIR,
        f"MI_label_entropy_{dataset_name}_{num_permutations}_permutations_{save_tag}.csv",
    )
    mi_df.to_csv(output_path, index=False)
    print(f"✅ Done. Results saved to {output_path}")
