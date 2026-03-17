#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
import os
import multiprocessing as mp
from numba import njit
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import List, Tuple

# ------------------ ARGPARSE ------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute per-gene MI vs barcode labels for a subset (topN + fixed randomN). "
            "Top genes are selected from an existing reference MI CSV (all-genes run). "
            "Outputs are written to a separate output directory."
        )
    )

    p.add_argument("dataset_name", type=str, help="e.g. treutlein / weinreb / junker")
    p.add_argument("--num_permutations", type=int, default=50)

    p.add_argument("--discretization", type=str, default="round", choices=["round", "bin", "none"])
    p.add_argument("--round_decimals", type=int, default=1)
    p.add_argument("--bin_width", type=float, default=0.05)

    p.add_argument("--reference_base_dir", type=str, required=True)
    p.add_argument("--output_base_dir", type=str, required=True)

    p.add_argument("--top_n", type=int, default=200)
    p.add_argument("--random_n", type=int, default=200)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--reference_nperm", type=int, default=50)

    return p.parse_args()

# ------------------ CONFIGURATION ------------------

EPSILON = 1e-9

def get_n_cores() -> int:
    """
    Prefer SLURM_CPUS_PER_TASK (SLURM), then NSLOTS (SGE-ish), else os.cpu_count().
    """
    for k in ("SLURM_CPUS_PER_TASK", "NSLOTS"):
        v = os.environ.get(k, "")
        if v.strip().isdigit():
            return max(1, int(v))
    return max(1, int(os.cpu_count() or 1))

n_cores = get_n_cores()

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

def calculate_entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())

def make_save_subdir(save_dir_base: str, discretization: str, round_decimals: int, bin_width: float) -> Tuple[str, str]:
    if discretization == "round":
        sub = f"discret_round_d{round_decimals}"
    elif discretization == "bin":
        bw = str(bin_width).replace(".", "p")
        sub = f"discret_bin_w{bw}"
    else:
        sub = "discret_none"
    out_dir = os.path.join(save_dir_base, sub)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, sub

def selection_cache_path(output_base_dir: str, save_tag: str, dataset_name: str,
                         top_n: int, random_n: int, seed: int, reference_nperm: int) -> Path:
    sel_dir = Path(output_base_dir) / "selected_gene_sets"
    sel_dir.mkdir(parents=True, exist_ok=True)
    return sel_dir / f"selected_{dataset_name}_{save_tag}_top{top_n}_rand{random_n}_seed{seed}_ref{reference_nperm}.tsv"

def find_reference_results_csv_strict(reference_base_dir: str, save_tag: str, dataset_name: str, reference_nperm: int) -> str:
    ref_path = os.path.join(
        reference_base_dir,
        save_tag,
        f"MI_label_entropy_{dataset_name}_{reference_nperm}_permutations_{save_tag}.csv"
    )
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            "Reference file not found:\n"
            f"  {ref_path}\n\n"
            "This must exist in the REFERENCE base dir (all-genes run)."
        )
    return ref_path

def select_top_and_fixed_random(
    reference_base_dir: str,
    output_base_dir: str,
    save_tag: str,
    dataset_name: str,
    all_gene_names: List[str],
    top_n: int,
    random_n: int,
    seed: int,
    reference_nperm: int,
) -> Tuple[List[str], List[str]]:
    cache = selection_cache_path(output_base_dir, save_tag, dataset_name, top_n, random_n, seed, reference_nperm)
    all_set = set(all_gene_names)

    if cache.exists():
        sel = pd.read_csv(cache, sep="\t")
        top_genes = sel.loc[sel["set"] == "top", "gene"].tolist()
        rand_genes = sel.loc[sel["set"] == "random", "gene"].tolist()

        top_genes = [g for g in top_genes if g in all_set]
        rand_genes = [g for g in rand_genes if g in all_set]
        if len(top_genes) == top_n and len(rand_genes) == random_n:
            print(f"✅ Loaded cached selection: {cache}")
            return top_genes, rand_genes

        print(f"[WARN] Cached selection invalid/incomplete; recomputing: {cache}")

    ref_csv = find_reference_results_csv_strict(reference_base_dir, save_tag, dataset_name, reference_nperm)
    ref_df = pd.read_csv(ref_csv)

    if "gene" not in ref_df.columns or "true_mi" not in ref_df.columns:
        raise ValueError(f"Reference CSV missing required columns 'gene' and/or 'true_mi': {ref_csv}")

    ref_df = ref_df[ref_df["gene"].isin(all_set)].copy()
    if ref_df.empty:
        raise ValueError(f"Reference CSV has no overlap with current dataset genes: {ref_csv}")

    ref_df = ref_df.sort_values("true_mi", ascending=False)
    top_genes = ref_df["gene"].head(top_n).tolist()
    if len(top_genes) < top_n:
        raise ValueError(f"Not enough genes to select top_n={top_n} from reference: got {len(top_genes)}")

    remaining = [g for g in all_gene_names if g not in set(top_genes)]
    if len(remaining) < random_n:
        raise ValueError(f"Not enough remaining genes for random_n={random_n}: remaining={len(remaining)}")

    rng = np.random.default_rng(seed)
    rand_genes = rng.choice(np.array(remaining), size=random_n, replace=False).tolist()

    sel_df = pd.DataFrame(
        {"gene": top_genes + rand_genes, "set": (["top"] * len(top_genes)) + (["random"] * len(rand_genes))}
    )
    sel_df.to_csv(cache, sep="\t", index=False)
    print(f"✅ Saved cached selection: {cache}")

    return top_genes, rand_genes

# ------------------ MI CORE (NUMBA) ------------------

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
    x = np.asarray(x)
    mask = np.isfinite(x)
    if not np.all(mask):
        finite_vals = x[mask]
        if finite_vals.size == 0:
            return np.zeros_like(x, dtype=np.int64)
        x = x.copy()
        x[~mask] = finite_vals.min()

    if discretization == "none":
        xq = x.astype(np.float32)
    elif discretization == "round":
        xq = np.round(x.astype(np.float32), decimals=round_decimals)
    elif discretization == "bin":
        x32 = x.astype(np.float32)
        xmin = np.min(x32)
        xq = np.floor((x32 - xmin) / np.float32(bin_width)).astype(np.int32)
        return xq.astype(np.int64)
    else:
        raise ValueError(f"Unknown discretization: {discretization}")

    _, inv = np.unique(xq, return_inverse=True)
    return inv.astype(np.int64)

def calc_mi_from_arrays(x, barcodes, discretization, round_decimals=1, bin_width=0.05):
    order = np.argsort(barcodes, kind="mergesort")
    b = barcodes[order]
    x = x[order]

    A = discretize_values(x, discretization, round_decimals, bin_width)
    max_index = int(A.max()) + 1 if A.size > 0 else 1

    change_indices = np.where(np.diff(b) != 0)[0] + 1
    split_arrays = np.split(A, change_indices)

    bincounts_all = np.stack([np.bincount(arr, minlength=max_index) for arr in split_arrays])
    comb_array = bincounts_all[:, ~np.all(bincounts_all == 0, axis=0)].T

    p_g_loc = comb_array / np.sum(comb_array)
    p_gene = np.sum(p_g_loc, axis=1)
    p_location = np.sum(p_g_loc, axis=0)

    return mutual_information(p_g_loc, p_gene, p_location)

def calc_noise_mi(x, barcodes, num_permutations, discretization, round_decimals=1, bin_width=0.05):
    if len(x) != len(barcodes):
        return np.nan, np.nan, np.nan

    shuffled_mis = np.empty(num_permutations, dtype=np.float64)
    for i in range(num_permutations):
        permuted = np.random.permutation(barcodes)
        shuffled_mis[i] = calc_mi_from_arrays(x, permuted, discretization, round_decimals, bin_width)

    mean_perm = np.mean(shuffled_mis)
    std_perm = np.std(shuffled_mis)
    perc95 = np.percentile(shuffled_mis, 95)
    return mean_perm, std_perm, perc95

def count_states(x, discretization, round_decimals=1, bin_width=0.05):
    state_ids = discretize_values(x, discretization, round_decimals, bin_width)
    n_states = int(state_ids.max()) + 1 if state_ids.size > 0 else 0
    return n_states

# ------------------ PARALLEL WORKER ------------------

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
            return {"gene": gene, "n_states": np.nan, "true_mi": np.nan,
                    "mean_perm_mi": np.nan, "95_perm_mi": np.nan, "std_perm_mi": np.nan,
                    "normalized_MI_label_entropy": np.nan}

        n_states = count_states(x, discretization, round_decimals, bin_width)
        true_mi = calc_mi_from_arrays(x, barcodes, discretization, round_decimals, bin_width)
        mean_perm, std_perm, perc95 = calc_noise_mi(x, barcodes, num_permutations, discretization, round_decimals, bin_width)
        norm_mi_label_entropy = true_mi / (H_barcode + EPSILON)

        return {"gene": gene, "n_states": n_states, "true_mi": true_mi,
                "mean_perm_mi": mean_perm, "95_perm_mi": perc95, "std_perm_mi": std_perm,
                "normalized_MI_label_entropy": norm_mi_label_entropy}

    except Exception as e:
        print(f"[PROCESS_GENE ERROR] {gene}: {e}")
        return {"gene": gene, "n_states": np.nan, "true_mi": np.nan,
                "mean_perm_mi": np.nan, "95_perm_mi": np.nan, "std_perm_mi": np.nan,
                "normalized_MI_label_entropy": np.nan}

# ------------------ MAIN ------------------

if __name__ == "__main__":
    args = parse_args()

    # Safety: match your intended mode
    if args.discretization != "round" or args.round_decimals != 1:
        raise ValueError("Run with --discretization round --round_decimals 1 (round d1).")

    dataset_name = args.dataset_name
    if dataset_name not in file_dict:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Options: {list(file_dict.keys())}")

    # OUTPUT: write subset results here
    OUTPUT_SAVE_DIR, save_tag = make_save_subdir(args.output_base_dir, args.discretization, args.round_decimals, args.bin_width)

    print(f"🔍 Processing dataset: {dataset_name}")
    print(f"    permutations: {args.num_permutations}")
    print(f"    discretization tag: {save_tag}")
    print(f"    CPUs requested (SLURM_CPUS_PER_TASK/NSLOTS): {n_cores}")
    print(f"    REFERENCE (read): {args.reference_base_dir}")
    print(f"    OUTPUT (write):   {args.output_base_dir}")
    print(f"    output_dir(tag):  {OUTPUT_SAVE_DIR}")

    # Load data
    with open(file_dict[dataset_name], "rb") as f:
        full_df = pickle.load(f)

    full_df = full_df.sort_values("barcode_cont", kind="mergesort")
    barcodes = full_df["barcode_cont"].values.copy()
    gene_data = full_df.iloc[:, :-1]
    all_genes = gene_data.columns.tolist()

    H_barcode = calculate_entropy(barcodes)
    print(f"📊 Entropy of barcodes H(Y): {H_barcode:.6f}")

    # Select subset
    top_genes, rand_genes = select_top_and_fixed_random(
        reference_base_dir=args.reference_base_dir,
        output_base_dir=args.output_base_dir,
        save_tag=save_tag,
        dataset_name=dataset_name,
        all_gene_names=all_genes,
        top_n=args.top_n,
        random_n=args.random_n,
        seed=args.random_seed,
        reference_nperm=args.reference_nperm,
    )
    selected_genes = top_genes + rand_genes
    print(f"✅ Selected genes: top={len(top_genes)} + random={len(rand_genes)} = total={len(selected_genes)}")
    print(f"    cache: {selection_cache_path(args.output_base_dir, save_tag, dataset_name, args.top_n, args.random_n, args.random_seed, args.reference_nperm)}")

    # Parallel pool uses n_cores
    with mp.Pool(
        processes=n_cores,
        initializer=initializer,
        initargs=(gene_data, barcodes, H_barcode, args.num_permutations, args.discretization, args.round_decimals, args.bin_width),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_gene, selected_genes),
                total=len(selected_genes),
                desc="⏳ Processing selected genes",
            )
        )

    mi_df = pd.DataFrame(results)

    output_path = os.path.join(
        OUTPUT_SAVE_DIR,
        f"MI_label_entropy_{dataset_name}_{args.num_permutations}_permutations_{save_tag}"
        f"_SUBSET_top{args.top_n}_rand{args.random_n}_seed{args.random_seed}_ref{args.reference_nperm}.csv"
    )
    mi_df.to_csv(output_path, index=False)
    print(f"✅ Done. Results saved to {output_path}")
