#!/usr/bin/env python3
# mi_cellid_vs_gene_group_numba.py
#
# Parallel MI(cell_id ; gene-group-expression) on random gene sets + mito gene set
# - Safe for many cores (e.g. 40) because we DO NOT pickle the full df per task.
# - Workers read the dataframe from a global (inherited via fork).
# - Uses SLURM_CPUS_PER_TASK / N_WORKERS / NSLOTS if available, otherwise falls back to 40.
#
# Notes:
#   MI(cell_id ; expr_bin) = H(expr_bin) because expr_bin is deterministic given cell_id.
#   So we compute entropy of the discretized expression-vector IDs.

import sys
import os
import pickle
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from numba import njit
from tqdm import tqdm
from pathlib import Path

# ------------------ CONFIGURATION ------------------

dataset_name = sys.argv[1]
n_combos = int(sys.argv[2]) if len(sys.argv) > 2 else 150

SAVE_DIR = "./mi_cellid_vs_gene_groups_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Same style as your numba/random-sets script (filtered pkls)
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
}

# Rounding for discretization (matches your previous logic: round(..., 1))
ROUND_DECIMALS = int(os.environ.get("ROUND_DECIMALS", "1"))

# ------------------ GLOBALS (for forked workers) ------------------

DF_GLOBAL = None  # set in __main__ before Pool is created

# ------------------ FUNCTIONS ------------------

def get_mito_names(df: pd.DataFrame):
    return [c for c in df.columns if str(c).lower().startswith("mt-")]

def choose_n_cores(default: int = 40) -> int:
    # Priority: explicit N_WORKERS (from SLURM wrapper) > SLURM_CPUS_PER_TASK > NSLOTS > default
    for key in ("N_WORKERS", "SLURM_CPUS_PER_TASK", "NSLOTS"):
        val = os.environ.get(key)
        if val and str(val).isdigit():
            return int(val)
    return default

@njit
def entropy_base2_from_counts(counts: np.ndarray) -> float:
    total = 0
    for i in range(counts.shape[0]):
        total += counts[i]
    if total == 0:
        return 0.0

    H = 0.0
    for i in range(counts.shape[0]):
        c = counts[i]
        if c > 0:
            p = c / total
            H -= p * np.log2(p)
    return H

def mi_cellid_vs_gene_group(full_df: pd.DataFrame, gene_set) -> float:
    """
    Computes MI(cell_id ; expr_bin) for the given gene_set.

    Since expr_bin is a deterministic function of cell_id (each cell maps to exactly one bin),
    I(cell;expr_bin) = H(expr_bin).

    Steps:
      1) extract expression matrix for gene_set
      2) discretize per cell by rounding
      3) map each unique discretized row -> expr_bin id
      4) compute entropy of expr_bin distribution
    """
    expr = full_df[gene_set].to_numpy(copy=False)
    expr_rounded = np.round(expr, ROUND_DECIMALS)

    # Unique discretized expression vectors (rows) -> expr_bin id for each cell
    # (np.unique(axis=0) is efficient and avoids python tuples/crosstab)
    _, expr_ids = np.unique(expr_rounded, axis=0, return_inverse=True)

    counts = np.bincount(expr_ids)
    return float(entropy_base2_from_counts(counts))

def process_combo(combo):
    # IMPORTANT: do not pass df through multiprocessing queues
    return mi_cellid_vs_gene_group(DF_GLOBAL, combo)

# ------------------ MAIN ------------------

if __name__ == "__main__":
    print(f"🚀 Processing {dataset_name} with {n_combos} random gene sets")
    n_cores = choose_n_cores(default=40)
    print(f"🧵 Using n_cores = {n_cores}")
    print(f"🔢 ROUND_DECIMALS = {ROUND_DECIMALS}")

    if dataset_name not in file_dict:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Allowed: {list(file_dict.keys())}")

    # Load df once in parent
    with open(file_dict[dataset_name], "rb") as f:
        df = pickle.load(f)

    # Make globally visible to forked workers BEFORE Pool creation
    DF_GLOBAL = df

    # Gene columns: follow your previous assumption (last column is barcode_cont/similar)
    gene_names = list(df.columns[:-1])

    mito_genes = get_mito_names(df)
    n_genes = len(mito_genes)
    if n_genes == 0:
        raise ValueError("No mito genes found (columns starting with 'mt-').")

    print(f"🧬 Mito genes: {n_genes}")

    print("🔀 Sampling gene sets...")
    random_sets = [random.sample(gene_names, n_genes) for _ in range(n_combos)]

    print("⚙️ Starting parallel MI calculation...")
    chunksize = max(1, len(random_sets) // (n_cores * 4))

    with mp.Pool(processes=n_cores, maxtasksperchild=10) as pool:
        MI_random_samples = list(
            tqdm(
                pool.imap_unordered(process_combo, random_sets, chunksize=chunksize),
                total=len(random_sets),
            )
        )

    print("➕ Calculating MI for mitochondrial genes...")
    mito_mi = mi_cellid_vs_gene_group(df, mito_genes)
    MI_random_samples.append(mito_mi)

    print("💾 Saving results...")
    output_path = os.path.join(
        SAVE_DIR,
        f"MI_cellid_vs_gene_groups_{dataset_name}_{n_combos}_numba.pkl"
    )
    with open(output_path, "wb") as f:
        pickle.dump(MI_random_samples, f)

    print(f"✅ Done. Results saved to {output_path}")
