# mi_random_gene_sets_numba.py
#
# Parallel MI on random gene sets + mito gene set
# - Safe for many cores (e.g. 40) because we DO NOT pickle the full df per task.
# - Workers read the dataframe from a global (inherited via fork).
# - Uses SLURM_CPUS_PER_TASK / N_WORKERS if available, otherwise falls back to 40.

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

SAVE_DIR = "./mi_group_mt_and_random_sets_results"
os.makedirs(SAVE_DIR, exist_ok=True)

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

# ------------------ GLOBALS (for forked workers) ------------------

DF_GLOBAL = None  # will be set in __main__ before Pool is created

# ------------------ FUNCTIONS ------------------

def get_mito_names(counts: pd.DataFrame):
    return [name for name in counts.columns if name.lower().startswith("mt-")]

@njit
def mutual_information(p_g_loc, p_gene, p_location):
    MI = 0.0
    for i in range(p_gene.shape[0]):
        for j in range(p_location.shape[0]):
            if p_gene[i] != 0 and p_g_loc[i, j] != 0:
                mu = p_g_loc[i, j] / (p_gene[i] * p_location[j])
                MI += p_g_loc[i, j] * np.log2(mu)
    return MI

def calc_mutual_inf_for_network_w_specific_gene_set(full_df: pd.DataFrame, gene_set):
    full_df = full_df.sort_values("barcode_cont")
    df = full_df[gene_set]
    barcodes = np.array(full_df["barcode_cont"])

    # NOTE: this creates Python objects; expensive but unchanged from your current logic
    Counts = [np.round(df.iloc[i, :], 1) for i in range(df.shape[0])]

    index_dict = {tuple(val): i for i, val in enumerate(np.unique(Counts, axis=0))}
    A = np.array([index_dict[tuple(i)] for i in Counts])

    max_index = A.max() + 1
    change_indices = np.where(np.diff(barcodes) != 0)[0] + 1
    split_arrays = np.split(A, change_indices)

    bincounts_all = np.stack([np.bincount(arr, minlength=max_index) for arr in split_arrays])
    comb_array = bincounts_all[:, ~np.all(bincounts_all == 0, axis=0)].T

    p_g_loc = comb_array / np.sum(comb_array)
    p_domains = np.sum(p_g_loc, axis=0)
    p_gene_comb = np.sum(p_g_loc, axis=1)

    return mutual_information(p_g_loc, p_gene_comb, p_domains)

def process_combo(combo):
    # IMPORTANT: do not pass df through multiprocessing queues
    return calc_mutual_inf_for_network_w_specific_gene_set(DF_GLOBAL, combo)

def choose_n_cores(default: int = 40) -> int:
    # Priority: explicit N_WORKERS (from SLURM wrapper) > SLURM_CPUS_PER_TASK > NSLOTS > default
    for key in ("N_WORKERS", "SLURM_CPUS_PER_TASK", "NSLOTS"):
        val = os.environ.get(key)
        if val and str(val).isdigit():
            return int(val)
    return default

# ------------------ MAIN ------------------

if __name__ == "__main__":
    print(f"🚀 Processing {dataset_name} with {n_combos} random gene sets")

    n_cores = choose_n_cores(default=40)
    print(f"🧵 Using n_cores = {n_cores}")

    if dataset_name not in file_dict:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Allowed: {list(file_dict.keys())}")

    # Load df once in parent
    with open(file_dict[dataset_name], "rb") as f:
        df = pickle.load(f)

    # Make globally visible to forked workers BEFORE Pool creation
    DF_GLOBAL = df

    gene_names = list(df.columns[:-1])  # assumes last column is barcode_cont or similar
    mito_genes = get_mito_names(df)
    no_of_genes = len(mito_genes)

    print(f"🧬 Mito genes: {no_of_genes}")

    print("🔀 Sampling gene sets...")
    random_sets = [random.sample(gene_names, no_of_genes) for _ in range(n_combos)]

    print("⚙️ Starting parallel MI calculation...")

    # Reduce IPC overhead; tune chunking for big tasks
    chunksize = max(1, len(random_sets) // (n_cores * 4))

    # Restart workers occasionally to reduce risk of hangs / leaks
    with mp.Pool(processes=n_cores, maxtasksperchild=10) as pool:
        MI_random_samples = list(
            tqdm(
                pool.imap_unordered(process_combo, random_sets, chunksize=chunksize),
                total=len(random_sets),
            )
        )

    print("➕ Calculating MI for mitochondrial genes...")
    mito_mi = calc_mutual_inf_for_network_w_specific_gene_set(df, mito_genes)
    MI_random_samples.append(mito_mi)

    print("💾 Saving results...")
    output_path = os.path.join(
        SAVE_DIR,
        f"MI_mito_random_samples_{dataset_name}_{n_combos}_numba.pkl"
    )
    with open(output_path, "wb") as f:
        pickle.dump(MI_random_samples, f)

    print(f"✅ Done. Results saved to {output_path}")
