#!/usr/bin/env python3
# mi_group_stratified_random.py
#
# Control: MT gene set vs RANDOM gene sets matched for gene-wise expression entropy.
# Computes MI(cell_id ; expr_bin) which equals entropy of discretized expression-vector IDs.
#
# Parallelization:
# - DF_GLOBAL inherited via fork (no pickling df per task)
# - pool.imap_unordered with chunking
# - maxtasksperchild to reduce long-run issues
#
# Usage:
#   python mi_group_stratified_random.py <dataset_name> [n_combos=150]
#
# Environment knobs:
#   ROUND_DECIMALS=1          # discretization for expression vectors
#   N_STRATA=10               # number of entropy strata
#   INCLUDE_PLAIN_RANDOM=1    # also compute non-stratified random sets (0/1)
#   SEED=0                    # RNG seed (0 => random via os.urandom)
#   N_WORKERS=40              # override cores (optional)
#   SLURM_CPUS_PER_TASK=...   # used automatically if set
#   NSLOTS=...                # used automatically if set (SGE)

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
from typing import List, Dict, Tuple, Optional

# ------------------ CLI ------------------

dataset_name = sys.argv[1]
n_combos = int(sys.argv[2]) if len(sys.argv) > 2 else 150

# ------------------ PATHS ------------------

SAVE_DIR = "./mi_cellid_stratified_random_results"
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

# ------------------ GLOBALS ------------------

DF_GLOBAL: Optional[pd.DataFrame] = None
ROUND_DECIMALS = int(os.environ.get("ROUND_DECIMALS", "1"))
N_STRATA = int(os.environ.get("N_STRATA", "10"))
INCLUDE_PLAIN_RANDOM = bool(int(os.environ.get("INCLUDE_PLAIN_RANDOM", "1")))
SEED_ENV = os.environ.get("SEED", "0")

# ------------------ HELPERS ------------------

def choose_n_cores(default: int = 40) -> int:
    for key in ("N_WORKERS", "SLURM_CPUS_PER_TASK", "NSLOTS"):
        val = os.environ.get(key)
        if val and str(val).isdigit():
            return int(val)
    return default

def get_mito_names(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).lower().startswith("mt-")]

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

def mi_cellid_vs_gene_group(full_df: pd.DataFrame, gene_set: List[str]) -> float:
    expr = full_df[gene_set].to_numpy(copy=False)
    expr_rounded = np.round(expr, ROUND_DECIMALS)
    _, expr_ids = np.unique(expr_rounded, axis=0, return_inverse=True)
    counts = np.bincount(expr_ids)
    return float(entropy_base2_from_counts(counts))

def gene_entropy_1d(x: np.ndarray) -> float:
    """
    1D entropy of a single gene across cells after rounding.
    This is just used to stratify genes by variability.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    x = np.round(x, ROUND_DECIMALS)
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def build_entropy_strata(
    df: pd.DataFrame,
    gene_names: List[str]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Returns:
      edges: array of bin edges (len = N_STRATA+1)
      gene_to_stratum: dict gene -> stratum idx in [0..N_STRATA-1]
    """
    ent = np.empty(len(gene_names), dtype=float)
    for i, g in enumerate(gene_names):
        ent[i] = gene_entropy_1d(df[g].to_numpy(copy=False))

    # handle NaNs by pushing them to lowest bin
    ent_no_nan = ent.copy()
    ent_no_nan[np.isnan(ent_no_nan)] = -np.inf

    # quantile edges => roughly equal genes per stratum
    qs = np.linspace(0, 1, N_STRATA + 1)
    edges = np.quantile(ent_no_nan, qs)

    # ensure monotonic edges (can have ties)
    edges[0] = -np.inf
    edges[-1] = np.inf

    # assign stratum by edges
    gene_to_stratum: Dict[str, int] = {}
    for g, e in zip(gene_names, ent_no_nan):
        # np.digitize returns 1..N_STRATA, subtract 1 -> 0..N_STRATA-1
        s = int(np.digitize(e, edges[1:-1], right=True))
        gene_to_stratum[g] = s

    return edges, gene_to_stratum

def sample_stratified_gene_set(
    gene_names: List[str],
    gene_to_stratum: Dict[str, int],
    target_genes: List[str],
    rng: random.Random
) -> List[str]:
    """
    Sample a random gene set matched to the target_genes' stratum composition.
    """
    # count composition of target set
    comp = np.zeros(N_STRATA, dtype=int)
    for g in target_genes:
        if g in gene_to_stratum:
            comp[gene_to_stratum[g]] += 1

    # pre-index candidate genes per stratum
    strata_to_genes: List[List[str]] = [[] for _ in range(N_STRATA)]
    for g in gene_names:
        strata_to_genes[gene_to_stratum[g]].append(g)

    sampled: List[str] = []
    used = set()
    for s in range(N_STRATA):
        k = int(comp[s])
        if k == 0:
            continue

        candidates = [g for g in strata_to_genes[s] if g not in used]
        if len(candidates) < k:
            # fallback: allow reuse across strata pool (rare if strata are tiny)
            candidates = strata_to_genes[s]

        pick = rng.sample(candidates, k)
        sampled.extend(pick)
        used.update(pick)

    # if due to missing genes in mapping we sampled fewer, top up randomly
    if len(sampled) < len(target_genes):
        remaining = [g for g in gene_names if g not in used]
        sampled.extend(rng.sample(remaining, len(target_genes) - len(sampled)))

    return sampled

def process_combo(combo: List[str]) -> float:
    # DF_GLOBAL is inherited via fork; this assumes Linux/HPC (not Windows spawn).
    return mi_cellid_vs_gene_group(DF_GLOBAL, combo)  # type: ignore[arg-type]

# ------------------ MAIN ------------------

if __name__ == "__main__":
    if dataset_name not in file_dict:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Allowed: {list(file_dict.keys())}")

    n_cores = choose_n_cores(default=40)
    print(f"🚀 Dataset={dataset_name}  n_combos={n_combos}  n_cores={n_cores}")
    print(f"🔢 ROUND_DECIMALS={ROUND_DECIMALS}  N_STRATA={N_STRATA}  INCLUDE_PLAIN_RANDOM={INCLUDE_PLAIN_RANDOM}")

    # seed
    if SEED_ENV and SEED_ENV.isdigit() and int(SEED_ENV) != 0:
        seed = int(SEED_ENV)
    else:
        seed = int.from_bytes(os.urandom(4), "little")
    rng = random.Random(seed)
    print(f"🎲 SEED={seed}")

    # load df
    with open(file_dict[dataset_name], "rb") as f:
        df = pickle.load(f)

    DF_GLOBAL = df

    # gene list (assumes last column is barcode_cont or similar)
    gene_names = list(df.columns[:-1])
    mito_genes = get_mito_names(df)
    if len(mito_genes) == 0:
        raise ValueError("No mito genes found (columns starting with 'mt-').")

    n_genes = len(mito_genes)
    n_cells = int(df.shape[0])
    print(f"🧬 mito_genes={n_genes}  n_cells={n_cells}")

    # entropy strata over genes
    print("🧩 Computing gene-wise entropies + strata...")
    edges, gene_to_stratum = build_entropy_strata(df, gene_names)

    # sample stratified sets
    print("🔀 Sampling stratified random gene sets...")
    strat_sets: List[List[str]] = [
        sample_stratified_gene_set(gene_names, gene_to_stratum, mito_genes, rng)
        for _ in range(n_combos)
    ]

    plain_sets: Optional[List[List[str]]] = None
    if INCLUDE_PLAIN_RANDOM:
        print("🔀 Sampling plain random gene sets...")
        plain_sets = [rng.sample(gene_names, n_genes) for _ in range(n_combos)]

    # parallel compute
    chunksize = max(1, len(strat_sets) // (n_cores * 4))

    print("⚙️ Parallel MI (stratified random)...")
    with mp.Pool(processes=n_cores, maxtasksperchild=10) as pool:
        strat_vals = list(
            tqdm(pool.imap_unordered(process_combo, strat_sets, chunksize=chunksize), total=len(strat_sets))
        )

    plain_vals: Optional[List[float]] = None
    if INCLUDE_PLAIN_RANDOM and plain_sets is not None:
        print("⚙️ Parallel MI (plain random)...")
        with mp.Pool(processes=n_cores, maxtasksperchild=10) as pool:
            plain_vals = list(
                tqdm(pool.imap_unordered(process_combo, plain_sets, chunksize=chunksize), total=len(plain_sets))
            )

    print("➕ MI for MT genes...")
    mt_val = mi_cellid_vs_gene_group(df, mito_genes)

    # save
    out = {
        "dataset": dataset_name,
        "seed": seed,
        "round_decimals": ROUND_DECIMALS,
        "n_strata": N_STRATA,
        "strata_edges": edges,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "mito_genes": mito_genes,
        "mt_value": mt_val,
        "random_stratified": strat_vals,
        "random_plain": plain_vals,
    }

    out_path = os.path.join(
        SAVE_DIR,
        f"MI_cellid_stratified_random_{dataset_name}_{n_combos}_round{ROUND_DECIMALS}_S{N_STRATA}.pkl"
    )
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print(f"✅ Done. Saved: {out_path}")
