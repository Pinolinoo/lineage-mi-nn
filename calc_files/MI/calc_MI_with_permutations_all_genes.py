import sys
import numpy as np
import pandas as pd
import pickle
import os
import multiprocessing as mp
from numba import njit
from tqdm import tqdm
from pathlib import Path

# ------------------ CONFIGURATION ------------------

dataset_name = sys.argv[1]          # e.g. "weinreb"
num_permutations = 50

SAVE_DIR = "./mi_analysis_results_MI_label_entropy_all_genes_parallel"
os.makedirs(SAVE_DIR, exist_ok=True)

EPSILON = 1e-9
n_cores = int(os.environ.get("NSLOTS", os.cpu_count()))

base_dir = Path("/fast/AG_Metzger/philipp/python_files/gene_signature/mutual_information_approach/November_25_ReDo/preprocessing/pkl_files_filtered")

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
    "morris_atac" : base_dir / "morris_ATAC.pkl",
}

# ------------------ HELPERS ------------------

def calculate_entropy(labels):
    """Entropy of discrete barcode labels H(Y)."""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -(probs * np.log2(probs + 1e-12)).sum()

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

def calc_mi_from_df(df):
    """
    df: DataFrame with columns ["gene", "barcode_cont"]
    """
    df = df.sort_values("barcode_cont")
    barcodes = np.array(df["barcode_cont"].values)
    values = np.round(df["gene"].values[:, None], decimals=1).astype(np.float32)

    uniq_vals, inv = np.unique(values, axis=0, return_inverse=True)
    A = inv
    max_index = A.max() + 1

    change_indices = np.where(np.diff(barcodes) != 0)[0] + 1
    split_arrays = np.split(A, change_indices)

    bincounts_all = np.stack(
        [np.bincount(arr, minlength=max_index) for arr in split_arrays]
    )

    comb_array = bincounts_all[:, ~np.all(bincounts_all == 0, axis=0)].T
    p_g_loc = comb_array / np.sum(comb_array)

    p_gene = np.sum(p_g_loc, axis=1)
    p_location = np.sum(p_g_loc, axis=0)

    return mutual_information(p_g_loc, p_gene, p_location)

def calc_noise_mi(X_col, barcodes, num_permutations):
    """
    Permutation-based null MI via barcode shuffling.
    X_col: (n_cells, 1) array of gene expression
    barcodes: (n_cells,) array of lineage labels
    """
    x_flat = X_col.flatten()
    if len(x_flat) != len(barcodes):
        return np.nan, np.nan, np.nan

    shuffled_mis = np.empty(num_permutations, dtype=np.float64)
    try:
        for i in range(num_permutations):
            permuted = np.random.permutation(barcodes)
            temp_df = pd.DataFrame({"gene": x_flat, "barcode_cont": permuted})
            shuffled_mis[i] = calc_mi_from_df(temp_df)
    except Exception as e:
        print(f"[ERROR in calc_noise_mi] issue: {e}")
        return np.nan, np.nan, np.nan

    mean_perm = np.mean(shuffled_mis)
    std_perm = np.std(shuffled_mis)
    perc95 = np.percentile(shuffled_mis, 95)
    return mean_perm, std_perm, perc95

# ------------------ PARALLEL WORKER ------------------

def initializer(_gene_data, _barcodes, _H_barcode):
    global gene_data, barcodes, H_barcode
    gene_data = _gene_data
    barcodes = _barcodes
    H_barcode = _H_barcode

def process_gene(gene):
    try:
        X = gene_data[gene].values[:, None]
        if X.shape[0] != barcodes.shape[0]:
            return {
                "gene": gene,
                "true_mi": np.nan,
                "mean_perm_mi": np.nan,
                "95_perm_mi": np.nan,
                "std_perm_mi": np.nan,
                "normalized_MI_label_entropy": np.nan,
            }

        # true MI for this gene
        df_true = pd.DataFrame({"gene": X.flatten(), "barcode_cont": barcodes})
        true_mi = calc_mi_from_df(df_true)

        # permutation-based null MI (noise)
        mean_perm, std_perm, perc95 = calc_noise_mi(X, barcodes, num_permutations)

        # normalize ONLY by barcode entropy H(Y)
        norm_mi_label_entropy = true_mi / (H_barcode + EPSILON)

        return {
            "gene": gene,
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
            "true_mi": np.nan,
            "mean_perm_mi": np.nan,
            "95_perm_mi": np.nan,
            "std_perm_mi": np.nan,
            "normalized_MI_label_entropy": np.nan,
        }

# ------------------ MAIN ------------------

if __name__ == "__main__":
    print(f"🔍 Processing dataset: {dataset_name} with {num_permutations} permutations...")

    # Load data
    with open(file_dict[dataset_name], "rb") as f:
        full_df = pickle.load(f)

    # Sort and split genes / barcodes
    full_df = full_df.sort_values("barcode_cont")
    barcodes = full_df["barcode_cont"].values.copy()
    gene_data = full_df.iloc[:, :-1]  # all gene columns
    genes = gene_data.columns.tolist()

    # Entropy of barcodes H(Y)
    H_barcode = calculate_entropy(barcodes)
    print(f"📊 Entropy of barcodes H(Y) for {dataset_name}: {H_barcode:.4f}")

    # Parallel processing
    with mp.Pool(
        processes=n_cores,
        initializer=initializer,
        initargs=(gene_data, barcodes, H_barcode),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_gene, genes),
                total=len(genes),
                desc="⏳ Processing genes",
            )
        )

    mi_df = pd.DataFrame(results)
    output_path = os.path.join(
        SAVE_DIR,
        f"MI_label_entropy_{dataset_name}_{num_permutations}_permutations.csv",
    )
    mi_df.to_csv(output_path, index=False)
    print(f"✅ Done. Results saved to {output_path}")
