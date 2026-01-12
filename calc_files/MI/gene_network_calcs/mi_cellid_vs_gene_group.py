# mi_cellid_vs_gene_group.py

import sys
import os
import pickle
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

# ------------------ CONFIG ------------------

dataset_name = sys.argv[1]
n_combos = int(sys.argv[2]) if len(sys.argv) > 2 else 150
SAVE_DIR = "./mi_cellid_vs_gene_groups_results"
os.makedirs(SAVE_DIR, exist_ok=True)

file_dict = {
    "treutlein": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/treutlein_all_genes_with_EGFP_tomato.pkl",
    "weinreb": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/weinreb_all_genes.pkl",
    "junker": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/junker_z5_all_endogenous_genes_no_dubl.pkl",
    "you1": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/you_stage1_all_genes.pkl",
    "you2": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/you_stage2_all_genes.pkl",
    "you3": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/you_stage3_all_genes.pkl",
    "you4": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/you_stage4_all_genes.pkl",
    "knoblich_day16_1": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/knoblich_day16_1.pkl",
    "knoblich_day16_2": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/knoblich_day16_2.pkl",
    "knoblich_day42_1": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/knoblich_day42_1.pkl",
    "knoblich_day42_2": "/fast/AG_Metzger/philipp/python_files/gene_signature/adata_saves/making_pandas_df_for_lukas/pkl_files_no_barcode_col/knoblich_day42_2.pkl"
}

# ------------------ FUNCTIONS ------------------

def get_mito_names(counts):
    return [name for name in counts.columns if name.lower().startswith("mt-")]

def calc_mi_cellid_vs_gene_group(full_df, gene_set):
    expr_df = full_df[gene_set]
    
    # Use row indices as proxy for unique cell IDs
    cell_ids = np.arange(expr_df.shape[0])
    
    # Discretize expression vectors per cell (row)
    discretized_expr = [tuple(np.round(expr_df.iloc[i, :], 1)) for i in range(expr_df.shape[0])]
    _, expr_ids = np.unique(discretized_expr, return_inverse=True)

    # Mutual Information between cell index (cell ID) and expression vector ID
    joint_counts = pd.crosstab(
        pd.Series(cell_ids, name="cell"),
        pd.Series(expr_ids, name="expr_bin")
    )

    joint_probs = joint_counts.values / np.sum(joint_counts.values)

    p_cell = np.sum(joint_probs, axis=1)
    p_expr = np.sum(joint_probs, axis=0)

    MI = 0.0
    for i in range(joint_probs.shape[0]):
        for j in range(joint_probs.shape[1]):
            p_ij = joint_probs[i, j]
            if p_ij > 0 and p_cell[i] > 0 and p_expr[j] > 0:
                MI += p_ij * np.log2(p_ij / (p_cell[i] * p_expr[j]))
    return MI


def process_combo(args):
    df, gene_set = args
    return calc_mi_cellid_vs_gene_group(df, gene_set)

# ------------------ MAIN ------------------

if __name__ == "__main__":
    print(f"🚀 Processing {dataset_name} with {n_combos} random gene sets")
    n_cores = int(os.environ.get("NSLOTS", os.cpu_count()))

    with open(file_dict[dataset_name], "rb") as f:
        df = pickle.load(f)

    gene_names = list(df.columns[:-1])
    mito_genes = get_mito_names(df)

    print("🔀 Sampling gene sets...")
    random_sets = [random.sample(gene_names, 13) for _ in range(n_combos)]
    combo_args = [(df, gene_set) for gene_set in random_sets]

    print("⚙️ Running parallel MI calculations...")
    with mp.Pool(processes=n_cores) as pool:
        MI_random = list(tqdm(pool.imap(process_combo, combo_args), total=len(combo_args)))

    print("➕ Calculating MI for mitochondrial gene group...")
    mito_mi = calc_mi_cellid_vs_gene_group(df, mito_genes)
    MI_random.append(mito_mi)

    print("💾 Saving results...")
    output_path = os.path.join(SAVE_DIR, f"MI_cellid_vs_gene_groups_{dataset_name}_{n_combos}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(MI_random, f)

    print(f"✅ Done. Saved to: {output_path}")
