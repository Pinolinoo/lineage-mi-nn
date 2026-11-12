# Mutual Information Analysis of Lineage Memory

This repository accompanies the paper  
**"Mutual information identifies metabolic genes as conserved carriers of lineage memory"**.

It contains all scripts and notebooks used for data analysis, MI calculation, and GEMLI benchmarking.

---

## Repository structure

### `analysis/`
Contains Jupyter notebooks for data analysis and downstream exploration.  
These notebooks were used to generate the figures and analyses presented in the paper.

### `analysis/GEMLI/`
Includes scripts for running **GEMLI**, the gene-expression memory lineage inference tool, used for comparison and validation of lineage-informative genes.

### `calc_files/MI`
Contains Python and bash scripts for:
- Mutual Information (MI) calculation including permutation testing for significance estimation  
- Normalization of MI values across datasets and conditions

### `calc_files/NN`
Contains scripts for neural network hyperparameter search and  xAI (LIME and Integrated Gradients) analysis. 

### `YAML files`
These files document the software dependencies and versions used for the analyses performed in this study.

Neural network and explainable AI (xAI) analyses, including LIME and Integrated Gradients, were performed using the Conda environment defined in environment_NN.yaml. Environment specifications for mutual information calculations and related analyses are stored in environment_MI.yaml.

For the R-based scATAC-seq and GEMLI analyses aswell as GSEA, the required packages and their versions are listed in the corresponding yaml files.

---

## Notes
- Large intermediate result files (e.g., plots and MI result folders) are intentionally excluded from the repository via `.gitignore`.
- For dataset availability pls get in touch with: "philipp.roth@mdc-berlin.de"
- All code was run on a CentOS-based HPC cluster environment using SLURM.

---

## Citation
If you use this code, please cite our paper:

> Roth et al., *Mutual information identifies metabolic genes as conserved carriers of lineage memory*, 2026.

---

## Contact
For questions or collaborations, please contact:  
**Philipp Roth** – [philipp.roth@mdc-berlin.de](mailto:philipp.roth@mdc-berlin.de)
