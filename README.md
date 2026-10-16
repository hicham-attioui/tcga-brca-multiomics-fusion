l# When Does Deep Fusion Help? Multi-omics Integration for Breast Cancer Subtyping

**Status: work in progress.** The data pipeline, all baselines, the deep models and the main results below are complete and reproducible. Interpretation analyses (modality ablation, latent space) are in progress; see [Next steps](#next-steps).

## Summary

This project asks whether deep learning models that fuse several omics layers classify breast cancer subtypes better than simple, well-regularized single-omics models. Using 617 TCGA breast cancer patients with gene expression, DNA methylation and miRNA data, I compared linear models, random forests, single-omics neural networks and a multi-omics fusion autoencoder under the same cross-validation protocol.

**Main finding so far:** with 365 labelled patients, no model outperformed L2-regularized logistic regression on gene expression alone (macro-F1 0.855). Simple concatenation of the three omics made results worse, and the fusion autoencoder did not close the gap. Adding 252 unlabelled patients to train the autoencoder improved the fusion model (0.785 to 0.808 mean macro-F1), suggesting that the unlabelled data is useful for representation learning, even if not enough here to beat the linear baseline.

## Research question

1. Does combining expression, methylation and miRNA improve PAM50 subtype classification compared with the best single omics?
2. Does a deep intermediate fusion model (one encoder per omics) do better than naive concatenation?
3. Do patients without a subtype label help, when used only to train the encoders (semi-supervised learning)?

## Data

**Source:** The Cancer Genome Atlas, breast invasive carcinoma (TCGA-BRCA), from the UCSC Xena **TCGA Hub** (legacy processing, files dated 2018-07-22, downloaded 2026-10-13). All data used is open access.

| Data type | Measurement | Primary tumours |
|---|---|---|
| Gene expression | RNA-seq, log2(RSEM normalized count + 1), 20,530 genes | 1,097 |
| DNA methylation | Illumina 450k array, beta values, 485,577 CpG probes | 785 |
| miRNA expression | Illumina HiSeq, 2,238 miRNAs | 749 |
| Clinical | PAM50 labels (`PAM50Call_RNAseq`) | |

**Final cohort:** 617 patients with all three omics, of which **365 have a usable PAM50 label** (LumA 188, LumB 97, Basal 55, Her2 25) and **252 are unlabelled** (238 without a label, 14 Normal-like).

![Data overview](figures/fig1_data_overview.png)

### Preprocessing and data decisions

| Decision | Reason |
|---|---|
| Primary tumour samples only (TCGA code `01`), one per patient | Avoid normal tissue and duplicate patients across splits |
| Remove the 50 PAM50 genes from the expression data | The labels were computed from these genes on this same RNA-seq data. Removing them reduces circularity; it does not eliminate it, since correlated genes remain |
| Exclude Normal-like as a class (14 patients) | Debated as a true subtype (possibly normal tissue contamination) and too few patients to evaluate. These patients are kept as unlabelled |
| miRNA: drop miRNAs missing in at least 20% of patients, fill the rest with 0 | Missing values in this file mean "not detected" (the file contains no zeros) |
| Methylation: drop probes missing in at least 5% of patients | 89,512 probes are missing in every sample; the rest are almost complete |
| No relabelling from the 2012 microarray PAM50 calls | Only 3 of the 238 unlabelled patients could have been recovered |

The PAM50 gene list (Parker et al. 2009, J Clin Oncol) is in `data/pam50_genes.txt`. `ORC6` appears as its older name `ORC6L` in the Xena file.

## Methods

### Evaluation protocol

- **Stratified 5-fold cross-validation, repeated with 3 seeds:** 15 train/test splits, saved once in `results/splits.csv` and **reused by every model**, so all comparisons are paired.
- **No information from the test fold is used for training:** feature selection (top features by variance), imputation and scaling are fitted on the training side of each split only. Hyperparameters are chosen by an inner cross-validation (logistic regression) or on a validation set taken from the training side (neural networks).
- **Metrics:** macro-F1 (main metric, robust to class imbalance) and balanced accuracy.
- **Features kept per split:** top 2,000 genes, all 521 miRNAs, top 5,000 CpG probes, ranked by variance on the training side.

### Models

| Model | Description |
|---|---|
| **LR** | L2-regularized logistic regression; regularization strength chosen by inner 3-fold CV over 8 values |
| **RF** | Random forest, 500 trees, default settings (no tuning, no class weighting) |
| **LR / RF concatenated** | The three omics side by side, each block selected and scaled separately (early fusion) |
| **MLP** | One hidden layer (256 units, ReLU), dropout, AdamW; weight decay chosen on validation; early stopping |
| **Fusion autoencoder** | One encoder per omics (to a 32-dimensional latent vector) and one decoder per omics. The three latent vectors are concatenated and passed to a classifier. Loss = classification loss + weighted reconstruction loss. Weight decay and reconstruction weight chosen on validation |
| **Fusion autoencoder + unlabelled** | Same model; the 252 unlabelled patients are added to the training side and contribute to the reconstruction loss only. They never appear in validation or test sets |

## Results

Macro-F1 and balanced accuracy over the 15 test folds (mean ± standard deviation). The last two columns compare each model with LR on expression, split by split.

| Model | Macro-F1 | Balanced accuracy | Diff. vs LR expression | Splits won vs LR expression |
|---|---|---|---|---|
| **LR expression** | **0.855 ± 0.057** | **0.856 ± 0.063** | reference | |
| LR miRNA | 0.795 ± 0.041 | 0.785 ± 0.047 | −0.060 | 2/15 |
| LR methylation | 0.754 ± 0.069 | 0.755 ± 0.071 | −0.101 | 0/15 |
| LR concatenated | 0.817 ± 0.055 | 0.812 ± 0.058 | −0.038 | 2/15 |
| RF expression | 0.808 ± 0.052 | 0.783 ± 0.059 | −0.047 | 5/15 |
| RF miRNA | 0.602 ± 0.028 | 0.609 ± 0.028 | −0.253 | 0/15 |
| RF methylation | 0.669 ± 0.053 | 0.645 ± 0.039 | −0.187 | 0/15 |
| RF concatenated | 0.749 ± 0.063 | 0.722 ± 0.053 | −0.106 | 1/15 |
| MLP expression | 0.840 ± 0.060 | 0.845 ± 0.063 | −0.015 | 5/15 |
| MLP miRNA | 0.759 ± 0.064 | 0.751 ± 0.052 | −0.096 | 2/15 |
| MLP methylation | 0.716 ± 0.075 | 0.717 ± 0.077 | −0.139 | 0/15 |
| Fusion autoencoder | 0.785 ± 0.069 | 0.783 ± 0.068 | −0.071 | 3/15 |
| Fusion autoencoder + unlabelled | 0.808 ± 0.057 | 0.801 ± 0.063 | −0.047 | 5/15 |

![Model comparison](figures/fig2_model_comparison.png)

### Key findings

1. **A regularized linear model on expression is the strongest model.** Neither non-linear classical models (random forest) nor neural networks outperformed it. With 365 labelled patients, the subtype signal appears to be captured well by a linear model, and more flexible models mainly add variance.
2. **Naive concatenation hurts.** Combining the three omics in one linear model performed worse than expression alone in 13 of 15 splits. A likely reason: methylation, the weakest omics, contributes two thirds of the concatenated features.
3. **Intermediate fusion did not solve this.** The fusion autoencoder, which gives each omics its own representation, also stayed below expression alone.
4. **Unlabelled patients help the fusion model.** Adding them for reconstruction raised mean macro-F1 from 0.785 to 0.808 and reduced the spread across splits [paired comparison: X of 15 splits won].
5. **Errors follow biology and label noise.** Most errors are between Luminal A and Luminal B, the same pair on which the two TCGA PAM50 label versions disagree (about 90% agreement overall). Basal tumours are almost always recognised (95%). Her2 is the hardest class (25 patients), often predicted as LumB or Basal.

![Confusion matrix](figures/fig3_confusion_fusion.png)

## Limitations

- **Small labelled cohort** (365 patients, 25 in the smallest class). Results are estimated with 15 splits, but the spread between splits is large (standard deviations around 0.05).
- **Splits are not independent** (training sets overlap), so differences are reported as paired win counts, not formal significance tests.
- **Labels are not independent of the input.** PAM50 labels come from expression data; removing the 50 PAM50 genes reduces but does not remove this circularity, which favours expression-based models.
- **Limited hyperparameter search** for the neural networks and no tuning for the random forest. Better tuned deep models could perform differently.
- **One cancer type, one label.** Conclusions may not hold for other tasks, such as survival prediction, where the omics may carry more complementary information.

## Next steps

- [ ] Modality ablation: performance of the fusion model when one omics is removed
- [ ] Latent space visualisation (UMAP), including where Normal-like patients fall
- [ ] Expression-only autoencoder with and without unlabelled patients, to separate the effect of unlabelled data from the effect of fusion
- [ ] Non-circular label: receptor status (ER, PR, HER2) measured by immunohistochemistry
- [ ] Survival prediction with the TCGA Pan-Cancer clinical data resource

## Repository structure

```
├── data/
│   ├── pam50_genes.txt           PAM50 gene list (with source)
│   ├── raw/                      downloaded files (not in the repository)
│   └── processed/                output of 02_build_dataset.py (not in the repository)
├── scripts/
│   ├── 02_build_dataset.py       filtering, QC, alignment of the three omics, labels
│   ├── 03_run_baselines.py       logistic regression and random forest, creates the CV splits
│   ├── 04_run_mlp.py             single-omics neural networks
│   ├── 05_run_fusion.py          fusion autoencoder, with and without unlabelled patients
│   └── 06_make_figures.py        figures and results table
├── notebooks/                    data exploration and learning exercises
├── results/                      metrics per split, summaries, predictions
└── figures/
```

## How to reproduce

**1. Environment** (Python 3.11):
```
conda create -n mofusion python=3.11
conda activate mofusion
pip install -r requirements.txt
```
For GPU support, install PyTorch with the command from pytorch.org for your system. All scripts also run on CPU.

**2. Data.** From [UCSC Xena](https://xenabrowser.net), TCGA Hub, cohort *TCGA Breast Cancer (BRCA)*, download into `data/raw/`:
- `TCGA.BRCA.sampleMap_HiSeqV2.gz` (gene expression RNAseq, IlluminaHiSeq)
- `TCGA.BRCA.sampleMap_HumanMethylation450.gz` (DNA methylation, Methylation450k)
- `TCGA.BRCA.sampleMap_miRNA_HiSeq_gene.gz` (miRNA mature strand expression, IlluminaHiseq)
- `TCGA.BRCA.sampleMap_BRCA_clinicalMatrix` (phenotypes)

**3. Run the pipeline** from the repository root:
```
python scripts/02_build_dataset.py
python scripts/03_run_baselines.py
python scripts/04_run_mlp.py
python scripts/05_run_fusion.py
python scripts/06_make_figures.py
```
Total runtime is a few minutes on a recent GPU (the methylation file takes the longest to load).

## Author

Hicham Attioui, bioinformatics engineer. GitHub: [HICHAM-CODES](https://github.com/hicham-attioui)
