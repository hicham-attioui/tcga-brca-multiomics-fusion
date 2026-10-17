from pathlib import Path
import pandas as pd

# Paths and settings
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

FILES = {
    "expr": RAW / "TCGA.BRCA.sampleMap_HiSeqV2.gz",
    "mirna": RAW / "TCGA.BRCA.sampleMap_miRNA_HiSeq_gene.gz",
    "meth": RAW / "TCGA.BRCA.sampleMap_HumanMethylation450.gz",
    "clin": RAW / "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix",
}
PAM50_FILE = ROOT / "data" / "pam50_genes.txt"

LABEL_COL = "PAM50Call_RNAseq"
KEEP_CLASSES = ["LumA", "LumB", "Basal", "Her2"]  
METH_MAX_MISSING = 0.05   
MIRNA_MAX_MISSING = 0.20  


# Loading
def load_pam50_genes(path):
    """Read the PAM50 gene list. Lines starting with # are comments (use one for the source)."""
    lines = path.read_text().splitlines()
    genes = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    assert len(genes) == 50, f"Expected 50 PAM50 genes, found {len(genes)}"
    return genes


def load_omics(path):
    """Read a Xena matrix (features x samples) and return it as samples x features, float32."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.T.astype("float32")


def keep_primary_tumours(df):
    """Keep rows whose sample ID has type code 01 (e.g. TCGA-A2-A0T2-01)."""
    is_tumour = df.index.str[13:15] == "01"
    return df.loc[is_tumour]


# Quality control 
def qc_expression(expr, pam50_genes):
    """Drop genes that are zero in every sample, then drop the PAM50 genes."""
    expr = expr.loc[:, (expr != 0).any(axis=0)]
    found = [g for g in pam50_genes if g in expr.columns]
    not_found = sorted(set(pam50_genes) - set(found))
    if not_found:
        print(f"  WARNING: {len(not_found)} PAM50 genes not found in expression: {not_found}")
    return expr.drop(columns=found), not_found


def qc_mirna(mirna):
    """Drop rarely detected miRNAs, then fill remaining missing values with 0 (= not detected)."""
    missing = mirna.isna().mean(axis=0)
    mirna = mirna.loc[:, missing < MIRNA_MAX_MISSING]
    return mirna.fillna(0.0)


def qc_methylation(meth):
    """Drop probes with too many missing values. Remaining NaNs are imputed later, inside the CV folds."""
    missing = meth.isna().mean(axis=0)
    return meth.loc[:, missing < METH_MAX_MISSING]


# Labels
def build_labels(clin, samples):
    """One row per sample, in the same order as the omics matrices."""
    original = clin[LABEL_COL].reindex(samples)            
    pam50 = original.where(original.isin(KEEP_CLASSES))     
    labels = pd.DataFrame({
        "sample_id": samples,
        "pam50_original": original.values,
        "pam50": pam50.values,
    })
    labels["labelled"] = labels["pam50"].notna()
    return labels


# Main
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    pam50_genes = load_pam50_genes(PAM50_FILE)

    print("Loading expression...")
    expr = keep_primary_tumours(load_omics(FILES["expr"]))
    print("Loading miRNA...")
    mirna = keep_primary_tumours(load_omics(FILES["mirna"]))
    print("Loading methylation (slow)...")
    meth = keep_primary_tumours(load_omics(FILES["meth"]))
    print("Loading clinical...")
    clin = pd.read_csv(FILES["clin"], sep="\t", index_col=0, low_memory=False)

    common = expr.index.intersection(mirna.index).intersection(meth.index).sort_values()
    expr, mirna, meth = expr.loc[common], mirna.loc[common], meth.loc[common]

    print("Running QC...")
    expr, pam50_not_found = qc_expression(expr, pam50_genes)
    mirna = qc_mirna(mirna)
    meth = qc_methylation(meth)

    labels = build_labels(clin, common)

    assert expr.index.is_unique
    assert expr.index.equals(mirna.index) and expr.index.equals(meth.index)
    assert (labels["sample_id"].values == expr.index.values).all()
    assert not expr.isna().any().any(), "Expression should have no missing values"
    assert not mirna.isna().any().any(), "miRNA should have no missing values after fillna"

    expr.to_pickle(OUT / "expr.pkl")
    mirna.to_pickle(OUT / "mirna.pkl")
    meth.to_pickle(OUT / "meth.pkl")
    labels.to_csv(OUT / "labels.csv", index=False)

    print("\n=== Summary ===")
    print(f"Patients with all 3 omics: {len(common)}")
    print(f"Expression:  {expr.shape}   (PAM50 genes not found: {len(pam50_not_found)})")
    print(f"miRNA:       {mirna.shape}")
    print(f"Methylation: {meth.shape}   (remaining missing: {meth.isna().mean().mean():.4%})")
    print(f"Labelled: {labels['labelled'].sum()}   Unlabelled: {(~labels['labelled']).sum()}")
    print("Class counts (training labels):")
    print(labels["pam50"].value_counts().to_string())
    print("Original labels of unlabelled patients:")
    print(labels.loc[~labels["labelled"], "pam50_original"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
