"""
Classical baselines for PAM50 subtype classification (TCGA-BRCA).

Two classifiers, each on
    - expression only
    - miRNA only
    - methylation only
    - all three omics concatenated (early fusion)

    L2 logistic regression (linear)      -> model names: expr, mirna, meth, concat
    Random forest (non-linear, not deep) -> model names: rf_expr, rf_mirna, rf_meth, rf_concat

Evaluation: stratified 5-fold CV, repeated with 3 seeds (15 train/test splits).
All preprocessing that learns from data (feature selection, imputation, scaling)
is fitted on the training side of each split only. The logistic regression
chooses its regularization with an inner CV inside the training side.

Input : data/processed/{expr,mirna,meth}.pkl and labels.csv
Output: results/splits.csv            (fold of each patient, reused by the deep models)
        results/baselines_metrics.csv (one row per model per split)
        results/baselines_summary.csv (mean and std per model)

Run from the repository root:
    python scripts/03_run_baselines.py           # full run
    python scripts/03_run_baselines.py --quick   # 1 seed, small grid, to test the code
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

CLASSES = ["LumA", "LumB", "Basal", "Her2"]   # fixed order: class 0, 1, 2, 3
OMICS = ["expr", "mirna", "meth"]
TOP_K = {"expr": 2000, "mirna": 521, "meth": 5000}  # features kept per omics (chosen on train only)
N_FOLDS = 5

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# A small custom preprocessing step: keep the k most variable features
# ---------------------------------------------------------------------------
class TopVariance(BaseEstimator, TransformerMixin):
    """Keep the k features with the highest variance, computed in fit() only.

    Inside a Pipeline, fit() only ever sees the training data, so the test
    patients never influence which features are kept.
    """

    def __init__(self, k):
        self.k = k

    def fit(self, X, y=None):
        variances = np.nanvar(X, axis=0)
        k = min(self.k, X.shape[1])
        self.keep_ = np.argsort(variances)[::-1][:k]
        return self

    def transform(self, X):
        return X[:, self.keep_]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_labelled_data():
    """Return omics matrices (numpy, labelled patients only), labels y (0..3) and sample IDs."""
    labels = pd.read_csv(PROC / "labels.csv")
    mask = labels["labelled"].values

    data = {}
    for name in OMICS:
        df = pd.read_pickle(PROC / f"{name}.pkl")
        assert (df.index.values == labels["sample_id"].values).all(), f"{name} not aligned with labels"
        data[name] = df.values[mask].astype("float32")

    y = labels.loc[mask, "pam50"].map({c: i for i, c in enumerate(CLASSES)}).values
    ids = labels.loc[mask, "sample_id"].values
    return data, y, ids


def make_splits(y, ids, seeds):
    """Assign each labelled patient to a test fold, once per seed. Saved so all models use the same splits."""
    splits = pd.DataFrame({"sample_id": ids})
    for seed in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        fold = np.empty(len(y), dtype=int)
        for f, (_, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            fold[test_idx] = f
        splits[f"fold_seed{seed}"] = fold
    return splits


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def omics_preprocessing(k):
    """Per-omics steps, all fitted on training data only."""
    return Pipeline([
        ("top_var", TopVariance(k)),
        ("impute", SimpleImputer(strategy="median")),  # only methylation has NaNs, harmless for others
        ("scale", StandardScaler()),                   # needed for LR; harmless for the random forest
    ])


def make_logreg(seed, quick):
    """L2-regularized logistic regression. The inner CV picks C from the training data."""
    return LogisticRegressionCV(
        Cs=np.logspace(-4, 1, 4 if quick else 8),
        l1_ratios=(0,),                              # 0 = pure L2 (ridge)
        solver="lbfgs",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),  # inner CV
        scoring="f1_macro",
        max_iter=2000,
        n_jobs=-1,
        random_state=seed,
        use_legacy_attributes=True,
    )


def make_rf(seed, quick):
    """Random forest with standard, fixed settings (robust defaults, no tuning)."""
    return RandomForestClassifier(
        n_estimators=200 if quick else 500,   # number of trees
        max_features="sqrt",                  # features tried at each split: sqrt(n_features)
        n_jobs=-1,
        random_state=seed,
    )


def build_model(name, clf_type, seed, quick, n_features):
    """Single-omics model, or concatenated model with separate preprocessing per omics."""
    if name in OMICS:
        prep = omics_preprocessing(TOP_K[name])
    else:  # "concat": each omics block gets its own branch, then everything is joined
        branches, start = [], 0
        for omics in OMICS:
            end = start + n_features[omics]
            branches.append((omics, omics_preprocessing(TOP_K[omics]), slice(start, end)))
            start = end
        prep = ColumnTransformer(branches)
    clf = make_logreg(seed, quick) if clf_type == "lr" else make_rf(seed, quick)
    return Pipeline([("prep", prep), ("clf", clf)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1 seed and a small grid, to test the code")
    args = parser.parse_args()
    seeds = [0] if args.quick else [0, 1, 2]

    RES.mkdir(exist_ok=True)
    data, y, ids = load_labelled_data()
    n_features = {k: v.shape[1] for k, v in data.items()}
    print(f"Labelled patients: {len(y)}   Class counts: {np.bincount(y)} ({CLASSES})")

    splits = make_splits(y, ids, seeds)
    if not args.quick:
        splits.to_csv(RES / "splits.csv", index=False)

    inputs = dict(data)
    inputs["concat"] = np.hstack([data[o] for o in OMICS])

    rows = []
    for clf_type in ["lr", "rf"]:
        for name in OMICS + ["concat"]:
            model_name = name if clf_type == "lr" else f"rf_{name}"
            X = inputs[name]
            t0 = time.time()
            for seed in seeds:
                for fold in range(N_FOLDS):
                    test = splits[f"fold_seed{seed}"].values == fold
                    train = ~test

                    model = build_model(name, clf_type, seed, args.quick, n_features)
                    model.fit(X[train], y[train])
                    pred = model.predict(X[test])

                    clf = model.named_steps["clf"]
                    rows.append({
                        "model": model_name, "seed": seed, "fold": fold,
                        "macro_f1": f1_score(y[test], pred, average="macro"),
                        "balanced_acc": balanced_accuracy_score(y[test], pred),
                        "C": float(clf.C_[0]) if clf_type == "lr" else np.nan,
                    })
            print(f"{model_name:10s} done in {time.time() - t0:.0f} s")

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("model")[["macro_f1", "balanced_acc"]].agg(["mean", "std"]).round(3)

    suffix = "_quick" if args.quick else ""
    metrics.to_csv(RES / f"baselines_metrics{suffix}.csv", index=False)
    summary.to_csv(RES / f"baselines_summary{suffix}.csv")
    print("\n=== Summary (mean and std over splits) ===")
    print(summary)


if __name__ == "__main__":
    main()
