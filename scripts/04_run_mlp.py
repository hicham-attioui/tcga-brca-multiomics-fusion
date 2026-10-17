"""
Deep single-omics models: one small MLP per omics, PAM50 subtype classification.

Same evaluation as the baselines: the 15 train/test splits saved in results/splits.csv.
Inside each split:
    - 20% of the training side is held out as validation (early stopping, weight decay choice)
    - preprocessing (top-variance features, median imputation, scaling) is fitted on train only
    - the test fold is used once, for the final score

Input : data/processed/*.pkl, labels.csv, results/splits.csv
Output: results/mlp_metrics.csv, results/mlp_summary.csv

Run from the repository root:
    python scripts/04_run_mlp.py           # full run (3 seeds x 5 folds)
    python scripts/04_run_mlp.py --quick   # seed 0 only, to test the code
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"

CLASSES = ["LumA", "LumB", "Basal", "Her2"]
OMICS = ["expr", "mirna", "meth"]
TOP_K = {"expr": 2000, "mirna": 521, "meth": 5000}   # same as the baselines
N_FOLDS = 5

HIDDEN = 256
DROPOUT = 0.5
LR = 1e-3
WEIGHT_DECAYS = [1e-3, 1e-2, 1e-1]   # chosen per split on the validation set
MAX_EPOCHS, PATIENCE = 2000, 100

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_labelled():
    labels = pd.read_csv(PROC / "labels.csv")
    mask = labels["labelled"].values
    data = {}
    for name in OMICS:
        df = pd.read_pickle(PROC / f"{name}.pkl")
        assert (df.index.values == labels["sample_id"].values).all()
        data[name] = df.values[mask].astype("float32")
    y = labels.loc[mask, "pam50"].map({c: i for i, c in enumerate(CLASSES)}).values
    splits = pd.read_csv(RES / "splits.csv")
    assert (splits["sample_id"].values == labels.loc[mask, "sample_id"].values).all()
    return data, y, splits


class Preprocessor:
    """Top-variance selection + median imputation + standard scaling, fitted on training data only.
    Same steps as the scikit-learn Pipeline in the baselines, written by hand on numpy arrays."""

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.keep_ = np.argsort(np.nanvar(X, axis=0))[::-1][: min(self.k, X.shape[1])]
        Xk = X[:, self.keep_]
        self.median_ = np.nanmedian(Xk, axis=0)
        Xk = np.where(np.isnan(Xk), self.median_, Xk)
        self.mean_ = Xk.mean(axis=0)
        self.std_ = Xk.std(axis=0) + 1e-8          # avoid division by zero
        return self

    def transform(self, X):
        Xk = X[:, self.keep_]
        Xk = np.where(np.isnan(Xk), self.median_, Xk)
        return ((Xk - self.mean_) / self.std_).astype("float32")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Input -> hidden layer (ReLU, dropout) -> 4 class scores."""

    def __init__(self, n_in, n_hidden=HIDDEN, n_out=len(CLASSES), dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.2),              # light dropout on the inputs too: thousands of noisy features
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),                    # the non-linearity: what makes this more than logistic regression
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_out),   # raw scores (logits), no softmax
        )

    def forward(self, x):
        return self.net(x)


def train_one(Xtr, ytr, Xva, yva, weight_decay, seed):
    """Train with early stopping on validation loss. Returns the best model and its validation loss."""
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1]).to(device)
    # AdamW: weight decay applied directly to the weights (decoupled), not through Adam's rescaling
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val


def tensor(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [0] if args.quick else [0, 1, 2]

    print("device:", device)
    data, y, splits = load_labelled()

    rows = []
    for name in OMICS:
        X, t0 = data[name], time.time()
        for seed in seeds:
            for fold in range(N_FOLDS):
                test = splits[f"fold_seed{seed}"].values == fold
                idx_trval = np.where(~test)[0]

                # validation = 20% of the training side, stratified; the test fold is never touched
                idx_tr, idx_va = train_test_split(
                    idx_trval, test_size=0.2, stratify=y[idx_trval], random_state=seed
                )

                prep = Preprocessor(TOP_K[name]).fit(X[idx_tr])
                Xtr, Xva, Xte = (tensor(prep.transform(X[i])) for i in (idx_tr, idx_va, test))
                ytr, yva = tensor(y[idx_tr], torch.long), tensor(y[idx_va], torch.long)

                # choose weight decay on the validation set
                best = None
                for wd in WEIGHT_DECAYS:
                    model, val_loss = train_one(Xtr, ytr, Xva, yva, wd, seed)
                    if best is None or val_loss < best[1]:
                        best = (model, val_loss, wd)
                model, _, wd = best

                model.eval()
                with torch.no_grad():
                    pred = model(Xte).argmax(dim=1).cpu().numpy()
                rows.append({
                    "model": f"mlp_{name}", "seed": seed, "fold": fold,
                    "macro_f1": f1_score(y[test], pred, average="macro"),
                    "balanced_acc": balanced_accuracy_score(y[test], pred),
                    "weight_decay": wd,
                })
        print(f"mlp_{name:6s} done in {time.time() - t0:.0f} s")

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("model")[["macro_f1", "balanced_acc"]].agg(["mean", "std"]).round(3)
    suffix = "_quick" if args.quick else ""
    metrics.to_csv(RES / f"mlp_metrics{suffix}.csv", index=False)
    summary.to_csv(RES / f"mlp_summary{suffix}.csv")
    print("\n=== Summary (mean and std over splits) ===")
    print(summary)


if __name__ == "__main__":
    main()
