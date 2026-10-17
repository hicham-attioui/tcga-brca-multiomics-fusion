"""
Intermediate fusion autoencoder for PAM50 subtype classification (main model).

Architecture (one branch per omics: expression, miRNA, methylation):
    omics_i -> encoder_i -> z_i (latent vector) -> decoder_i -> reconstruction of omics_i
    concat(z_expr, z_mirna, z_meth) -> classifier -> 4 class scores

Loss = classification loss (labelled patients only, label -1 is ignored)
     + recon_weight * reconstruction loss (all training patients)

Two versions, evaluated on the same 15 splits as all other models:
    fusion       : trained on labelled patients only (like-for-like with the baselines)
    fusion_semi  : + the 252 unlabelled patients, used for reconstruction only

Inside each split, everything that learns from data (feature selection, imputation,
scaling, network weights, hyperparameter choice) uses the training side only.
Unlabelled patients are added to the training side only, never to validation or test.

Input : data/processed/*.pkl, labels.csv, results/splits.csv
Output: results/fusion_metrics.csv, fusion_summary.csv, fusion_predictions.csv

Run from the repository root:
    python scripts/05_run_fusion.py           # full run
    python scripts/05_run_fusion.py --quick   # seed 0, one hyperparameter setting
"""
import argparse
import itertools
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
TOP_K = {"expr": 2000, "mirna": 521, "meth": 5000}
N_FOLDS = 5

HIDDEN, LATENT, DROPOUT = 256, 32, 0.5
LR = 1e-3
MAX_EPOCHS, PATIENCE = 2000, 100
GRID = {"weight_decay": [1e-3, 1e-2, 1e-1], "recon_weight": [0.1, 1.0]}  # chosen on validation

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_all():
    """All 617 patients. y = class index for labelled patients, -1 for unlabelled."""
    labels = pd.read_csv(PROC / "labels.csv")
    data = {}
    for name in OMICS:
        df = pd.read_pickle(PROC / f"{name}.pkl")
        assert (df.index.values == labels["sample_id"].values).all()
        data[name] = df.values.astype("float32")

    y = labels["pam50"].map({c: i for i, c in enumerate(CLASSES)}).fillna(-1).astype(int).values
    lab_pos = np.where(labels["labelled"].values)[0]     # positions of labelled patients
    unlab_pos = np.where(~labels["labelled"].values)[0]  # positions of unlabelled patients

    splits = pd.read_csv(RES / "splits.csv")             # covers labelled patients only, same order
    assert (splits["sample_id"].values == labels["sample_id"].values[lab_pos]).all()
    return data, y, lab_pos, unlab_pos, splits, labels["sample_id"].values


class Preprocessor:
    """Top-variance selection + median imputation + scaling, fitted on training data only."""

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.keep_ = np.argsort(np.nanvar(X, axis=0))[::-1][: min(self.k, X.shape[1])]
        Xk = X[:, self.keep_]
        self.median_ = np.nanmedian(Xk, axis=0)
        Xk = np.where(np.isnan(Xk), self.median_, Xk)
        self.mean_, self.std_ = Xk.mean(axis=0), Xk.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        Xk = X[:, self.keep_]
        Xk = np.where(np.isnan(Xk), self.median_, Xk)
        return ((Xk - self.mean_) / self.std_).astype("float32")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionAE(nn.Module):
    def __init__(self, dims, n_classes=len(CLASSES)):
        super().__init__()
        # One encoder per omics: compress thousands of features into LATENT numbers
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.2),                 # input noise: the decoder must rebuild the clean input
                nn.Linear(d, HIDDEN), nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, LATENT),
            ) for d in dims
        ])
        # One decoder per omics: rebuild the omics from its latent vector only
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, d))
            for d in dims
        ])
        # Classifier on the fused representation (concatenated latent vectors)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(LATENT * len(dims), 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, xs):
        zs = [enc(x) for enc, x in zip(self.encoders, xs)]
        recons = [dec(z) for dec, z in zip(self.decoders, zs)]
        logits = self.classifier(torch.cat(zs, dim=1))   # the fusion step
        return logits, recons, zs


def train_fusion(Xtr, ytr, Xva, yva, weight_decay, recon_weight, seed):
    """Early stopping on the validation CLASSIFICATION loss (comparable across recon weights)."""
    torch.manual_seed(seed)
    model = FusionAE([x.shape[1] for x in Xtr]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss(ignore_index=-1)   # unlabelled patients (label -1) add nothing here
    mse = nn.MSELoss()

    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits, recons, _ = model(Xtr)
        clf_loss = ce(logits, ytr)
        rec_loss = sum(mse(r, x) for r, x in zip(recons, Xtr)) / len(Xtr)  # all patients
        loss = clf_loss + recon_weight * rec_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = ce(model(Xva)[0], yva).item()
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
    grid = [(1e-2, 1.0)] if args.quick else list(itertools.product(GRID["weight_decay"], GRID["recon_weight"]))

    print("device:", device)
    data, y, lab_pos, unlab_pos, splits, sample_ids = load_all()
    print(f"labelled: {len(lab_pos)}   unlabelled: {len(unlab_pos)}")

    metric_rows, pred_rows = [], []
    for model_name, use_unlabelled in [("fusion", False), ("fusion_semi", True)]:
        t0 = time.time()
        for seed in seeds:
            for fold in range(N_FOLDS):
                is_test = splits[f"fold_seed{seed}"].values == fold
                test_idx = lab_pos[is_test]                 # labelled patients only
                trval_idx = lab_pos[~is_test]
                tr_idx, va_idx = train_test_split(
                    trval_idx, test_size=0.2, stratify=y[trval_idx], random_state=seed
                )
                if use_unlabelled:
                    tr_idx = np.concatenate([tr_idx, unlab_pos])   # training side only

                assert (y[test_idx] >= 0).all() and (y[va_idx] >= 0).all()
                assert not set(tr_idx) & set(test_idx)

                # preprocessing fitted on the training side (incl. unlabelled if used)
                Xtr, Xva, Xte = [], [], []
                for name in OMICS:
                    prep = Preprocessor(TOP_K[name]).fit(data[name][tr_idx])
                    Xtr.append(tensor(prep.transform(data[name][tr_idx])))
                    Xva.append(tensor(prep.transform(data[name][va_idx])))
                    Xte.append(tensor(prep.transform(data[name][test_idx])))
                ytr, yva = tensor(y[tr_idx], torch.long), tensor(y[va_idx], torch.long)

                best = None
                for wd, rw in grid:
                    model, val_loss = train_fusion(Xtr, ytr, Xva, yva, wd, rw, seed)
                    if best is None or val_loss < best[1]:
                        best = (model, val_loss, wd, rw)
                model, _, wd, rw = best

                model.eval()
                with torch.no_grad():
                    pred = model(Xte)[0].argmax(dim=1).cpu().numpy()
                y_test = y[test_idx]
                metric_rows.append({
                    "model": model_name, "seed": seed, "fold": fold,
                    "macro_f1": f1_score(y_test, pred, average="macro"),
                    "balanced_acc": balanced_accuracy_score(y_test, pred),
                    "weight_decay": wd, "recon_weight": rw,
                })
                for sid, t, p in zip(sample_ids[test_idx], y_test, pred):
                    pred_rows.append({"model": model_name, "seed": seed, "fold": fold,
                                      "sample_id": sid, "true": CLASSES[t], "pred": CLASSES[p]})
        print(f"{model_name:12s} done in {time.time() - t0:.0f} s")

    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby("model")[["macro_f1", "balanced_acc"]].agg(["mean", "std"]).round(3)
    suffix = "_quick" if args.quick else ""
    metrics.to_csv(RES / f"fusion_metrics{suffix}.csv", index=False)
    summary.to_csv(RES / f"fusion_summary{suffix}.csv")
    pd.DataFrame(pred_rows).to_csv(RES / f"fusion_predictions{suffix}.csv", index=False)
    print("\n=== Summary (mean and std over splits) ===")
    print(summary)


if __name__ == "__main__":
    main()
