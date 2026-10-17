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

# Settings
ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"

CLASSES = ["LumA", "LumB", "Basal", "Her2"]
OMICS = ["expr", "mirna", "meth"]
TOP_K = {"expr": 2000, "mirna": 521, "meth": 5000}
N_FOLDS = 5

HIDDEN, LATENT, DROPOUT = 256, 32, 0.5
LR = 1e-3
MAX_EPOCHS, PATIENCE = 2000, 100

JOINT_GRID = list(itertools.product([1e-3, 1e-2, 1e-1], [0.1, 1.0]))      

# two-stage training
PRE_MAX_EPOCHS, PRE_PATIENCE, PRE_WD = 3000, 100, 1e-4
FT_LR = 1e-4                 
FT_RECON_WEIGHT = 1.0       
TWO_STAGE_GRID = list(itertools.product(["frozen", "finetune"], [1e-3, 1e-2, 1e-1]))  

VARIANTS = [  
    ("fusion", False, "joint"),
    ("fusion_semi", True, "joint"),
    ("fusion_2stage", False, "two_stage"),
    ("fusion_2stage_semi", True, "two_stage"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
ce = nn.CrossEntropyLoss(ignore_index=-1)   
mse = nn.MSELoss()


# Data
def load_all():
    """All 617 patients. y = class index for labelled patients, -1 for unlabelled."""
    labels = pd.read_csv(PROC / "labels.csv")
    data = {}
    for name in OMICS:
        df = pd.read_pickle(PROC / f"{name}.pkl")
        assert (df.index.values == labels["sample_id"].values).all()
        data[name] = df.values.astype("float32")

    y = labels["pam50"].map({c: i for i, c in enumerate(CLASSES)}).fillna(-1).astype(int).values
    lab_pos = np.where(labels["labelled"].values)[0]
    unlab_pos = np.where(~labels["labelled"].values)[0]

    splits = pd.read_csv(RES / "splits.csv")
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


# Model
class FusionAE(nn.Module):
    def __init__(self, dims, n_classes=len(CLASSES)):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.2),                
                nn.Linear(d, HIDDEN), nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, LATENT),
            ) for d in dims
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, d))
            for d in dims
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(LATENT * len(dims), 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, xs):
        zs = [enc(x) for enc, x in zip(self.encoders, xs)]
        recons = [dec(z) for dec, z in zip(self.decoders, zs)]
        logits = self.classifier(torch.cat(zs, dim=1))   
        return logits, recons, zs


def recon_loss(recons, xs):
    """Mean reconstruction error over the three omics."""
    return sum(mse(r, x) for r, x in zip(recons, xs)) / len(xs)


def copy_state(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


# Strategy 1: joint training
def train_joint(dims, Xtr, ytr, Xva, yva, weight_decay, recon_weight, seed):
    """Early stopping on the validation CLASSIFICATION loss (comparable across settings)."""
    torch.manual_seed(seed)
    model = FusionAE(dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)

    best_val, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits, recons, _ = model(Xtr)
        clf, rec = ce(logits, ytr), recon_loss(recons, Xtr)
        (clf + recon_weight * rec).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val = ce(model(Xva)[0], yva).item()
        history.append({"stage": "joint", "epoch": epoch, "train_clf": clf.item(),
                        "train_rec": rec.item(), "val_clf": val})
        if val < best_val:
            best_val, best_state, wait = val, copy_state(model), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val, history


# Strategy 2: two-stage training
def pretrain(dims, Xtr, Xva, seed):
    """Stage 1: encoders + decoders on reconstruction only. Early stopping on validation reconstruction."""
    torch.manual_seed(seed)
    model = FusionAE(dims).to(device)
    params = list(model.encoders.parameters()) + list(model.decoders.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=PRE_WD)

    best_val, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(PRE_MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        _, recons, _ = model(Xtr)
        rec = recon_loss(recons, Xtr)
        rec.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val = recon_loss(model(Xva)[1], Xva).item()
        history.append({"stage": "pretrain", "epoch": epoch, "train_rec": rec.item(), "val_rec": val})
        if val < best_val:
            best_val, best_state, wait = val, copy_state(model), 0
        else:
            wait += 1
            if wait >= PRE_PATIENCE:
                break

    return best_state, history


def finetune(dims, pre_state, Xtr, ytr, Xva, yva, mode, weight_decay, seed):
    """Stage 2: train the classifier from the pretrained encoders (frozen or fine-tuned)."""
    torch.manual_seed(seed)
    model = FusionAE(dims).to(device)
    model.load_state_dict(pre_state)
    frozen = mode == "frozen"

    if frozen:
        for p in list(model.encoders.parameters()) + list(model.decoders.parameters()):
            p.requires_grad = False
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=weight_decay)

    best_val, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(MAX_EPOCHS):
        model.train()
        if frozen:
            model.encoders.eval()          
        optimizer.zero_grad()
        logits, recons, _ = model(Xtr)
        clf, rec = ce(logits, ytr), recon_loss(recons, Xtr)
        loss = clf if frozen else clf + FT_RECON_WEIGHT * rec
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val = ce(model(Xva)[0], yva).item()
        history.append({"stage": "finetune", "epoch": epoch, "train_clf": clf.item(),
                        "train_rec": rec.item(), "val_clf": val})
        if val < best_val:
            best_val, best_state, wait = val, copy_state(model), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val, history


def tensor(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype, device=device)


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [0] if args.quick else [0, 1, 2]
    joint_grid = [(1e-2, 1.0)] if args.quick else JOINT_GRID
    two_stage_grid = [("finetune", 1e-2)] if args.quick else TWO_STAGE_GRID

    print("device:", device)
    data, y, lab_pos, unlab_pos, splits, sample_ids = load_all()
    print(f"labelled: {len(lab_pos)}   unlabelled: {len(unlab_pos)}")

    metric_rows, pred_rows, hist_rows = [], [], []
    for model_name, use_unlabelled, strategy in VARIANTS:
        t0 = time.time()
        for seed in seeds:
            for fold in range(N_FOLDS):
                is_test = splits[f"fold_seed{seed}"].values == fold
                test_idx = lab_pos[is_test]
                trval_idx = lab_pos[~is_test]
                tr_idx, va_idx = train_test_split(
                    trval_idx, test_size=0.2, stratify=y[trval_idx], random_state=seed
                )
                if use_unlabelled:
                    tr_idx = np.concatenate([tr_idx, unlab_pos])   

                assert (y[test_idx] >= 0).all() and (y[va_idx] >= 0).all()
                assert not set(tr_idx) & set(test_idx)

                Xtr, Xva, Xte = [], [], []
                for name in OMICS:
                    prep = Preprocessor(TOP_K[name]).fit(data[name][tr_idx])
                    Xtr.append(tensor(prep.transform(data[name][tr_idx])))
                    Xva.append(tensor(prep.transform(data[name][va_idx])))
                    Xte.append(tensor(prep.transform(data[name][test_idx])))
                ytr, yva = tensor(y[tr_idx], torch.long), tensor(y[va_idx], torch.long)
                dims = [x.shape[1] for x in Xtr]

                best = None
                if strategy == "joint":
                    for wd, rw in joint_grid:
                        model, val, hist = train_joint(dims, Xtr, ytr, Xva, yva, wd, rw, seed)
                        if best is None or val < best[1]:
                            best = (model, val, f"wd={wd}, recon={rw}", hist)
                else:
                    pre_state, pre_hist = pretrain(dims, Xtr, Xva, seed)   
                    for mode, wd in two_stage_grid:
                        model, val, hist = finetune(dims, pre_state, Xtr, ytr, Xva, yva, mode, wd, seed)
                        if best is None or val < best[1]:
                            best = (model, val, f"{mode}, wd={wd}", pre_hist + hist)
                model, _, setting, hist = best
                hist_rows += [{"model": model_name, "seed": seed, "fold": fold, **h} for h in hist]

                model.eval()
                with torch.no_grad():
                    pred = model(Xte)[0].argmax(dim=1).cpu().numpy()
                y_test = y[test_idx]
                metric_rows.append({
                    "model": model_name, "seed": seed, "fold": fold,
                    "macro_f1": f1_score(y_test, pred, average="macro"),
                    "balanced_acc": balanced_accuracy_score(y_test, pred),
                    "setting": setting,
                })
                for sid, t, p in zip(sample_ids[test_idx], y_test, pred):
                    pred_rows.append({"model": model_name, "seed": seed, "fold": fold,
                                      "sample_id": sid, "true": CLASSES[t], "pred": CLASSES[p]})
        print(f"{model_name:20s} done in {time.time() - t0:.0f} s")

    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby("model")[["macro_f1", "balanced_acc"]].agg(["mean", "std"]).round(3)
    suffix = "_quick" if args.quick else ""
    metrics.to_csv(RES / f"fusion_metrics{suffix}.csv", index=False)
    summary.to_csv(RES / f"fusion_summary{suffix}.csv")
    pd.DataFrame(pred_rows).to_csv(RES / f"fusion_predictions{suffix}.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(RES / f"fusion_history{suffix}.csv", index=False)
    print("\n=== Summary (mean and std over splits) ===")
    print(summary)
    print("\nSettings chosen on validation (two-stage):")
    print(metrics[metrics["model"].str.contains("2stage")].groupby("model")["setting"]
          .value_counts().to_string())


if __name__ == "__main__":
    main()
