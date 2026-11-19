import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"

CLASSES = ["LumA", "LumB", "Basal", "Her2"]
OMICS = ["expr", "mirna", "meth"]
TOP_K = {"expr": 2000, "mirna": 521, "meth": 5000}     
N_FOLDS = 5

HIDDEN, SHARED, LATENT = 256, 128, 32
LR, WEIGHT_DECAY = 1e-3, 1e-3
MAX_EPOCHS, PATIENCE, WARMUP = 3000, 100, 200
BETA = 1.0
VARIANTS = {"ae": False, "vae": True}                   

device = "cuda" if torch.cuda.is_available() else "cpu"


# Data

def load_all():
    labels = pd.read_csv(PROC / "labels.csv")
    data, names = {}, {}
    for name in OMICS:
        df = pd.read_pickle(PROC / f"{name}.pkl")
        assert (df.index.values == labels["sample_id"].values).all()
        data[name] = df.values.astype("float32")
        names[name] = df.columns.values
    y = labels["pam50"].map({c: i for i, c in enumerate(CLASSES)}).fillna(-1).astype(int).values
    lab_pos = np.where(labels["labelled"].values)[0]
    unlab_pos = np.where(~labels["labelled"].values)[0]
    splits = pd.read_csv(RES / "splits.csv")
    assert (splits["sample_id"].values == labels["sample_id"].values[lab_pos]).all()
    return data, names, y, lab_pos, unlab_pos, splits


class Preprocessor:

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
class MultiOmicsVAE(nn.Module):
    def __init__(self, dims, variational=True):
        super().__init__()
        self.variational = variational
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Dropout(0.2), nn.Linear(d, HIDDEN), nn.ReLU()) for d in dims
        ])
        self.shared = nn.Sequential(nn.Linear(HIDDEN * len(dims), SHARED), nn.ReLU())
        self.to_mu = nn.Linear(SHARED, LATENT)
        self.to_logvar = nn.Linear(SHARED, LATENT)
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, d)) for d in dims
        ])

    def encode(self, xs):
        h = torch.cat([enc(x) for enc, x in zip(self.encoders, xs)], dim=1)   # fusion
        h = self.shared(h)
        return self.to_mu(h), self.to_logvar(h).clamp(-10, 10)

    def forward(self, xs):
        mu, logvar = self.encode(xs)
        if self.variational and self.training:
            eps = torch.randn_like(mu)                       
            z = mu + torch.exp(0.5 * logvar) * eps           
        else:
            z = mu                                          
        return [dec(z) for dec in self.decoders], mu, logvar


def loss_terms(model, xs, omics_weights):
    recons, mu, logvar = model(xs)
    rec = sum(w * 0.5 * ((r - x) ** 2).sum(dim=1)
              for w, r, x in zip(omics_weights, recons, xs)).mean()
    kl_per_dim = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1)    
    kl = kl_per_dim.sum(dim=1).mean()
    return rec, kl, kl_per_dim.mean(dim=0)


def train_model(dims, Xtr, Xva, variational, seed, max_epochs):
    torch.manual_seed(seed)
    model = MultiOmicsVAE(dims, variational).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total = sum(dims)
    omics_weights = [(total / len(dims)) / d for d in dims]      

    best_val, best_state, wait = np.inf, None, 0
    warmup = min(WARMUP, max_epochs // 2)
    for epoch in range(max_epochs):
        beta_t = beta * min(1.0, (epoch + 1) / warmup)          
        model.train()
        optimizer.zero_grad()
        rec, kl, _ = loss_terms(model, Xtr, omics_weights)
        (rec + beta_t * kl).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vrec, vkl, _ = loss_terms(model, Xva, omics_weights)
            val = (vrec + beta * vkl).item()
        if epoch + 1 < warmup:                                  
            continue
        if val < best_val:
            best_val, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, _, kl_dims = loss_terms(model, Xtr, omics_weights)
        mu_tr, _ = model.encode(Xtr)
    diagnostics = {
        "epochs": epoch + 1,
        "active_units": int((mu_tr.var(dim=0) > 0.01).sum().item()),
        "collapsed_dims": int((kl_dims < 0.01).sum().item()) if variational else np.nan,
    }
    return model, diagnostics


def encode_mu(model, xs):
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(xs)
    return mu.cpu().numpy()


def tensor(a):
    return torch.tensor(a, dtype=torch.float32, device=device)


def make_classifier(seed):
    """Same classifier family as the baselines: L2 logistic regression, C by inner CV."""
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=np.logspace(-4, 1, 8), l1_ratios=(0,), solver="lbfgs",
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        scoring="f1_macro", max_iter=2000, use_legacy_attributes=True))


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    runs = [(0, 0)] if args.quick else [(s, f) for s in [0, 1, 2] for f in range(N_FOLDS)]
    max_epochs = 400 if args.quick else MAX_EPOCHS

    print("device:", device)
    data, names, y, lab_pos, unlab_pos, splits = load_all()

    rows = []
    for variant, variational in VARIANTS.items():
        t0 = time.time()
        for seed, fold in runs:
            is_test = splits[f"fold_seed{seed}"].values == fold
            test_idx, trval_idx = lab_pos[is_test], lab_pos[~is_test]
            tr_lab, va_idx = train_test_split(trval_idx, test_size=0.2,
                                              stratify=y[trval_idx], random_state=seed)
            tr_idx = np.concatenate([tr_lab, unlab_pos])          
            assert not set(tr_idx) & set(test_idx) and (y[test_idx] >= 0).all()

            preps = {n: Preprocessor(TOP_K[n]).fit(data[n][tr_idx]) for n in OMICS}

            def X(idx):
                return [tensor(preps[n].transform(data[n][idx])) for n in OMICS]

            Xtr = X(tr_idx)
            dims = [x.shape[1] for x in Xtr]
            model, diag = train_model(dims, Xtr, X(va_idx), variational, seed, max_epochs)

            clf = make_classifier(seed).fit(encode_mu(model, X(trval_idx)), y[trval_idx])
            pred = clf.predict(encode_mu(model, X(test_idx)))
            rows.append({"model": f"{variant}_lr", "seed": seed, "fold": fold,
                         "macro_f1": f1_score(y[test_idx], pred, average="macro"),
                         "balanced_acc": balanced_accuracy_score(y[test_idx], pred), **diag})

            if variant == "vae" and seed == 0 and fold == 0:       
                torch.save({
                    "state_dict": model.state_dict(), "dims": dims,
                    "preprocessing": {n: {"keep": p.keep_, "median": p.median_,
                                          "mean": p.mean_, "std": p.std_} for n, p in preps.items()},
                    "feature_names": {n: names[n][preps[n].keep_] for n in OMICS},
                    "train_idx": tr_idx, "trval_idx": trval_idx, "test_idx": test_idx,
                }, RES / ("vae_seed0_fold0_quick.pt" if args.quick else "vae_seed0_fold0.pt"))
            r = rows[-1]
            print(f"{variant} seed {seed} fold {fold}: macro-F1 {r['macro_f1']:.3f}, "
                  f"{r['epochs']} epochs, active units {r['active_units']}/{LATENT}")
        print(f"{variant} done in {time.time() - t0:.0f} s")

    metrics = pd.DataFrame(rows)
    suffix = "_quick" if args.quick else ""
    metrics.to_csv(RES / f"vae_metrics{suffix}.csv", index=False)
    summary = metrics.groupby("model")[["macro_f1", "balanced_acc", "active_units", "collapsed_dims"]] \
                     .agg(["mean", "std"]).round(3)
    summary.to_csv(RES / f"vae_summary{suffix}.csv")
    pd.set_option("display.width", 200)
    print("\n=== Summary ===")
    print(summary)

    if not args.quick:
        base = pd.read_csv(RES / "baselines_metrics.csv")
        both = pd.concat([base[base["model"] == "expr"], metrics])
        w = both.pivot_table(index=["seed", "fold"], columns="model", values="macro_f1")
        print()
        for a, b in [("vae_lr", "expr"), ("ae_lr", "expr"), ("vae_lr", "ae_lr")]:
            d = w[a] - w[b]
            print(f"{a} vs {b}: mean diff {d.mean():+.3f}, wins {(d > 0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
