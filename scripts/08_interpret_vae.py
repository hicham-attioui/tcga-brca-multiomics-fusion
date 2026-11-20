import sys
from importlib import import_module
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import GradientShap, IntegratedGradients
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))
v = import_module("07_run_vae")                    

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
OMICS_LABEL = {"expr": "Expression", "mirna": "miRNA", "meth": "Methylation"}
OMICS_COLOR = {"expr": "#2a78d6", "mirna": "#1baf7a", "meth": "#eb6834"}
TOP_N, N_BACKGROUND, IG_STEPS, SHAP_SAMPLES = 50, 100, 50, 50


class LatentMean(nn.Module):

    def __init__(self, vae, dims):
        super().__init__()
        self.vae, self.dims = vae, dims

    def forward(self, x):
        mu, _ = self.vae.encode(list(torch.split(x, self.dims, dim=1)))
        return mu


def preprocess(data, ck, idx):
    blocks = []
    for n in v.OMICS:
        p = ck["preprocessing"][n]
        Xk = data[n][idx][:, p["keep"]]
        Xk = np.where(np.isnan(Xk), p["median"], Xk)
        blocks.append((Xk - p["mean"]) / p["std"])
    return v.tensor(np.concatenate(blocks, axis=1).astype("float32"))


def main():
    torch.manual_seed(0)
    ck = torch.load(RES / "vae_seed0_fold0.pt", weights_only=False)
    data, _, y, *_ = v.load_all()
    dims = ck["dims"]

    vae = v.MultiOmicsVAE(dims).to(v.device)
    vae.load_state_dict(ck["state_dict"])
    vae.eval()
    f = LatentMean(vae, dims).eval()

    feat_names = np.concatenate([ck["feature_names"][n] for n in v.OMICS])
    feat_omics = np.concatenate([[n] * d for n, d in zip(v.OMICS, dims)])

    X_trval = preprocess(data, ck, ck["trval_idx"])
    X_test = preprocess(data, ck, ck["test_idx"])
    X_train = preprocess(data, ck, ck["train_idx"])

    with torch.no_grad():
        mu_trval = f(X_trval).cpu().numpy()
    clf = v.make_classifier(0).fit(mu_trval, y[ck["trval_idx"]])
    coef = clf[-1].coef_                                    
    chosen = {}
    for c, name in enumerate(v.CLASSES):
        k = int(np.argmax(np.abs(coef[c])))
        chosen.setdefault(k, []).append(name)
    print("Latent dimensions explained (dimension: subtypes it matters most for):")
    for k, subtypes in chosen.items():
        print(f"  z{k}: {', '.join(subtypes)}")

    ig, gs = IntegratedGradients(f), GradientShap(f)
    zero_baseline = torch.zeros_like(X_test)                 
    bg_idx = torch.randperm(len(X_train))[:N_BACKGROUND]
    background = X_train[bg_idx]

    comp_rows, imp_rows, top_rows = [], [], []
    for k, subtypes in chosen.items():
        a_ig, delta = ig.attribute(X_test, baselines=zero_baseline, target=k, n_steps=IG_STEPS,
                                   internal_batch_size=512, return_convergence_delta=True)
        a_gs = gs.attribute(X_test, baselines=background, target=k, n_samples=SHAP_SAMPLES)

        with torch.no_grad():                                
            gap = (f(X_test)[:, k] - f(zero_baseline)[:, k]).abs().mean().item()
        rel_delta = delta.abs().mean().item() / (gap + 1e-12)

        imp_ig = a_ig.abs().mean(0).detach().cpu().numpy()
        imp_gs = a_gs.abs().mean(0).detach().cpu().numpy()

        rho = spearmanr(imp_ig, imp_gs).correlation
        top_ig = set(np.argsort(imp_ig)[::-1][:TOP_N])
        top_gs = set(np.argsort(imp_gs)[::-1][:TOP_N])
        row = {"dimension": f"z{k}", "subtypes": "+".join(subtypes),
               "spearman_all_features": rho, f"top{TOP_N}_overlap": len(top_ig & top_gs),
               "ig_completeness_error": rel_delta}
        for n in v.OMICS:                                   
            m = feat_omics == n
            row[f"ig_share_{n}"] = imp_ig[m].sum() / imp_ig.sum()
            row[f"shap_share_{n}"] = imp_gs[m].sum() / imp_gs.sum()
            row[f"top{TOP_N}_overlap_{n}"] = len({i for i in top_ig & top_gs if feat_omics[i] == n})
        comp_rows.append(row)

        imp_rows.append(pd.DataFrame({"dimension": f"z{k}", "feature": feat_names, "omics": feat_omics,
                                      "ig_importance": imp_ig, "shap_importance": imp_gs}))
        for method, imp in [("IG", imp_ig), ("GradientSHAP", imp_gs)]:
            for rank, i in enumerate(np.argsort(imp)[::-1][:15], start=1):
                top_rows.append({"dimension": f"z{k}", "method": method, "rank": rank,
                                 "feature": feat_names[i], "omics": feat_omics[i], "importance": imp[i]})

    comp = pd.DataFrame(comp_rows)
    imps = pd.concat(imp_rows)
    tops = pd.DataFrame(top_rows)
    comp.to_csv(RES / "interpret_comparison.csv", index=False)
    imps.to_csv(RES / "interpret_importance.csv", index=False)
    tops.to_csv(RES / "interpret_top_features.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== IG vs GradientSHAP, per latent dimension ===")
    print(comp.round(3).to_string(index=False))
    print("\n=== Top 10 features per dimension ===")
    for (k, method), g in tops[tops["rank"] <= 10].groupby(["dimension", "method"], sort=False):
        print(f"{k} {method:13s}: " + ", ".join(f"{r.feature} ({r.omics})" for r in g.itertuples()))

    first = comp.iloc[0]["dimension"]
    d = imps[imps["dimension"] == first]
    from matplotlib.ticker import NullFormatter
    fig, grid = plt.subplots(2, 3, figsize=(15, 9), gridspec_kw={"height_ratios": [1, 1.1]})
    within_rows = []
    for col, n in enumerate(v.OMICS):
        sc = grid[0, col]
        s_all = d[d["omics"] == n]
        sc.scatter(s_all["ig_importance"], s_all["shap_importance"], s=6, alpha=0.4, color=OMICS_COLOR[n])
        lims = [min(s_all["ig_importance"].min(), s_all["shap_importance"].min()),
                max(s_all["ig_importance"].max(), s_all["shap_importance"].max())]
        sc.plot(lims, lims, ls="--", lw=0.8, color="#888780", label="perfect agreement (y = x)")
        sc.set_xscale("log"); sc.set_yscale("log")
        for axis in (sc.xaxis, sc.yaxis):                   
            axis.set_minor_formatter(NullFormatter())
        sc.set_xlabel("Integrated Gradients importance")
        if col == 0:
            sc.set_ylabel("GradientSHAP importance")
        sc.legend(frameon=False, fontsize=8, loc="upper left")
        sc.spines[["top", "right"]].set_visible(False)

        ax = grid[1, col]
        s = d[d["omics"] == n].copy()
        n_top = min(TOP_N, len(s) // 5)                     
        top_ig = set(s.nlargest(n_top, "ig_importance")["feature"])
        top_gs = set(s.nlargest(n_top, "shap_importance")["feature"])
        rho = spearmanr(s["ig_importance"], s["shap_importance"]).correlation
        within_rows.append({"dimension": first, "omics": n, "n_features": len(s),
                            "spearman": rho, "top_overlap": len(top_ig & top_gs), "top_n": n_top})

        s["IG"] = s["ig_importance"] / s["ig_importance"].max()
        s["GradientSHAP"] = s["shap_importance"] / s["shap_importance"].max()
        top10 = s.assign(mean=(s["IG"] + s["GradientSHAP"]) / 2).nlargest(10, "mean")[::-1]
        ypos = np.arange(len(top10))
        ax.barh(ypos + 0.2, top10["IG"], height=0.4, color=OMICS_COLOR[n], label="Integrated Gradients")
        ax.barh(ypos - 0.2, top10["GradientSHAP"], height=0.4, color="#5f5e5a", label="GradientSHAP")
        ax.set_yticks(ypos, top10["feature"], fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Importance (relative to the top feature)")
        sc.set_title(f"{OMICS_LABEL[n]} ({len(s)} features)\n"
                     f"Spearman {rho:.2f}, top-{n_top} overlap {len(top_ig & top_gs)}/{n_top}", fontsize=10)
        ax.set_title(f"{OMICS_LABEL[n]}: top 10 features", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Latent dimension {first}, the most important for {comp.iloc[0]['subtypes']}: "
                 f"IG vs GradientSHAP. Top: all features. Bottom: top 10 features.", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_vae_ig_vs_shap_by_omics.png", dpi=300)
    fig.savefig(FIG / "fig6_vae_ig_vs_shap_by_omics.pdf")
    plt.close(fig)
    within = pd.DataFrame(within_rows)
    within.to_csv(RES / "interpret_within_omics.csv", index=False)
    print(f"\n=== {first}: agreement within each omics ===")
    print(within.round(3).to_string(index=False))

    fig, b = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(comp))
    width = 0.25
    for i, n in enumerate(v.OMICS):
        b.bar(x + (i - 1) * width, comp[f"top{TOP_N}_overlap_{n}"], width,
              color=OMICS_COLOR[n], label=OMICS_LABEL[n])
    b.set_xticks(x, [f"{r.dimension}\n({r.subtypes})" for r in comp.itertuples()])
    b.set_ylabel(f"Features in BOTH overall top-{TOP_N} lists")
    b.set_title("Where IG and GradientSHAP agree, by omics", fontsize=10)
    b.legend(frameon=False)
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_vae_agreement_summary.png", dpi=300)
    fig.savefig(FIG / "fig7_vae_agreement_summary.pdf")
    print("\nsaved figures/fig6_vae_ig_vs_shap_by_omics and fig7_vae_agreement_summary")


if __name__ == "__main__":
    main()
