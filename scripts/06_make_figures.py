from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, FIG = ROOT / "data" / "processed", ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

CLASSES = ["LumA", "LumB", "Basal", "Her2"]
SUBTYPE_COLORS = {"LumA": "#2a78d6", "LumB": "#1baf7a", "Basal": "#e34948", "Her2": "#eda100",
                  "Unlabelled": "#b4b2a9"}

MODELS = {
    "expr": ("LR expression", "Classical"),
    "mirna": ("LR miRNA", "Classical"),
    "meth": ("LR methylation", "Classical"),
    "concat": ("LR concatenated (early fusion)", "Classical"),
    "rf_expr": ("RF expression", "Classical"),
    "rf_mirna": ("RF miRNA", "Classical"),
    "rf_meth": ("RF methylation", "Classical"),
    "rf_concat": ("RF concatenated (early fusion)", "Classical"),
    "mlp_expr": ("MLP expression", "Deep, single omics"),
    "mlp_mirna": ("MLP miRNA", "Deep, single omics"),
    "mlp_meth": ("MLP methylation", "Deep, single omics"),
    "fusion": ("Fusion autoencoder", "Deep, intermediate fusion"),
    "fusion_semi": ("Fusion autoencoder + unlabelled", "Deep, intermediate fusion"),
    "fusion_2stage": ("Fusion AE, two-stage", "Deep, intermediate fusion"),
    "fusion_2stage_semi": ("Fusion AE, two-stage + unlabelled", "Deep, intermediate fusion"),
}
FAMILY_COLORS = {"Classical": "#5f5e5a", "Deep, single omics": "#2a78d6",
                 "Deep, intermediate fusion": "#eb6834"}
REFERENCE = "expr"   

FLOW = [("Expression, primary tumours", 1097), ("Methylation, primary tumours", 785),
        ("miRNA, primary tumours", 749), ("All three omics", 617)]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


def fig1_data_overview():
    labels = pd.read_csv(PROC / "labels.csv")
    counts = labels["pam50"].value_counts().reindex(CLASSES)
    n_unlab = int((~labels["labelled"]).sum())

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))

    names, values = zip(*FLOW)
    a.barh(names[::-1], values[::-1], color="#5f5e5a", height=0.6)
    for i, v in enumerate(values[::-1]):
        a.text(v + 15, i, str(v), va="center", fontsize=9)
    a.set_xlabel("Number of patients")
    a.set_title("TCGA-BRCA patients per data type")
    a.set_xlim(0, max(values) * 1.15)

    bar_names = CLASSES + ["Unlabelled"]
    bar_values = list(counts.values) + [n_unlab]
    a_bars = b.bar(bar_names, bar_values, color=[SUBTYPE_COLORS[c] for c in bar_names], width=0.6)
    b.bar_label(a_bars, fontsize=9)
    b.set_ylabel("Number of patients")
    b.set_title(f"PAM50 labels, {len(labels)} patients with all three omics\n"
                f"(unlabelled includes 14 Normal-like)", fontsize=10)

    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig1_data_overview")


def load_metrics():
    files = ["baselines_metrics.csv", "mlp_metrics.csv", "fusion_metrics.csv"]
    return pd.concat([pd.read_csv(RES / f) for f in files if (RES / f).exists()])


def results_table(metrics):
    wide = metrics.pivot_table(index=["seed", "fold"], columns="model", values="macro_f1")
    rows = []
    for m in [m for m in MODELS if m in wide.columns]:
        sub = metrics[metrics["model"] == m]
        diff = wide[m] - wide[REFERENCE]
        rows.append({
            "Model": MODELS[m][0],
            "Macro-F1": f"{sub['macro_f1'].mean():.3f} ± {sub['macro_f1'].std():.3f}",
            "Balanced accuracy": f"{sub['balanced_acc'].mean():.3f} ± {sub['balanced_acc'].std():.3f}",
            "Diff. vs LR expression": "reference" if m == REFERENCE else f"{diff.mean():+.3f}",
            "Splits won vs LR expression": "" if m == REFERENCE else f"{(diff > 0).sum()}/{len(diff)}",
        })
    table = pd.DataFrame(rows)
    table.to_csv(RES / "results_table.csv", index=False)
    (RES / "results_table.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    print(table.to_string(index=False))

    pairs = [("fusion_semi", "fusion"), ("fusion_2stage_semi", "fusion_2stage"),
             ("fusion_2stage", "fusion"), ("fusion_2stage_semi", "fusion_semi")]
    print()
    for a, b in pairs:
        if {a, b} <= set(wide.columns):
            d = wide[a] - wide[b]
            print(f"{MODELS[a][0]} vs {MODELS[b][0]}: mean diff {d.mean():+.3f}, "
                  f"wins {(d > 0).sum()}/{len(d)}")


def fig2_model_comparison(metrics):
    stats = metrics.groupby("model")["macro_f1"].agg(["mean", "std"])
    order = [m for m in stats.sort_values("mean").index if m in MODELS]   
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(order):
        color = FAMILY_COLORS[MODELS[m][1]]
        vals = metrics.loc[metrics["model"] == m, "macro_f1"].values
        ax.scatter(vals, i + rng.uniform(-0.15, 0.15, len(vals)), s=12, color=color, alpha=0.35)
        ax.errorbar(stats.loc[m, "mean"], i, xerr=stats.loc[m, "std"], fmt="o", color=color,
                    ms=7, capsize=3, lw=1.5)
    ax.axvline(stats.loc[REFERENCE, "mean"], ls="--", lw=1, color="#888780")
    ax.set_yticks(range(len(order)), [MODELS[m][0] for m in order])
    ax.set_xlabel("Macro-F1 on test folds (15 splits: 5 folds x 3 seeds)")
    ax.set_title("PAM50 subtype classification (365 labelled patients)")
    for fam, c in FAMILY_COLORS.items():
        ax.scatter([], [], color=c, label=fam)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.text(stats.loc[REFERENCE, "mean"], -0.9, " LR expression mean", fontsize=7, color="#888780")
    ax.set_ylim(-1.1, len(order) - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig2_model_comparison")


def fig3_confusion(model="fusion_semi"):
    p = pd.read_csv(RES / "fusion_predictions.csv")
    p = p[p["model"] == model]
    cm = confusion_matrix(p["true"], p["pred"], labels=CLASSES, normalize="true")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.6 else "black", fontsize=10)
    n_per_class = p.drop_duplicates("sample_id")["true"].value_counts()
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), [f"{c} (n={n_per_class.get(c, 0)})" for c in CLASSES])
    ax.set_xlabel("Predicted subtype")
    ax.set_ylabel("True subtype")
    ax.set_title(f"{MODELS[model][0]}\n(row fractions, test predictions pooled over 3 seeds)", fontsize=10)
    save(fig, "fig3_confusion_fusion")


def fig4_learning_curves():
    """Training and validation losses of the fusion models.
    Faint lines: validation loss in every split. Bold lines: one representative split (seed 0, fold 0)."""
    path = RES / "fusion_history.csv"
    if not path.exists():
        print("fusion_history.csv not found: rerun 05_run_fusion.py to get learning curves")
        return
    h = pd.read_csv(path)
    if "stage" in h.columns:
        h = h[h["stage"] == "joint"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, m in zip(axes, ["fusion", "fusion_semi"]):
        hm = h[h["model"] == m]
        for _, split in hm.groupby(["seed", "fold"]):
            ax.plot(split["epoch"], split["val_clf"], color="#eb6834", alpha=0.15, lw=0.8)
        rep = hm[(hm["seed"] == 0) & (hm["fold"] == 0)]
        ax.plot(rep["epoch"], rep["train_clf"], color="#2a78d6", lw=2, label="train classification loss")
        ax.plot(rep["epoch"], rep["val_clf"], color="#eb6834", lw=2, label="validation classification loss")
        ax.plot(rep["epoch"], rep["train_rec"], color="#5f5e5a", lw=1.5, ls="--",
                label="train reconstruction loss")
        best = rep.loc[rep["val_clf"].idxmin(), "epoch"]
        ax.axvline(best, color="#888780", lw=1, ls=":")
        ax.text(best, ax.get_ylim()[1] * 0.95, " best epoch", fontsize=8, color="#888780")
        ax.set_title(MODELS[m][0], fontsize=11)
        ax.set_xlabel("epoch")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Learning curves (bold: seed 0, fold 0; faint: validation loss in all 15 splits)",
                 fontsize=10)
    save(fig, "fig4_learning_curves")


def fig5_two_stage_curves(model="fusion_2stage_semi"):
    """Two-stage training, seed 0 fold 0: pretraining (reconstruction) then fine-tuning (classification)."""
    path = RES / "fusion_history.csv"
    if not path.exists():
        return
    h = pd.read_csv(path)
    if "stage" not in h.columns or model not in h["model"].values:
        return
    rep = h[(h["model"] == model) & (h["seed"] == 0) & (h["fold"] == 0)]
    pre, ft = rep[rep["stage"] == "pretrain"], rep[rep["stage"] == "finetune"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(pre["epoch"], pre["train_rec"], color="#5f5e5a", lw=2, label="train reconstruction loss")
    a.plot(pre["epoch"], pre["val_rec"], color="#eb6834", lw=2, label="validation reconstruction loss")
    a.set_title("Stage 1: pretraining (reconstruction only)", fontsize=11)

    b.plot(ft["epoch"], ft["train_clf"], color="#2a78d6", lw=2, label="train classification loss")
    b.plot(ft["epoch"], ft["val_clf"], color="#eb6834", lw=2, label="validation classification loss")
    b.plot(ft["epoch"], ft["train_rec"], color="#5f5e5a", lw=1.5, ls="--", label="train reconstruction loss")
    b.axvline(ft.loc[ft["val_clf"].idxmin(), "epoch"], color="#888780", lw=1, ls=":")
    b.set_title("Stage 2: fine-tuning (classification)", fontsize=11)

    for ax in (a, b):
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{MODELS[model][0]}, seed 0, fold 0", fontsize=10)
    save(fig, "fig5_two_stage_curves")


def main():
    fig1_data_overview()
    metrics = load_metrics()
    results_table(metrics)
    fig2_model_comparison(metrics)
    fig3_confusion()
    fig4_learning_curves()
    fig5_two_stage_curves()


if __name__ == "__main__":
    main()
