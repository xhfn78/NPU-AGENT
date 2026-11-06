"""완료된 XGBoost 트리 수/학습 시드 실험을 정적인 그래프로 저장합니다."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "xgboost_1000_trees"


def main():
    counts = pd.read_csv(OUT / "tree_comparison.csv")
    seeds = pd.read_csv(OUT / "seed_scores.csv")
    summary = json.loads((OUT / "seed_summary.json").read_text(encoding="utf-8"))
    assert seeds.model_seed.tolist() == list(range(1, 42)) and summary["count"] == 41
    assert np.isfinite(seeds.select_dtypes(include="number")).all().all()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    first, second = axes
    first.plot(counts.trees, counts.auc, "o-", color="#2563eb", linewidth=1.8, markersize=4)
    first.set(title="Tree count comparison | model seed 42", xlabel="Number of trees", ylabel="Validation ROC-AUC")
    first.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    second.plot(seeds.model_seed, seeds.auc_trees_800, "o-", label="800 trees", color="#0f766e", linewidth=1.4, markersize=3)
    second.plot(seeds.model_seed, seeds.auc_trees_1000, "o-", label="1000 trees", color="#c2410c", linewidth=1.4, markersize=3)
    second.axhline(summary["reference42_auc"], linestyle="--", color="#64748b", linewidth=1.2, label="Seed 42, 1000 trees")
    second.axhline(summary["seed_ensemble_auc"], linestyle=":", color="#7c3aed", linewidth=1.7, label="Mean prediction of 41 seeds, 1000 trees")
    second.set(title="Learning seeds 1-41 | same data split and preprocessing", xlabel="Model learning seed", ylabel="Validation ROC-AUC")
    second.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 41])
    second.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    second.legend(loc="best", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("XGBoost | fixed 80:20 holdout (139,927 validation rows)", fontsize=14)
    fig.savefig(OUT / "comparison.png", dpi=180)
    fig.savefig(OUT / "comparison.svg")
    plt.close(fig)
    print("SAVED", OUT / "comparison.png")


if __name__ == "__main__":
    main()
