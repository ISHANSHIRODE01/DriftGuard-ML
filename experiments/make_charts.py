"""Render the evidence figures from experiment_results.csv."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("reports")


def main() -> None:
    df = pd.read_csv(OUT / "experiment_results.csv")
    summary = json.loads((OUT / "experiment_summary.json").read_text())

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        "DriftGuard on ELEC2: drift-triggered retraining vs a frozen model",
        fontsize=14,
        fontweight="bold",
    )

    # --- Panel 1: F1 over time -------------------------------------------
    ax = axes[0]
    ax.plot(df.batch_index, df.static_f1, "o--", color="#c0392b",
            label=f"Static (no retrain) — mean {summary['static_mean_f1']:.3f}",
            lw=1.8, ms=4)
    ax.plot(df.batch_index, df.driftguard_f1, "o-", color="#1e8449",
            label=f"DriftGuard — mean {summary['driftguard_mean_f1']:.3f}",
            lw=2.0, ms=4)
    for _, r in df[df.promoted].iterrows():
        ax.axvline(r.batch_index, color="#2980b9", ls=":", alpha=0.65, lw=1.4)
    ax.set_ylabel("F1 (minority class)")
    ax.set_title(
        f"F1 improves {summary['driftguard_mean_f1'] - summary['static_mean_f1']:+.3f} "
        f"({(summary['driftguard_mean_f1'] / summary['static_mean_f1'] - 1) * 100:+.1f}%). "
        "Dotted blue = model promotion",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    # --- Panel 2: accuracy over time -------------------------------------
    ax = axes[1]
    ax.plot(df.batch_index, df.static_accuracy, "o--", color="#c0392b",
            label=f"Static — mean {summary['static_mean_accuracy']:.3f}", lw=1.8, ms=4)
    ax.plot(df.batch_index, df.driftguard_accuracy, "o-", color="#1e8449",
            label=f"DriftGuard — mean {summary['driftguard_mean_accuracy']:.3f}",
            lw=2.0, ms=4)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Accuracy gain is small (+1.1%) — the win is in minority-class recall, "
        "not overall accuracy",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    # --- Panel 3: drift signal -------------------------------------------
    ax = axes[2]
    colors = ["#e67e22" if d >= 3 else "#95a5a6" for d in df.n_features_drifted]
    ax.bar(df.batch_index, df.n_features_drifted, color=colors, alpha=0.85)
    ax.axhline(2.5, color="#c0392b", ls="--", lw=1.4,
               label="dataset-drift gate (>50% of features)")
    ax.set_ylabel("Features drifted (of 5)")
    ax.set_xlabel("Batch index (each = 1,440 records ≈ 30 days)")
    ax.set_title(
        f"KS + PSI drift signal. {summary['n_retrains_triggered']} retrains triggered, "
        f"{summary['n_promotions']} passed the promotion gate "
        f"({summary['n_retrains_triggered'] - summary['n_promotions']} rejected)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUT / "drift_timeline.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}/drift_timeline.png")

    # --- Feature drift frequency ----------------------------------------
    feats = pd.DataFrame(summary["most_drifted_features"])
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(feats.feature, feats.drift_rate * 100, color="#8e44ad", alpha=0.85)
    ax2.set_xlabel("% of batches where feature drifted (KS p<0.05 AND PSI>=0.10)")
    ax2.set_title("Which features actually moved", fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3, axis="x")
    for i, v in enumerate(feats.drift_rate * 100):
        ax2.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    plt.tight_layout()
    fig2.savefig(OUT / "feature_drift.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}/feature_drift.png")


if __name__ == "__main__":
    main()
