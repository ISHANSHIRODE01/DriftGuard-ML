"""
The headline experiment: does drift-triggered retraining actually help?

Two arms over the identical chronological batch stream:

    A. STATIC    -- train once on batches 0-2, never retrain. The control.
    B. DRIFTGUARD -- monitor every batch, retrain when the drift or
                     performance gate fires, promote only on a win.

Reporting arm B alone would prove nothing: accuracy could rise simply because
later batches are easier. The static baseline is what makes the delta a
measurement rather than an assertion.

    python experiments/run_experiment.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.loader import load_elec2, make_batches
from src.db.repository import DriftRepository
from src.models.trainer import evaluate, train_model
from src.pipeline.monitor import DriftMonitor, MonitorConfig

BOOTSTRAP_BATCHES = 3


def run_static_baseline(batches, artifact_dir="artifacts/baseline") -> list[dict]:
    """Arm A: one model, frozen, scored on every subsequent batch."""
    boot = batches[:BOOTSTRAP_BATCHES]
    X = pd.concat([b.X for b in boot], ignore_index=True)
    y = pd.concat([b.y for b in boot], ignore_index=True)

    res = train_model(
        X, y, version=0, artifact_dir=artifact_dir, log_to_mlflow=False
    )
    print(
        f"[static] baseline trained on {res.n_train_rows} rows | "
        f"val_acc={res.val_accuracy:.4f}"
    )

    rows = []
    for b in batches[BOOTSTRAP_BATCHES:]:
        m = evaluate(res.model, b.X, b.y)
        rows.append(
            {"batch_index": b.index, "accuracy": m["accuracy"], "f1": m["f1"]}
        )
    return rows


def run_driftguard(batches, db_url: str) -> list[dict]:
    """Arm B: full monitoring loop with gated retraining."""
    repo = DriftRepository(db_url)
    repo.drop_schema()
    repo.create_schema()

    monitor = DriftMonitor(repo, MonitorConfig())
    monitor.fit_initial(batches[:BOOTSTRAP_BATCHES])
    print(
        f"[driftguard] champion v{monitor.champion_version} | "
        f"val_acc={monitor.champion_val_accuracy:.4f}"
    )

    rows = []
    for b in batches[BOOTSTRAP_BATCHES:]:
        out = monitor.process(b)
        rows.append(
            {
                "batch_index": out.batch_index,
                "accuracy": out.batch_accuracy,
                "f1": out.batch_f1,
                "n_features_drifted": out.n_features_drifted,
                "drift_share": out.drift_share,
                "retrain_triggered": out.retrain_triggered,
                "promoted": out.promoted,
                "champion_version": out.champion_version,
                "reason": out.trigger_reason,
            }
        )
        flag = ""
        if out.retrain_triggered:
            flag = " [RETRAIN->PROMOTED]" if out.promoted else " [RETRAIN->rejected]"
        print(
            f"  batch {out.batch_index:2d} | acc={out.batch_accuracy:.4f} "
            f"| drift {out.n_features_drifted}/5 | v{out.champion_version}{flag}"
        )

    return rows, repo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/elec2.csv")
    ap.add_argument("--batch-size", type=int, default=1440)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--db", default="sqlite:///driftguard.db")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    df = load_elec2(args.data)
    batches = make_batches(df, args.batch_size, args.max_batches)
    print(f"ELEC2: {len(df):,} rows -> {len(batches)} batches of {args.batch_size}\n")

    print("=" * 62)
    print("ARM A: STATIC BASELINE (no retraining)")
    print("=" * 62)
    static_rows = run_static_baseline(batches)

    print()
    print("=" * 62)
    print("ARM B: DRIFTGUARD (drift-triggered gated retraining)")
    print("=" * 62)
    dg_rows, repo = run_driftguard(batches, args.db)

    static = pd.DataFrame(static_rows).rename(
        columns={"accuracy": "static_accuracy", "f1": "static_f1"}
    )
    dg = pd.DataFrame(dg_rows).rename(
        columns={"accuracy": "driftguard_accuracy", "f1": "driftguard_f1"}
    )
    merged = static.merge(dg, on="batch_index")
    merged["delta_accuracy"] = (
        merged["driftguard_accuracy"] - merged["static_accuracy"]
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "experiment_results.csv", index=False)

    summary = {
        "dataset": "ELEC2 (Harries 1999)",
        "n_rows": int(len(df)),
        "n_batches": len(batches),
        "batch_size": args.batch_size,
        "bootstrap_batches": BOOTSTRAP_BATCHES,
        "n_evaluated_batches": int(len(merged)),
        "static_mean_accuracy": round(float(merged.static_accuracy.mean()), 4),
        "driftguard_mean_accuracy": round(
            float(merged.driftguard_accuracy.mean()), 4
        ),
        "absolute_improvement": round(float(merged.delta_accuracy.mean()), 4),
        "relative_improvement_pct": round(
            float(
                merged.delta_accuracy.mean() / merged.static_accuracy.mean() * 100
            ),
            2,
        ),
        "static_mean_f1": round(float(merged.static_f1.mean()), 4),
        "driftguard_mean_f1": round(float(merged.driftguard_f1.mean()), 4),
        "batches_where_driftguard_better": int((merged.delta_accuracy > 0).sum()),
        "batches_where_static_better": int((merged.delta_accuracy < 0).sum()),
        "n_retrains_triggered": int(merged.retrain_triggered.sum()),
        "n_promotions": int(merged.promoted.sum()),
        "final_model_version": int(merged.champion_version.iloc[-1]),
        "most_drifted_features": repo.most_drifted_features(5),
    }

    with open(out_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    for k, v in summary.items():
        if k != "most_drifted_features":
            print(f"  {k:34s} {v}")
    print("\n  most drifted features:")
    for r in summary["most_drifted_features"]:
        print(
            f"    {r['feature']:12s} drifted {r['times_drifted']}/{r['n']} "
            f"({r['drift_rate']:.0%})"
        )
    print(f"\n  written -> {out_dir}/experiment_results.csv, experiment_summary.json")


if __name__ == "__main__":
    main()
