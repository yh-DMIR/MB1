from __future__ import annotations

from pathlib import Path

from tabicl.train.result_viz import (
    aggregate_case_summaries,
    aggregate_scale_rows,
    save_case_comparison_plot,
    save_grouped_scale_plot,
    save_summary_csv,
)


def aggregate_same_seed(root_dir: Path):
    rows = aggregate_case_summaries(root_dir)
    if not rows:
        print(f"No final_metrics.json found under {root_dir}")
        return
    for row in rows:
        row["case_name"] = Path(row["case_dir"]).name
    save_summary_csv(root_dir / "summary.csv", rows)
    save_case_comparison_plot(root_dir / "accuracy_comparison.png", rows, "accuracy", "Accuracy by Case")
    save_case_comparison_plot(root_dir / "roc_auc_comparison.png", rows, "roc_auc", "ROC-AUC by Case")
    save_case_comparison_plot(root_dir / "log_loss_comparison.png", rows, "log_loss", "Log Loss by Case")
    print(f"Saved summary.csv and comparison plots under {root_dir}")


def aggregate_scale_sweep(root_dir: Path):
    rows = aggregate_scale_rows(root_dir)
    if not rows:
        print(f"No final_metrics.json found under {root_dir}")
        return
    save_summary_csv(root_dir / "summary.csv", rows)
    save_grouped_scale_plot(root_dir / "accuracy_by_scale.png", rows, "accuracy", "Accuracy Across Scales")
    save_grouped_scale_plot(root_dir / "roc_auc_by_scale.png", rows, "roc_auc", "ROC-AUC Across Scales")
    save_grouped_scale_plot(root_dir / "log_loss_by_scale.png", rows, "log_loss", "Log Loss Across Scales")
    print(f"Saved scale summary.csv and grouped plots under {root_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Result root directory")
    parser.add_argument("--mode", type=str, required=True, choices=["same_seed", "scale_sweep"])
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if args.mode == "same_seed":
        aggregate_same_seed(root_dir)
    else:
        aggregate_scale_sweep(root_dir)


if __name__ == "__main__":
    main()
