from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency at runtime
    plt = None


def _to_serializable(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def save_json(path: str | Path, payload: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: _to_serializable(value) for key, value in payload.items()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)


def save_metrics_history_csv(path: str | Path, history: List[Dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = sorted({key for row in history for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: _to_serializable(row.get(key, "")) for key in fieldnames})


def save_text_summary(path: str | Path, title: str, metrics: Dict, extra_lines: Optional[Iterable[str]] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        if extra_lines:
            for line in extra_lines:
                handle.write(f"{line}\n")
            handle.write("\n")
        for key, value in metrics.items():
            handle.write(f"{key}: {_to_serializable(value)}\n")


def save_metric_curves(path: str | Path, history: List[Dict], metric_names: List[str], title: str):
    if plt is None or not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(8, 3 * len(metric_names)), squeeze=False)
    for axis, metric_name in zip(axes.flatten(), metric_names):
        values = [row.get(metric_name) for row in history]
        axis.plot(steps, values, marker="o")
        axis.set_title(metric_name)
        axis.set_xlabel("step")
        axis.set_ylabel(metric_name)
        axis.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_mb_score_plot(
    path: str | Path,
    feature_scores: List[float],
    oracle_labels: Optional[List[float]] = None,
    title: str = "MB Scores",
):
    if plt is None or not feature_scores:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    feature_indices = list(range(len(feature_scores)))
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(feature_indices, feature_scores, color="#4C78A8", alpha=0.8, label="predicted / source score")
    if oracle_labels is not None:
        axis.plot(feature_indices, oracle_labels, color="#E45756", marker="o", linewidth=1.5, label="oracle MB")
    axis.set_xlabel("feature index")
    axis.set_ylabel("score")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_case_summaries(root_dir: str | Path) -> List[Dict]:
    root_dir = Path(root_dir)
    summaries = []
    for path in root_dir.rglob("final_metrics.json"):
        metrics = load_json(path)
        metrics["case_dir"] = str(path.parent)
        summaries.append(metrics)
    return summaries


def save_summary_csv(path: str | Path, rows: List[Dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_serializable(row.get(key, "")) for key in fieldnames})


def save_case_comparison_plot(path: str | Path, rows: List[Dict], metric_name: str, title: str):
    if plt is None or not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [row.get("case_name", Path(row["case_dir"]).name) for row in rows]
    values = [row.get(metric_name, 0.0) for row in rows]

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, values, color="#4C78A8")
    axis.set_title(title)
    axis.set_ylabel(metric_name)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def aggregate_scale_rows(root_dir: str | Path) -> List[Dict]:
    root_dir = Path(root_dir)
    rows = []
    for final_json in root_dir.rglob("final_metrics.json"):
        parent = final_json.parent
        case_name = parent.name
        scale_name = parent.parent.name if parent.parent != root_dir else "default"
        metrics = load_json(final_json)
        metrics["case_name"] = case_name
        metrics["scale_name"] = scale_name
        metrics["case_dir"] = str(parent)
        rows.append(metrics)
    return rows


def save_grouped_scale_plot(path: str | Path, rows: List[Dict], metric_name: str, title: str):
    if plt is None or not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scale_names = sorted({row["scale_name"] for row in rows})
    case_names = sorted({row["case_name"] for row in rows})
    grouped = {scale: {case: 0.0 for case in case_names} for scale in scale_names}
    for row in rows:
        grouped[row["scale_name"]][row["case_name"]] = row.get(metric_name, 0.0)

    width = 0.12 if len(case_names) > 4 else 0.18
    x_positions = list(range(len(scale_names)))

    fig, axis = plt.subplots(figsize=(10, 4.5))
    for idx, case_name in enumerate(case_names):
        values = [grouped[scale][case_name] for scale in scale_names]
        offsets = [x + (idx - (len(case_names) - 1) / 2) * width for x in x_positions]
        axis.bar(offsets, values, width=width, label=case_name)

    axis.set_xticks(x_positions)
    axis.set_xticklabels(scale_names)
    axis.set_ylabel(metric_name)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

