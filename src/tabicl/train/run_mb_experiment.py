from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import optim
import torch.nn.functional as F

from tabicl import InferenceConfig
from tabicl.prior.synthetic_scm_mb import SyntheticSCMBatchDataset, smoke_test_synthetic_scm_mb
from tabicl.train.mb_utils import (
    build_mb_predictor,
    build_score_provider,
    compute_prediction_metrics,
    compute_mb_metrics,
    load_tabicl_model,
    split_support_query,
)
from tabicl.train.result_viz import (
    save_json,
    save_metric_curves,
    save_metrics_history_csv,
    save_mb_score_plot,
    save_text_summary,
)
from tabicl.train.train_config import build_parser


def _save_checkpoint(model, path: str, extra: dict | None = None):
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _average_history(history: list[dict]) -> dict:
    if not history:
        return {}

    excluded = {"step", "train_mode", "mb_score_source", "mb_injection"}
    metric_keys = sorted({key for row in history for key in row.keys() if key not in excluded})
    averaged = {}
    for key in metric_keys:
        values = [float(row[key]) for row in history if key in row]
        if values:
            averaged[key] = float(sum(values) / len(values))

    averaged["num_datasets"] = len(history)
    averaged["aggregation"] = "mean_over_datasets"
    averaged["train_mode"] = history[-1]["train_mode"]
    averaged["mb_score_source"] = history[-1]["mb_score_source"]
    averaged["mb_injection"] = history[-1]["mb_injection"]
    return averaged


def main():
    parser = build_parser()
    config = parser.parse_args()
    if config.task_type != "classification":
        raise ValueError("MB-TabICL main runner currently supports classification only.")

    smoke = smoke_test_synthetic_scm_mb(
        scm_num_features=config.scm_num_features,
        scm_mb_size=config.scm_mb_size,
        scm_num_samples=config.scm_num_samples,
        scm_noise_dim=config.scm_noise_dim,
        scm_redundant_dim=config.scm_redundant_dim,
        scm_nonlinear=config.scm_nonlinear,
        scm_task_type=config.scm_task_type,
        scm_num_classes=config.scm_num_classes,
        scm_noise_std=config.scm_noise_std,
        scm_seed=config.scm_seed,
        n_support=config.n_support,
        n_query=config.n_query,
    )
    print("Synthetic SCM smoke test:", smoke)

    dataset = SyntheticSCMBatchDataset(
        batch_size=config.batch_size,
        scm_num_features=config.scm_num_features,
        scm_mb_size=config.scm_mb_size,
        scm_num_samples=config.scm_num_samples,
        scm_noise_dim=config.scm_noise_dim,
        scm_redundant_dim=config.scm_redundant_dim,
        scm_nonlinear=config.scm_nonlinear,
        scm_task_type=config.scm_task_type,
        scm_num_classes=config.scm_num_classes,
        scm_noise_std=config.scm_noise_std,
        scm_seed=config.scm_seed,
        n_support=config.n_support,
        n_query=config.n_query,
    )
    model = load_tabicl_model(config, strict=False)
    inference_config = InferenceConfig(
        COL_CONFIG={"device": config.device, "use_amp": config.amp, "verbose": False},
        ROW_CONFIG={"device": config.device, "use_amp": config.amp, "verbose": False},
        ICL_CONFIG={"device": config.device, "use_amp": config.amp, "verbose": False},
    )
    predictor = None
    if config.mb_score_source == "predicted":
        predictor = build_mb_predictor(config, model)
        predictor.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if config.mb_train_calibrator or config.train_mode == "calibrator":
        if not trainable_params:
            raise ValueError("No trainable parameters found for calibrator mode.")
        optimizer = optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = None

    score_provider = build_score_provider(config)
    case_dir = Path(config.checkpoint_dir or "checkpoints")
    os.makedirs(case_dir, exist_ok=True)
    metrics_history = []
    last_metrics = {}
    aggregate_metrics = {}
    last_feature_scores = None
    last_oracle_labels = None
    num_eval_steps = config.eval_num_datasets if config.eval_num_datasets > 0 else max(1, config.max_steps)

    for step in range(num_eval_steps):
        batch = next(iter(dataset))
        split_batch = split_support_query(batch)
        X = split_batch["X"].to(config.device)
        y = split_batch["y"].to(config.device)
        y_support = split_batch["y_support"].to(config.device)
        y_query = split_batch["y_query"].to(config.device)
        X_support = split_batch["X_support"].to(config.device)
        mb_labels = split_batch["mb_labels"].to(config.device)
        d = split_batch["d"].to(config.device)

        predicted_scores = None
        predictor_metrics = {}
        if predictor is not None:
            with torch.no_grad():
                predictor_output = predictor(X_support, y_support, d=d, task_type=config.scm_task_type)
            predicted_scores = predictor_output.mb_scores
            predictor_metrics = compute_mb_metrics(predicted_scores.detach(), mb_labels.detach())

        score_result = score_provider.get_scores(
            X_support=X_support,
            y_support=y_support,
            task_type=config.scm_task_type,
            d=d,
            mb_labels=mb_labels,
            predicted_scores=predicted_scores,
        )

        if optimizer is not None:
            model.train()
            if config.freeze_tabicl_backbone:
                model.col_embedder.eval()
                model.row_interactor.eval()
                model.icl_predictor.eval()
            optimizer.zero_grad(set_to_none=True)
            logits, diagnostics = model(X, y_support, d=d, mb_scores=score_result.scores, return_mb_diagnostics=True)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_query.reshape(-1).long())
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                logits, diagnostics = model(
                    X,
                    y_support,
                    d=d,
                    mb_scores=score_result.scores,
                    return_mb_diagnostics=True,
                    return_logits=True,
                    inference_config=inference_config,
                )
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_query.reshape(-1).long())

        metrics = compute_prediction_metrics(logits.reshape(-1, logits.shape[-1]), y_query.reshape(-1), task_type="classification")
        metrics["loss"] = float(loss.item())
        sparsity_value = score_result.diagnostics["sparsity"]
        metrics["mb_sparsity"] = float(sparsity_value.item() if hasattr(sparsity_value, "item") else sparsity_value)
        if "mean_score" in score_result.diagnostics:
            mean_score_value = score_result.diagnostics["mean_score"]
            metrics["mb_mean_score"] = float(
                mean_score_value.item() if hasattr(mean_score_value, "item") else mean_score_value
            )
        if "masked_features" in diagnostics:
            metrics["masked_features"] = float(diagnostics["masked_features"].item())
        if "selected_ratio" in diagnostics:
            metrics["selected_ratio"] = float(diagnostics["selected_ratio"].item())
        for key, value in predictor_metrics.items():
            metrics[f"predictor_{key}"] = value

        row = {
            "step": step,
            "train_mode": config.train_mode,
            "mb_score_source": config.mb_score_source,
            "mb_injection": config.mb_injection,
            **metrics,
        }
        metrics_history.append(row)
        last_metrics = dict(row)
        last_feature_scores = score_result.scores.detach().mean(dim=0).cpu().tolist()
        last_oracle_labels = mb_labels.detach().float().mean(dim=0).cpu().tolist()

        if step % max(1, config.save_temp_every) == 0 or step == num_eval_steps - 1:
            print(f"step={step} mode={config.train_mode} source={config.mb_score_source} injection={config.mb_injection}")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")
            if optimizer is not None:
                ckpt_path = os.path.join(config.checkpoint_dir or "checkpoints", f"mb_calibrator_step_{step}.ckpt")
                _save_checkpoint(model, ckpt_path, extra={"step": step, "metrics": metrics})

    aggregate_metrics = _average_history(metrics_history)
    if aggregate_metrics:
        print("aggregate metrics across datasets:")
        for key, value in aggregate_metrics.items():
            if isinstance(value, (int, float)) and key not in {"num_datasets"}:
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    save_json(
        case_dir / "run_config.json",
        {
            "train_mode": config.train_mode,
            "task_type": config.task_type,
            "mb_score_source": config.mb_score_source,
            "mb_injection": config.mb_injection,
            "scm_num_features": config.scm_num_features,
            "scm_mb_size": config.scm_mb_size,
            "scm_num_samples": config.scm_num_samples,
            "n_support": config.n_support,
            "n_query": config.n_query,
            "seed": config.scm_seed,
            "device": config.device,
            "eval_num_datasets": config.eval_num_datasets,
        },
    )
    save_json(case_dir / "smoke_test.json", smoke)
    save_json(case_dir / "final_metrics.json", last_metrics)
    if aggregate_metrics:
        save_json(case_dir / "aggregate_metrics.json", aggregate_metrics)
    save_metrics_history_csv(case_dir / "metrics_history.csv", metrics_history)
    save_text_summary(
        case_dir / "summary.txt",
        title=f"MB-TabICL Step1 Result: {config.mb_score_source} / {config.mb_injection}",
        metrics=aggregate_metrics if aggregate_metrics else last_metrics,
        extra_lines=[
            f"checkpoint_dir={case_dir}",
            f"seed={config.scm_seed}",
            f"support={config.n_support}",
            f"query={config.n_query}",
            f"features={config.scm_num_features}",
            f"mb_size={config.scm_mb_size}",
            f"eval_num_datasets={config.eval_num_datasets or 1}",
        ],
    )
    save_metric_curves(
        case_dir / "metric_curves.png",
        metrics_history,
        metric_names=["accuracy", "balanced_accuracy", "roc_auc", "log_loss", "mb_sparsity"],
        title=f"{config.mb_score_source} / {config.mb_injection}",
    )
    save_mb_score_plot(
        case_dir / "mb_scores.png",
        feature_scores=last_feature_scores or [],
        oracle_labels=last_oracle_labels,
        title=f"Mean MB Scores: {config.mb_score_source}",
    )
    print(f"Saved evaluation artifacts to {case_dir}")


if __name__ == "__main__":
    main()
