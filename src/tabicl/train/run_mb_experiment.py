from __future__ import annotations

import os

import torch
from torch import optim
import torch.nn.functional as F

from tabicl.prior.synthetic_scm_mb import SyntheticSCMBatchDataset, smoke_test_synthetic_scm_mb
from tabicl.train.mb_utils import (
    build_mb_predictor,
    build_score_provider,
    compute_prediction_metrics,
    compute_mb_metrics,
    load_tabicl_model,
    split_support_query,
)
from tabicl.train.train_config import build_parser


def _save_checkpoint(model, path: str, extra: dict | None = None):
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


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
    os.makedirs(config.checkpoint_dir or "checkpoints", exist_ok=True)

    for step in range(max(1, config.max_steps)):
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
                    X, y_support, d=d, mb_scores=score_result.scores, return_mb_diagnostics=True, return_logits=True
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

        if step % max(1, config.save_temp_every) == 0 or step == config.max_steps - 1:
            print(f"step={step} mode={config.train_mode} source={config.mb_score_source} injection={config.mb_injection}")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")
            if optimizer is not None:
                ckpt_path = os.path.join(config.checkpoint_dir or "checkpoints", f"mb_calibrator_step_{step}.ckpt")
                _save_checkpoint(model, ckpt_path, extra={"step": step, "metrics": metrics})


if __name__ == "__main__":
    main()
