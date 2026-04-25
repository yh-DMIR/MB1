from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

import torch
import torch.nn.functional as F

from tabicl import TabICL
from tabicl.model.mb_injection import MBScoreProvider
from tabicl.model.mb_predictor import MultiViewMBPredictor


def build_model_config(config) -> Dict[str, object]:
    return {
        "max_classes": config.max_classes,
        "embed_dim": config.embed_dim,
        "col_num_blocks": config.col_num_blocks,
        "col_nhead": config.col_nhead,
        "col_num_inds": config.col_num_inds,
        "row_num_blocks": config.row_num_blocks,
        "row_nhead": config.row_nhead,
        "row_num_cls": config.row_num_cls,
        "row_rope_base": config.row_rope_base,
        "icl_num_blocks": config.icl_num_blocks,
        "icl_nhead": config.icl_nhead,
        "ff_factor": config.ff_factor,
        "dropout": config.dropout,
        "activation": config.activation,
        "norm_first": config.norm_first,
        "mb_injection": config.mb_injection,
        "mb_bias_init": config.mb_bias_init,
        "mb_bias_trainable": config.mb_bias_trainable,
        "mb_bias_per_layer": config.mb_bias_per_layer,
        "mb_bias_per_head": config.mb_bias_per_head,
        "mb_bias_target": config.mb_bias_target,
        "mb_hard_mask_threshold": config.mb_hard_mask_threshold,
        "mb_topk_extra": config.mb_topk_extra,
        "mb_hard_select_mode": config.mb_hard_select_mode,
        "mb_hard_select_topk": config.mb_hard_select_topk,
        "mb_embedding_dim": config.mb_embedding_dim,
        "mb_embedding_add_position": config.mb_embedding_add_position,
    }


def load_tabicl_model(config, strict: bool = True) -> TabICL:
    model = TabICL(**build_model_config(config))
    model.to(config.device)

    ckpt_path = config.tabicl_checkpoint_path or config.checkpoint_path
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=True)
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=strict)
        if not strict:
            if missing:
                print(f"Missing keys while loading TabICL checkpoint: {missing}")
            if unexpected:
                print(f"Unexpected keys while loading TabICL checkpoint: {unexpected}")

    if config.freeze_tabicl_backbone:
        model.col_embedder.eval()
        model.row_interactor.eval()
        model.icl_predictor.eval()
        for parameter in model.col_embedder.parameters():
            parameter.requires_grad = False
        for parameter in model.row_interactor.parameters():
            parameter.requires_grad = False
        for parameter in model.icl_predictor.parameters():
            parameter.requires_grad = False
    return model


def build_mb_predictor(config, model: TabICL) -> MultiViewMBPredictor:
    predictor = MultiViewMBPredictor(
        col_embedder=model.col_embedder,
        embed_dim=model.embed_dim,
        reserve_cls_tokens=model.row_num_cls,
        max_classes=model.max_classes,
        mb_num_subsets=config.mb_num_subsets,
        mb_subset_size=config.mb_subset_size,
        mb_stratified_sampling=config.mb_stratified_sampling,
        mb_regression_num_bins=config.mb_regression_num_bins,
        mb_sampling_with_replacement=config.mb_sampling_with_replacement,
        mb_min_subset_size=config.mb_min_subset_size,
        mb_pooling=config.mb_pooling,
        mb_token_dim=config.mb_token_dim,
        mb_pool_mlp_hidden_dim=config.mb_pool_mlp_hidden_dim,
        mb_pool_dropout=config.mb_pool_dropout,
        mb_use_target_token=config.mb_use_target_token,
        mb_target_token_dim=config.mb_target_token_dim,
        mb_target_stats=config.mb_target_stats,
        mb_use_feature_target_stats=config.mb_use_feature_target_stats,
        mb_stats_dim=config.mb_stats_dim,
        mb_stats_mlp_hidden_dim=config.mb_stats_mlp_hidden_dim,
        mb_stats_nan_to_num=config.mb_stats_nan_to_num,
        mb_column_transformer_layers=config.mb_column_transformer_layers,
        mb_column_transformer_heads=config.mb_column_transformer_heads,
        mb_column_transformer_ffn_dim=config.mb_column_transformer_ffn_dim,
        mb_column_transformer_dropout=config.mb_column_transformer_dropout,
        mb_column_transformer_norm_first=config.mb_column_transformer_norm_first,
        mb_subset_aggregation=config.mb_subset_aggregation,
        mb_head_hidden_dim=config.mb_head_hidden_dim,
        mb_head_dropout=config.mb_head_dropout,
        mb_score_temperature=config.mb_score_temperature,
    )
    predictor.to(config.device)

    if config.mb_predictor_checkpoint_path and os.path.exists(config.mb_predictor_checkpoint_path):
        checkpoint = torch.load(config.mb_predictor_checkpoint_path, map_location=config.device, weights_only=True)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        predictor.load_state_dict(state_dict, strict=True)
    return predictor


def focal_bce_loss(scores: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, pos_weight: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy(scores, targets, reduction="none")
    pt = torch.where(targets > 0.5, scores, 1 - scores).clamp_min(1e-6)
    focal_weight = (1 - pt).pow(gamma)
    class_weight = torch.where(targets > 0.5, torch.full_like(targets, pos_weight), torch.ones_like(targets))
    return (focal_weight * class_weight * bce).mean()


def mb_supervision_loss(scores: torch.Tensor, labels: torch.Tensor, config) -> torch.Tensor:
    labels = labels.float()
    if config.mb_use_focal_loss:
        return focal_bce_loss(scores, labels, gamma=config.mb_focal_gamma, pos_weight=config.mb_pos_weight)
    weight = torch.where(labels > 0.5, torch.full_like(labels, config.mb_pos_weight), torch.ones_like(labels))
    return F.binary_cross_entropy(scores, labels, weight=weight)


def compute_prediction_metrics(logits: torch.Tensor, targets: torch.Tensor, task_type: str = "classification") -> Dict[str, float]:
    outputs: Dict[str, float] = {}
    if task_type == "classification":
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = probabilities.argmax(axis=-1)
        true = targets.detach().cpu().numpy()
        outputs["accuracy"] = float(accuracy_score(true, preds))
        outputs["balanced_accuracy"] = float(balanced_accuracy_score(true, preds))
        outputs["log_loss"] = float(log_loss(true, probabilities, labels=np.arange(probabilities.shape[-1])))
        try:
            if probabilities.shape[-1] == 2:
                outputs["roc_auc"] = float(roc_auc_score(true, probabilities[:, 1]))
            else:
                outputs["roc_auc"] = float(roc_auc_score(true, probabilities, multi_class="ovr"))
        except Exception:
            outputs["roc_auc"] = float("nan")
        return outputs

    preds = logits.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    outputs["rmse"] = float(np.sqrt(mean_squared_error(true, preds)))
    outputs["mae"] = float(mean_absolute_error(true, preds))
    outputs["r2"] = float(r2_score(true, preds))
    return outputs


def compute_mb_metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    scores_np = scores.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    preds_np = (scores_np >= threshold).astype(np.int64)
    outputs: Dict[str, float] = {}
    try:
        outputs["auroc"] = float(roc_auc_score(labels_np, scores_np))
    except Exception:
        outputs["auroc"] = float("nan")
    try:
        outputs["auprc"] = float(average_precision_score(labels_np, scores_np))
    except Exception:
        outputs["auprc"] = float("nan")
    outputs["f1"] = float(f1_score(labels_np, preds_np, zero_division=0))

    batch_scores = scores.detach().cpu()
    batch_labels = labels.detach().cpu()
    precision_k = []
    recall_k = []
    topk_recall = []
    for sample_scores, sample_labels in zip(batch_scores, batch_labels):
        true_k = int(sample_labels.sum().item())
        k = max(1, true_k)
        indices = torch.topk(sample_scores, k=k).indices
        predicted_mask = torch.zeros_like(sample_labels, dtype=torch.bool)
        predicted_mask[indices] = True
        tp = (predicted_mask & sample_labels.bool()).sum().item()
        precision_k.append(tp / max(1, k))
        recall_k.append(tp / max(1, true_k))
        topk_recall.append(tp / max(1, true_k))
    outputs["precision_at_k"] = float(np.mean(precision_k))
    outputs["recall_at_k"] = float(np.mean(recall_k))
    outputs["topk_recall"] = float(np.mean(topk_recall))
    outputs["mb_mean_score"] = float(scores[labels > 0.5].mean().item()) if (labels > 0.5).any() else 0.0
    outputs["non_mb_mean_score"] = float(scores[labels <= 0.5].mean().item()) if (labels <= 0.5).any() else 0.0
    outputs["score_sparsity"] = float((scores > threshold).float().mean().item())
    return outputs


def build_score_provider(config) -> MBScoreProvider:
    return MBScoreProvider(
        mb_score_source=config.mb_score_source,
        mb_estimator=config.mb_estimator,
        mb_shuffle_within_batch=config.mb_shuffle_within_batch,
        mb_random_seed=config.mb_random_seed,
    )


def split_support_query(batch: Dict[str, torch.Tensor | str]) -> Dict[str, torch.Tensor]:
    X = batch["X"]
    y = batch["y"]
    support_size = int(batch["support_size"][0].item())
    return {
        "X": X,
        "y": y,
        "X_support": X[:, :support_size],
        "y_support": y[:, :support_size],
        "X_query": X[:, support_size:],
        "y_query": y[:, support_size:],
        "support_size": support_size,
        "d": batch["d"],
        "mb_labels": batch["mb_labels"],
    }
