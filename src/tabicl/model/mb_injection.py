from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import torch
from torch import nn, Tensor


def _safe_corrcoef(x: Tensor, y: Tensor) -> Tensor:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False).clamp_min(1e-6) * y.std(unbiased=False).clamp_min(1e-6)
    return (x * y).mean() / denom


def _normalize_scores(scores: Tensor, feature_mask: Optional[Tensor] = None) -> Tensor:
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if feature_mask is not None:
        scores = scores * feature_mask.float()
    mins = scores.min(dim=-1, keepdim=True).values
    maxs = scores.max(dim=-1, keepdim=True).values
    scores = (scores - mins) / (maxs - mins).clamp_min(1e-6)
    if feature_mask is not None:
        scores = scores * feature_mask.float()
    return scores


def build_feature_mask(num_features: int, d: Optional[Tensor], device: torch.device) -> Tensor:
    if d is None:
        return torch.ones(1, num_features, dtype=torch.bool, device=device)
    indices = torch.arange(num_features, device=device).view(1, num_features)
    return indices < d.view(-1, 1)


def compute_corr_scores(
    X_support: Tensor,
    y_support: Tensor,
    task_type: str = "classification",
    d: Optional[Tensor] = None,
) -> Tensor:
    """Compute simple support-only feature-target correlation scores."""

    batch_size, _, num_features = X_support.shape
    feature_mask = build_feature_mask(num_features, d, X_support.device)
    scores = torch.zeros(batch_size, num_features, device=X_support.device)

    for batch_idx in range(batch_size):
        current_mask = feature_mask[min(batch_idx, feature_mask.shape[0] - 1)]
        for feat_idx in torch.where(current_mask)[0]:
            feature_values = X_support[batch_idx, :, feat_idx]
            if task_type == "classification":
                labels = y_support[batch_idx].float()
                score = _safe_corrcoef(feature_values, labels).abs()
            else:
                score = _safe_corrcoef(feature_values, y_support[batch_idx]).abs()
            scores[batch_idx, feat_idx] = score

    return _normalize_scores(scores, feature_mask if feature_mask.shape[0] == batch_size else feature_mask.expand(batch_size, -1))


def compute_bootstrap_corr_scores(
    X_support: Tensor,
    y_support: Tensor,
    task_type: str = "classification",
    d: Optional[Tensor] = None,
    num_bootstrap: int = 4,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Average correlation scores over bootstrap resamples."""

    if generator is None:
        generator = torch.Generator(device=X_support.device if X_support.is_cuda else "cpu")
        generator.manual_seed(1234)

    batch_size, num_rows, _ = X_support.shape
    aggregated = []
    for _ in range(num_bootstrap):
        subset_indices = torch.randint(0, num_rows, (batch_size, num_rows), generator=generator, device=X_support.device)
        batch_indices = torch.arange(batch_size, device=X_support.device).unsqueeze(-1)
        X_sub = X_support[batch_indices, subset_indices]
        y_sub = y_support[batch_indices, subset_indices]
        aggregated.append(compute_corr_scores(X_sub, y_sub, task_type=task_type, d=d))
    return torch.stack(aggregated, dim=0).mean(dim=0)


def compute_mi_scores(
    X_support: Tensor,
    y_support: Tensor,
    task_type: str = "classification",
    d: Optional[Tensor] = None,
) -> Tensor:
    """Compute mutual-information scores on CPU with sklearn."""

    batch_size, _, num_features = X_support.shape
    feature_mask = build_feature_mask(num_features, d, X_support.device)
    outputs = torch.zeros(batch_size, num_features, device=X_support.device)
    for batch_idx in range(batch_size):
        valid_dim = int(feature_mask[min(batch_idx, feature_mask.shape[0] - 1)].sum().item())
        X_cpu = X_support[batch_idx, :, :valid_dim].detach().cpu().numpy()
        y_cpu = y_support[batch_idx].detach().cpu().numpy()
        if task_type == "classification":
            scores = mutual_info_classif(X_cpu, y_cpu, discrete_features=False)
        else:
            scores = mutual_info_regression(X_cpu, y_cpu, discrete_features=False)
        outputs[batch_idx, :valid_dim] = torch.from_numpy(scores).to(outputs.device, dtype=torch.float32)
    return _normalize_scores(outputs, feature_mask if feature_mask.shape[0] == batch_size else feature_mask.expand(batch_size, -1))


@dataclass
class MBScoreResult:
    scores: Tensor
    diagnostics: Dict[str, Tensor | str | float]


class MBScoreProvider:
    """Provide MB scores from support sets or oracle labels."""

    def __init__(
        self,
        mb_score_source: str = "none",
        mb_estimator: str = "bootstrap_corr",
        mb_shuffle_within_batch: bool = True,
        mb_random_seed: int = 42,
    ) -> None:
        self.mb_score_source = mb_score_source
        self.mb_estimator = mb_estimator
        self.mb_shuffle_within_batch = mb_shuffle_within_batch
        self.mb_random_seed = mb_random_seed

    def _random_scores(self, shape: torch.Size, device: torch.device) -> Tensor:
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
        generator.manual_seed(self.mb_random_seed)
        return torch.rand(shape, generator=generator, device=device)

    def _shuffle_scores(self, scores: Tensor) -> Tensor:
        if not self.mb_shuffle_within_batch:
            generator = torch.Generator(device=scores.device if scores.is_cuda else "cpu")
            generator.manual_seed(self.mb_random_seed)
            perm = torch.randperm(scores.shape[1], generator=generator, device=scores.device)
            return scores[:, perm]

        shuffled = []
        for batch_idx in range(scores.shape[0]):
            generator = torch.Generator(device=scores.device if scores.is_cuda else "cpu")
            generator.manual_seed(self.mb_random_seed + batch_idx)
            perm = torch.randperm(scores.shape[1], generator=generator, device=scores.device)
            shuffled.append(scores[batch_idx, perm])
        return torch.stack(shuffled, dim=0)

    def get_scores(
        self,
        X_support: Tensor,
        y_support: Tensor,
        task_type: str = "classification",
        d: Optional[Tensor] = None,
        mb_labels: Optional[Tensor] = None,
        predicted_scores: Optional[Tensor] = None,
    ) -> MBScoreResult:
        num_features = X_support.shape[-1]
        feature_mask = build_feature_mask(num_features, d, X_support.device)
        if feature_mask.shape[0] == 1:
            feature_mask = feature_mask.expand(X_support.shape[0], -1)

        source = self.mb_score_source
        diagnostics: Dict[str, Tensor | str | float] = {"source": source}

        if source == "none":
            scores = torch.zeros(X_support.shape[0], num_features, device=X_support.device)
        elif source == "oracle":
            if mb_labels is None:
                raise ValueError("oracle MB source requires mb_labels")
            scores = mb_labels.float()
        elif source == "predicted":
            if predicted_scores is None:
                raise ValueError("predicted MB source requires predicted_scores")
            scores = predicted_scores.float()
        elif source == "random":
            scores = self._random_scores((X_support.shape[0], num_features), X_support.device)
        elif source == "corr":
            scores = compute_corr_scores(X_support, y_support, task_type=task_type, d=d)
        elif source == "mi":
            scores = compute_mi_scores(X_support, y_support, task_type=task_type, d=d)
        elif source == "estimated":
            if self.mb_estimator == "bootstrap_corr":
                scores = compute_bootstrap_corr_scores(X_support, y_support, task_type=task_type, d=d)
            else:
                scores = compute_corr_scores(X_support, y_support, task_type=task_type, d=d)
                diagnostics["estimator_fallback"] = self.mb_estimator
        elif source == "shuffled":
            if mb_labels is not None:
                base_scores = mb_labels.float()
                diagnostics["shuffle_base"] = "oracle"
            elif predicted_scores is not None:
                base_scores = predicted_scores.float()
                diagnostics["shuffle_base"] = "predicted"
            else:
                base_scores = compute_corr_scores(X_support, y_support, task_type=task_type, d=d)
                diagnostics["shuffle_base"] = "corr"
            scores = self._shuffle_scores(base_scores)
        else:
            raise ValueError(f"Unsupported mb_score_source: {source}")

        scores = _normalize_scores(scores, feature_mask)
        diagnostics["sparsity"] = (scores > 0.5).float().mean()
        diagnostics["mean_score"] = scores.mean()
        return MBScoreResult(scores=scores, diagnostics=diagnostics)


class MBAttentionBias(nn.Module):
    """Build additive attention bias tensors from MB scores."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_cls: int,
        bias_init: float = 0.0,
        trainable: bool = True,
        per_layer: bool = True,
        per_head: bool = True,
        target: str = "cls_to_feature",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_cls = num_cls
        self.target = target

        if per_layer and per_head:
            shape = (num_layers, num_heads)
        elif per_layer:
            shape = (num_layers, 1)
        elif per_head:
            shape = (1, num_heads)
        else:
            shape = (1, 1)

        init_tensor = torch.full(shape, float(bias_init))
        if trainable:
            self.scale = nn.Parameter(init_tensor)
        else:
            self.register_buffer("scale", init_tensor)

    def forward(self, mb_scores: Tensor, num_tokens: int) -> list[Tensor]:
        batch_size, num_features = mb_scores.shape
        if num_tokens < self.num_cls + num_features:
            raise ValueError("num_tokens must cover CLS and feature tokens")

        outputs = []
        for layer_idx in range(self.num_layers):
            layer_scale = self.scale[min(layer_idx, self.scale.shape[0] - 1)]
            layer_scale = layer_scale.view(1, -1, 1, 1).to(device=mb_scores.device, dtype=mb_scores.dtype)
            attn_bias = torch.zeros(
                batch_size, self.num_heads, num_tokens, num_tokens, device=mb_scores.device, dtype=mb_scores.dtype
            )
            feature_slice = slice(self.num_cls, self.num_cls + num_features)
            if self.target == "cls_to_feature":
                bias_values = layer_scale * mb_scores.view(batch_size, 1, 1, num_features)
                attn_bias[:, :, : self.num_cls, feature_slice] = bias_values
            elif self.target == "all_to_feature":
                bias_values = layer_scale * mb_scores.view(batch_size, 1, 1, num_features)
                attn_bias[:, :, :, feature_slice] = bias_values
            elif self.target == "feature_to_feature":
                bias_values = layer_scale * mb_scores.view(batch_size, 1, 1, num_features)
                attn_bias[:, :, feature_slice, feature_slice] = bias_values.expand(-1, -1, num_features, -1)
            else:
                raise ValueError(f"Unsupported mb_bias_target: {self.target}")
            outputs.append(attn_bias)
        return outputs


class MBEmbeddingAdapter(nn.Module):
    """Add score-conditioned embeddings to feature tokens."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, feature_embeddings: Tensor, mb_scores: Tensor) -> Tensor:
        score_embeddings = self.net(mb_scores.unsqueeze(-1)).unsqueeze(1)
        return feature_embeddings + score_embeddings


class MBHardSelector:
    """Select features with threshold or top-k rules."""

    def __init__(
        self,
        threshold: float = 0.5,
        mode: str = "threshold",
        topk: int = 0,
        topk_extra: int = 0,
    ) -> None:
        self.threshold = threshold
        self.mode = mode
        self.topk = topk
        self.topk_extra = topk_extra

    def get_mask(self, mb_scores: Tensor) -> Tensor:
        if self.mode == "topk":
            k = max(1, min(self.topk + self.topk_extra, mb_scores.shape[1]))
            topk_indices = torch.topk(mb_scores, k=k, dim=-1).indices
            mask = torch.zeros_like(mb_scores, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            return mask

        mask = mb_scores >= self.threshold
        if self.topk_extra > 0:
            topk_extra_idx = torch.topk(mb_scores, k=min(self.topk_extra, mb_scores.shape[1]), dim=-1).indices
            mask.scatter_(1, topk_extra_idx, True)
        empty_rows = mask.sum(dim=-1) == 0
        if empty_rows.any():
            best_idx = mb_scores.argmax(dim=-1, keepdim=True)
            mask[empty_rows] = False
            mask.scatter_(1, best_idx, True)
        return mask


def build_cls_hard_mask_bias(
    mb_scores: Tensor,
    num_layers: int,
    num_heads: int,
    num_cls: int,
    threshold: float = 0.5,
    topk_extra: int = 0,
    neg_large: float = -1e4,
) -> list[Tensor]:
    selector = MBHardSelector(threshold=threshold, mode="threshold", topk_extra=topk_extra)
    keep_mask = selector.get_mask(mb_scores)
    num_tokens = num_cls + mb_scores.shape[1]
    outputs = []
    for _ in range(num_layers):
        bias = torch.zeros(
            mb_scores.shape[0], num_heads, num_tokens, num_tokens, device=mb_scores.device, dtype=mb_scores.dtype
        )
        feature_slice = slice(num_cls, num_cls + mb_scores.shape[1])
        blocked = (~keep_mask).float().view(mb_scores.shape[0], 1, 1, mb_scores.shape[1]) * neg_large
        bias[:, :, :num_cls, feature_slice] = blocked
        outputs.append(bias)
    return outputs


def build_summary_token(feature_embeddings: Tensor, mb_scores: Tensor) -> Tensor:
    weights = mb_scores / mb_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return (feature_embeddings * weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2, keepdim=True)
