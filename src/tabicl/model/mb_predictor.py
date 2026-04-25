from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

import torch
from torch import nn, Tensor

from .encoders import Encoder


def _resolve_subset_size(subset_size: float | int, support_size: int, min_subset_size: int) -> int:
    if isinstance(subset_size, float) and 0 < subset_size <= 1:
        resolved = int(round(support_size * subset_size))
    else:
        resolved = int(subset_size)
    return max(min_subset_size, min(support_size, resolved))


def _safe_std(input_tensor: Tensor, dim: int) -> Tensor:
    if input_tensor.shape[dim] <= 1:
        return torch.zeros_like(input_tensor.mean(dim=dim))
    return input_tensor.std(dim=dim, unbiased=False)


@dataclass
class MBPredictorOutput:
    mb_scores: Tensor
    local_mb_scores: Tensor
    diagnostics: Dict[str, Tensor | list | str]


class SubsetSampler:
    """Sample multiple stratified support subsets."""

    def __init__(
        self,
        num_subsets: int = 8,
        subset_size: float | int = 0.5,
        stratified_sampling: bool = True,
        regression_num_bins: int = 5,
        sampling_with_replacement: bool = False,
        min_subset_size: int = 16,
    ) -> None:
        self.num_subsets = num_subsets
        self.subset_size = subset_size
        self.stratified_sampling = stratified_sampling
        self.regression_num_bins = regression_num_bins
        self.sampling_with_replacement = sampling_with_replacement
        self.min_subset_size = min_subset_size

    def _stratified_indices(self, labels: Tensor, sample_size: int, task_type: str) -> Tensor:
        device = labels.device
        if not self.stratified_sampling:
            if self.sampling_with_replacement:
                return torch.randint(0, len(labels), (sample_size,), device=device)
            return torch.randperm(len(labels), device=device)[:sample_size]

        if task_type == "classification":
            unique_vals, counts = labels.unique(return_counts=True)
            if len(unique_vals) <= 1:
                return torch.randperm(len(labels), device=device)[:sample_size]
            sampled = []
            for value, count in zip(unique_vals, counts):
                group = torch.where(labels == value)[0]
                expected = max(1, int(round(sample_size * count.item() / len(labels))))
                if expected >= len(group) and not self.sampling_with_replacement:
                    sampled.append(group)
                else:
                    if self.sampling_with_replacement:
                        picked = group[torch.randint(0, len(group), (expected,), device=device)]
                    else:
                        picked = group[torch.randperm(len(group), device=device)[:expected]]
                    sampled.append(picked)
            sampled = torch.cat(sampled, dim=0)
            if len(sampled) < sample_size:
                extra = torch.randperm(len(labels), device=device)[: sample_size - len(sampled)]
                sampled = torch.cat([sampled, extra], dim=0)
            return sampled[:sample_size]

        # regression
        if len(labels) < self.regression_num_bins * 2:
            return torch.randperm(len(labels), device=device)[:sample_size]
        quantiles = torch.linspace(0.0, 1.0, self.regression_num_bins + 1, device=device)[1:-1]
        bins = torch.bucketize(labels, torch.quantile(labels, quantiles))
        return self._stratified_indices(bins, sample_size, task_type="classification")

    def sample(self, y_support: Tensor, task_type: str = "classification") -> Tensor:
        batch_size, support_size = y_support.shape
        sample_size = _resolve_subset_size(self.subset_size, support_size, self.min_subset_size)
        samples = []
        for batch_idx in range(batch_size):
            subsets = []
            for _ in range(self.num_subsets):
                subsets.append(self._stratified_indices(y_support[batch_idx], sample_size, task_type))
            samples.append(torch.stack(subsets, dim=0))
        return torch.stack(samples, dim=0)


class TargetTokenEncoder(nn.Module):
    """Encode support-label statistics into a target token."""

    def __init__(self, token_dim: int, max_classes: int = 10) -> None:
        super().__init__()
        self.max_classes = max_classes
        classification_dim = max_classes + 4
        regression_dim = 8
        self.classification_mlp = nn.Sequential(
            nn.Linear(classification_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.regression_mlp = nn.Sequential(
            nn.Linear(regression_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, y_support: Tensor, task_type: str = "classification") -> Tensor:
        if task_type == "classification":
            batch_size, subset_size = y_support.shape
            histograms = []
            for batch_idx in range(batch_size):
                labels = y_support[batch_idx].long()
                hist = torch.bincount(labels, minlength=self.max_classes)[: self.max_classes].float()
                hist = hist / hist.sum().clamp_min(1.0)
                entropy = -(hist * (hist + 1e-6).log()).sum()
                stats = torch.cat(
                    [
                        hist,
                        torch.tensor(
                            [
                                float((hist > 0).sum().item()),
                                entropy.item(),
                                float(subset_size),
                                float(hist.max().item()),
                            ],
                            device=y_support.device,
                        ),
                    ],
                    dim=0,
                )
                histograms.append(stats)
            return self.classification_mlp(torch.stack(histograms, dim=0))

        quantiles = torch.tensor([0.25, 0.5, 0.75], device=y_support.device)
        q = torch.quantile(y_support.float(), quantiles, dim=1).transpose(0, 1)
        stats = torch.cat(
            [
                y_support.mean(dim=1, keepdim=True),
                _safe_std(y_support.float(), dim=1).unsqueeze(-1),
                y_support.min(dim=1, keepdim=True).values,
                y_support.max(dim=1, keepdim=True).values,
                q,
                torch.full((y_support.shape[0], 1), float(y_support.shape[1]), device=y_support.device),
            ],
            dim=1,
        )
        return self.regression_mlp(stats.float())


class FeatureTargetStatsEncoder(nn.Module):
    """Encode per-feature statistics with respect to the support labels."""

    def __init__(self, output_dim: int, hidden_dim: int = 64, nan_to_num: bool = True) -> None:
        super().__init__()
        self.nan_to_num = nan_to_num
        self.net = nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def _classification_score(self, feature: Tensor, labels: Tensor) -> Tensor:
        global_mean = feature.mean()
        overall_var = feature.var(unbiased=False).clamp_min(1e-6)
        class_means = []
        class_counts = []
        for value in labels.unique(sorted=True):
            group = feature[labels == value]
            class_means.append(group.mean())
            class_counts.append(float(len(group)))
        means = torch.stack(class_means)
        counts = torch.tensor(class_counts, device=feature.device)
        between_var = (counts * (means - global_mean).square()).sum() / counts.sum().clamp_min(1.0)
        return between_var / overall_var

    def forward(self, X_support: Tensor, y_support: Tensor, task_type: str = "classification") -> Tensor:
        batch_size, _, num_features = X_support.shape
        features = []
        for batch_idx in range(batch_size):
            feature_stats = []
            for feat_idx in range(num_features):
                values = X_support[batch_idx, :, feat_idx]
                missing_rate = torch.isnan(values).float().mean()
                cleaned = torch.nan_to_num(values.float(), nan=0.0)
                unique_ratio = cleaned.unique().numel() / max(1, len(cleaned))
                variance = cleaned.var(unbiased=False)
                mean_abs = cleaned.abs().mean()
                max_abs = cleaned.abs().max()
                if task_type == "classification":
                    target_score = self._classification_score(cleaned, y_support[batch_idx].long())
                else:
                    centered_x = cleaned - cleaned.mean()
                    centered_y = y_support[batch_idx].float() - y_support[batch_idx].float().mean()
                    denom = centered_x.std(unbiased=False).clamp_min(1e-6) * centered_y.std(unbiased=False).clamp_min(
                        1e-6
                    )
                    target_score = (centered_x * centered_y).mean() / denom
                feature_stats.append(
                    torch.tensor(
                        [target_score, missing_rate, unique_ratio, variance, mean_abs, max_abs],
                        device=X_support.device,
                    )
                )
            features.append(torch.stack(feature_stats, dim=0))
        stats = torch.stack(features, dim=0).float()
        if self.nan_to_num:
            stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        return self.net(stats)


class SubsetAggregator(nn.Module):
    """Aggregate multi-view feature representations across subsets."""

    def __init__(self, token_dim: int, aggregation: str = "mean_std_max") -> None:
        super().__init__()
        self.aggregation = aggregation
        if aggregation == "attention":
            self.attn = nn.Linear(token_dim, 1)
            input_dim = token_dim
        elif aggregation == "mean":
            input_dim = token_dim
        elif aggregation == "mean_std":
            input_dim = token_dim * 2
        else:
            input_dim = token_dim * 3
        self.projector = nn.Sequential(nn.Linear(input_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))

    def forward(self, subset_tokens: Tensor) -> Tensor:
        if self.aggregation == "attention":
            attn_logits = self.attn(subset_tokens).squeeze(-1)
            weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
            pooled = (subset_tokens * weights).sum(dim=1)
            return self.projector(pooled)

        mean = subset_tokens.mean(dim=1)
        if self.aggregation == "mean":
            aggregated = mean
        elif self.aggregation == "mean_std":
            aggregated = torch.cat([mean, _safe_std(subset_tokens, dim=1)], dim=-1)
        else:
            aggregated = torch.cat([mean, _safe_std(subset_tokens, dim=1), subset_tokens.max(dim=1).values], dim=-1)
        return self.projector(aggregated)


class MultiViewMBPredictor(nn.Module):
    """Predict feature-level MB scores from support sets using frozen TabICL embeddings."""

    def __init__(
        self,
        col_embedder: nn.Module,
        embed_dim: int,
        reserve_cls_tokens: int,
        max_classes: int = 10,
        mb_num_subsets: int = 8,
        mb_subset_size: float | int = 0.5,
        mb_stratified_sampling: bool = True,
        mb_regression_num_bins: int = 5,
        mb_sampling_with_replacement: bool = False,
        mb_min_subset_size: int = 16,
        mb_pooling: str = "mean_std_max",
        mb_token_dim: int = 128,
        mb_pool_mlp_hidden_dim: int = 128,
        mb_pool_dropout: float = 0.0,
        mb_use_target_token: bool = True,
        mb_target_token_dim: int = 128,
        mb_target_stats: bool = True,
        mb_use_feature_target_stats: bool = True,
        mb_stats_dim: int = 64,
        mb_stats_mlp_hidden_dim: int = 64,
        mb_stats_nan_to_num: bool = True,
        mb_column_transformer_layers: int = 2,
        mb_column_transformer_heads: int = 4,
        mb_column_transformer_ffn_dim: int = 256,
        mb_column_transformer_dropout: float = 0.0,
        mb_column_transformer_norm_first: bool = True,
        mb_subset_aggregation: str = "mean_std_max",
        mb_head_hidden_dim: int = 128,
        mb_head_dropout: float = 0.0,
        mb_score_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.col_embedder = col_embedder
        self.reserve_cls_tokens = reserve_cls_tokens
        self.mb_use_target_token = mb_use_target_token
        self.mb_use_feature_target_stats = mb_use_feature_target_stats
        self.mb_target_stats = mb_target_stats
        self.mb_score_temperature = mb_score_temperature
        self.sampler = SubsetSampler(
            num_subsets=mb_num_subsets,
            subset_size=mb_subset_size,
            stratified_sampling=mb_stratified_sampling,
            regression_num_bins=mb_regression_num_bins,
            sampling_with_replacement=mb_sampling_with_replacement,
            min_subset_size=mb_min_subset_size,
        )

        if mb_pooling == "mean":
            pool_in_dim = embed_dim
        elif mb_pooling == "mean_std":
            pool_in_dim = embed_dim * 2
        else:
            pool_in_dim = embed_dim * 3
        self.mb_pooling = mb_pooling
        self.pool_projector = nn.Sequential(
            nn.Linear(pool_in_dim, mb_pool_mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(mb_pool_dropout),
            nn.Linear(mb_pool_mlp_hidden_dim, mb_token_dim),
        )

        self.target_encoder = TargetTokenEncoder(token_dim=mb_target_token_dim, max_classes=max_classes)
        self.target_to_token = (
            nn.Linear(mb_target_token_dim, mb_token_dim) if mb_target_token_dim != mb_token_dim else nn.Identity()
        )
        self.stats_encoder = FeatureTargetStatsEncoder(
            output_dim=mb_stats_dim,
            hidden_dim=mb_stats_mlp_hidden_dim,
            nan_to_num=mb_stats_nan_to_num,
        )
        stats_input_dim = mb_token_dim + (mb_stats_dim if mb_use_feature_target_stats else 0)
        self.feature_projector = nn.Sequential(nn.Linear(stats_input_dim, mb_token_dim), nn.GELU(), nn.Linear(mb_token_dim, mb_token_dim))
        self.column_transformer = Encoder(
            num_blocks=mb_column_transformer_layers,
            d_model=mb_token_dim,
            nhead=mb_column_transformer_heads,
            dim_feedforward=mb_column_transformer_ffn_dim,
            dropout=mb_column_transformer_dropout,
            activation="gelu",
            norm_first=mb_column_transformer_norm_first,
        )
        self.aggregator = SubsetAggregator(mb_token_dim, aggregation=mb_subset_aggregation)
        self.head = nn.Sequential(
            nn.Linear(mb_token_dim, mb_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(mb_head_dropout),
            nn.Linear(mb_head_hidden_dim, 1),
        )

        for parameter in self.col_embedder.parameters():
            parameter.requires_grad = False

    def _pool_support_embeddings(self, support_embeddings: Tensor) -> Tensor:
        mean = support_embeddings.mean(dim=1)
        if self.mb_pooling == "mean":
            pooled = mean
        elif self.mb_pooling == "mean_std":
            pooled = torch.cat([mean, _safe_std(support_embeddings, dim=1)], dim=-1)
        else:
            pooled = torch.cat([mean, _safe_std(support_embeddings, dim=1), support_embeddings.max(dim=1).values], dim=-1)
        return self.pool_projector(pooled)

    def forward(
        self,
        X_support: Tensor,
        y_support: Tensor,
        d: Optional[Tensor] = None,
        task_type: str = "classification",
    ) -> MBPredictorOutput:
        batch_size, _, num_features = X_support.shape
        subset_indices = self.sampler.sample(y_support, task_type=task_type)
        num_subsets, subset_size = subset_indices.shape[1], subset_indices.shape[2]

        batch_index = torch.arange(batch_size, device=X_support.device).view(batch_size, 1, 1)
        X_subsets = X_support[batch_index, subset_indices]
        y_subsets = y_support[batch_index, subset_indices]

        flat_X = X_subsets.reshape(batch_size * num_subsets, subset_size, num_features)
        flat_y = y_subsets.reshape(batch_size * num_subsets, subset_size)
        if d is not None:
            flat_d = d.unsqueeze(1).expand(batch_size, num_subsets).reshape(-1)
        else:
            flat_d = None

        with torch.no_grad():
            self.col_embedder.eval()
            subset_embeddings = self.col_embedder(flat_X, d=flat_d, train_size=subset_size)
        subset_embeddings = subset_embeddings[:, :, self.reserve_cls_tokens :, :]
        pooled_tokens = self._pool_support_embeddings(subset_embeddings)

        if self.mb_use_feature_target_stats:
            stats = self.stats_encoder(flat_X, flat_y, task_type=task_type)
            pooled_tokens = self.feature_projector(torch.cat([pooled_tokens, stats], dim=-1))
        else:
            pooled_tokens = self.feature_projector(pooled_tokens)

        if self.mb_use_target_token and self.mb_target_stats:
            target_token = self.target_to_token(self.target_encoder(flat_y, task_type=task_type)).unsqueeze(1)
        else:
            target_token = torch.zeros(batch_size * num_subsets, 1, pooled_tokens.shape[-1], device=X_support.device)

        tokens = torch.cat([target_token, pooled_tokens], dim=1)
        encoded_tokens = self.column_transformer(tokens)
        feature_tokens = encoded_tokens[:, 1:, :]
        feature_tokens = feature_tokens.view(batch_size, num_subsets, num_features, -1)

        aggregated_tokens = self.aggregator(feature_tokens.transpose(1, 2).reshape(batch_size * num_features, num_subsets, -1))
        aggregated_tokens = aggregated_tokens.view(batch_size, num_features, -1)
        logits = self.head(aggregated_tokens).squeeze(-1) / max(self.mb_score_temperature, 1e-6)
        mb_scores = torch.sigmoid(logits)
        local_scores = torch.sigmoid(self.head(feature_tokens).squeeze(-1) / max(self.mb_score_temperature, 1e-6))

        if d is not None:
            feature_mask = (torch.arange(num_features, device=X_support.device).view(1, num_features) < d.view(-1, 1)).float()
            mb_scores = mb_scores * feature_mask
            local_scores = local_scores * feature_mask.unsqueeze(1)

        return MBPredictorOutput(
            mb_scores=mb_scores,
            local_mb_scores=local_scores,
            diagnostics={"subset_indices": subset_indices},
        )
