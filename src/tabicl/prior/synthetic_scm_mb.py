from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

import torch
from torch.utils.data import IterableDataset


@dataclass
class SyntheticSCMTask:
    """Container for a synthetic SCM task with MB supervision."""

    X: torch.Tensor
    y: torch.Tensor
    mb_labels: torch.Tensor
    support_size: int
    query_size: int
    metadata: Dict[str, torch.Tensor | int | float | str]


def _ensure_generator(seed: Optional[int] = None) -> torch.Generator:
    generator = torch.Generator()
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator.manual_seed(seed)
    return generator


def _sample_without_replacement(population_size: int, sample_size: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(population_size, generator=generator)[:sample_size]


def _apply_nonlinearity(values: torch.Tensor) -> torch.Tensor:
    return torch.tanh(values) + 0.25 * values.square() - 0.1 * values


def _sample_mb_partition(scm_mb_size: int, generator: torch.Generator) -> tuple[int, int, int]:
    """Sample a variable MB partition so different tasks can come from different DAG topologies."""

    if scm_mb_size == 2:
        return 1, 1, 0

    max_parents = max(1, scm_mb_size - 1)
    num_parents = int(torch.randint(1, max_parents + 1, (1,), generator=generator).item())
    remaining = scm_mb_size - num_parents
    if remaining <= 0:
        num_parents = scm_mb_size - 1
        remaining = 1

    num_children = int(torch.randint(1, remaining + 1, (1,), generator=generator).item())
    num_spouses = remaining - num_children
    return num_parents, num_children, num_spouses


def generate_scm_task(
    scm_num_features: int = 32,
    scm_mb_size: int = 6,
    scm_num_samples: int = 128,
    scm_noise_dim: int = 8,
    scm_redundant_dim: int = 4,
    scm_nonlinear: bool = False,
    scm_task_type: str = "classification",
    scm_num_classes: int = 2,
    scm_noise_std: float = 0.1,
    scm_seed: Optional[int] = None,
    n_support: int = 64,
    n_query: int = 64,
) -> SyntheticSCMTask:
    """Generate a synthetic SCM task with MB labels.

    Notes
    -----
    The generator builds a small DAG with parents -> y -> children and spouse noise
    variables that also influence the children. The MB label is therefore
    parents U children U spouses.
    """

    if scm_num_features < 3:
        raise ValueError("scm_num_features must be at least 3")
    if scm_mb_size < 2:
        raise ValueError("scm_mb_size must be at least 2")

    total_samples = n_support + n_query
    if total_samples > scm_num_samples:
        scm_num_samples = total_samples

    generator = _ensure_generator(scm_seed)
    device = torch.device("cpu")

    num_parents, num_children, num_spouses = _sample_mb_partition(scm_mb_size, generator)

    mb_size = num_parents + num_children + num_spouses
    remaining = scm_num_features - mb_size
    if remaining < 0:
        raise ValueError("scm_mb_size cannot exceed scm_num_features")

    redundant_dim = min(scm_redundant_dim, remaining)
    remaining -= redundant_dim
    noise_dim = min(scm_noise_dim, remaining)
    remaining -= noise_dim
    distractor_dim = remaining

    num_rows = scm_num_samples
    parents = torch.randn(num_rows, num_parents, generator=generator, device=device)
    spouse_latents = torch.randn(num_rows, num_spouses, generator=generator, device=device)

    parent_weights = 0.5 + torch.rand(num_parents, generator=generator, device=device)
    parent_sign = torch.where(
        torch.rand(num_parents, generator=generator, device=device) > 0.5,
        torch.ones(num_parents, device=device),
        -torch.ones(num_parents, device=device),
    )
    y_latent = parents @ (parent_weights * parent_sign)
    if scm_nonlinear:
        y_latent = _apply_nonlinearity(y_latent)
    y_latent = y_latent + scm_noise_std * torch.randn(num_rows, generator=generator, device=device)

    children = []
    child_parent_edges = []
    child_spouse_edges = []
    for idx in range(num_children):
        y_coeff = 0.6 + 0.6 * torch.rand(1, generator=generator, device=device)
        child = y_coeff * y_latent

        parent_edge_mask = torch.zeros(num_parents, dtype=torch.float32, device=device)
        num_parent_inputs = int(torch.randint(1, min(num_parents, 3) + 1, (1,), generator=generator).item())
        parent_indices = _sample_without_replacement(num_parents, num_parent_inputs, generator)
        parent_edge_mask[parent_indices] = 1.0
        parent_weights_to_child = 0.15 + 0.35 * torch.rand(
            num_parent_inputs, generator=generator, device=device
        )
        child = child + parents[:, parent_indices] @ parent_weights_to_child
        child_parent_edges.append(parent_edge_mask)

        spouse_edge_mask = torch.zeros(num_spouses, dtype=torch.float32, device=device)
        if num_spouses > 0:
            num_spouse_inputs = int(torch.randint(1, num_spouses + 1, (1,), generator=generator).item())
            spouse_indices = _sample_without_replacement(num_spouses, num_spouse_inputs, generator)
            spouse_edge_mask[spouse_indices] = 1.0
            spouse_weights_to_child = 0.25 + 0.45 * torch.rand(
                num_spouse_inputs, generator=generator, device=device
            )
            child = child + spouse_latents[:, spouse_indices] @ spouse_weights_to_child
        child_spouse_edges.append(spouse_edge_mask)

        if scm_nonlinear:
            child = _apply_nonlinearity(child)
        child = child + scm_noise_std * torch.randn(num_rows, generator=generator, device=device)
        children.append(child.unsqueeze(-1))
    children = torch.cat(children, dim=-1) if children else torch.empty(num_rows, 0, device=device)

    spouses = []
    for idx in range(num_spouses):
        spouse = spouse_latents[:, idx]
        if scm_nonlinear:
            spouse = torch.sin(spouse)
        spouse = spouse + scm_noise_std * torch.randn(num_rows, generator=generator, device=device)
        spouses.append(spouse.unsqueeze(-1))
    spouses = torch.cat(spouses, dim=-1) if spouses else torch.empty(num_rows, 0, device=device)

    redundant = []
    blanket_stack = torch.cat([parents, children, spouses], dim=-1)
    for idx in range(redundant_dim):
        source_idx = idx % blanket_stack.shape[1]
        redundant_feature = blanket_stack[:, source_idx] + 0.2 * torch.randn(
            num_rows, generator=generator, device=device
        )
        redundant.append(redundant_feature.unsqueeze(-1))
    redundant = torch.cat(redundant, dim=-1) if redundant else torch.empty(num_rows, 0, device=device)

    noise = torch.randn(num_rows, noise_dim, generator=generator, device=device)

    distractors = []
    for idx in range(distractor_dim):
        coeffs = torch.randn(num_parents, generator=generator, device=device) * 0.15
        distractor = parents @ coeffs + 0.9 * torch.randn(num_rows, generator=generator, device=device)
        if scm_nonlinear and idx % 2 == 0:
            distractor = torch.sin(distractor)
        distractors.append(distractor.unsqueeze(-1))
    distractors = torch.cat(distractors, dim=-1) if distractors else torch.empty(num_rows, 0, device=device)

    X_parts = [parents, children, spouses, redundant, noise, distractors]
    X = torch.cat([part for part in X_parts if part.numel() > 0], dim=-1)
    if X.shape[1] != scm_num_features:
        raise RuntimeError(f"Expected {scm_num_features} features but generated {X.shape[1]}")

    if scm_task_type == "classification":
        if scm_num_classes < 2:
            raise ValueError("scm_num_classes must be >= 2 for classification tasks")
        quantiles = torch.linspace(0.0, 1.0, scm_num_classes + 1, device=device)[1:-1]
        thresholds = torch.quantile(y_latent, quantiles) if len(quantiles) > 0 else torch.empty(0, device=device)
        y = torch.bucketize(y_latent, thresholds).long()
    elif scm_task_type == "regression":
        y = y_latent.float()
    else:
        raise ValueError(f"Unsupported scm_task_type: {scm_task_type}")

    mb_labels = torch.zeros(scm_num_features, dtype=torch.float32, device=device)
    mb_labels[:mb_size] = 1.0

    feature_perm = torch.randperm(scm_num_features, generator=generator, device=device)
    X = X[:, feature_perm]
    mb_labels = mb_labels[feature_perm]

    row_perm = torch.randperm(num_rows, generator=generator, device=device)
    X = X[row_perm]
    y = y[row_perm]

    X = X[:total_samples]
    y = y[:total_samples]

    metadata = {
        "task_type": scm_task_type,
        "num_parents": num_parents,
        "num_children": num_children,
        "num_spouses": num_spouses,
        "num_redundant": redundant_dim,
        "num_noise": noise_dim,
        "seed": scm_seed if scm_seed is not None else -1,
        "dag_parent_sign_sum": float(parent_sign.sum().item()),
        "dag_child_parent_density": float(
            torch.stack(child_parent_edges).mean().item() if child_parent_edges else 0.0
        ),
        "dag_child_spouse_density": float(
            torch.stack(child_spouse_edges).mean().item() if child_spouse_edges else 0.0
        ),
    }

    return SyntheticSCMTask(
        X=X.float(),
        y=y.float() if scm_task_type == "regression" else y.long(),
        mb_labels=mb_labels.float(),
        support_size=n_support,
        query_size=n_query,
        metadata=metadata,
    )


class SyntheticSCMBatchDataset(IterableDataset):
    """Infinite iterable dataset that yields synthetic SCM tasks with MB labels."""

    def __init__(
        self,
        batch_size: int = 8,
        scm_num_features: int = 32,
        scm_mb_size: int = 6,
        scm_num_samples: int = 128,
        scm_noise_dim: int = 8,
        scm_redundant_dim: int = 4,
        scm_nonlinear: bool = False,
        scm_task_type: str = "classification",
        scm_num_classes: int = 2,
        scm_noise_std: float = 0.1,
        scm_seed: int = 42,
        n_support: int = 64,
        n_query: int = 64,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.scm_num_features = scm_num_features
        self.scm_mb_size = scm_mb_size
        self.scm_num_samples = scm_num_samples
        self.scm_noise_dim = scm_noise_dim
        self.scm_redundant_dim = scm_redundant_dim
        self.scm_nonlinear = scm_nonlinear
        self.scm_task_type = scm_task_type
        self.scm_num_classes = scm_num_classes
        self.scm_noise_std = scm_noise_std
        self.scm_seed = scm_seed
        self.n_support = n_support
        self.n_query = n_query
        self._counter = 0

    def get_batch(self) -> Dict[str, torch.Tensor | str]:
        tasks = []
        for batch_idx in range(self.batch_size):
            task = generate_scm_task(
                scm_num_features=self.scm_num_features,
                scm_mb_size=self.scm_mb_size,
                scm_num_samples=self.scm_num_samples,
                scm_noise_dim=self.scm_noise_dim,
                scm_redundant_dim=self.scm_redundant_dim,
                scm_nonlinear=self.scm_nonlinear,
                scm_task_type=self.scm_task_type,
                scm_num_classes=self.scm_num_classes,
                scm_noise_std=self.scm_noise_std,
                scm_seed=self.scm_seed + self._counter + batch_idx,
                n_support=self.n_support,
                n_query=self.n_query,
            )
            tasks.append(task)

        self._counter += self.batch_size

        X = torch.stack([task.X for task in tasks], dim=0)
        y = torch.stack([task.y for task in tasks], dim=0)
        mb_labels = torch.stack([task.mb_labels for task in tasks], dim=0)
        d = torch.full((self.batch_size,), self.scm_num_features, dtype=torch.long)
        return {
            "X": X,
            "y": y,
            "mb_labels": mb_labels,
            "d": d,
            "support_size": torch.full((self.batch_size,), self.n_support, dtype=torch.long),
            "query_size": torch.full((self.batch_size,), self.n_query, dtype=torch.long),
            "task_type": self.scm_task_type,
        }

    def __iter__(self) -> "SyntheticSCMBatchDataset":
        return self

    def __next__(self) -> Dict[str, torch.Tensor | str]:
        return self.get_batch()

    def __repr__(self) -> str:
        return (
            "SyntheticSCMBatchDataset("
            f"batch_size={self.batch_size}, "
            f"features={self.scm_num_features}, "
            f"mb_size={self.scm_mb_size}, "
            f"samples={self.scm_num_samples}, "
            f"task_type={self.scm_task_type}, "
            f"support={self.n_support}, "
            f"query={self.n_query})"
        )


def smoke_test_synthetic_scm_mb(**kwargs) -> Dict[str, tuple[int, ...] | float]:
    """Run a small smoke test and return summary statistics."""

    task = generate_scm_task(**kwargs)
    mb_mask = task.mb_labels.bool()
    X = task.X
    if task.metadata["task_type"] == "classification":
        y = task.y.float()
    else:
        y = task.y

    mb_mean = X[:, mb_mask].abs().mean().item() if mb_mask.any() else 0.0
    non_mb_mean = X[:, ~mb_mask].abs().mean().item() if (~mb_mask).any() else 0.0
    return {
        "X_shape": tuple(X.shape),
        "y_shape": tuple(task.y.shape),
        "mb_shape": tuple(task.mb_labels.shape),
        "mb_positive": float(task.mb_labels.sum().item()),
        "target_std": float(y.float().std().item()),
        "mb_abs_mean": mb_mean,
        "non_mb_abs_mean": non_mb_mean,
    }
