from __future__ import annotations

import torch

from tabicl.prior.synthetic_scm_mb import SyntheticSCMBatchDataset
from tabicl.train.mb_utils import build_mb_predictor, compute_mb_metrics, load_tabicl_model
from tabicl.train.train_config import build_parser


def main():
    parser = build_parser()
    config = parser.parse_args()
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
    predictor = build_mb_predictor(config, model)
    predictor.eval()

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for _ in range(max(1, config.max_steps)):
            batch = next(iter(dataset))
            X_support = batch["X"][:, : config.n_support].to(config.device)
            y_support = batch["y"][:, : config.n_support].to(config.device)
            d = batch["d"].to(config.device)
            output = predictor(X_support, y_support, d=d, task_type=config.scm_task_type)
            all_scores.append(output.mb_scores.cpu())
            all_labels.append(batch["mb_labels"])

    metrics = compute_mb_metrics(torch.cat(all_scores, dim=0), torch.cat(all_labels, dim=0))
    print("MB predictor evaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
