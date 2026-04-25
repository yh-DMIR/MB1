from __future__ import annotations

import os

import torch
from torch import optim

from tabicl.prior.synthetic_scm_mb import SyntheticSCMBatchDataset
from tabicl.train.mb_utils import build_mb_predictor, compute_mb_metrics, load_tabicl_model, mb_supervision_loss
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
    predictor.train()

    optimizer = optim.AdamW(predictor.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    os.makedirs(config.checkpoint_dir or "checkpoints", exist_ok=True)

    for step in range(config.max_steps):
        batch = next(iter(dataset))
        X_support = batch["X"][:, : config.n_support].to(config.device)
        y_support = batch["y"][:, : config.n_support].to(config.device)
        mb_labels = batch["mb_labels"].to(config.device)
        d = batch["d"].to(config.device)

        optimizer.zero_grad(set_to_none=True)
        output = predictor(X_support, y_support, d=d, task_type=config.scm_task_type)
        loss = config.mb_loss_weight * mb_supervision_loss(output.mb_scores, mb_labels, config)
        loss.backward()
        optimizer.step()

        if step % max(1, config.save_temp_every) == 0 or step == config.max_steps - 1:
            metrics = compute_mb_metrics(output.mb_scores.detach(), mb_labels.detach())
            metrics["loss"] = float(loss.item())
            print(f"step={step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            checkpoint_path = os.path.join(config.checkpoint_dir or "checkpoints", f"mb_predictor_step_{step}.ckpt")
            torch.save({"state_dict": predictor.state_dict(), "step": step}, checkpoint_path)


if __name__ == "__main__":
    main()
