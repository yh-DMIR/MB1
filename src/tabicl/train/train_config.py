"""Define argument parser for TabICL training."""

import argparse


def str2bool(value):
    return value.lower() == "true"


def train_size_type(value):
    """Custom type function to handle both int and float train sizes."""
    value = float(value)
    if 0 < value < 1:
        return value
    elif value.is_integer():
        return int(value)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be either an integer (absolute position) "
            "or a float between 0 and 1 (ratio of sequence length)."
        )


def float_or_int_type(value):
    value = float(value)
    if value > 0 and value.is_integer():
        return int(value)
    if 0 < value <= 1:
        return value
    raise argparse.ArgumentTypeError("Value must be a positive integer or a float in (0, 1].")


def build_parser():
    """Build parser with all TabICL training arguments."""
    parser = argparse.ArgumentParser()

    ###########################################################################
    ###### Wandb Config #######################################################
    ###########################################################################
    parser.add_argument("--wandb_log", default=False, type=str2bool, help="Log results using wandb")
    parser.add_argument("--wandb_project", type=str, default="TabICL", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run ID")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Wandb logging directory")
    parser.add_argument(
        "--wandb_mode", default="offline", type=str, help="Wandb logging mode: online, offline, or disabled"
    )

    ###########################################################################
    ###### Training Config ####################################################
    ###########################################################################
    parser.add_argument("--device", default="cuda", type=str, help="Device for training: cpu, cuda, cuda:0")
    parser.add_argument(
        "--dtype", default="float32", type=str, help="Data type (supported for float16, float32) used for training"
    )
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--max_steps", type=int, default=60000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=8, help="Size of micro-batches for gradient accumulation"
    )

    # Optimization Config
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--scheduler", type=str, default="cosine_warmup", help="Learning rate scheduler: see optim.py for options."
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.2,
        help="The proportion of total steps over which we warmup."
        "If this value is set to -1, we warmup for a fixed number of steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="The number of steps over which we warm up. Only used when warmup_proportion is set to -1",
    )
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="If > 0, clip gradients.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay / L2 regularization penalty")
    parser.add_argument(
        "--cosine_num_cycles",
        type=int,
        default=1,
        help="Number of hard restarts for cosine schedule. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument(
        "--cosine_amplitude_decay",
        type=float,
        default=1.0,
        help="Amplitude scaling factor per cycle. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument("--cosine_lr_end", type=float, default=0, help="Final learning rate for cosine_with_restarts")
    parser.add_argument(
        "--poly_decay_lr_end", type=float, default=1e-7, help="Final learning rate for polynomial decay scheduler"
    )
    parser.add_argument(
        "--poly_decay_power", type=float, default=1.0, help="Power factor for polynomial decay scheduler"
    )

    # Prior Dataset Config
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="If set, load pre-generated prior datasets directly from this directory on disk instead of generating them on the fly.",
    )
    parser.add_argument(
        "--load_prior_start",
        type=int,
        default=0,
        help="Batch index to start loading from pre-generated prior data. Only used when prior_dir is set.",
    )
    parser.add_argument(
        "--delete_after_load",
        default=False,
        type=str2bool,
        help="Delete prior data after loading. Only used when prior_dir is set.",
    )
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=5, help="The minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="The maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="The maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum samples per dataset")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum samples per dataset")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size",
        type=train_size_type,
        default=0.1,
        help="Starting position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--max_train_size",
        type=train_size_type,
        default=0.9,
        help="Ending position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type", default="mix_scm", type=str, help="Prior type: dummy, mlp_scm, tree_scm, mix_scm"
    )
    parser.add_argument("--prior_device", default="cpu", type=str, help="Device for prior data generation")

    ###########################################################################
    ##### Model Architecture Config ###########################################
    ###########################################################################
    parser.add_argument(
        "--amp",
        default=True,
        type=str2bool,
        help="If True, use automatic mixed precision (AMP) which can provide significant speedups on compatible GPU",
    )
    parser.add_argument(
        "--model_compile",
        default=False,
        type=str2bool,
        help="If True, compile the model using torch.compile for speedup",
    )

    # Column Embedding Config
    parser.add_argument("--embed_dim", type=int, default=128, help="Base embedding dimension")
    parser.add_argument("--col_num_blocks", type=int, default=3, help="Number of blocks in column embedder")
    parser.add_argument("--col_nhead", type=int, default=4, help="Number of attention heads in column embedder")
    parser.add_argument("--col_num_inds", type=int, default=128, help="Number of inducing points in column embedder")
    parser.add_argument("--freeze_col", default=False, type=str2bool, help="Whether to freeze the column embedder")

    # Row Interaction Config
    parser.add_argument("--row_num_blocks", type=int, default=3, help="Number of blocks in row interactor")
    parser.add_argument("--row_nhead", type=int, default=8, help="Number of attention heads in row interactor")
    parser.add_argument("--row_num_cls", type=int, default=4, help="Number of CLS tokens in row interactor")
    parser.add_argument("--row_rope_base", type=float, default=100000, help="RoPE base value for row interactor")
    parser.add_argument("--freeze_row", default=False, type=str2bool, help="Whether to freeze the row interactor")

    # ICL Config
    parser.add_argument("--icl_num_blocks", type=int, default=12, help="Number of transformer blocks in ICL predictor")
    parser.add_argument("--icl_nhead", type=int, default=4, help="Number of attention heads in ICL predictor")
    parser.add_argument("--freeze_icl", default=False, type=str2bool, help="Whether to freeze the ICL predictor")

    # Shared Architecture Config
    parser.add_argument("--ff_factor", type=int, default=2, help="Expansion factor for feedforward dimensions")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function type")
    parser.add_argument(
        "--norm_first", default=True, type=str2bool, help="If True, use pre-norm transformer architecture"
    )

    ###########################################################################
    ###### Checkpointing ######################################################
    ###########################################################################
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory for checkpoint saving and loading")
    parser.add_argument("--save_temp_every", default=50, type=int, help="Steps between temporary checkpoints")
    parser.add_argument("--save_perm_every", default=5000, type=int, help="Steps between permanent checkpoints")
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of temporary checkpoints to keep. Permanent checkpoints are not counted.",
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to specific checkpoint file to load")
    parser.add_argument("--only_load_model", default=False, type=str2bool, help="Whether to only load model weights")

    ###########################################################################
    ###### MB-TabICL / Research Config ########################################
    ###########################################################################
    parser.add_argument("--train_mode", type=str, default="baseline", help="baseline, mb_predictor, calibrator")
    parser.add_argument("--eval_mode", type=str, default="none", help="none, mb_predictor, closed_loop")
    parser.add_argument("--task_type", type=str, default="classification", help="classification or regression")
    parser.add_argument("--tabicl_checkpoint_path", type=str, default=None, help="Checkpoint for pretrained TabICL")
    parser.add_argument(
        "--mb_predictor_checkpoint_path", type=str, default=None, help="Checkpoint for MultiViewMBPredictor"
    )
    parser.add_argument(
        "--freeze_tabicl_backbone", default=True, type=str2bool, help="Freeze TabICL backbone in MB experiments"
    )

    parser.add_argument(
        "--mb_score_source",
        type=str,
        default="none",
        help="none, oracle, estimated, predicted, shuffled, corr, mi, random",
    )
    parser.add_argument(
        "--mb_estimator",
        type=str,
        default="bootstrap_corr",
        help="bootstrap_corr, conditional_mi, permutation, placeholder",
    )
    parser.add_argument("--mb_shuffle_within_batch", default=True, type=str2bool, help="Shuffle MB scores per batch")
    parser.add_argument("--mb_random_seed", type=int, default=42, help="Random seed for random/shuffled MB scores")
    parser.add_argument(
        "--mb_injection",
        type=str,
        default="none",
        help="none, cls_soft_bias, embedding_add, cls_hard_mask, summary_token, hard_select",
    )
    parser.add_argument("--mb_bias_init", type=float, default=0.0, help="Initial MB attention bias scale")
    parser.add_argument("--mb_bias_trainable", default=True, type=str2bool, help="Train MB bias scale")
    parser.add_argument("--mb_bias_per_layer", default=True, type=str2bool, help="Use per-layer MB bias scale")
    parser.add_argument("--mb_bias_per_head", default=True, type=str2bool, help="Use per-head MB bias scale")
    parser.add_argument(
        "--mb_bias_target",
        type=str,
        default="cls_to_feature",
        help="cls_to_feature, all_to_feature, feature_to_feature",
    )
    parser.add_argument("--mb_hard_mask_threshold", type=float, default=0.5, help="Threshold for hard masks")
    parser.add_argument("--mb_topk_extra", type=int, default=0, help="Keep extra top-k features in hard variants")
    parser.add_argument("--mb_hard_select_mode", type=str, default="threshold", help="threshold or topk")
    parser.add_argument("--mb_hard_select_topk", type=int, default=0, help="Top-k for hard_select mode")
    parser.add_argument("--mb_embedding_dim", type=int, default=None, help="Hidden dim for embedding_add adapter")
    parser.add_argument(
        "--mb_embedding_add_position",
        type=str,
        default="before_rowinteraction",
        help="before_rowinteraction or after_colembedding",
    )

    parser.add_argument("--mb_num_subsets", type=int, default=8, help="Number of support subsets per task")
    parser.add_argument(
        "--mb_subset_size", type=float_or_int_type, default=0.5, help="Subset size as ratio or absolute count"
    )
    parser.add_argument(
        "--mb_stratified_sampling", default=True, type=str2bool, help="Use stratified sampling for support subsets"
    )
    parser.add_argument("--mb_regression_num_bins", type=int, default=5, help="Regression stratification bins")
    parser.add_argument(
        "--mb_sampling_with_replacement", default=False, type=str2bool, help="Sample support subsets with replacement"
    )
    parser.add_argument("--mb_min_subset_size", type=int, default=16, help="Minimum support subset size")
    parser.add_argument(
        "--mb_pooling", type=str, default="mean_std_max", help="mean, mean_std, mean_std_max"
    )
    parser.add_argument("--mb_token_dim", type=int, default=128, help="MB predictor token dimension")
    parser.add_argument("--mb_pool_mlp_hidden_dim", type=int, default=128, help="Pooling MLP hidden dim")
    parser.add_argument("--mb_pool_dropout", type=float, default=0.0, help="Pooling dropout")
    parser.add_argument("--mb_use_target_token", default=True, type=str2bool, help="Use target token")
    parser.add_argument("--mb_target_token_dim", type=int, default=128, help="Target token dimension")
    parser.add_argument("--mb_target_stats", default=True, type=str2bool, help="Use target summary statistics")
    parser.add_argument(
        "--mb_use_feature_target_stats", default=True, type=str2bool, help="Use feature-target statistics"
    )
    parser.add_argument("--mb_stats_dim", type=int, default=64, help="Feature-target stats embedding dim")
    parser.add_argument("--mb_stats_mlp_hidden_dim", type=int, default=64, help="Feature stats MLP hidden dim")
    parser.add_argument("--mb_stats_nan_to_num", default=True, type=str2bool, help="NaN-safe stats encoding")
    parser.add_argument("--mb_column_transformer_layers", type=int, default=2, help="MB column TF layers")
    parser.add_argument("--mb_column_transformer_heads", type=int, default=4, help="MB column TF heads")
    parser.add_argument("--mb_column_transformer_ffn_dim", type=int, default=256, help="MB column TF FFN dim")
    parser.add_argument("--mb_column_transformer_dropout", type=float, default=0.0, help="MB column TF dropout")
    parser.add_argument(
        "--mb_column_transformer_norm_first", default=True, type=str2bool, help="Use pre-norm MB column TF"
    )
    parser.add_argument(
        "--mb_subset_aggregation",
        type=str,
        default="mean_std_max",
        help="mean, mean_std, mean_std_max, attention",
    )
    parser.add_argument("--mb_head_hidden_dim", type=int, default=128, help="MB head hidden dim")
    parser.add_argument("--mb_head_dropout", type=float, default=0.0, help="MB head dropout")
    parser.add_argument("--mb_score_temperature", type=float, default=1.0, help="MB score temperature")

    parser.add_argument("--mb_loss_weight", type=float, default=1.0, help="Weight for MB supervision loss")
    parser.add_argument("--mb_pos_weight", type=float, default=1.0, help="Positive class weight for MB BCE")
    parser.add_argument("--mb_use_focal_loss", default=False, type=str2bool, help="Use focal loss for MB labels")
    parser.add_argument("--mb_focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--mb_train_calibrator", default=False, type=str2bool, help="Train only bias calibrator")

    parser.add_argument("--scm_num_features", type=int, default=32, help="Synthetic SCM number of features")
    parser.add_argument("--scm_mb_size", type=int, default=6, help="Synthetic SCM MB size")
    parser.add_argument("--scm_num_samples", type=int, default=128, help="Synthetic SCM number of samples")
    parser.add_argument("--scm_noise_dim", type=int, default=8, help="Synthetic SCM pure noise features")
    parser.add_argument("--scm_redundant_dim", type=int, default=4, help="Synthetic SCM redundant features")
    parser.add_argument("--scm_nonlinear", default=False, type=str2bool, help="Use nonlinear SCM generator")
    parser.add_argument(
        "--scm_task_type", type=str, default="classification", help="classification or regression for SCM generator"
    )
    parser.add_argument("--scm_num_classes", type=int, default=2, help="Synthetic SCM number of classes")
    parser.add_argument("--scm_noise_std", type=float, default=0.1, help="Synthetic SCM noise std")
    parser.add_argument("--scm_seed", type=int, default=42, help="Synthetic SCM seed")
    parser.add_argument("--n_support", type=int, default=64, help="Support size for SCM tasks")
    parser.add_argument("--n_query", type=int, default=64, help="Query size for SCM tasks")

    return parser
