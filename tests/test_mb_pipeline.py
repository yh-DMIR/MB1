import torch

from tabicl.model.mb_injection import MBScoreProvider
from tabicl.prior.synthetic_scm_mb import smoke_test_synthetic_scm_mb
from tabicl.train.mb_utils import build_model_config
from tabicl import TabICL


class DummyConfig:
    max_classes = 10
    embed_dim = 32
    col_num_blocks = 1
    col_nhead = 4
    col_num_inds = 16
    row_num_blocks = 1
    row_nhead = 4
    row_num_cls = 2
    row_rope_base = 100000
    icl_num_blocks = 1
    icl_nhead = 2
    ff_factor = 2
    dropout = 0.0
    activation = "gelu"
    norm_first = True
    mb_injection = "cls_soft_bias"
    mb_bias_init = 0.1
    mb_bias_trainable = True
    mb_bias_per_layer = True
    mb_bias_per_head = True
    mb_bias_target = "cls_to_feature"
    mb_hard_mask_threshold = 0.5
    mb_topk_extra = 0
    mb_hard_select_mode = "threshold"
    mb_hard_select_topk = 0
    mb_embedding_dim = None
    mb_embedding_add_position = "before_rowinteraction"


def test_synthetic_scm_smoke():
    smoke = smoke_test_synthetic_scm_mb(
        scm_num_features=12,
        scm_mb_size=4,
        scm_num_samples=24,
        scm_noise_dim=2,
        scm_redundant_dim=2,
        n_support=12,
        n_query=12,
    )
    assert smoke["X_shape"] == (24, 12)
    assert smoke["y_shape"] == (24,)
    assert smoke["mb_shape"] == (12,)


def test_tabicl_accepts_mb_scores():
    config = DummyConfig()
    model = TabICL(**build_model_config(config))
    model.train()
    X = torch.randn(2, 10, 6)
    y_train = torch.randint(0, 2, (2, 5))
    provider = MBScoreProvider(mb_score_source="random")
    result = provider.get_scores(X[:, :5], y_train.float(), task_type="classification")
    logits, diagnostics = model(X, y_train.float(), mb_scores=result.scores, return_mb_diagnostics=True)
    assert logits.shape[0] == 2
    assert logits.shape[1] == 5
    assert "mb_score_mean" in diagnostics
