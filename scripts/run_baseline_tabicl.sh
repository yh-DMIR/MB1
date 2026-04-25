#!/usr/bin/env bash

# 原始 TabICL baseline 训练脚本。默认不启用任何 MB 逻辑。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

DATASET="${DATASET:-synthetic_prior}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/baseline}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TABICL_CKPT="${TABICL_CKPT:-${CHECKPOINT_PATH}}"
MB_PREDICTOR_CKPT="${MB_PREDICTOR_CKPT:-}"
TASK_TYPE="${TASK_TYPE:-classification}"
N_SUPPORT="${N_SUPPORT:-64}"
N_QUERY="${N_QUERY:-64}"

mkdir -p "${OUTPUT_DIR}"

python "${ROOT_DIR}/src/tabicl/train/run.py" \
  --device "${DEVICE}" \
  --np_seed "${SEED}" \
  --torch_seed "${SEED}" \
  --checkpoint_dir "${OUTPUT_DIR}" \
  --checkpoint_path "${TABICL_CKPT}" \
  --max_steps 5 \
  --batch_size 8 \
  --micro_batch_size 4 \
  --prior_type mix_scm \
  --prior_device cpu \
  --min_features 2 \
  --max_features 32 \
  --max_classes 10 \
  --max_seq_len "$((N_SUPPORT + N_QUERY))" \
  --min_train_size "${N_SUPPORT}" \
  --max_train_size "$((N_SUPPORT + 1))"
