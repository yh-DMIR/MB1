#!/usr/bin/env bash

# 训练 MultiViewMBPredictor。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

DATASET="${DATASET:-synthetic_scm_mb}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/mb_predictor_train}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TABICL_CKPT="${TABICL_CKPT:-${CHECKPOINT_PATH}}"
MB_PREDICTOR_CKPT="${MB_PREDICTOR_CKPT:-}"
TASK_TYPE="${TASK_TYPE:-classification}"
N_SUPPORT="${N_SUPPORT:-64}"
N_QUERY="${N_QUERY:-64}"

mkdir -p "${OUTPUT_DIR}"

python "${ROOT_DIR}/src/tabicl/train/train_mb_predictor.py" \
  --device "${DEVICE}" \
  --task_type "${TASK_TYPE}" \
  --scm_task_type "${TASK_TYPE}" \
  --scm_seed "${SEED}" \
  --tabicl_checkpoint_path "${TABICL_CKPT}" \
  --mb_predictor_checkpoint_path "${MB_PREDICTOR_CKPT}" \
  --checkpoint_dir "${OUTPUT_DIR}" \
  --batch_size 4 \
  --max_steps 10 \
  --save_temp_every 2 \
  --n_support "${N_SUPPORT}" \
  --n_query "${N_QUERY}" \
  --scm_num_samples "$((N_SUPPORT + N_QUERY))" \
  --train_mode mb_predictor
