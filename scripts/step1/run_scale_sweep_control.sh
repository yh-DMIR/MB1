#!/usr/bin/env bash

# 在同一随机种子设定下，对不同规模 synthetic SCM 任务分别做对照。
#
# 默认划分为 small / medium / large 三档。
# 每一档都会调用 run_same_seed_control.sh，保证：
# 1. 同一档内部 baseline/oracle/shuffled/corr 等条件使用同一套参数
# 2. 每一档内部共用同一个 SEED
# 3. 各档规模可以分别比较

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/step1/test1/scale_sweep}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TABICL_VERSION="${TABICL_VERSION:-v1.0}"
TABICL_CKPT_V10="${TABICL_CKPT_V10:-${ROOT_DIR}/ckpt/tabicl-classifier-v1-0208.ckpt}"
TABICL_CKPT_V11="${TABICL_CKPT_V11:-${ROOT_DIR}/ckpt/tabicl-classifier-v1.1-0506.ckpt}"

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  TABICL_CKPT="${TABICL_CKPT:-${CHECKPOINT_PATH}}"
elif [[ -n "${TABICL_CKPT:-}" ]]; then
  TABICL_CKPT="${TABICL_CKPT}"
elif [[ "${TABICL_VERSION}" == "v1.1" ]]; then
  TABICL_CKPT="${TABICL_CKPT_V11}"
else
  TABICL_CKPT="${TABICL_CKPT_V10}"
fi

MB_PREDICTOR_CKPT="${MB_PREDICTOR_CKPT:-}"
TASK_TYPE="${TASK_TYPE:-classification}"
MAX_STEPS="${MAX_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SCM_NONLINEAR="${SCM_NONLINEAR:-False}"
SCM_NUM_CLASSES="${SCM_NUM_CLASSES:-2}"
SCM_NOISE_STD="${SCM_NOISE_STD:-0.1}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${TABICL_CKPT}" ]]; then
  echo "ERROR: TabICL checkpoint not found: ${TABICL_CKPT}"
  echo "Run: bash ${ROOT_DIR}/scripts/step1/download_tabicl_ckpts.sh"
  echo "Or set TABICL_CKPT=/path/to/your/checkpoint.ckpt"
  exit 1
fi

run_scale() {
  local SCALE_NAME="$1"
  local N_SUPPORT="$2"
  local N_QUERY="$3"
  local SCM_NUM_FEATURES="$4"
  local SCM_MB_SIZE="$5"
  local SCM_NOISE_DIM="$6"
  local SCM_REDUNDANT_DIM="$7"

  local SCALE_OUT="${OUTPUT_DIR}/${SCALE_NAME}"
  mkdir -p "${SCALE_OUT}"

  echo "############################################################"
  echo "Running scale=${SCALE_NAME}"
  echo "support=${N_SUPPORT} query=${N_QUERY} features=${SCM_NUM_FEATURES} mb_size=${SCM_MB_SIZE}"
  echo "tabicl_ckpt=${TABICL_CKPT}"
  echo "############################################################"

  OUTPUT_DIR="${SCALE_OUT}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
  TABICL_CKPT="${TABICL_CKPT}" \
  MB_PREDICTOR_CKPT="${MB_PREDICTOR_CKPT}" \
  TASK_TYPE="${TASK_TYPE}" \
  N_SUPPORT="${N_SUPPORT}" \
  N_QUERY="${N_QUERY}" \
  SCM_NUM_FEATURES="${SCM_NUM_FEATURES}" \
  SCM_MB_SIZE="${SCM_MB_SIZE}" \
  SCM_NUM_SAMPLES="$((N_SUPPORT + N_QUERY))" \
  SCM_NOISE_DIM="${SCM_NOISE_DIM}" \
  SCM_REDUNDANT_DIM="${SCM_REDUNDANT_DIM}" \
  SCM_NONLINEAR="${SCM_NONLINEAR}" \
  SCM_NUM_CLASSES="${SCM_NUM_CLASSES}" \
  SCM_NOISE_STD="${SCM_NOISE_STD}" \
  MAX_STEPS="${MAX_STEPS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  bash "${ROOT_DIR}/scripts/step1/run_same_seed_control.sh"
}

# small
run_scale small 32 32 16 4 4 2

# medium
run_scale medium 64 64 32 6 8 4

# large
run_scale large 128 128 64 10 16 8
