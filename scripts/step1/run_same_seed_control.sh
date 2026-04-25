#!/usr/bin/env bash

# 在同一套 synthetic SCM 任务配置、同一随机种子下，
# 对比 baseline / oracle / shuffled / corr / random / mi。
#
# 说明：
# 1. 所有条件都调用同一个 runner: src/tabicl/train/run_mb_experiment.py
# 2. 所有条件共享完全相同的 synthetic SCM 参数和 SEED
# 3. baseline 也走同一数据生成器，只是关闭 MB 注入
# 4. 这样可以避免“baseline 用的是另一套数据分布”带来的比较偏差

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

DATASET="${DATASET:-synthetic_scm_mb}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/step1/test1/mb_bias_init_1.0/same_seed_control}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TABICL_VERSION="${TABICL_VERSION:-v1.0}"
TABICL_CKPT_V10="${TABICL_CKPT_V10:-${ROOT_DIR}/ckpt/tabicl-classifier-v1-0208.ckpt}"
TABICL_CKPT_V11="${TABICL_CKPT_V11:-${ROOT_DIR}/ckpt/tabicl-classifier-v1.1-0506.ckpt}"
MB_BIAS_INIT="${MB_BIAS_INIT:-1.0}"
MB_BIAS_TRAINABLE="${MB_BIAS_TRAINABLE:-False}"

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
N_SUPPORT="${N_SUPPORT:-64}"
N_QUERY="${N_QUERY:-64}"

# synthetic SCM 参数
SCM_NUM_FEATURES="${SCM_NUM_FEATURES:-32}"
SCM_MB_SIZE="${SCM_MB_SIZE:-6}"
SCM_NUM_SAMPLES="${SCM_NUM_SAMPLES:-$((N_SUPPORT + N_QUERY))}"
SCM_NOISE_DIM="${SCM_NOISE_DIM:-8}"
SCM_REDUNDANT_DIM="${SCM_REDUNDANT_DIM:-4}"
SCM_NONLINEAR="${SCM_NONLINEAR:-False}"
SCM_NUM_CLASSES="${SCM_NUM_CLASSES:-2}"
SCM_NOISE_STD="${SCM_NOISE_STD:-0.1}"

# 运行步数
NUM_DATASETS="${NUM_DATASETS:-30}"
MAX_STEPS="${MAX_STEPS:-${NUM_DATASETS}}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${ROOT_DIR}/ckpt"

if [[ ! -f "${TABICL_CKPT}" ]]; then
  echo "ERROR: TabICL checkpoint not found: ${TABICL_CKPT}"
  echo "Run: bash ${ROOT_DIR}/scripts/step1/download_tabicl_ckpts.sh"
  echo "Or set TABICL_CKPT=/path/to/your/checkpoint.ckpt"
  exit 1
fi

run_case() {
  local CASE_NAME="$1"
  local MB_SOURCE="$2"
  local MB_INJECTION="$3"

  local CASE_OUT="${OUTPUT_DIR}/${CASE_NAME}"
  mkdir -p "${CASE_OUT}"

  echo "============================================================"
  echo "Running case=${CASE_NAME}"
  echo "seed=${SEED} features=${SCM_NUM_FEATURES} samples=${SCM_NUM_SAMPLES} support=${N_SUPPORT} query=${N_QUERY}"
  echo "tabicl_ckpt=${TABICL_CKPT}"
  echo "num_datasets=${NUM_DATASETS}"
  echo "mb_bias_init=${MB_BIAS_INIT} mb_bias_trainable=${MB_BIAS_TRAINABLE}"
  echo "mb_score_source=${MB_SOURCE} mb_injection=${MB_INJECTION}"
  echo "output_dir=${CASE_OUT}"
  echo "============================================================"

  python "${ROOT_DIR}/src/tabicl/train/run_mb_experiment.py" \
    --device "${DEVICE}" \
    --task_type "${TASK_TYPE}" \
    --scm_task_type "${TASK_TYPE}" \
    --scm_seed "${SEED}" \
    --tabicl_checkpoint_path "${TABICL_CKPT}" \
    --mb_predictor_checkpoint_path "${MB_PREDICTOR_CKPT}" \
    --checkpoint_dir "${CASE_OUT}" \
    --batch_size "${BATCH_SIZE}" \
    --max_steps "${MAX_STEPS}" \
    --eval_num_datasets "${NUM_DATASETS}" \
    --n_support "${N_SUPPORT}" \
    --n_query "${N_QUERY}" \
    --scm_num_features "${SCM_NUM_FEATURES}" \
    --scm_mb_size "${SCM_MB_SIZE}" \
    --scm_num_samples "${SCM_NUM_SAMPLES}" \
    --scm_noise_dim "${SCM_NOISE_DIM}" \
    --scm_redundant_dim "${SCM_REDUNDANT_DIM}" \
    --scm_nonlinear "${SCM_NONLINEAR}" \
    --scm_num_classes "${SCM_NUM_CLASSES}" \
    --scm_noise_std "${SCM_NOISE_STD}" \
    --mb_bias_init "${MB_BIAS_INIT}" \
    --mb_bias_trainable "${MB_BIAS_TRAINABLE}" \
    --mb_score_source "${MB_SOURCE}" \
    --mb_injection "${MB_INJECTION}"
}

# baseline：同一 synthetic task，但不使用 MB
run_case baseline none none

# 真实 MB 标签注入
run_case oracle oracle cls_soft_bias

# 负对照
run_case shuffled shuffled cls_soft_bias

# 简单统计 proxy
run_case corr corr cls_soft_bias
run_case mi mi cls_soft_bias
run_case random random cls_soft_bias

python "${ROOT_DIR}/src/tabicl/train/aggregate_step1_results.py" \
  --root_dir "${OUTPUT_DIR}" \
  --mode same_seed
