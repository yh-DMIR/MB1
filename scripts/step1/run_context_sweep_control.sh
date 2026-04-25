#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
TABICL_VERSION="${TABICL_VERSION:-v1.0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TABICL_CKPT_V10="${TABICL_CKPT_V10:-${ROOT_DIR}/ckpt/tabicl-classifier-v1-0208.ckpt}"
TABICL_CKPT_V11="${TABICL_CKPT_V11:-${ROOT_DIR}/ckpt/tabicl-classifier-v1.1-0506.ckpt}"
MB_BIAS_INIT="${MB_BIAS_INIT:-2.0}"
MB_BIAS_TRAINABLE="${MB_BIAS_TRAINABLE:-False}"
NUM_DATASETS="${NUM_DATASETS:-30}"
TASK_TYPE="${TASK_TYPE:-classification}"

# Keep batch_size=1 for 10G GPUs.
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-${NUM_DATASETS}}"
SCM_NONLINEAR="${SCM_NONLINEAR:-False}"
SCM_NUM_CLASSES="${SCM_NUM_CLASSES:-2}"
SCM_NOISE_STD="${SCM_NOISE_STD:-0.1}"

# Five scales. Context grows, and feature complexity grows with it.
# The last two levels are intentionally aggressive and may need patience or smaller overrides on 10G GPUs.
CTX500_SUPPORT="${CTX500_SUPPORT:-384}"
CTX500_QUERY="${CTX500_QUERY:-128}"
CTX500_FEATURES="${CTX500_FEATURES:-48}"
CTX500_MB_SIZE="${CTX500_MB_SIZE:-12}"
CTX500_NOISE_DIM="${CTX500_NOISE_DIM:-12}"
CTX500_REDUNDANT_DIM="${CTX500_REDUNDANT_DIM:-6}"

CTX1000_SUPPORT="${CTX1000_SUPPORT:-896}"
CTX1000_QUERY="${CTX1000_QUERY:-128}"
CTX1000_FEATURES="${CTX1000_FEATURES:-64}"
CTX1000_MB_SIZE="${CTX1000_MB_SIZE:-16}"
CTX1000_NOISE_DIM="${CTX1000_NOISE_DIM:-16}"
CTX1000_REDUNDANT_DIM="${CTX1000_REDUNDANT_DIM:-8}"

CTX3000_SUPPORT="${CTX3000_SUPPORT:-2816}"
CTX3000_QUERY="${CTX3000_QUERY:-256}"
CTX3000_FEATURES="${CTX3000_FEATURES:-80}"
CTX3000_MB_SIZE="${CTX3000_MB_SIZE:-20}"
CTX3000_NOISE_DIM="${CTX3000_NOISE_DIM:-20}"
CTX3000_REDUNDANT_DIM="${CTX3000_REDUNDANT_DIM:-10}"

CTX4000_SUPPORT="${CTX4000_SUPPORT:-3840}"
CTX4000_QUERY="${CTX4000_QUERY:-256}"
CTX4000_FEATURES="${CTX4000_FEATURES:-96}"
CTX4000_MB_SIZE="${CTX4000_MB_SIZE:-24}"
CTX4000_NOISE_DIM="${CTX4000_NOISE_DIM:-24}"
CTX4000_REDUNDANT_DIM="${CTX4000_REDUNDANT_DIM:-12}"

CTX5000_SUPPORT="${CTX5000_SUPPORT:-4864}"
CTX5000_QUERY="${CTX5000_QUERY:-256}"
CTX5000_FEATURES="${CTX5000_FEATURES:-112}"
CTX5000_MB_SIZE="${CTX5000_MB_SIZE:-28}"
CTX5000_NOISE_DIM="${CTX5000_NOISE_DIM:-28}"
CTX5000_REDUNDANT_DIM="${CTX5000_REDUNDANT_DIM:-14}"

RUN_LEVELS="${RUN_LEVELS:-ctx500 ctx1000 ctx3000 ctx4000 ctx5000}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/step1/context_sweep/mb_bias_init_${MB_BIAS_INIT}}"

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  TABICL_CKPT="${TABICL_CKPT:-${CHECKPOINT_PATH}}"
elif [[ -n "${TABICL_CKPT:-}" ]]; then
  TABICL_CKPT="${TABICL_CKPT}"
elif [[ "${TABICL_VERSION}" == "v1.1" ]]; then
  TABICL_CKPT="${TABICL_CKPT_V11}"
else
  TABICL_CKPT="${TABICL_CKPT_V10}"
fi

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${TABICL_CKPT}" ]]; then
  echo "ERROR: TabICL checkpoint not found: ${TABICL_CKPT}"
  echo "Run: bash ${ROOT_DIR}/scripts/step1/download_tabicl_ckpts.sh"
  echo "Or set TABICL_CKPT=/path/to/your/checkpoint.ckpt"
  exit 1
fi

run_context() {
  local CONTEXT_NAME="$1"
  local N_SUPPORT="$2"
  local N_QUERY="$3"
  local SCM_NUM_FEATURES="$4"
  local SCM_MB_SIZE="$5"
  local SCM_NOISE_DIM="$6"
  local SCM_REDUNDANT_DIM="$7"
  local TOTAL_CONTEXT="$((N_SUPPORT + N_QUERY))"
  local CONTEXT_OUT="${OUTPUT_DIR}/${CONTEXT_NAME}"

  mkdir -p "${CONTEXT_OUT}"

  echo "############################################################"
  echo "Running context=${CONTEXT_NAME}"
  echo "support=${N_SUPPORT} query=${N_QUERY} total_context=${TOTAL_CONTEXT}"
  echo "features=${SCM_NUM_FEATURES} mb_size=${SCM_MB_SIZE} noise_dim=${SCM_NOISE_DIM} redundant_dim=${SCM_REDUNDANT_DIM}"
  echo "tabicl_ckpt=${TABICL_CKPT}"
  echo "num_datasets=${NUM_DATASETS}"
  echo "mb_bias_init=${MB_BIAS_INIT} mb_bias_trainable=${MB_BIAS_TRAINABLE}"
  if [[ "${TOTAL_CONTEXT}" -ge 3000 ]]; then
    echo "note=${CONTEXT_NAME} is memory-intensive on a 10G GPU"
  fi
  echo "############################################################"

  OUTPUT_DIR="${CONTEXT_OUT}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
  TABICL_CKPT="${TABICL_CKPT}" \
  MB_BIAS_INIT="${MB_BIAS_INIT}" \
  MB_BIAS_TRAINABLE="${MB_BIAS_TRAINABLE}" \
  NUM_DATASETS="${NUM_DATASETS}" \
  TASK_TYPE="${TASK_TYPE}" \
  N_SUPPORT="${N_SUPPORT}" \
  N_QUERY="${N_QUERY}" \
  SCM_NUM_FEATURES="${SCM_NUM_FEATURES}" \
  SCM_MB_SIZE="${SCM_MB_SIZE}" \
  SCM_NUM_SAMPLES="${TOTAL_CONTEXT}" \
  SCM_NOISE_DIM="${SCM_NOISE_DIM}" \
  SCM_REDUNDANT_DIM="${SCM_REDUNDANT_DIM}" \
  SCM_NONLINEAR="${SCM_NONLINEAR}" \
  SCM_NUM_CLASSES="${SCM_NUM_CLASSES}" \
  SCM_NOISE_STD="${SCM_NOISE_STD}" \
  MAX_STEPS="${MAX_STEPS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  bash "${ROOT_DIR}/scripts/step1/run_same_seed_control.sh"
}

for LEVEL in ${RUN_LEVELS}; do
  case "${LEVEL}" in
    ctx500)
      run_context ctx500 \
        "${CTX500_SUPPORT}" "${CTX500_QUERY}" \
        "${CTX500_FEATURES}" "${CTX500_MB_SIZE}" "${CTX500_NOISE_DIM}" "${CTX500_REDUNDANT_DIM}"
      ;;
    ctx1000)
      run_context ctx1000 \
        "${CTX1000_SUPPORT}" "${CTX1000_QUERY}" \
        "${CTX1000_FEATURES}" "${CTX1000_MB_SIZE}" "${CTX1000_NOISE_DIM}" "${CTX1000_REDUNDANT_DIM}"
      ;;
    ctx3000)
      run_context ctx3000 \
        "${CTX3000_SUPPORT}" "${CTX3000_QUERY}" \
        "${CTX3000_FEATURES}" "${CTX3000_MB_SIZE}" "${CTX3000_NOISE_DIM}" "${CTX3000_REDUNDANT_DIM}"
      ;;
    ctx4000)
      run_context ctx4000 \
        "${CTX4000_SUPPORT}" "${CTX4000_QUERY}" \
        "${CTX4000_FEATURES}" "${CTX4000_MB_SIZE}" "${CTX4000_NOISE_DIM}" "${CTX4000_REDUNDANT_DIM}"
      ;;
    ctx5000)
      run_context ctx5000 \
        "${CTX5000_SUPPORT}" "${CTX5000_QUERY}" \
        "${CTX5000_FEATURES}" "${CTX5000_MB_SIZE}" "${CTX5000_NOISE_DIM}" "${CTX5000_REDUNDANT_DIM}"
      ;;
    *)
      echo "ERROR: unsupported RUN_LEVELS entry: ${LEVEL}"
      echo "Supported values: ctx500 ctx1000 ctx3000 ctx4000 ctx5000"
      exit 1
      ;;
  esac
done

python "${ROOT_DIR}/src/tabicl/train/aggregate_step1_results.py" \
  --root_dir "${OUTPUT_DIR}" \
  --mode scale_sweep
