#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

# Context sweep is meant to isolate the effect of support-set context length,
# so feature dimensionality stays fixed across runs.
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

# Use batch_size=1 for stability on 10G GPUs.
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-${NUM_DATASETS}}"

# Fixed talent-like feature scale.
SCM_NUM_FEATURES="${SCM_NUM_FEATURES:-48}"
SCM_MB_SIZE="${SCM_MB_SIZE:-12}"
SCM_NOISE_DIM="${SCM_NOISE_DIM:-12}"
SCM_REDUNDANT_DIM="${SCM_REDUNDANT_DIM:-6}"
SCM_NONLINEAR="${SCM_NONLINEAR:-False}"
SCM_NUM_CLASSES="${SCM_NUM_CLASSES:-2}"
SCM_NOISE_STD="${SCM_NOISE_STD:-0.1}"

# Approximate total context sizes: ~500 / ~1000 / ~3000.
# Query size is kept small and fixed so the sweep mainly reflects support context.
CTX500_SUPPORT="${CTX500_SUPPORT:-256}"
CTX500_QUERY="${CTX500_QUERY:-256}"
CTX1000_SUPPORT="${CTX1000_SUPPORT:-512}"
CTX1000_QUERY="${CTX1000_QUERY:-512}"
CTX3000_SUPPORT="${CTX3000_SUPPORT:-1024}"
CTX3000_QUERY="${CTX3000_QUERY:-1024}"

RUN_LEVELS="${RUN_LEVELS:-ctx500 ctx1000 ctx3000}"
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
  local TOTAL_CONTEXT="$((N_SUPPORT + N_QUERY))"
  local CONTEXT_OUT="${OUTPUT_DIR}/${CONTEXT_NAME}"

  mkdir -p "${CONTEXT_OUT}"

  echo "############################################################"
  echo "Running context=${CONTEXT_NAME}"
  echo "support=${N_SUPPORT} query=${N_QUERY} total_context=${TOTAL_CONTEXT}"
  echo "features=${SCM_NUM_FEATURES} mb_size=${SCM_MB_SIZE}"
  echo "tabicl_ckpt=${TABICL_CKPT}"
  echo "num_datasets=${NUM_DATASETS}"
  echo "mb_bias_init=${MB_BIAS_INIT} mb_bias_trainable=${MB_BIAS_TRAINABLE}"
  if [[ "${TOTAL_CONTEXT}" -ge 3000 ]]; then
    echo "note=ctx3000 is the most memory-intensive level on a 10G GPU"
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
      run_context ctx500 "${CTX500_SUPPORT}" "${CTX500_QUERY}"
      ;;
    ctx1000)
      run_context ctx1000 "${CTX1000_SUPPORT}" "${CTX1000_QUERY}"
      ;;
    ctx3000)
      run_context ctx3000 "${CTX3000_SUPPORT}" "${CTX3000_QUERY}"
      ;;
    *)
      echo "ERROR: unsupported RUN_LEVELS entry: ${LEVEL}"
      echo "Supported values: ctx500 ctx1000 ctx3000"
      exit 1
      ;;
  esac
done

python "${ROOT_DIR}/src/tabicl/train/aggregate_step1_results.py" \
  --root_dir "${OUTPUT_DIR}" \
  --mode scale_sweep
