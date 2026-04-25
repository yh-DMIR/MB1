#!/usr/bin/env bash

# 下载官方 TabICL 分类 checkpoint:
# - v1.0: tabicl-classifier-v1-0208.ckpt
# - v1.1: tabicl-classifier-v1.1-0506.ckpt
#
# 默认保存到仓库根目录的 ckpt/ 下。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/ckpt}"

V10_URL="https://huggingface.co/jingang/TabICL-clf/resolve/main/tabicl-classifier-v1-0208.ckpt"
V11_URL="https://huggingface.co/jingang/TabICL-clf/resolve/main/tabicl-classifier-v1.1-0506.ckpt"

mkdir -p "${CKPT_DIR}"

download_one() {
  local URL="$1"
  local OUT="$2"

  echo "Downloading -> ${OUT}"

  if command -v wget >/dev/null 2>&1; then
    wget -O "${OUT}" "${URL}"
    return
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -L "${URL}" -o "${OUT}"
    return
  fi

  echo "ERROR: neither wget nor curl is available."
  exit 1
}

download_one "${V10_URL}" "${CKPT_DIR}/tabicl-classifier-v1-0208.ckpt"
download_one "${V11_URL}" "${CKPT_DIR}/tabicl-classifier-v1.1-0506.ckpt"

echo
echo "Downloaded checkpoints:"
ls -lh "${CKPT_DIR}"/tabicl-classifier-v1-0208.ckpt "${CKPT_DIR}"/tabicl-classifier-v1.1-0506.ckpt

