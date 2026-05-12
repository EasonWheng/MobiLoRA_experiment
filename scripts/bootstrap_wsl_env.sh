#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="${MOBILORA_ASSET_ROOT:-/mnt/d/MobiLoRA_assets}"
HF_CACHE_DIR="${HF_HOME:-${ASSET_ROOT}/hf_cache}"
DATASETS_CACHE_DIR="${HF_DATASETS_CACHE:-${ASSET_ROOT}/datasets}"
ADAPTERS_DIR="${MOBILORA_ADAPTER_DIR:-${ASSET_ROOT}/adapters}"
MINICONDA_ROOT="${HOME}/miniconda3"
ENV_NAME="nn-lesson-SEU"
MINICONDA_DIR_ON_D="${ASSET_ROOT}/miniconda"
MINICONDA_INSTALLER="${MINICONDA_DIR_ON_D}/Miniconda3-latest-Linux-x86_64.sh"

mkdir -p "${ASSET_ROOT}" "${HF_CACHE_DIR}" "${DATASETS_CACHE_DIR}" "${ADAPTERS_DIR}" "${MINICONDA_DIR_ON_D}" "${ASSET_ROOT}/bench_outputs"

APT_PACKAGES=()
command -v curl >/dev/null 2>&1 || APT_PACKAGES+=("curl")
command -v git >/dev/null 2>&1 || APT_PACKAGES+=("git")
command -v gcc >/dev/null 2>&1 || APT_PACKAGES+=("build-essential")

if [ "${#APT_PACKAGES[@]}" -gt 0 ]; then
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    apt-get install -y "${APT_PACKAGES[@]}"
  else
    sudo apt-get update
    sudo apt-get install -y "${APT_PACKAGES[@]}"
  fi
fi

if [ ! -f "${MINICONDA_INSTALLER}" ]; then
  curl -L "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "${MINICONDA_INSTALLER}"
fi

if [ ! -d "${MINICONDA_ROOT}" ]; then
  bash "${MINICONDA_INSTALLER}" -b -p "${MINICONDA_ROOT}"
fi

source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"

# Newer conda releases may block non-interactive env creation until the
# default Anaconda channels' Terms of Service are accepted.
if conda tos --help >/dev/null 2>&1; then
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.11
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  transformers peft accelerate bitsandbytes safetensors \
  datasets bert-score pandas matplotlib pyyaml sentencepiece huggingface_hub

grep -q "MOBILORA_ASSET_ROOT" "${HOME}/.bashrc" 2>/dev/null || cat <<EOF >> "${HOME}/.bashrc"
export MOBILORA_ASSET_ROOT="${ASSET_ROOT}"
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${DATASETS_CACHE_DIR}"
EOF

echo "Bootstrap complete."
echo "Activate with: source ${MINICONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
