#!/usr/bin/env bash
# Recreate both venvs on the pod's local disk after every (re)start.
#
# Why local: /workspace is a network FUSE volume; importing torch/vLLM from it took
# ~110 s vs ~22 s from local disk. The uv cache lives on /workspace, so rebuilding
# is fast (~40 s per venv) and survives pod stops. Code, models, results stay on /workspace.
#
# Usage: bash scripts/pod_setup.sh          (idempotent; skips venvs that already exist)
set -euo pipefail

REPO=/workspace/nano-vllm-serve
export UV_CACHE_DIR=/workspace/.uv-cache
export HF_HOME=/workspace/hf-models/.hf-home

TORCH_VERSION=2.8.0
FLASH_ATTN_WHEEL=/workspace/wheels/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
FLASH_ATTN_SHA256=61ab39b7038d6a05cb97664f3481a92ced300d11aa14529ae3b675300a99f5e6
VLLM_VERSION=0.33.0

mkdir -p ~/huggingface && ln -sfn /workspace/hf-models/Qwen3-0.6B ~/huggingface/Qwen3-0.6B

# venv A: this repo (nano-vllm) + flash-attn prebuilt wheel
if [ ! -x /root/.venv/bin/python ]; then
  if [ ! -f "$FLASH_ATTN_WHEEL" ]; then
    mkdir -p /workspace/wheels
    curl -sSL -o "$FLASH_ATTN_WHEEL" "$FLASH_ATTN_URL"
  fi
  echo "$FLASH_ATTN_SHA256  $FLASH_ATTN_WHEEL" | sha256sum -c -
  uv venv -q --python 3.12 /root/.venv
  uv pip install -q --link-mode=copy --python /root/.venv/bin/python \
    "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --link-mode=copy --python /root/.venv/bin/python "$FLASH_ATTN_WHEEL"
fi
# Always (cheap when satisfied): picks up dependencies added to pyproject.toml since the last run.
uv pip install -q --link-mode=copy --python /root/.venv/bin/python -e "$REPO[test]" "huggingface_hub[cli]" py-spy

# venv B: vLLM (pins its own torch, so it is kept separate)
if [ ! -x /root/.venv-vllm/bin/python ]; then
  uv venv -q --python 3.12 /root/.venv-vllm
  uv pip install -q --link-mode=copy --python /root/.venv-vllm/bin/python "vllm==$VLLM_VERSION"
  # vLLM's default sampler JIT-compiles FlashInfer kernels, which needs nvcc + ninja
  # (absent in the RunPod image). Install FlashInfer's precompiled kernels instead,
  # so vLLM runs with its default configuration.
  /root/.venv-vllm/bin/vllm download-kernels
fi

/root/.venv/bin/python -c "import torch, flash_attn, nanovllm; print('venv A ok: torch', torch.__version__, 'flash_attn', flash_attn.__version__)"
/root/.venv-vllm/bin/python -c "import vllm; print('venv B ok: vllm', vllm.__version__)"
