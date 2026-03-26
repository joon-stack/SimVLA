#!/bin/bash
set -euo pipefail

BASE_DIR="${1:-/shared/s2/lab01/youngjoonjeong}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_DIR="${SIMVLA_VENV_DIR:-$BASE_DIR/venvs/simvla}"
CACHE_ROOT="${SIMVLA_CACHE_ROOT:-$BASE_DIR/.cache}"
HF_HOME_DIR="${SIMVLA_HF_HOME:-$CACHE_ROOT/huggingface}"
HF_HUB_CACHE_DIR="${SIMVLA_HUGGINGFACE_HUB_CACHE:-$HF_HOME_DIR/hub}"
TORCH_HOME_DIR="${SIMVLA_TORCH_HOME:-$CACHE_ROOT/torch}"
UV_CACHE_DIR_VALUE="${SIMVLA_UV_CACHE_DIR:-$CACHE_ROOT/uv}"
WANDB_DIR_VALUE="${SIMVLA_WANDB_DIR:-$BASE_DIR/wandb}"
RUNS_DIR="${SIMVLA_RUNS_DIR:-$BASE_DIR/runs}"
ARTIFACTS_DIR="${SIMVLA_ARTIFACTS_DIR:-$BASE_DIR/artifacts}"
ENV_FILE="${SIMVLA_ENV_FILE:-$BASE_DIR/venvs/simvla_env.sh}"

PYTHON_VERSION="${SIMVLA_PYTHON_VERSION:-3.10}"
TORCH_SPEC="${SIMVLA_TORCH_SPEC:-torch==2.6.0+cu124}"
TORCHVISION_SPEC="${SIMVLA_TORCHVISION_SPEC:-torchvision==0.21.0+cu124}"
FLASH_ATTN_SPEC="${SIMVLA_FLASH_ATTN_SPEC:-flash-attn==2.5.6}"
SKIP_FLASH_ATTN="${SIMVLA_SKIP_FLASH_ATTN:-0}"

MODEL_REPO_ID="${SIMVLA_MODEL_REPO_ID:-HuggingFaceTB/SmolVLM-500M-Instruct}"
DATASET_REPO_ID="${SIMVLA_DATASET_REPO_ID:-HuggingFaceVLA/libero}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is not installed or not on PATH." >&2
  exit 1
}

mkdir -p \
  "$BASE_DIR/venvs" \
  "$CACHE_ROOT" \
  "$HF_HOME_DIR" \
  "$HF_HUB_CACHE_DIR" \
  "$TORCH_HOME_DIR" \
  "$UV_CACHE_DIR_VALUE" \
  "$WANDB_DIR_VALUE" \
  "$RUNS_DIR" \
  "$ARTIFACTS_DIR"

export XDG_CACHE_HOME="$CACHE_ROOT"
export UV_CACHE_DIR="$UV_CACHE_DIR_VALUE"
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE_DIR"
export TORCH_HOME="$TORCH_HOME_DIR"
export WANDB_DIR="$WANDB_DIR_VALUE"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

uv pip install --index-url https://download.pytorch.org/whl/cu124 \
  "$TORCH_SPEC" "$TORCHVISION_SPEC"

uv pip install \
  'transformers>=4.57.0' \
  peft \
  accelerate \
  fastapi \
  tensorboard \
  uvicorn \
  json_numpy \
  safetensors \
  scipy \
  einops \
  timm \
  mmengine \
  pyarrow \
  h5py \
  mediapy \
  num2words \
  av \
  wandb \
  websockets \
  msgpack_numpy \
  huggingface_hub \
  omegaconf \
  opencv-python \
  pillow

if [ "$SKIP_FLASH_ATTN" != "1" ]; then
  uv pip install --no-build-isolation "$FLASH_ATTN_SPEC"
fi

export SIMVLA_SETUP_MODEL_REPO_ID="$MODEL_REPO_ID"
export SIMVLA_SETUP_DATASET_REPO_ID="$DATASET_REPO_ID"
mapfile -t DOWNLOAD_PATHS < <(python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_root = snapshot_download(repo_id=os.environ["SIMVLA_SETUP_MODEL_REPO_ID"])
dataset_root = snapshot_download(
    repo_id=os.environ["SIMVLA_SETUP_DATASET_REPO_ID"],
    repo_type="dataset",
    allow_patterns=["meta/*", "meta/**/*", "data/*/*.parquet"],
)
print(model_root)
print(dataset_root)
PY
)

MODEL_SNAPSHOT="${DOWNLOAD_PATHS[0]}"
DATASET_SNAPSHOT="${DOWNLOAD_PATHS[1]}"

mkdir -p "$(dirname "$ENV_FILE")"
cat > "$ENV_FILE" <<ENVEOF
#!/bin/bash
export XDG_CACHE_HOME="$CACHE_ROOT"
export UV_CACHE_DIR="$UV_CACHE_DIR_VALUE"
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE_DIR"
export TORCH_HOME="$TORCH_HOME_DIR"
export WANDB_DIR="$WANDB_DIR_VALUE"
export SIMVLA_REPO_ROOT="$REPO_ROOT"
export SIMVLA_VENV_DIR="$VENV_DIR"
export SIMVLA_LEROBOT_ROOT="$DATASET_SNAPSHOT"
export SIMVLA_SMOLVLM_SNAPSHOT="$MODEL_SNAPSHOT"
source "$VENV_DIR/bin/activate"
ENVEOF
chmod +x "$ENV_FILE"

cat <<MSG
============================================================
SimVLA shared setup complete.
============================================================
Base dir:                $BASE_DIR
Repo root:               $REPO_ROOT
Virtual env:             $VENV_DIR
HF cache:                $HF_HOME_DIR
Torch cache:             $TORCH_HOME_DIR
WandB dir:               $WANDB_DIR_VALUE
Runs dir:                $RUNS_DIR
Artifacts dir:           $ARTIFACTS_DIR
SmolVLM snapshot:        $MODEL_SNAPSHOT
LeRobot LIBERO snapshot: $DATASET_SNAPSHOT
Env file:                $ENV_FILE
============================================================
Next time, just run:
source "$ENV_FILE"
cd "$REPO_ROOT"
============================================================
MSG
