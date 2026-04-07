#!/bin/bash
set -euo pipefail

# Example:
#   export SIMVLA_WANDB_API_KEY=...
#   export SIMVLA_LATENT_TEACHER_REPO_ROOT=/path/to/latent_action
#   export SIMVLA_LATENT_TEACHER_CONFIG=/path/to/hydra_resolved.yaml
#   export SIMVLA_LATENT_TEACHER_CHECKPOINT=/path/to/checkpoint_latest.pt
#   export SIMVLA_LATENT_TRAINING_STAGE=latent_only
#   bash train_smolvlm_small_lerobot_seqfm_staged.sh
#
# Stage 3:
#   export SIMVLA_LATENT_TRAINING_STAGE=action_only
#   export SIMVLA_INIT_CKPT=/path/to/stage2_latent_ckpt
#   bash train_smolvlm_small_lerobot_seqfm_staged.sh

export WANDB_PROJECT="${WANDB_PROJECT:-simvla-libero}"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ -n "${SIMVLA_WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY="${SIMVLA_WANDB_API_KEY}"
fi

export SIMVLA_LATENT_MODE=sequential_fm
export SIMVLA_LATENT_TRAINING_STAGE="${SIMVLA_LATENT_TRAINING_STAGE:-joint}"
export SIMVLA_LATENT_LOSS_WEIGHT="${SIMVLA_LATENT_LOSS_WEIGHT:-1.0}"
export SIMVLA_LATENT_STRIDE_K="${SIMVLA_LATENT_STRIDE_K:-20}"
export SIMVLA_LATENT_SAMPLE_STEPS="${SIMVLA_LATENT_SAMPLE_STEPS:-1}"
export SIMVLA_LATENT_TEACHER_TARGET="${SIMVLA_LATENT_TEACHER_TARGET:-z_t_tokens_raw}"

BATCH_SIZE=${1:-64}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./runs/simvla_libero_small_lerobot_seqfm_staged/${SIMVLA_LATENT_TRAINING_STAGE}}
RESUME_CKPT=${4:-""}
TASK_SUITE_NAME=${5:-""}
CAMERA_MODE=${6:-dual}
GRAD_ACCUM_STEPS=${7:-4}
INIT_CKPT=${SIMVLA_INIT_CKPT:-""}

echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   init_ckpt: ${INIT_CKPT:-'None'}"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"
echo "   task_suite_name: ${TASK_SUITE_NAME:-'all'}"
echo "   camera_mode: ${CAMERA_MODE}"
echo "   grad_accum_steps: ${GRAD_ACCUM_STEPS}"

GPU_DEVICES=${SIMVLA_CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${SIMVLA_NUM_PROCESSES:-1}
MAIN_PROCESS_PORT=${SIMVLA_MAIN_PROCESS_PORT:-29504}
MIXED_PRECISION=${SIMVLA_MIXED_PRECISION:-bf16}
EFFECTIVE_GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_PROCESSES))

export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}
export TF_CPP_MIN_LOG_LEVEL=2

DATASET_ROOT=${SIMVLA_LEROBOT_ROOT:-""}
DATASET_REPO_ID=${SIMVLA_LEROBOT_REPO_ID:-HuggingFaceVLA/libero}
SMOLVLM_MODEL=${SIMVLA_SMOLVLM_SNAPSHOT:-HuggingFaceTB/SmolVLM-500M-Instruct}

LEARNING_RATE=1e-4
NUM_ACTIONS=10
ITERS=200000
WARMUP_STEPS=0
FREEZE_STEPS=1000
SAVE_INTERVAL=5000
LOG_INTERVAL=20
NUM_WORKERS=4
MAX_GRAD_NORM=1.0

HIDDEN_SIZE=768
DEPTH=12
NUM_HEADS=12
USE_ADALN=false

ARGS="--output_dir ${OUTPUT_DIR} \
    --dataset_backend lerobot_hf \
    --dataset_repo_id ${DATASET_REPO_ID} \
    --camera_mode ${CAMERA_MODE} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode libero_joint \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --num_actions ${NUM_ACTIONS} \
    --iters ${ITERS} \
    --warmup_steps ${WARMUP_STEPS} \
    --freeze_steps ${FREEZE_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --image_size 384 \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --latent_mode ${SIMVLA_LATENT_MODE} \
    --latent_training_stage ${SIMVLA_LATENT_TRAINING_STAGE} \
    --latent_loss_weight ${SIMVLA_LATENT_LOSS_WEIGHT} \
    --latent_stride_k ${SIMVLA_LATENT_STRIDE_K} \
    --latent_sample_steps ${SIMVLA_LATENT_SAMPLE_STEPS}"

if [ -n "${DATASET_ROOT}" ]; then
    ARGS="${ARGS} --dataset_root ${DATASET_ROOT}"
fi

if [ "${SIMVLA_LOCAL_FILES_ONLY:-0}" = "1" ]; then
    ARGS="${ARGS} --local_files_only"
fi

if [ -n "${TASK_SUITE_NAME}" ]; then
    ARGS="${ARGS} --task_suite_name ${TASK_SUITE_NAME}"
fi

if [ "${USE_ADALN}" = true ]; then
    ARGS="${ARGS} --use_adaln"
fi

if [ -n "${INIT_CKPT}" ] && [ -n "${RESUME_CKPT}" ]; then
    echo "SIMVLA_INIT_CKPT and RESUME_CKPT cannot both be set."
    exit 1
fi

if [ -n "${INIT_CKPT}" ]; then
    ARGS="${ARGS} --models ${INIT_CKPT}"
    echo "Initializing from ${INIT_CKPT}"
fi

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

if [ "${SIMVLA_LATENT_TRAINING_STAGE}" = "action_only" ]; then
    if [ -z "${INIT_CKPT}" ] && [ -z "${RESUME_CKPT}" ]; then
        echo "action_only stage requires SIMVLA_INIT_CKPT or RESUME_CKPT."
        exit 1
    fi
else
    : "${SIMVLA_LATENT_TEACHER_REPO_ROOT:?SIMVLA_LATENT_TEACHER_REPO_ROOT is required}"
    : "${SIMVLA_LATENT_TEACHER_CONFIG:?SIMVLA_LATENT_TEACHER_CONFIG is required}"
    : "${SIMVLA_LATENT_TEACHER_CHECKPOINT:?SIMVLA_LATENT_TEACHER_CHECKPOINT is required}"

    ARGS="${ARGS} --latent_teacher_repo_root ${SIMVLA_LATENT_TEACHER_REPO_ROOT}"
    ARGS="${ARGS} --latent_teacher_config ${SIMVLA_LATENT_TEACHER_CONFIG}"
    ARGS="${ARGS} --latent_teacher_checkpoint ${SIMVLA_LATENT_TEACHER_CHECKPOINT}"
    ARGS="${ARGS} --latent_teacher_target ${SIMVLA_LATENT_TEACHER_TARGET}"
fi

if [ -n "${SIMVLA_LATENT_TEACHER_OBS_KEY:-}" ]; then
    ARGS="${ARGS} --latent_teacher_obs_key ${SIMVLA_LATENT_TEACHER_OBS_KEY}"
fi

echo "============================================================"
echo "Starting SimVLA Training on LeRobot LIBERO (Small, Stagewise SeqFM)"
echo "============================================================"
echo "Dataset repo: ${DATASET_REPO_ID}"
echo "Dataset root: ${DATASET_ROOT:-'auto'}"
echo "Task suite: ${TASK_SUITE_NAME:-'all'}"
echo "Camera mode: ${CAMERA_MODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "num_processes: ${NUM_PROCESSES}"
echo "grad_accumulation_steps: ${GRAD_ACCUM_STEPS}"
echo "effective_global_batch_size: ${EFFECTIVE_GLOBAL_BATCH_SIZE}"
echo "mixed_precision: ${MIXED_PRECISION}"
echo "latent_mode: ${SIMVLA_LATENT_MODE}"
echo "latent_training_stage: ${SIMVLA_LATENT_TRAINING_STAGE}"
echo "latent_stride_k: ${SIMVLA_LATENT_STRIDE_K}"
echo "latent_sample_steps: ${SIMVLA_LATENT_SAMPLE_STEPS}"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port ${MAIN_PROCESS_PORT} \
    --mixed_precision ${MIXED_PRECISION} \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
