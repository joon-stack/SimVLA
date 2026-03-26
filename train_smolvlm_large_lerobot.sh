#!/bin/bash
# SimVLA Training Script for LeRobot LIBERO (Large Model)

set -e

BATCH_SIZE=${1:-128}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./runs/simvla_libero_large_lerobot}
RESUME_CKPT=${4:-""}
TASK_SUITE_NAME=${5:-""}
CAMERA_MODE=${6:-dual}
GRAD_ACCUM_STEPS=${7:-2}

LATENT_AUX_ENABLED=${SIMVLA_LATENT_AUX_ENABLED:-false}
LATENT_AUX_WEIGHT=${SIMVLA_LATENT_AUX_WEIGHT:-1.0}
LATENT_TEACHER_REPO_ROOT=${SIMVLA_LATENT_TEACHER_REPO_ROOT:-""}
LATENT_TEACHER_CONFIG=${SIMVLA_LATENT_TEACHER_CONFIG:-""}
LATENT_TEACHER_CHECKPOINT=${SIMVLA_LATENT_TEACHER_CHECKPOINT:-""}
LATENT_TEACHER_FUTURE_OFFSET=${SIMVLA_LATENT_TEACHER_FUTURE_OFFSET:-""}
echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"
echo "   task_suite_name: ${TASK_SUITE_NAME:-'all'}"
echo "   camera_mode: ${CAMERA_MODE}"
echo "   grad_accum_steps: ${GRAD_ACCUM_STEPS}"

GPU_DEVICES=${SIMVLA_CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${SIMVLA_NUM_PROCESSES:-1}
MAIN_PROCESS_PORT=${SIMVLA_MAIN_PROCESS_PORT:-29505}
MIXED_PRECISION=${SIMVLA_MIXED_PRECISION:-bf16}
EFFECTIVE_GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_PROCESSES))

export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}
export TF_CPP_MIN_LOG_LEVEL=2

DATASET_ROOT=${SIMVLA_LEROBOT_ROOT:-""}
DATASET_REPO_ID=${SIMVLA_LEROBOT_REPO_ID:-HuggingFaceVLA/libero}
SMOLVLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"

LEARNING_RATE=2e-4
NUM_ACTIONS=10
ITERS=200000
WARMUP_STEPS=0
FREEZE_STEPS=1000
SAVE_INTERVAL=10000
LOG_INTERVAL=20
NUM_WORKERS=4
MAX_GRAD_NORM=1.0

HIDDEN_SIZE=1024
DEPTH=24
NUM_HEADS=16
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
    --max_grad_norm ${MAX_GRAD_NORM}"

if [ -n "${DATASET_ROOT}" ]; then
    ARGS="${ARGS} --dataset_root ${DATASET_ROOT}"
fi

if [ -n "${TASK_SUITE_NAME}" ]; then
    ARGS="${ARGS} --task_suite_name ${TASK_SUITE_NAME}"
fi

if [ "${USE_ADALN}" = true ]; then
    ARGS="${ARGS} --use_adaln"
fi

if [ "${LATENT_AUX_ENABLED}" = true ]; then
    ARGS="${ARGS} --latent_aux_enabled --latent_aux_weight ${LATENT_AUX_WEIGHT}"
    ARGS="${ARGS} --latent_teacher_repo_root ${LATENT_TEACHER_REPO_ROOT}"
    ARGS="${ARGS} --latent_teacher_config ${LATENT_TEACHER_CONFIG}"
    ARGS="${ARGS} --latent_teacher_checkpoint ${LATENT_TEACHER_CHECKPOINT}"
    if [ -n "${LATENT_TEACHER_FUTURE_OFFSET}" ]; then
        ARGS="${ARGS} --latent_teacher_future_offset ${LATENT_TEACHER_FUTURE_OFFSET}"
    fi
fi

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

echo "============================================================"
echo "Starting SimVLA Training on LeRobot LIBERO (Large)"
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
if [ "${LATENT_AUX_ENABLED}" = true ]; then
    echo "latent_aux_enabled: true"
    echo "latent_aux_weight: ${LATENT_AUX_WEIGHT}"
    echo "latent_teacher_repo_root: ${LATENT_TEACHER_REPO_ROOT}"
    echo "latent_teacher_config: ${LATENT_TEACHER_CONFIG}"
    echo "latent_teacher_checkpoint: ${LATENT_TEACHER_CHECKPOINT}"
fi
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port ${MAIN_PROCESS_PORT} \
    --mixed_precision ${MIXED_PRECISION} \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
