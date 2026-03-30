"""
SmolVLM-VLA Training Script

Training script for SmolVLM-VLA using SmolVLM-500M-Instruct as backbone.
Uses 512x512 image resolution and unified VLM features (no aux_visual_inputs).

Usage:
    python train_smolvlm.py \
        --output_dir ./runs/smolvlm_vla \
        --train_metas_path ./train_metas.json \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --action_mode galaxea_joint \
        --num_actions 10
"""

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import (
    DEFAULT_LEROBOT_LIBERO_REPO_ID,
    create_lerobot_libero_dataloader,
    create_smolvlm_dataloader,
    resolve_lerobot_libero_dataset_root,
    resolve_lerobot_libero_norm_stats_path,
)
from models.dinolam_teacher import DinoLAMTeacher
from models.latent_aux_head import LatentAuxHead
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

import logging
import sys

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# ============================================================
# Logger
# ============================================================
def get_logger(name="train_smolvlm", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train_smolvlm.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("SmolVLM-VLA Training", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, default=None, 
                        help="Path to pretrained SmolVLM-VLA checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="runnings_smolvlm", 
                        help="Directory to save checkpoints")

    # SmolVLM backbone
    parser.add_argument("--smolvlm_model_path", type=str, 
                        default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="Path or HF repo for SmolVLM backbone")
    
    # Data
    parser.add_argument("--train_metas_path", type=str, default=None,
                        help="Path to training metadata (required for libero_hdf5 backend)")
    parser.add_argument("--dataset_backend", type=str, default="libero_hdf5",
                        choices=["libero_hdf5", "lerobot_hf"],
                        help="Dataset backend to use for training")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Optional local dataset root for lerobot_hf backend")
    parser.add_argument("--dataset_repo_id", type=str, default=DEFAULT_LEROBOT_LIBERO_REPO_ID,
                        help="Dataset repo id for lerobot_hf backend")
    parser.add_argument("--task_suite_name", type=str, default=None,
                        help="Optional LIBERO suite filter: libero_10/libero_goal/libero_object/libero_spatial")
    parser.add_argument("--camera_mode", type=str, default="single",
                        choices=["single", "dual"],
                        help="Camera inputs for lerobot_hf backend")
    parser.add_argument("--local_files_only", action="store_true", default=False,
                        help="Use local Hugging Face cache only when resolving lerobot_hf snapshots")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of micro-batches to accumulate before each optimizer step")
    parser.add_argument("--image_size", type=int, default=384, 
                        help="Image size for SmolVLM (default: 384, can be 384 or 512)")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0, 
                        help="LR multiplier for VLM backbone")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=1000000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)
    
    # Action mode
    parser.add_argument("--action_mode", type=str, default="galaxea_joint",
                        help="Action mode: galaxea_joint, galaxea, libero_joint, etc.")
    
    # Data loading
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Normalization
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="Path to normalization statistics JSON file")

    # DinoLAM latent auxiliary supervision
    parser.add_argument("--latent_aux_enabled", action="store_true", default=False,
                        help="Enable DinoLAM latent auxiliary supervision")
    parser.add_argument("--latent_aux_weight", type=float, default=1.0,
                        help="Weight for DinoLAM latent auxiliary loss")
    parser.add_argument("--latent_teacher_repo_root", type=str, default=None,
                        help="Path to the local latent_action repo root")
    parser.add_argument("--latent_teacher_config", type=str, default=None,
                        help="Resolved DinoLAM run config path")
    parser.add_argument("--latent_teacher_checkpoint", type=str, default=None,
                        help="Stage1 DinoLAM checkpoint path")
    parser.add_argument("--latent_teacher_future_offset", type=int, default=None,
                        help="Future offset used for DinoLAM teacher targets")
    parser.add_argument("--latent_teacher_image_size", type=int, default=0,
                        help="Optional resize for teacher images; <=0 infers from teacher config")
    
    # Action horizon
    parser.add_argument("--num_actions", type=int, default=10,
                        help="Action horizon (number of future actions to predict)")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    
    # Resume control
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from checkpoint")
    
    # DiT/AdaLN mode
    parser.add_argument("--use_adaln", action="store_true", default=False,
                        help="Use DiT-style AdaLN conditioning")
    
    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Hidden size for action transformer")
    parser.add_argument("--depth", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def ensure_model_has_latent_head(model: SmolVLMVLA, latent_num_tokens: int, latent_token_dim: int):
    """Attach a latent auxiliary head to a loaded checkpoint when needed."""
    current_enabled = bool(getattr(model, "latent_aux_enabled", False))
    current_num_tokens = int(getattr(model, "latent_num_tokens", 0))
    current_token_dim = int(getattr(model, "latent_token_dim", 0))
    if (
        current_enabled
        and current_num_tokens == int(latent_num_tokens)
        and current_token_dim == int(latent_token_dim)
        and getattr(model, "latent_head", None) is not None
    ):
        return model

    model.latent_aux_enabled = True
    model.latent_num_tokens = int(latent_num_tokens)
    model.latent_token_dim = int(latent_token_dim)
    model.config.latent_aux_enabled = True
    model.config.latent_num_tokens = int(latent_num_tokens)
    model.config.latent_token_dim = int(latent_token_dim)
    model.latent_head = LatentAuxHead(
        hidden_size=model.transformer.hidden_size,
        num_tokens=int(latent_num_tokens),
        token_dim=int(latent_token_dim),
    )
    return model


def build_optimizer(model: SmolVLMVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_vlm=1.0):
    """Build optimizer with separate param groups."""
    vlm_params = list(model.vlm.parameters())
    
    # Get action output params based on mode
    if hasattr(model.transformer, 'final_layer'):
        action_params = list(model.transformer.final_layer.parameters()) + list(model.transformer.action_encoder.parameters())
    else:
        action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    if getattr(model, "latent_head", None) is not None:
        action_params += list(model.latent_head.parameters())

    exclude = set(map(id, vlm_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups:
        if g["name"] == name:
            g["lr"] = lr


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    for g in optim.param_groups:
        if g["name"] == name:
            return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Linear warmup followed by cosine decay."""
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    """Update learning rates for all param groups."""
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "action_heads": args.learning_rate,
    }
    
    def schedule(step, base_lr):
        return linear_warmup_cosine(
            step, args.freeze_steps, args.warmup_steps, 
            args.iters, base_lr, args.min_lr_ratio
        )
    
    if step < args.freeze_steps:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "action_heads", base["action_heads"])
    else:
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    train_camera_mode = args.camera_mode if args.dataset_backend == "lerobot_hf" else "dual"
    
    # WandB setup
    wandb_api_key = os.environ.get("WANDB_API_KEY") or args.wandb_api_key
    wandb_project = os.environ.get("WANDB_PROJECT") or args.wandb_project
    use_wandb = WANDB_AVAILABLE and wandb_api_key

    log_with = ["tensorboard"]
    if use_wandb:
        log_with.append("wandb")
        os.environ["WANDB_API_KEY"] = wandb_api_key

    # Accelerator setup
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with=log_with,
        project_dir=output_dir,
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Initialize trackers
    tracker_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "micro_batch_size": args.batch_size,
        "effective_global_batch_size": (
            args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
        ),
        "iters": args.iters,
        "dataset_backend": args.dataset_backend,
        "dataset_repo_id": args.dataset_repo_id,
        "task_suite_name": args.task_suite_name,
        "camera_mode": args.camera_mode,
        "smolvlm_model_path": args.smolvlm_model_path,
        "freeze_steps": args.freeze_steps,
        "warmup_steps": args.warmup_steps,
        "save_interval": args.save_interval,
        "action_mode": args.action_mode,
        "num_actions": args.num_actions,
        "image_size": args.image_size,
        "hidden_size": args.hidden_size,
        "depth": args.depth,
        "use_adaln": args.use_adaln,
        "latent_aux_enabled": args.latent_aux_enabled,
        "latent_aux_weight": args.latent_aux_weight,
        "latent_teacher_repo_root": args.latent_teacher_repo_root,
        "latent_teacher_config": args.latent_teacher_config,
        "latent_teacher_checkpoint": args.latent_teacher_checkpoint,
        "latent_teacher_future_offset": args.latent_teacher_future_offset,
    }
    
    if use_wandb:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"smolvlm-{time.strftime('%Y%m%d-%H%M%S')}"}}
        )
    else:
        accelerator.init_trackers("SmolVLM-VLA-Training", config=tracker_config)

    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")
    logger.info(f"Using SmolVLM backbone: {args.smolvlm_model_path}")
    logger.info(f"Image size: {args.image_size}x{args.image_size}")

    if args.dataset_backend == "libero_hdf5" and not args.train_metas_path:
        raise ValueError("--train_metas_path is required when --dataset_backend=libero_hdf5.")
    if args.dataset_backend == "lerobot_hf" and args.action_mode != "libero_joint":
        raise ValueError(
            "LeRobot LIBERO training currently supports --action_mode libero_joint only."
        )

    resolved_lerobot_root = None
    if args.dataset_backend == "lerobot_hf":
        resolved_lerobot_root = resolve_lerobot_libero_dataset_root(
            args.dataset_root,
            repo_id=args.dataset_repo_id,
            local_files_only=args.local_files_only,
        )
        logger.info(f"Using LeRobot LIBERO root: {resolved_lerobot_root}")
        if not args.norm_stats_path:
            auto_norm_stats_path = resolve_lerobot_libero_norm_stats_path(resolved_lerobot_root)
            if auto_norm_stats_path:
                args.norm_stats_path = auto_norm_stats_path
                logger.info(f"Using LeRobot normalization stats from: {args.norm_stats_path}")

    latent_teacher = None
    latent_num_tokens = 0
    latent_token_dim = 0
    latent_teacher_image_size = int(args.latent_teacher_image_size)
    if args.latent_aux_enabled:
        if args.dataset_backend != "lerobot_hf":
            raise ValueError(
                "DinoLAM latent auxiliary supervision currently requires --dataset_backend lerobot_hf."
            )
        required_args = {
            "--latent_teacher_repo_root": args.latent_teacher_repo_root,
            "--latent_teacher_config": args.latent_teacher_config,
            "--latent_teacher_checkpoint": args.latent_teacher_checkpoint,
        }
        missing_args = [name for name, value in required_args.items() if not value]
        if missing_args:
            raise ValueError(
                "latent_aux_enabled requires the following arguments: "
                + ", ".join(missing_args)
            )
        latent_teacher = DinoLAMTeacher(
            repo_root=args.latent_teacher_repo_root,
            resolved_config_path=args.latent_teacher_config,
            checkpoint_path=args.latent_teacher_checkpoint,
            device=accelerator.device,
        )
        latent_num_tokens = int(latent_teacher.latent_num_tokens)
        latent_token_dim = int(latent_teacher.latent_token_dim)
        if latent_teacher_image_size <= 0:
            if latent_teacher.image_hw[0] == latent_teacher.image_hw[1]:
                latent_teacher_image_size = int(latent_teacher.image_hw[0])
            else:
                latent_teacher_image_size = int(max(latent_teacher.image_hw))
        logger.info(
            "Using DinoLAM teacher: "
            f"image_key={latent_teacher.image_key}, "
            f"image_range={latent_teacher.image_value_range}, "
            f"image_hw={latent_teacher.image_hw}, "
            f"latent_shape=[{latent_num_tokens}, {latent_token_dim}]"
        )

    # Load model
    from models.configuration_smolvlm_vla import SmolVLMVLAConfig
    from models.action_hub import build_action_space
    
    action_space_kwargs = {}
    if args.norm_stats_path:
        action_space_kwargs["norm_stats_path"] = args.norm_stats_path
        logger.info(f"Using normalization stats from: {args.norm_stats_path}")
    
    load_path = args.models
    
    if load_path and os.path.isdir(load_path) and os.path.exists(os.path.join(load_path, "model.safetensors")):
        logger.info(f"Loading SmolVLM-VLA from checkpoint: {load_path}")
        model = SmolVLMVLA.from_pretrained(load_path)
        
        if args.action_mode != model.action_mode:
            logger.warning(f"Overriding model action_mode from '{model.action_mode}' to '{args.action_mode}'")
            model.action_mode = args.action_mode
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
        elif action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
            
        if args.num_actions != model.num_actions:
            logger.warning(f"Overriding model num_actions from {model.num_actions} to {args.num_actions}")
            model.config.num_actions = args.num_actions
            model.num_actions = args.num_actions
            
        model_use_adaln = getattr(model, 'use_adaln', False)
        if args.use_adaln != model_use_adaln:
            logger.warning(f"⚠️ Cannot change use_adaln when loading from checkpoint")
        if args.latent_aux_enabled:
            ensure_model_has_latent_head(model, latent_num_tokens, latent_token_dim)
            logger.info(
                "Attached DinoLAM latent auxiliary head to loaded checkpoint: "
                f"num_tokens={latent_num_tokens}, token_dim={latent_token_dim}"
            )
    else:
        logger.info(f"Initializing SmolVLM-VLA from config")
        logger.info(f"  smolvlm_model_path: {args.smolvlm_model_path}")
        logger.info(f"  action_mode: {args.action_mode}")
        logger.info(f"  num_actions: {args.num_actions}")
        logger.info(f"  use_adaln: {args.use_adaln}")
        logger.info(f"  camera_mode: {train_camera_mode}")
        
        config = SmolVLMVLAConfig(
            smolvlm_model_path=args.smolvlm_model_path,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            action_mode=args.action_mode,
            num_actions=args.num_actions,
            use_adaln=args.use_adaln,
            image_size=args.image_size,
            camera_mode=train_camera_mode,
            latent_aux_enabled=args.latent_aux_enabled,
            latent_num_tokens=latent_num_tokens if args.latent_aux_enabled else 4,
            latent_token_dim=latent_token_dim if args.latent_aux_enabled else 32,
        )
        model = SmolVLMVLA(config)
        
        if action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)

    model.config.camera_mode = train_camera_mode
    
    # Build processor
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)

    if args.dataset_backend == "lerobot_hf":
        train_dataloader = create_lerobot_libero_dataloader(
            batch_size=args.batch_size,
            dataset_root=resolved_lerobot_root,
            num_actions=model.num_actions,
            action_mode=model.action_mode,
            training=True,
            num_workers=args.num_workers,
            image_size=args.image_size,
            camera_mode=args.camera_mode,
            task_suite_name=args.task_suite_name,
            emit_latent_teacher_fields=args.latent_aux_enabled,
            latent_teacher_future_offset=args.latent_teacher_future_offset,
            latent_teacher_image_size=latent_teacher_image_size if args.latent_aux_enabled else 224,
        )
    else:
        train_dataloader = create_smolvlm_dataloader(
            batch_size=args.batch_size,
            metas_path=args.train_metas_path,
            num_actions=model.num_actions,
            action_mode=model.action_mode,
            training=True,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )

    # Optimizer
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_vlm=args.learning_coef,
    )
    model, optim = accelerator.prepare(model, optim)

    # Training loop
    model.train()
    
    start_step = 0
    if args.resume and load_path and os.path.isdir(load_path):
        state_json = os.path.join(load_path, "state.json")
        if os.path.exists(state_json):
            try:
                with open(state_json, "r") as f:
                    start_step = int(json.load(f).get("global_step", 0))
                logger.info(f"Resuming from step: {start_step}")
            except Exception:
                pass
    
    global_step, t0 = start_step, time.time()
    effective_global_batch_size = (
        args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    )
    logger.info(f"🚀 Start SmolVLM-VLA training for {args.iters} iterations")
    logger.info(f"   world_size={accelerator.num_processes}")
    logger.info(f"   micro_batch_size={args.batch_size}")
    logger.info(f"   grad_accumulation={args.gradient_accumulation_steps}")
    logger.info(f"   effective_global_batch_size={effective_global_batch_size}")

    optim.zero_grad()
    for batch in train_dataloader:
        latent_obs_image = batch.pop("latent_obs_image", None)
        latent_future_obs_image = batch.pop("latent_future_obs_image", None)
        batch.pop("latent_teacher_future_offset", None)

        # Encode language
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.to(accelerator.device, non_blocking=True) for k, v in inputs.items()}

        if latent_teacher is not None:
            if latent_obs_image is None or latent_future_obs_image is None:
                raise RuntimeError(
                    "latent_aux_enabled is set but the dataloader did not return teacher image pairs."
                )
            inputs["latent_target_tokens"] = latent_teacher.predict_z_t_tokens(
                latent_obs_image,
                latent_future_obs_image,
            ).to(accelerator.device)

        with accelerator.accumulate(model):
            # Update LR in optimizer-step units.
            update_group_lrs(optim, global_step, args)

            # Forward
            loss_dict: Dict[str, torch.Tensor] = model(**inputs)
            loss = loss_dict["velocity_loss"]
            if "latent_aux_loss" in loss_dict:
                loss = loss + (args.latent_aux_weight * loss_dict["latent_aux_loss"])

            # Backward / optimize
            accelerator.backward(loss)
            if accelerator.sync_gradients and args.max_grad_norm:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            optim.zero_grad()

        if not accelerator.sync_gradients:
            continue

        # Logging
        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['loss_total']:.4f} "
                    f"lr_core={logs['lr_transformer_core']:.2e} "
                    f"lr_action={logs['lr_action_heads']:.2e} "
                    f"lr_vlm={logs['lr_vlm']:.2e} ({dt:.2f}s/it)"
                )

        # Checkpointing
        global_step += 1
        if accelerator.is_main_process:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"💾 Saving model to {save_dir}")
                accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump({"global_step": global_step}, f)

        if global_step >= args.iters:
            break

    accelerator.end_training()


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SmolVLM-VLA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
