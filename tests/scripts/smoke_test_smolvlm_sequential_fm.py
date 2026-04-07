#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.configuration_smolvlm_vla import SmolVLMVLAConfig
from models.dinolam_teacher import DinoLAMTeacher
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the SimVLA sequential FM latent path.")
    parser.add_argument("--smolvlm_model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--latent_stride_k", type=int, default=4)
    parser.add_argument("--latent_sample_steps", type=int, default=2)
    parser.add_argument("--camera_mode", type=str, default="dual")
    parser.add_argument("--check_save_reload", action="store_true")
    parser.add_argument("--teacher_repo_root", type=str, default="")
    parser.add_argument("--teacher_resolved_config_path", type=str, default="")
    parser.add_argument("--teacher_checkpoint_path", type=str, default="")
    parser.add_argument("--teacher_target_key", type=str, default="z_t_tokens_raw")
    return parser.parse_args()


def make_random_image(size: int) -> Image.Image:
    return Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))


def maybe_build_teacher(args: argparse.Namespace, device: torch.device) -> DinoLAMTeacher | None:
    if not (args.teacher_repo_root and args.teacher_resolved_config_path and args.teacher_checkpoint_path):
        return None
    return DinoLAMTeacher(
        repo_root=args.teacher_repo_root,
        resolved_config_path=args.teacher_resolved_config_path,
        checkpoint_path=args.teacher_checkpoint_path,
        device=device,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    teacher = maybe_build_teacher(args, device)
    latent_num_tokens = teacher.latent_num_tokens if teacher is not None else 4
    latent_token_dim = teacher.latent_token_dim if teacher is not None else 32

    config = SmolVLMVLAConfig(
        smolvlm_model_path=args.smolvlm_model_path,
        action_mode="libero_joint",
        num_actions=args.num_actions,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        image_size=384,
        camera_mode=args.camera_mode,
        num_views=2 if args.camera_mode == "dual" else 1,
        latent_mode="sequential_fm",
        latent_num_tokens=latent_num_tokens,
        latent_token_dim=latent_token_dim,
        latent_stride_k=args.latent_stride_k,
        latent_loss_weight=1.0,
        latent_sample_steps=args.latent_sample_steps,
    )

    model = SmolVLMVLA(config).to(device)
    processor = SmolVLMVLAProcessor(smolvlm_model_path=args.smolvlm_model_path)

    images = [make_random_image(processor.image_size)]
    if args.camera_mode == "dual":
        images.append(make_random_image(processor.image_size))
    encoded = processor(images, "open the drawer")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    proprio = torch.randn(1, 8, device=device)
    action = torch.randn(1, args.num_actions, 7, device=device)

    if teacher is not None:
        current = torch.rand(1, 3, teacher.image_hw[0], teacher.image_hw[1])
        boundaries = torch.rand(1, config.n_segment_steps, 3, teacher.image_hw[0], teacher.image_hw[1])
        boundary_valid = torch.ones(1, config.n_segment_steps, dtype=torch.bool)
        if config.n_segment_steps > 1:
            boundary_valid[:, -1] = False
        latent_target_memory, latent_target_mask = teacher.predict_segment_latents(
            current,
            boundaries,
            boundary_valid=boundary_valid,
            target_key=args.teacher_target_key,
        )
        latent_target_memory = latent_target_memory.to(device)
        latent_target_mask = latent_target_mask.to(device)
    else:
        latent_target_memory = torch.randn(
            1,
            config.latent_memory_steps,
            config.latent_token_dim,
            device=device,
        )
        latent_target_mask = torch.ones(1, config.latent_memory_steps, dtype=torch.bool, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            image_input=encoded["image_input"],
            image_mask=encoded["image_mask"],
            proprio=proprio,
            action=action,
            latent_target_memory=latent_target_memory,
            latent_target_mask=latent_target_mask,
        )
    print(f"latent_boundaries={config.latent_boundaries}")
    print(f"latent_memory_steps={config.latent_memory_steps}")
    print(f"forward_keys={sorted(outputs.keys())}")
    for key, value in outputs.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                print(f"{key}={float(value):.6f}")
            else:
                print(f"{key}_shape={tuple(value.shape)}")

    with torch.no_grad():
        actions = model.generate_actions(
            input_ids=encoded["input_ids"],
            image_input=encoded["image_input"],
            image_mask=encoded["image_mask"],
            proprio=proprio,
            steps=2,
        )
    print(f"generated_actions_shape={tuple(actions.shape)}")
    print(f"generated_actions_dtype={actions.dtype}")

    if args.check_save_reload:
        temp_dir = Path(tempfile.mkdtemp(prefix="simvla_seqfm_smoke_"))
        try:
            model.save_pretrained(temp_dir, safe_serialization=True)
            reloaded = SmolVLMVLA.from_pretrained(temp_dir).to(device)
            with torch.no_grad():
                reloaded_actions = reloaded.generate_actions(
                    input_ids=encoded["input_ids"],
                    image_input=encoded["image_input"],
                    image_mask=encoded["image_mask"],
                    proprio=proprio,
                    steps=2,
                )
            print(f"reload_generated_actions_shape={tuple(reloaded_actions.shape)}")
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
