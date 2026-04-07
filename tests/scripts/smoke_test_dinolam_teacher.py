#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.dinolam_teacher import DinoLAMTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for the DinoLAM teacher wrapper.")
    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--resolved_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--target_key", type=str, default="z_t_tokens_raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher = DinoLAMTeacher(
        repo_root=args.repo_root,
        resolved_config_path=args.resolved_config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )

    height, width = teacher.image_hw
    current = torch.rand(args.batch_size, 3, height, width)
    boundaries = torch.rand(args.batch_size, args.segments, 3, height, width)
    boundary_valid = torch.ones(args.batch_size, args.segments, dtype=torch.bool)
    boundary_valid[:, -1] = False

    teacher_latents, valid_mask = teacher.predict_segment_latents(
        current,
        boundaries,
        boundary_valid=boundary_valid,
        target_key=args.target_key,
    )

    print(f"device={teacher.device}")
    print(f"image_key={teacher.image_key}")
    print(f"image_value_range={teacher.image_value_range}")
    print(f"image_hw={teacher.image_hw}")
    print(f"latent_num_tokens={teacher.latent_num_tokens}")
    print(f"latent_token_dim={teacher.latent_token_dim}")
    print(f"teacher_latents_shape={tuple(teacher_latents.shape)}")
    print(f"valid_mask_shape={tuple(valid_mask.shape)}")
    print(f"valid_mask_row0={valid_mask[0].tolist()}")


if __name__ == "__main__":
    main()
