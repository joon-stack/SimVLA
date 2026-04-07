"""
Frozen DinoLAM teacher wrapper for SimVLA latent auxiliary experiments.

This module is intentionally standalone. It loads a local `latent_action`
checkout at runtime, restores a frozen stage1 DinoLAM checkpoint, infers the
teacher image preprocessing contract from the resolved config, and exposes the
teacher latent tokens for a current/future image pair.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn


@dataclass(frozen=True)
class DinoLAMTeacherSpec:
    repo_root: str
    resolved_config_path: str
    checkpoint_path: str
    image_key: str
    image_value_range: str
    image_hw: tuple[int, int]
    latent_num_tokens: int
    latent_token_dim: int


def _as_resolved_dict(path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(str(path))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Resolved config must be a mapping, got {type(resolved)!r}.")
    return resolved


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    device_name = str(device).strip().lower()
    if device_name in {"auto", "cuda:auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _resolve_hw_from_cfg(data_cfg: dict[str, Any]) -> tuple[int, int]:
    image_resize = data_cfg.get("image_resize", None)
    if image_resize is not None:
        if isinstance(image_resize, (int, float)):
            size = int(image_resize)
            return (size, size)
        if isinstance(image_resize, dict):
            if "size" in image_resize:
                size = int(image_resize["size"])
                return (size, size)
            if "hw" in image_resize:
                hw = image_resize["hw"]
                if len(hw) != 2:
                    raise ValueError(f"image_resize.hw must have length 2, got {hw}.")
                return (int(hw[0]), int(hw[1]))
        raise ValueError("image_resize must be int or contain 'size' or 'hw'.")

    image_chw = data_cfg.get("image_chw", None)
    if image_chw is not None:
        if len(image_chw) != 3:
            raise ValueError(f"image_chw must have length 3, got {image_chw}.")
        return (int(image_chw[1]), int(image_chw[2]))

    return (224, 224)


def _ensure_repo_root_on_path(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


class DinoLAMTeacher(nn.Module):
    """
    Frozen runtime wrapper around a stage1 DinoLAM checkpoint.

    The primary output is `z_t_tokens`, which matches the teacher latent token
    shape used by the auxiliary-loss branch in SimVLA.
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        resolved_config_path: str | Path,
        checkpoint_path: str | Path,
        device: str | torch.device | None = "auto",
        strict: bool = True,
    ) -> None:
        super().__init__()

        self.repo_root = Path(repo_root).expanduser().resolve()
        self.resolved_config_path = Path(resolved_config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.repo_root.exists():
            raise FileNotFoundError(f"latent_action repo root does not exist: {self.repo_root}")
        if not self.resolved_config_path.is_file():
            raise FileNotFoundError(
                f"resolved config does not exist: {self.resolved_config_path}"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {self.checkpoint_path}")

        _ensure_repo_root_on_path(self.repo_root)

        resolved_cfg = _as_resolved_dict(self.resolved_config_path)
        model_cfg = copy.deepcopy(resolved_cfg.get("model", {}) or {})
        data_cfg = copy.deepcopy(resolved_cfg.get("data", {}) or {})
        if not model_cfg:
            raise ValueError(
                f"Resolved config {self.resolved_config_path} is missing a model section."
            )

        self.image_key = str(
            model_cfg.get("encoders", {}).get(
                "image_key", data_cfg.get("output_image_key", "image")
            )
        )
        self.image_value_range = str(
            data_cfg.get(
                "image_value_range",
                model_cfg.get("encoders", {}).get("input_value_range", "zero_to_one"),
            )
        ).strip().lower()
        if self.image_value_range not in {"zero_to_one", "minus_one_to_one"}:
            raise ValueError(
                "DinoLAM teacher image_value_range must be "
                "'zero_to_one' or 'minus_one_to_one', got "
                f"{self.image_value_range!r}."
            )
        self.image_hw = _resolve_hw_from_cfg(data_cfg)

        model_cfg = OmegaConf.create(model_cfg)

        from src.model.dino_lam import DinoLAM

        teacher = DinoLAM(model_cfg)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", None)
        if state_dict is None:
            raise KeyError("Teacher checkpoint is missing model_state_dict.")
        try:
            teacher.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            stripped_state = {
                key.removeprefix("module."): value for key, value in state_dict.items()
            }
            teacher.load_state_dict(stripped_state, strict=strict)

        self.device = _resolve_device(device)
        teacher.to(self.device)
        teacher.eval()
        teacher.requires_grad_(False)
        self.teacher = teacher

        self.latent_num_tokens = int(getattr(self.teacher, "num_action_tokens", 0))
        self.latent_token_dim = int(getattr(self.teacher, "action_token_dim", 0))
        if self.latent_num_tokens <= 0 or self.latent_token_dim <= 0:
            latent_outputs = self.teacher(
                obs={self.image_key: self._dummy_image_batch(1, device="cpu")},
                obs_future={self.image_key: self._dummy_image_batch(1, device="cpu")},
            )
            z_t_tokens = latent_outputs["z_t_tokens"]
            self.latent_num_tokens = int(z_t_tokens.shape[1])
            self.latent_token_dim = int(z_t_tokens.shape[2])

        self.spec = DinoLAMTeacherSpec(
            repo_root=str(self.repo_root),
            resolved_config_path=str(self.resolved_config_path),
            checkpoint_path=str(self.checkpoint_path),
            image_key=self.image_key,
            image_value_range=self.image_value_range,
            image_hw=self.image_hw,
            latent_num_tokens=self.latent_num_tokens,
            latent_token_dim=self.latent_token_dim,
        )

    @staticmethod
    def _to_tensor(image: Any) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image
        if isinstance(image, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(image))
        raise TypeError(
            "Teacher images must be torch.Tensor or numpy.ndarray, got "
            f"{type(image)!r}."
        )

    def _ensure_bchw(self, image: Any) -> torch.Tensor:
        tensor = self._to_tensor(image)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            pass
        elif tensor.ndim == 5 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        else:
            raise ValueError(
                "Teacher images must have shape [C,H,W], [B,C,H,W], or "
                f"[B,1,C,H,W], got {tuple(tensor.shape)}."
            )

        if tensor.ndim != 4:
            raise ValueError(f"Teacher image batch must be 4D, got {tuple(tensor.shape)}.")

        if tensor.shape[-1] == 3 and tensor.shape[1] != 3:
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.shape[1] != 3:
            raise ValueError(
                f"Teacher expects RGB images with channel dimension 3, got {tuple(tensor.shape)}."
            )
        return tensor

    def _to_zero_to_one(self, image: torch.Tensor) -> torch.Tensor:
        orig_dtype = image.dtype
        image = image.float()
        if orig_dtype == torch.uint8:
            image = image.div(255.0)
        if self.image_value_range == "minus_one_to_one":
            return image.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        return image.clamp(0.0, 1.0)

    def _resize(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-2:]) == tuple(self.image_hw):
            return image
        return F.interpolate(
            image,
            size=self.image_hw,
            mode="bilinear",
            align_corners=False,
        )

    def preprocess_image(self, image: Any) -> torch.Tensor:
        """Convert an RGB image or batch to DinoLAM teacher input format."""
        tensor = self._ensure_bchw(image)
        tensor = self._to_zero_to_one(tensor)
        tensor = self._resize(tensor)
        return tensor

    def _dummy_image_batch(
        self,
        batch_size: int,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        return torch.zeros(
            int(batch_size),
            3,
            int(self.image_hw[0]),
            int(self.image_hw[1]),
            dtype=torch.float32,
            device=device,
        )

    @torch.inference_mode()
    def forward_with_outputs(
        self,
        current_image: Any,
        future_image: Any,
    ) -> dict[str, torch.Tensor]:
        current = self.preprocess_image(current_image).to(self.device)
        future = self.preprocess_image(future_image).to(self.device)
        if current.shape[0] != future.shape[0]:
            raise ValueError(
                "Teacher current/future batch size mismatch: "
                f"{current.shape[0]} vs {future.shape[0]}."
            )
        outputs = self.teacher(
            obs={self.image_key: current},
            obs_future={self.image_key: future},
        )
        return {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in outputs.items()
        }

    @torch.inference_mode()
    def forward(self, current_image: Any, future_image: Any) -> torch.Tensor:
        """Return DinoLAM teacher z_t_tokens for a current/future image pair."""
        outputs = self.forward_with_outputs(current_image, future_image)
        z_t_tokens = outputs.get("z_t_tokens", None)
        if z_t_tokens is None:
            raise KeyError("Teacher forward did not return z_t_tokens.")
        return z_t_tokens

    @torch.inference_mode()
    def predict_z_t_tokens(self, current_image: Any, future_image: Any) -> torch.Tensor:
        return self.forward(current_image, future_image)

    @torch.inference_mode()
    def predict_segment_latents(
        self,
        current_image: Any,
        boundary_images: Any,
        *,
        boundary_valid: torch.Tensor | None = None,
        target_key: str = "z_t_tokens_raw",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if boundary_images is None:
            raise ValueError("boundary_images is required for segment latent prediction.")

        if isinstance(boundary_images, torch.Tensor):
            boundary_tensor = boundary_images
        else:
            boundary_tensor = self._to_tensor(boundary_images)
        if boundary_tensor.ndim == 4:
            boundary_tensor = boundary_tensor.unsqueeze(0)
        if boundary_tensor.ndim != 5:
            raise ValueError(
                "boundary_images must have shape [S,C,H,W] or [B,S,C,H,W], got "
                f"{tuple(boundary_tensor.shape)}."
            )
        B, S = boundary_tensor.shape[:2]
        current = self.preprocess_image(current_image)
        if current.shape[0] != B:
            raise ValueError(
                "Teacher current/boundary batch size mismatch: "
                f"{current.shape[0]} vs {B}."
            )
        flat_boundaries = self.preprocess_image(boundary_tensor.reshape(B * S, *boundary_tensor.shape[2:]))
        flat_boundaries = flat_boundaries.reshape(B, S, *flat_boundaries.shape[1:])

        all_endpoints = torch.cat([current.unsqueeze(1), flat_boundaries], dim=1)
        obs_start = all_endpoints[:, :-1].reshape(B * S, *all_endpoints.shape[2:])
        obs_end = all_endpoints[:, 1:].reshape(B * S, *all_endpoints.shape[2:])
        obs_start = obs_start.to(self.device)
        obs_end = obs_end.to(self.device)
        outputs = self.teacher(
            obs={self.image_key: obs_start},
            obs_future={self.image_key: obs_end},
        )
        outputs = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in outputs.items()
        }

        resolved_target_key = str(target_key).strip()
        if resolved_target_key == "auto":
            if "z_t_tokens_raw" in outputs:
                resolved_target_key = "z_t_tokens_raw"
            elif "z_t_tokens" in outputs:
                resolved_target_key = "z_t_tokens"
            else:
                raise KeyError("Teacher forward did not return a token latent target.")

        teacher_latents = outputs.get(resolved_target_key, None)
        if teacher_latents is None:
            raise KeyError(f"Teacher forward did not return target {resolved_target_key!r}.")
        if teacher_latents.ndim != 3:
            raise ValueError(
                "Teacher token target must have shape [B*S,Q,Dz], got "
                f"{tuple(teacher_latents.shape)}."
            )
        teacher_latents = teacher_latents.reshape(B, S, teacher_latents.shape[1], teacher_latents.shape[2])
        teacher_latents = teacher_latents.reshape(B, S * teacher_latents.shape[2], teacher_latents.shape[3])

        if boundary_valid is None:
            valid_mask = torch.ones(B, S, device=teacher_latents.device, dtype=torch.bool)
        else:
            valid_mask = boundary_valid.to(device=teacher_latents.device, dtype=torch.bool)
            if valid_mask.ndim == 1:
                valid_mask = valid_mask.unsqueeze(0)
            if valid_mask.shape != (B, S):
                raise ValueError(
                    "boundary_valid must have shape [B,S], got "
                    f"{tuple(valid_mask.shape)}."
                )
        valid_mask = valid_mask.unsqueeze(-1).expand(B, S, self.latent_num_tokens)
        valid_mask = valid_mask.reshape(B, S * self.latent_num_tokens)
        return teacher_latents, valid_mask


__all__ = [
    "DinoLAMTeacher",
    "DinoLAMTeacherSpec",
]
