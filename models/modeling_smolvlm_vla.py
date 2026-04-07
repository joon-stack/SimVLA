"""
SmolVLM-VLA model with optional DinoLAM latent branches.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import traceback
from typing import Any, Dict

import cv2
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json_numpy
import numpy as np
from PIL import Image
import torch
import uvicorn
from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedModel
from transformers.modeling_utils import load_state_dict as hf_load_state_dict

from .action_hub import build_action_space
from .configuration_smolvlm_vla import SmolVLMVLAConfig
from .latent_aux_head import LatentAuxHead
from .transformer_smolvlm import SmolVLMActionTransformer, SmolVLMFlowTransformer


def _module_float_dtype(module: torch.nn.Module) -> torch.dtype:
    for tensor in module.parameters():
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in module.buffers():
        if tensor.is_floating_point():
            return tensor.dtype
    raise RuntimeError(f"Module {module.__class__.__name__} has no floating-point tensors.")


def _resolve_latent_mode(config: SmolVLMVLAConfig) -> str:
    latent_mode = str(getattr(config, "latent_mode", "disabled")).strip().lower()
    if latent_mode == "disabled" and bool(getattr(config, "latent_aux_enabled", False)):
        return "aux_only"
    if latent_mode not in {"disabled", "aux_only", "sequential_fm"}:
        raise ValueError(f"Unsupported latent_mode={latent_mode!r}.")
    return latent_mode


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = torch.square(pred - target)
    if valid_mask is None:
        return diff.mean()
    mask = valid_mask.to(device=diff.device, dtype=diff.dtype)
    while mask.ndim < diff.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(diff)
    denom = mask.sum().clamp(min=1.0)
    return (diff * mask).sum() / denom


class SmolVLMVLA(PreTrainedModel):
    config_class = SmolVLMVLAConfig
    base_model_prefix = "smolvlm_vla"
    supports_gradient_checkpointing = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_path = pretrained_model_name_or_path
        if isinstance(model_path, (str, bytes, Path)) and Path(model_path).expanduser().is_dir():
            model_dir = Path(model_path).expanduser().resolve()
            config = kwargs.pop("config", None)
            if config is None:
                config = cls.config_class.from_pretrained(model_dir)
            model = cls(config, *model_args)

            safe_weights = model_dir / "model.safetensors"
            pt_weights = model_dir / "pytorch_model.bin"
            safe_index = model_dir / "model.safetensors.index.json"
            pt_index = model_dir / "pytorch_model.bin.index.json"

            checkpoint_files: list[Path] = []
            if safe_weights.is_file():
                checkpoint_files = [safe_weights]
            elif pt_weights.is_file():
                checkpoint_files = [pt_weights]
            elif safe_index.is_file() or pt_index.is_file():
                index_path = safe_index if safe_index.is_file() else pt_index
                with open(index_path, "r") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                checkpoint_files = sorted({model_dir / shard_name for shard_name in weight_map.values()})

            if len(checkpoint_files) == 0:
                raise FileNotFoundError(
                    f"No supported checkpoint weights found in {model_dir}."
                )

            merged_state_dict: dict[str, torch.Tensor] = {}
            for checkpoint_file in checkpoint_files:
                merged_state_dict.update(hf_load_state_dict(str(checkpoint_file), map_location="cpu"))
            missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)
            if missing_keys:
                logging.warning("Missing keys while loading %s: %s", model_dir, missing_keys)
            if unexpected_keys:
                logging.warning("Unexpected keys while loading %s: %s", model_dir, unexpected_keys)
            return model

        # Composite eager init conflicts with Transformers' meta-init path, so
        # only use the base implementation for non-local restore flows.
        kwargs.setdefault("low_cpu_mem_usage", False)
        kwargs.setdefault("device_map", None)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def __init__(self, config: SmolVLMVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        self.image_size: int = config.image_size
        self.num_views: int = config.num_views

        self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        logging.info(f"Loading SmolVLM from: {config.smolvlm_model_path}")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            config.smolvlm_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            config.smolvlm_model_path,
            trust_remote_code=True,
        )

        self.vlm_hidden_size = int(self.vlm.config.text_config.hidden_size)
        logging.info(f"SmolVLM hidden size: {self.vlm_hidden_size}")

        self.use_adaln = bool(getattr(config, "use_adaln", False))
        self.dim_proprio = int(dim_proprio)

        self.transformer = self._build_action_transformer(
            dim_action=dim_action,
            use_cross_attention=_resolve_latent_mode(config) == "sequential_fm",
            memory_dim=int(getattr(config, "latent_token_dim", 32)),
        )
        self.latent_transformer: SmolVLMFlowTransformer | None = None
        self.latent_head: LatentAuxHead | None = None

        self.latent_mode = "disabled"
        self.latent_enabled = False
        self.latent_aux_enabled = False
        self.latent_num_tokens = int(getattr(config, "latent_num_tokens", 4))
        self.latent_token_dim = int(getattr(config, "latent_token_dim", 32))
        self.latent_loss_weight = float(getattr(config, "latent_loss_weight", 1.0))
        self.latent_sample_steps = int(getattr(config, "latent_sample_steps", 10))
        self.configure_latent_mode(
            latent_mode=_resolve_latent_mode(config),
            latent_num_tokens=self.latent_num_tokens,
            latent_token_dim=self.latent_token_dim,
            latent_stride_k=int(getattr(config, "latent_stride_k", 0)),
            latent_loss_weight=self.latent_loss_weight,
            latent_sample_steps=self.latent_sample_steps,
        )

        if self.use_adaln:
            logging.info("✓ DiT/AdaLN mode enabled: conditions injected via Adaptive Layer Norm")
        else:
            logging.info("✓ Concat mode enabled: conditions concatenated to sequence")

        self.app: FastAPI | None = None

    def _build_action_transformer(
        self,
        *,
        dim_action: int,
        use_cross_attention: bool,
        memory_dim: int | None,
    ) -> SmolVLMActionTransformer:
        return SmolVLMActionTransformer(
            hidden_size=self.config.hidden_size,
            vlm_hidden_size=self.vlm_hidden_size,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            dim_action=dim_action,
            dim_propio=self.dim_proprio,
            dim_time=self.config.dim_time,
            max_len_seq=self.config.max_len_seq,
            use_adaln=self.use_adaln,
            use_cross_attention=use_cross_attention,
            memory_dim=memory_dim,
        )

    def _build_latent_transformer(self) -> SmolVLMFlowTransformer:
        return SmolVLMFlowTransformer(
            hidden_size=self.config.hidden_size,
            vlm_hidden_size=self.vlm_hidden_size,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            dim_input=self.latent_token_dim,
            dim_output=self.latent_token_dim,
            dim_propio=self.dim_proprio,
            dim_time=self.config.dim_time,
            max_len_seq=self.config.max_len_seq,
            use_adaln=self.use_adaln,
            use_cross_attention=False,
            memory_dim=None,
        )

    def _rebuild_action_transformer_if_needed(self, use_cross_attention: bool) -> None:
        current = self.transformer
        if (
            current.use_cross_attention == bool(use_cross_attention)
            and (not use_cross_attention or current.memory_dim == self.latent_token_dim)
        ):
            return
        new_transformer = self._build_action_transformer(
            dim_action=self.action_space.dim_action,
            use_cross_attention=use_cross_attention,
            memory_dim=self.latent_token_dim if use_cross_attention else None,
        )
        new_transformer.load_state_dict(current.state_dict(), strict=False)
        self.transformer = new_transformer

    def configure_latent_mode(
        self,
        *,
        latent_mode: str,
        latent_num_tokens: int,
        latent_token_dim: int,
        latent_stride_k: int,
        latent_loss_weight: float,
        latent_sample_steps: int,
    ) -> None:
        latent_mode = str(latent_mode).strip().lower()
        if latent_mode not in {"disabled", "aux_only", "sequential_fm"}:
            raise ValueError(f"Unsupported latent_mode={latent_mode!r}.")

        self.config.latent_mode = latent_mode
        self.config.latent_aux_enabled = latent_mode == "aux_only"
        self.config.latent_num_tokens = int(latent_num_tokens)
        self.config.latent_token_dim = int(latent_token_dim)
        self.config.latent_stride_k = int(latent_stride_k)
        self.config.latent_loss_weight = float(latent_loss_weight)
        self.config.latent_sample_steps = int(latent_sample_steps)

        self.latent_mode = latent_mode
        self.latent_enabled = latent_mode != "disabled"
        self.latent_aux_enabled = latent_mode == "aux_only"
        self.latent_num_tokens = int(latent_num_tokens)
        self.latent_token_dim = int(latent_token_dim)
        self.latent_loss_weight = float(latent_loss_weight)
        self.latent_sample_steps = int(latent_sample_steps)

        if self.latent_aux_enabled:
            self.latent_head = LatentAuxHead(
                hidden_size=self.config.hidden_size,
                num_tokens=self.latent_num_tokens,
                token_dim=self.latent_token_dim,
            )
        else:
            self.latent_head = None

        if self.latent_mode == "sequential_fm":
            self.latent_transformer = self._build_latent_transformer()
            self._rebuild_action_transformer_if_needed(use_cross_attention=True)
        else:
            self.latent_transformer = None
            self._rebuild_action_transformer_if_needed(use_cross_attention=False)

    def forward_vlm(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        language_instruction: list[str] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]

        B, V, _, _, _ = pixel_values.shape
        device = pixel_values.device
        batch_features = []

        for b in range(B):
            valid_mask = image_mask[b].bool()
            valid_images = pixel_values[b][valid_mask]

            if valid_images.shape[0] == 0:
                raise ValueError("At least one image view must be valid per batch.")

            pil_images = []
            for img_tensor in valid_images:
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))

            content = [{"type": "image", "image": img} for img in pil_images]
            if language_instruction is not None and b < len(language_instruction):
                content.append({"type": "text", "text": language_instruction[b]})
            else:
                content.append({"type": "text", "text": "Describe the robot's observation."})

            messages = [{"role": "user", "content": content}]
            inputs = self.vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.vlm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            batch_features.append(outputs.hidden_states[-1].squeeze(0))

        max_len = max(f.shape[0] for f in batch_features)
        hidden_size = batch_features[0].shape[-1]
        padded_features = torch.zeros(B, max_len, hidden_size, device=device, dtype=batch_features[0].dtype)
        for b, feat in enumerate(batch_features):
            padded_features[b, :feat.shape[0]] = feat
        return {"vlm_features": padded_features}

    def forward_vlm_efficient(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        input_ids: torch.LongTensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]

        B, V, _, _, _ = pixel_values.shape
        device = pixel_values.device
        vision_model = self.vlm.model.vision_model
        text_model = self.vlm.model.text_model
        vision_dtype = _module_float_dtype(vision_model)
        text_dtype = _module_float_dtype(text_model)

        flat_images = pixel_values.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()
        valid_images = flat_images[flat_mask]

        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")
        valid_images = valid_images.to(dtype=vision_dtype)

        vision_outputs = vision_model(
            pixel_values=valid_images,
            output_hidden_states=True,
            return_dict=True,
        )
        image_features = vision_outputs.last_hidden_state

        if hasattr(self.vlm.model, "connector"):
            image_features = self.vlm.model.connector(image_features)
        elif hasattr(self.vlm.model, "multi_modal_projector"):
            image_features = self.vlm.model.multi_modal_projector(image_features)
        image_features = image_features.to(dtype=text_dtype)

        text_embeds = text_model.get_input_embeddings()(input_ids).to(dtype=text_dtype)

        hidden_size = image_features.shape[-1]
        num_patches = image_features.shape[1]
        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)
        valid_per_sample = image_mask.sum(dim=1).int()

        batch_inputs_embeds = []
        max_seq_len = 0
        for b in range(B):
            num_valid = valid_per_sample[b].item()
            sample_image_feats = full_image_features[b, :num_valid].reshape(-1, hidden_size)
            sample_text_embeds = text_embeds[b]
            combined = torch.cat([sample_image_feats, sample_text_embeds], dim=0)
            batch_inputs_embeds.append(combined)
            max_seq_len = max(max_seq_len, combined.shape[0])

        padded_inputs_embeds = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=text_dtype)
        attention_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
        for b, embeds in enumerate(batch_inputs_embeds):
            seq_len = embeds.shape[0]
            padded_inputs_embeds[b, :seq_len] = embeds
            attention_mask[b, :seq_len] = 1

        lm_outputs = text_model(
            inputs_embeds=padded_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return {"vlm_features": lm_outputs.last_hidden_state}

    def _normalize_model_inputs(
        self,
        action: torch.Tensor | None,
        proprio: torch.Tensor,
        vlm_features: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.dtype]:
        policy_dtype = _module_float_dtype(self.transformer)

        action_norm = None
        if action is not None:
            if hasattr(self.action_space, "normalize_action"):
                action_norm = self.action_space.normalize_action(action)
            elif hasattr(self.action_space, "normalize"):
                action_norm = self.action_space.normalize(action)
            else:
                action_norm = action
            action_norm = action_norm.to(dtype=policy_dtype)

        if hasattr(self.action_space, "normalize_state"):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, "normalize"):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio
        proprio_norm = proprio_norm.to(dtype=policy_dtype)
        vlm_features = vlm_features.to(dtype=policy_dtype)
        return action_norm, proprio_norm, vlm_features, policy_dtype

    def _sample_fm_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        beta_dist = torch.distributions.Beta(
            torch.tensor(1.5, device=device),
            torch.tensor(1.0, device=device),
        )
        return (beta_dist.sample((batch_size,)) * 0.999 + 0.001).to(dtype=dtype)

    def _compute_fm_loss(
        self,
        *,
        transformer: SmolVLMFlowTransformer,
        target_sequence: torch.Tensor,
        proprio: torch.Tensor,
        vlm_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = target_sequence.shape[0]
        device = target_sequence.device
        dtype = target_sequence.dtype
        t = self._sample_fm_time(B, device=device, dtype=dtype)
        noise = torch.randn_like(target_sequence)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * noise + (1 - t_expanded) * target_sequence
        u_t = noise - target_sequence
        v_t, policy_hidden = transformer.forward_with_features(
            vlm_features=vlm_features,
            sequence_with_noise=x_t,
            proprio=proprio,
            t=t,
            memory=memory,
            memory_mask=memory_mask,
        )
        return _masked_mse(v_t, u_t, valid_mask=valid_mask), policy_hidden

    @torch.no_grad()
    def _sample_sequence(
        self,
        *,
        transformer: SmolVLMFlowTransformer,
        sequence_shape: tuple[int, int, int],
        proprio: torch.Tensor,
        vlm_features: torch.Tensor,
        steps: int,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, sequence_len, dim = sequence_shape
        device = proprio.device
        dtype = _module_float_dtype(transformer)

        x_t = torch.randn(B, sequence_len, dim, device=device, dtype=dtype)
        steps = max(1, int(steps))
        dt = -1.0 / steps
        t_value = 1.0

        while t_value > -dt / 2:
            t_tensor = torch.full((B,), t_value, device=device, dtype=dtype)
            v_t = transformer(
                vlm_features=vlm_features,
                sequence_with_noise=x_t,
                proprio=proprio,
                t=t_tensor,
                memory=memory,
                memory_mask=memory_mask,
            )
            x_t = x_t + dt * v_t
            t_value = t_value + dt

        return x_t

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        latent_target_tokens: torch.Tensor | None = None,
        latent_target_memory: torch.Tensor | None = None,
        latent_target_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)
        action_norm, proprio_norm, vlm_features, policy_dtype = self._normalize_model_inputs(
            action=action,
            proprio=proprio,
            vlm_features=enc["vlm_features"],
        )

        if self.latent_mode == "sequential_fm":
            if self.latent_transformer is None:
                raise RuntimeError("latent_mode='sequential_fm' requires latent_transformer.")
            if latent_target_memory is None:
                raise RuntimeError("sequential_fm requires latent_target_memory during training.")
            if latent_target_memory.ndim != 3:
                raise ValueError(
                    "latent_target_memory must have shape [B, M, Dz], got "
                    f"{tuple(latent_target_memory.shape)}."
                )
            if latent_target_memory.shape[1] != self.config.latent_memory_steps:
                raise ValueError(
                    "latent_target_memory step mismatch: expected "
                    f"{self.config.latent_memory_steps}, got {latent_target_memory.shape[1]}."
                )
            if latent_target_memory.shape[2] != self.latent_token_dim:
                raise ValueError(
                    "latent_target_memory dim mismatch: expected "
                    f"{self.latent_token_dim}, got {latent_target_memory.shape[2]}."
                )

            latent_target_memory = latent_target_memory.to(dtype=policy_dtype)
            if latent_target_mask is not None:
                latent_target_mask = latent_target_mask.to(device=latent_target_memory.device, dtype=torch.bool)

            latent_loss, _ = self._compute_fm_loss(
                transformer=self.latent_transformer,
                target_sequence=latent_target_memory,
                proprio=proprio_norm,
                vlm_features=vlm_features,
                valid_mask=latent_target_mask,
            )

            predicted_latents = self._sample_sequence(
                transformer=self.latent_transformer,
                sequence_shape=(
                    input_ids.shape[0],
                    self.config.latent_memory_steps,
                    self.latent_token_dim,
                ),
                proprio=proprio_norm,
                vlm_features=vlm_features,
                steps=self.latent_sample_steps,
            ).detach()

            action_loss, _ = self._compute_fm_loss(
                transformer=self.transformer,
                target_sequence=action_norm,
                proprio=proprio_norm,
                vlm_features=vlm_features,
                memory=predicted_latents,
            )
            total_loss = action_loss + (self.latent_loss_weight * latent_loss)
            info_dtype = action_loss.dtype
            info_device = action_loss.device
            return {
                "velocity_loss": action_loss,
                "loss_action": action_loss,
                "loss_latent": latent_loss,
                "loss_total": total_loss,
                "n_segment_steps": torch.tensor(
                    float(self.config.n_segment_steps),
                    device=info_device,
                    dtype=info_dtype,
                ),
                "latent_memory_steps": torch.tensor(
                    float(self.config.latent_memory_steps),
                    device=info_device,
                    dtype=info_dtype,
                ),
            }

        action_loss, policy_hidden = self._compute_fm_loss(
            transformer=self.transformer,
            target_sequence=action_norm,
            proprio=proprio_norm,
            vlm_features=vlm_features,
        )
        losses: Dict[str, torch.Tensor] = {
            "velocity_loss": action_loss,
            "loss_action": action_loss,
            "loss_total": action_loss,
        }

        if self.latent_aux_enabled and latent_target_tokens is not None:
            if self.latent_head is None:
                raise RuntimeError("latent_mode='aux_only' requires latent_head.")
            if latent_target_tokens.ndim != 3:
                raise ValueError(
                    "latent_target_tokens must have shape [B, Q, Dz], got "
                    f"{tuple(latent_target_tokens.shape)}."
                )
            if latent_target_tokens.shape[1] != self.latent_num_tokens:
                raise ValueError(
                    "latent_target_tokens token count mismatch: expected "
                    f"{self.latent_num_tokens}, got {latent_target_tokens.shape[1]}."
                )
            if latent_target_tokens.shape[2] != self.latent_token_dim:
                raise ValueError(
                    "latent_target_tokens feature size mismatch: expected "
                    f"{self.latent_token_dim}, got {latent_target_tokens.shape[2]}."
                )

            latent_pred = self.latent_head(policy_hidden)
            latent_target_tokens = latent_target_tokens.to(
                device=latent_pred.device,
                dtype=latent_pred.dtype,
            )
            latent_aux_loss = torch.mean(torch.square(latent_pred - latent_target_tokens))
            losses["latent_aux_loss"] = latent_aux_loss
            losses["loss_latent"] = latent_aux_loss
            losses["loss_total"] = action_loss + (self.latent_loss_weight * latent_aux_loss)

        return losses

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        self.eval()
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)
        _, proprio_norm, vlm_features, _ = self._normalize_model_inputs(
            action=None,
            proprio=proprio,
            vlm_features=enc["vlm_features"],
        )

        memory = None
        if self.latent_mode == "sequential_fm":
            if self.latent_transformer is None:
                raise RuntimeError("latent_mode='sequential_fm' requires latent_transformer.")
            memory = self._sample_sequence(
                transformer=self.latent_transformer,
                sequence_shape=(
                    input_ids.shape[0],
                    self.config.latent_memory_steps,
                    self.latent_token_dim,
                ),
                proprio=proprio_norm,
                vlm_features=vlm_features,
                steps=self.latent_sample_steps,
            )

        x_t = self._sample_sequence(
            transformer=self.transformer,
            sequence_shape=(
                input_ids.shape[0],
                self.num_actions,
                self.action_space.dim_action,
            ),
            proprio=proprio_norm,
            vlm_features=vlm_features,
            steps=steps,
            memory=memory,
        )
        return self.action_space.postprocess(x_t)

    def _build_app(self, processor):
        if self.app is not None:
            return

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                images = []
                for key in ("image0", "image1", "image2"):
                    if key not in payload:
                        continue
                    value = json_numpy.loads(payload[key])
                    if isinstance(value, np.ndarray):
                        if value.ndim == 1:
                            value = cv2.imdecode(value, cv2.IMREAD_COLOR)
                        images.append(Image.fromarray(value))
                    elif isinstance(value, (list, tuple)):
                        images.append(Image.fromarray(np.array(value)))
                    elif isinstance(value, str):
                        images.append(Image.open(value))

                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                inputs = processor(images, payload["language_instruction"])
                if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))

                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype

                def to_model(value: torch.Tensor) -> torch.Tensor:
                    if not isinstance(value, torch.Tensor):
                        value = torch.as_tensor(value)
                    if value.is_floating_point():
                        return value.to(device=device, dtype=dtype)
                    return value.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs["proprio"] = to_model(proprio.unsqueeze(0))

                steps = int(payload.get("steps", 10))
                action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)
