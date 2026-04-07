"""
SmolVLM sequence transformer blocks used by the SimVLA flow-matching heads.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Final, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(x) -> Tuple:
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    return hasattr(F, "scaled_dot_product_attention")


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def basic_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )
        self.use_cross_attention = bool(use_cross_attention)
        if self.use_cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_size)
            self.cross_attn = nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=0.1,
                batch_first=True,
            )
        else:
            self.norm_cross = None
            self.cross_attn = None

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        if self.cross_attn is not None and memory is not None:
            attn_out, _ = self.cross_attn(
                query=self.norm_cross(x),
                key=memory,
                value=memory,
                key_padding_mask=memory_mask,
                need_weights=False,
            )
            x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )
        self.use_cross_attention = bool(use_cross_attention)
        if self.use_cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=0.1,
                batch_first=True,
            )
            out_mult = 9
        else:
            self.norm_cross = None
            self.cross_attn = None
            out_mult = 6

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, out_mult * hidden_size, bias=True),
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation_params = self.adaLN_modulation(c)
        if self.use_cross_attention:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_cross,
                scale_cross,
                gate_cross,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = modulation_params.chunk(9, dim=-1)
        else:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = modulation_params.chunk(6, dim=-1)
            shift_cross = scale_cross = gate_cross = None

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)

        if self.cross_attn is not None and memory is not None:
            x_norm = modulate(self.norm_cross(x), shift_cross, scale_cross)
            attn_out, _ = self.cross_attn(
                query=x_norm,
                key=memory,
                value=memory,
                key_padding_mask=memory_mask,
                need_weights=False,
            )
            x = x + gate_cross.unsqueeze(1) * attn_out

        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class SmolVLMFlowTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        vlm_hidden_size: int = 576,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_input: int = 26,
        dim_output: int | None = None,
        dim_propio: int = 21,
        dim_time: int = 32,
        max_len_seq: int = 1024,
        use_adaln: bool = False,
        use_cross_attention: bool = False,
        memory_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_input = dim_input
        self.dim_output = dim_input if dim_output is None else int(dim_output)
        self.dim_time = dim_time
        self.dim_propio = dim_propio
        self.use_adaln = use_adaln
        self.use_cross_attention = bool(use_cross_attention)
        self.memory_dim = int(memory_dim) if memory_dim is not None else None

        if use_adaln:
            self.blocks = nn.ModuleList(
                [
                    DiTBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        use_cross_attention=self.use_cross_attention,
                    )
                    for _ in range(depth)
                ]
            )
            self.time_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.vlm_cond_proj = nn.Linear(vlm_hidden_size, hidden_size)
            self.proprio_proj = nn.Linear(dim_propio, hidden_size)
            self.sequence_encoder = nn.Linear(self.dim_input, hidden_size)
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)
            self.final_layer = FinalLayer(hidden_size, self.dim_output)
            self.norm = None
            self.sequence_decoder = None
            self.vlm_proj = None
        else:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        use_cross_attention=self.use_cross_attention,
                    )
                    for _ in range(depth)
                ]
            )
            self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)
            self.norm = nn.LayerNorm(hidden_size)
            self.sequence_encoder = nn.Linear(self.dim_input + dim_time + dim_propio, hidden_size)
            self.sequence_decoder = nn.Linear(hidden_size, self.dim_output)
            self.final_layer = None
            self.time_proj = None
            self.vlm_cond_proj = None
            self.proprio_proj = None

        if self.use_cross_attention:
            if self.memory_dim is None:
                raise ValueError("memory_dim must be set when use_cross_attention=True.")
            self.memory_proj = nn.Linear(self.memory_dim, hidden_size)
        else:
            self.memory_proj = None

        self.apply(basic_init)

    def forward(
        self,
        vlm_features: torch.Tensor,
        sequence_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        velocity, _ = self.forward_with_features(
            vlm_features=vlm_features,
            sequence_with_noise=sequence_with_noise,
            proprio=proprio,
            t=t,
            memory=memory,
            memory_mask=memory_mask,
        )
        return velocity

    def forward_with_features(
        self,
        vlm_features: torch.Tensor,
        sequence_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_adaln:
            return self._forward_adaln(
                vlm_features=vlm_features,
                sequence_with_noise=sequence_with_noise,
                proprio=proprio,
                t=t,
                memory=memory,
                memory_mask=memory_mask,
            )
        return self._forward_concat(
            vlm_features=vlm_features,
            sequence_with_noise=sequence_with_noise,
            proprio=proprio,
            t=t,
            memory=memory,
            memory_mask=memory_mask,
        )

    def _project_memory(
        self,
        memory: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if memory is None:
            return None, None
        if self.memory_proj is None:
            raise RuntimeError("This transformer was not initialized for cross-attention memory.")
        if memory.ndim != 3:
            raise ValueError(f"memory must have shape [B, M, D], got {tuple(memory.shape)}.")
        projected = self.memory_proj(memory)
        key_padding_mask = None
        if memory_mask is not None:
            if memory_mask.ndim != 2:
                raise ValueError(
                    f"memory_mask must have shape [B, M], got {tuple(memory_mask.shape)}."
                )
            key_padding_mask = ~memory_mask.bool()
        return projected, key_padding_mask

    def _forward_concat(
        self,
        vlm_features: torch.Tensor,
        sequence_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, sequence_len = sequence_with_noise.shape[:2]

        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(B, sequence_len, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, sequence_len, proprio.shape[-1])

        seq_tokens = torch.cat([sequence_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.sequence_encoder(seq_tokens)
        x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)

        seq_plus_vlm_len = x.shape[1]
        if seq_plus_vlm_len > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {seq_plus_vlm_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
            )
        x = x + self.pos_emb[:, :seq_plus_vlm_len, :]

        projected_memory, key_padding_mask = self._project_memory(memory, memory_mask)
        for block in self.blocks:
            x = block(x, memory=projected_memory, memory_mask=key_padding_mask)

        policy_hidden = self.norm(x[:, :sequence_len])
        velocity = self.sequence_decoder(policy_hidden)
        return velocity, policy_hidden

    def _forward_adaln(
        self,
        vlm_features: torch.Tensor,
        sequence_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, sequence_len = sequence_with_noise.shape[:2]

        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_proj(t_emb)
        vlm_cond = self.vlm_cond_proj(vlm_features.mean(dim=1))
        proprio_cond = self.proprio_proj(proprio)
        c = t_emb + vlm_cond + proprio_cond

        x = self.sequence_encoder(sequence_with_noise)
        x = x + self.pos_emb[:, :sequence_len, :]

        projected_memory, key_padding_mask = self._project_memory(memory, memory_mask)
        for block in self.blocks:
            x = block(x, c, memory=projected_memory, memory_mask=key_padding_mask)

        policy_hidden = x
        velocity = self.final_layer(policy_hidden, c)
        return velocity, policy_hidden


class SmolVLMActionTransformer(SmolVLMFlowTransformer):
    def __init__(
        self,
        hidden_size: int = 768,
        vlm_hidden_size: int = 576,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_action: int = 26,
        dim_propio: int = 21,
        dim_time: int = 32,
        max_len_seq: int = 1024,
        use_adaln: bool = False,
        use_cross_attention: bool = False,
        memory_dim: int | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            vlm_hidden_size=vlm_hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dim_input=dim_action,
            dim_output=dim_action,
            dim_propio=dim_propio,
            dim_time=dim_time,
            max_len_seq=max_len_seq,
            use_adaln=use_adaln,
            use_cross_attention=use_cross_attention,
            memory_dim=memory_dim,
        )
        self.dim_action = dim_action

    def forward(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        sequence_with_noise: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_with_noise is None:
            action_with_noise = sequence_with_noise
        if action_with_noise is None or proprio is None or t is None:
            raise ValueError("action_with_noise/sequence_with_noise, proprio, and t are required.")
        return super().forward(
            vlm_features=vlm_features,
            sequence_with_noise=action_with_noise,
            proprio=proprio,
            t=t,
            memory=memory,
            memory_mask=memory_mask,
        )

    def forward_with_features(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        sequence_with_noise: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_with_noise is None:
            action_with_noise = sequence_with_noise
        if action_with_noise is None or proprio is None or t is None:
            raise ValueError("action_with_noise/sequence_with_noise, proprio, and t are required.")
        return super().forward_with_features(
            vlm_features=vlm_features,
            sequence_with_noise=action_with_noise,
            proprio=proprio,
            t=t,
            memory=memory,
            memory_mask=memory_mask,
        )


__all__ = [
    "SmolVLMFlowTransformer",
    "SmolVLMActionTransformer",
    "TransformerBlock",
    "DiTBlock",
    "FinalLayer",
    "Attention",
    "Mlp",
    "timestep_embedding",
]
