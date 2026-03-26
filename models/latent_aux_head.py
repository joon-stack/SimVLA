"""
Latent auxiliary head for SimVLA.

This head reads the shared policy hidden produced by the action transformer and
maps it to DinoLAM-style latent tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentAuxHead(nn.Module):
    """
    Project policy hidden states to latent tokens.

    The head first pools the temporal dimension from the action horizon to the
    latent token count, then applies a token-wise MLP to predict the latent
    token vectors.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_tokens: int,
        token_dim: int,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}.")

        self.hidden_size = int(hidden_size)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)

        self.pool = nn.AdaptiveAvgPool1d(self.num_tokens)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.token_dim),
        )

    def forward(self, policy_hidden: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        policy_hidden:
            Tensor of shape [B, T, H].

        Returns
        -------
        Tensor
            Latent token predictions of shape [B, Q, Dz].
        """
        if policy_hidden.ndim != 3:
            raise ValueError(
                "policy_hidden must have shape [B, T, H], got "
                f"{tuple(policy_hidden.shape)}."
            )
        if policy_hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                "policy_hidden hidden size mismatch: expected "
                f"{self.hidden_size}, got {policy_hidden.shape[-1]}."
            )

        x = policy_hidden.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.proj(x)
        return x


__all__ = ["LatentAuxHead"]
