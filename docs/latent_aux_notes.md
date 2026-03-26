# Latent Auxiliary Notes

Current implementation scope is v1 aux-only supervision.

Implemented idea:
- Read SimVLA policy hidden immediately before the final action projection.
- Predict DinoLAM stage1 `z_t_tokens` with an auxiliary head.
- Optimize `L_total = L_action + lambda * L_latent` during training only.
- Keep the action denoising path and inference loop unchanged.

Deferred variants:
- `v2` predicted-latent-conditioned denoising
  - Feed the student-predicted latent back into the denoising policy.
  - Deferred because it changes the generation path and makes representation effects harder to isolate.
- `v3` joint latent+action sequence output
  - Emit latent and action tokens from the same sequence model.
  - Deferred because it is a larger architectural change than the current aux-only question needs.
