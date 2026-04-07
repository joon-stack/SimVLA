"""
SmolVLM-VLA Configuration

Configuration class for SmolVLM-500M-Instruct based VLA model.
Uses SmolVLM as the vision-language backbone instead of Florence2.
"""

from transformers.configuration_utils import PretrainedConfig


class SmolVLMVLAConfig(PretrainedConfig):
    """
    Configuration class for the **SmolVLM-VLA (SmolVLM Vision-Language-Action)** model.

    This configuration defines all submodules of SmolVLM-VLA:
      - The visual-language backbone (SmolVLM-500M-Instruct)
      - The temporal/action transformer
      - The action/proprio setup
      
    Key differences from FlorenceVLA:
      - Uses SmolVLM (500M) instead of Florence2
      - Input image size: 512x512 (SmolVLM-500M uses 512x512 patches)
      - All views input to VLM directly, no aux_visual_inputs
      - Efficient model suitable for on-device applications
    """

    model_type = "smolvlm_vla"

    def __init__(
        self,
        # === SmolVLM backbone ===
        smolvlm_model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        
        # === Transformer head ===
        hidden_size: int = 768,  # Action transformer hidden size
        depth: int = 12,  # Number of transformer layers
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_time: int = 32,
        max_len_seq: int = 512,  

        # === Action & proprio ===
        num_actions: int = 30,
        action_mode: str = "galaxea_joint",
        use_proprio: bool = True,
        
        # === DiT/AdaLN Mode ===
        use_adaln: bool = False,

        # === Latent auxiliary head ===
        latent_mode: str = "disabled",
        latent_training_stage: str = "joint",
        latent_aux_enabled: bool = False,
        latent_loss_weight: float = 1.0,
        latent_stride_k: int = 0,
        latent_sample_steps: int = 10,
        latent_teacher_target: str = "z_t_tokens_raw",
        latent_teacher_obs_key: str | None = None,
        latent_num_tokens: int = 4,
        latent_token_dim: int = 32,
        
        # === Image settings ===
        image_size: int = 384,  # Can be 384 or 512
        num_views: int = 3,  # Number of camera views
        camera_mode: str = "dual",

        **kwargs,
    ):
        # SmolVLM backbone path
        self.smolvlm_model_path = smolvlm_model_path
        
        # Transformer hyperparameters
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio
        
        # DiT/AdaLN settings
        self.use_adaln = use_adaln

        # Latent auxiliary settings
        self.latent_mode = str(latent_mode).strip().lower()
        self.latent_training_stage = str(latent_training_stage).strip().lower()
        self.latent_aux_enabled = latent_aux_enabled
        self.latent_loss_weight = float(latent_loss_weight)
        self.latent_stride_k = int(latent_stride_k)
        self.latent_sample_steps = int(latent_sample_steps)
        self.latent_teacher_target = str(latent_teacher_target)
        self.latent_teacher_obs_key = latent_teacher_obs_key
        self.latent_num_tokens = latent_num_tokens
        self.latent_token_dim = latent_token_dim
        
        # Image settings
        self.image_size = image_size
        self.num_views = num_views
        self.camera_mode = camera_mode

        # Initialize base HF config attributes
        super().__init__(**kwargs)

    def to_dict(self):
        """
        Convert this configuration into a fully serializable dictionary.
        """
        output = super().to_dict()
        return output

    @property
    def n_obs_steps(self) -> int:
        return 1

    @property
    def future_horizon(self) -> int:
        return int(self.num_actions)

    @property
    def latent_boundaries(self) -> list[int]:
        future_horizon = int(self.future_horizon)
        if future_horizon <= 0:
            return [0]
        if self.latent_mode != "sequential_fm":
            return [0, future_horizon]
        stride_k = int(self.latent_stride_k)
        if stride_k <= 0:
            stride_k = future_horizon
        boundaries = [0]
        offset = stride_k
        while offset < future_horizon:
            boundaries.append(int(offset))
            offset += stride_k
        if boundaries[-1] != future_horizon:
            boundaries.append(future_horizon)
        return boundaries

    @property
    def latent_boundary_offsets(self) -> list[int]:
        return [int(v) for v in self.latent_boundaries[1:]]

    @property
    def n_segment_steps(self) -> int:
        return max(1, len(self.latent_boundaries) - 1)

    @property
    def n_latent_steps(self) -> int:
        return int(self.latent_num_tokens)

    @property
    def latent_memory_steps(self) -> int:
        if self.latent_mode == "sequential_fm":
            return int(self.n_segment_steps * self.latent_num_tokens)
        return int(self.latent_num_tokens)

    @property
    def observation_delta_indices(self) -> list[int]:
        history = list(range(1 - self.n_obs_steps, 1))
        return history + self.latent_boundary_offsets
