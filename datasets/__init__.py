import torch

from .dataset_lerobot_libero import (
    DEFAULT_LEROBOT_LIBERO_REPO_ID,
    create_lerobot_libero_dataloader,
    normalize_libero_task_suite_name,
    resolve_lerobot_libero_dataset_root,
    resolve_lerobot_libero_norm_stats_path,
)


_SMOLVLM_EXPORTS = {
    "SmolVLMDataReader",
    "SmolVLMDataReaderWithPadding",
    "create_smolvlm_dataloader",
}


def __getattr__(name: str):
    if name in _SMOLVLM_EXPORTS:
        from . import dataset_smolvlm as _dataset_smolvlm

        return getattr(_dataset_smolvlm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def worker_init_fn(worker_id: int):
    """Worker process initialization: set random seeds and configure TensorFlow."""
    base_seed = torch.initial_seed() % (2**32)
    import random, numpy as np

    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)

    # Configure TensorFlow environment to avoid GPU contention
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass


__all__ = [
    "SmolVLMDataReader",
    "SmolVLMDataReaderWithPadding",
    "create_smolvlm_dataloader",
    "DEFAULT_LEROBOT_LIBERO_REPO_ID",
    "create_lerobot_libero_dataloader",
    "normalize_libero_task_suite_name",
    "resolve_lerobot_libero_dataset_root",
    "resolve_lerobot_libero_norm_stats_path",
    "worker_init_fn",
]
