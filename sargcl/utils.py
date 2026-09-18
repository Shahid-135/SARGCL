"""Shared utilities for SARGCL."""

import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exists(x):
    """Check if a value is not None."""
    return x is not None


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity between two sets of vectors."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return torch.matmul(a, b.transpose(-1, -2))


def append_training_log(log_path: str, message: str):
    """Append one timestamped message to the persistent training log.

    This helper is used by the training loop at startup, after saving
    checkpoints, and when training finishes.
    """
    log_path = str(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
