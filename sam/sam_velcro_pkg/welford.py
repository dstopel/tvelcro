"""Welford online variance accumulator (numerically stable, float64)."""

from typing import Optional

import numpy as np
import torch


class WelfordAccumulator:
    """Single-pass online mean and variance using Welford's algorithm.

    All internal state is kept in float64 for numerical stability.
    Results are exported as float32 numpy arrays.
    """

    def __init__(self):
        self.count: int = 0
        self.mean: Optional[torch.Tensor] = None
        self.M2: Optional[torch.Tensor] = None
        self.entropy_sum: float = 0.0
        self.entropy_count: int = 0

    def update(self, x: torch.Tensor) -> None:
        arr = x.double()
        if self.mean is None:
            self.mean = torch.zeros_like(arr)
            self.M2 = torch.zeros_like(arr)

        self.count += 1
        delta = arr - self.mean
        self.mean += delta / self.count
        delta2 = arr - self.mean
        self.M2 += delta * delta2

    def update_entropy(self, x: torch.Tensor) -> None:
        magnitudes = torch.sqrt(torch.sum(x.float() ** 2, dim=-1) + 1e-10)
        probs = magnitudes / (torch.sum(magnitudes) + 1e-10)
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
        self.entropy_sum += entropy
        self.entropy_count += 1

    def get_mean(self) -> Optional[np.ndarray]:
        if self.count < 1 or self.mean is None:
            return None
        return self.mean.numpy().astype(np.float32)

    def get_variance(self) -> Optional[np.ndarray]:
        if self.count < 2 or self.M2 is None:
            return None
        return (self.M2 / (self.count - 1)).numpy().astype(np.float32)

    def get_entropy(self) -> float:
        if self.entropy_count == 0:
            return 0.0
        return self.entropy_sum / self.entropy_count

    def __repr__(self) -> str:
        shape = tuple(self.mean.shape) if self.mean is not None else "uninitialized"
        return f"WelfordAccumulator(count={self.count}, shape={shape})"
