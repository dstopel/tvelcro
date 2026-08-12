"""Profile and freeze hooks for SAM ViT-H MLP neurons."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .constants import (
    GLOBAL_ATTN_INDEXES,
    HIDDEN_DIM,
    MLP_DIM,
    NUM_BLOCKS,
)
from .welford import WelfordAccumulator


class MLPProfileHookManager:
    """Registers forward-pre-hooks on mlp.lin2 to profile per-neuron variance."""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.accumulators: Dict[int, WelfordAccumulator] = {}
        self.hooks: List[torch.utils.hooks.RemovableHook] = []
        self.sample_count = 0

    def register(self) -> None:
        for block_idx in range(NUM_BLOCKS):
            lin2 = self.model.image_encoder.blocks[block_idx].mlp.lin2
            hook = lin2.register_forward_pre_hook(self._make_profile_hook(block_idx))
            self.hooks.append(hook)

    def _make_profile_hook(self, block_idx: int):
        def hook(module, args):
            x = args[0].detach()  # (B_eff, H, W, 5120) or (B_eff, seq_len, 5120)
            # Flatten all dimensions except the last (neuron dim) and average.
            # This handles both 3D and 4D tensors correctly.
            neuron_means = x.reshape(-1, x.shape[-1]).mean(dim=0).cpu()  # (5120,)

            if block_idx not in self.accumulators:
                self.accumulators[block_idx] = WelfordAccumulator()
            self.accumulators[block_idx].update(neuron_means)

        return hook

    def increment_sample_count(self) -> None:
        self.sample_count += 1

    def save(self, out_dir: Path, group_name: str) -> dict:
        save_dir = out_dir / group_name
        save_dir.mkdir(parents=True, exist_ok=True)

        summary = {"group": group_name, "sample_count": self.sample_count, "neurons": {}}

        for block_idx in sorted(self.accumulators.keys()):
            acc = self.accumulators[block_idx]
            prefix = f"block{block_idx}_mlp"
            mean = acc.get_mean()
            variance = acc.get_variance()
            if mean is None or variance is None:
                continue

            mean_path = save_dir / f"{prefix}_mean.npy"
            var_path = save_dir / f"{prefix}_variance.npy"
            np.save(mean_path, mean)
            np.save(var_path, variance)

            block_type = "global" if block_idx in GLOBAL_ATTN_INDEXES else "windowed"

            summary["neurons"][f"block{block_idx}"] = {
                "block": block_idx,
                "block_type": block_type,
                "total_neurons": MLP_DIM,
                "rank_score_sum": float(np.sum(variance)),
                "rank_score_mean": float(np.mean(variance)),
                "rank_score_max": float(np.max(variance)),
                "rank_score_min": float(np.min(variance)),
                "mean_shape": list(mean.shape),
                "variance_shape": list(variance.shape),
                "count": acc.count,
            }

        return summary

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class MLPFreezeHookManager:
    """Registers forward-pre-hooks on mlp.lin2 to replace frozen neurons with means."""

    def __init__(self, model, frozen_neurons: Dict[Tuple[int, int], float],
                 device: torch.device, collect_diagnostics: bool = False):
        self.model = model
        self.device = device
        self.frozen_neurons = frozen_neurons
        self.collect_diagnostics = collect_diagnostics
        self.hooks: List[torch.utils.hooks.RemovableHook] = []
        self.diagnostics: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        # Group frozen neurons by block for efficient per-block hooks.
        self._frozen_per_block: Dict[int, Dict[int, float]] = defaultdict(dict)
        for (block_idx, neuron_idx), mean_val in frozen_neurons.items():
            self._frozen_per_block[block_idx][neuron_idx] = mean_val

    def register(self) -> None:
        for block_idx, neuron_means in self._frozen_per_block.items():
            lin2 = self.model.image_encoder.blocks[block_idx].mlp.lin2
            hook = lin2.register_forward_pre_hook(
                self._make_freeze_hook(block_idx, neuron_means)
            )
            self.hooks.append(hook)

    def _make_freeze_hook(self, block_idx: int, neuron_means: Dict[int, float]):
        collect = self.collect_diagnostics
        diag = self.diagnostics
        # Pre-compute indices and values as tensors for vectorized replacement.
        neuron_indices = list(neuron_means.keys())
        neuron_values = torch.tensor(
            [neuron_means[idx] for idx in neuron_indices],
            dtype=torch.float32, device=self.device
        )

        def hook(module, args):
            x = args[0]  # (B_eff, H, W, 5120) or (B_eff, seq_len, 5120)
            x_new = x.clone()

            if collect:
                for i, neuron_idx in enumerate(neuron_indices):
                    original = x_new[..., neuron_idx].detach()
                    cos = torch.nn.functional.cosine_similarity(
                        original.flatten().unsqueeze(0),
                        torch.full_like(original.flatten().unsqueeze(0),
                                        neuron_values[i].item()),
                        dim=1,
                    ).mean().item()
                    diag[(block_idx, neuron_idx)].append(cos)

            # Vectorized replacement of frozen neuron columns (last dimension).
            for i, neuron_idx in enumerate(neuron_indices):
                x_new[..., neuron_idx] = neuron_values[i].to(x.dtype)

            return (x_new,)

        return hook

    def get_diagnostics_summary(self) -> Dict[str, float]:
        summary = {}
        for (b, n), vals in sorted(self.diagnostics.items()):
            if vals:
                summary[f"block{b}_neuron{n}"] = {
                    "mean_cosine_sim": float(np.mean(vals)),
                    "std_cosine_sim": float(np.std(vals)),
                    "min_cosine_sim": float(np.min(vals)),
                    "num_samples": len(vals),
                }
        return summary

    def reset_diagnostics(self) -> None:
        self.diagnostics = defaultdict(list)

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def select_mlp_neurons_to_freeze(
    profile_dir: Path,
    group_name: str,
    freeze_fraction: float,
) -> Dict[Tuple[int, int], float]:
    """Load profiles and select lowest-variance neurons to freeze.

    Args:
        profile_dir: Root directory containing profile outputs.
        group_name: Subdirectory name for the domain group.
        freeze_fraction: Fraction of total MLP neurons to freeze (0.0–1.0).

    Returns:
        Dictionary of (block_idx, neuron_idx) -> mean_value for frozen neurons.
    """
    save_dir = profile_dir / group_name

    # Collect all (block, neuron, variance, mean) tuples.
    all_neurons: List[Tuple[int, int, float, float]] = []

    for block_idx in range(NUM_BLOCKS):
        var_path = save_dir / f"block{block_idx}_mlp_variance.npy"
        mean_path = save_dir / f"block{block_idx}_mlp_mean.npy"
        if not var_path.exists() or not mean_path.exists():
            continue

        variance = np.load(var_path).flatten()  # (5120,)
        mean = np.load(mean_path).flatten()     # (5120,)

        for neuron_idx in range(min(MLP_DIM, len(variance))):
            all_neurons.append((
                block_idx,
                neuron_idx,
                float(variance[neuron_idx].item()),
                float(mean[neuron_idx].item()),
            ))

    if not all_neurons:
        return {}

    # Sort by variance (ascending) — lowest-variance neurons are frozen first.
    all_neurons.sort(key=lambda t: t[2])

    num_to_freeze = int(len(all_neurons) * freeze_fraction)
    frozen = all_neurons[:num_to_freeze]

    return {(block_idx, neuron_idx): mean_val
            for block_idx, neuron_idx, _, mean_val in frozen}
