"""Profile and freeze hooks for SAM ViT-H attention heads."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .constants import (
    GLOBAL_ATTN_INDEXES,
    HEAD_DIM,
    HIDDEN_DIM,
    NUM_BLOCKS,
    NUM_HEADS,
)
from .welford import WelfordAccumulator


class ProfileHookManager:
    """Registers forward-pre-hooks on attn.proj to profile per-head variance."""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.accumulators: Dict[Tuple[int, int], WelfordAccumulator] = {}
        self.hooks: List[torch.utils.hooks.RemovableHook] = []
        self.sample_count = 0

    def register(self) -> None:
        for block_idx in range(NUM_BLOCKS):
            proj = self.model.image_encoder.blocks[block_idx].attn.proj
            hook = proj.register_forward_pre_hook(self._make_profile_hook(block_idx))
            self.hooks.append(hook)

    def _make_profile_hook(self, block_idx: int):
        def hook(module, args):
            x = args[0].detach()  # (B_eff, H, W, C) where C=1280
            B_eff = x.shape[0]
            S = x.shape[1] * x.shape[2]  # H*W: 196 (windowed) or 4096 (global)
            x_flat = x.reshape(B_eff, S, NUM_HEADS, HEAD_DIM).cpu()

            for bi in range(B_eff):
                for h in range(NUM_HEADS):
                    key = (block_idx, h)
                    if key not in self.accumulators:
                        self.accumulators[key] = WelfordAccumulator()
                    head_data = x_flat[bi, :, h, :]  # (S, HEAD_DIM)
                    self.accumulators[key].update(head_data)
                    self.accumulators[key].update_entropy(head_data)

        return hook

    def increment_sample_count(self) -> None:
        self.sample_count += 1

    def save(self, out_dir: Path, group_name: str) -> dict:
        save_dir = out_dir / group_name
        save_dir.mkdir(parents=True, exist_ok=True)

        summary = {"group": group_name, "sample_count": self.sample_count, "heads": {}}

        for (block_idx, head_idx), acc in sorted(self.accumulators.items()):
            prefix = f"block{block_idx}_head{head_idx}"
            mean = acc.get_mean()
            variance = acc.get_variance()
            if mean is None or variance is None:
                continue

            mean_path = save_dir / f"{prefix}_mean.npy"
            var_path = save_dir / f"{prefix}_variance.npy"
            np.save(mean_path, mean)
            np.save(var_path, variance)

            rank_score = float(np.sum(variance))
            block_type = "global" if block_idx in GLOBAL_ATTN_INDEXES else "windowed"
            entropy = acc.get_entropy()

            summary["heads"][f"{block_idx}_{head_idx}"] = {
                "block": block_idx,
                "head": head_idx,
                "block_type": block_type,
                "rank_score": rank_score,
                "entropy": entropy,
                "mean_shape": list(mean.shape),
                "variance_shape": list(variance.shape),
                "count": acc.count,
            }

        return summary

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class FreezeHookManager:
    """Registers forward-pre-hooks on attn.proj to replace frozen heads with means."""

    def __init__(self, model, frozen_heads: Dict[Tuple[int, int], np.ndarray],
                 device: torch.device, collect_diagnostics: bool = False):
        self.model = model
        self.device = device
        self.frozen_heads = frozen_heads
        self.collect_diagnostics = collect_diagnostics
        self.hooks: List[torch.utils.hooks.RemovableHook] = []
        self.diagnostics: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        self._frozen_per_block: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        for (block_idx, head_idx), mean_np in frozen_heads.items():
            mean_t = torch.from_numpy(mean_np).to(device)
            self._frozen_per_block[block_idx][head_idx] = mean_t

    def register(self) -> None:
        for block_idx, head_means in self._frozen_per_block.items():
            proj = self.model.image_encoder.blocks[block_idx].attn.proj
            hook = proj.register_forward_pre_hook(
                self._make_freeze_hook(block_idx, head_means)
            )
            self.hooks.append(hook)

    def _make_freeze_hook(self, block_idx: int, head_means: Dict[int, torch.Tensor]):
        collect = self.collect_diagnostics
        diag = self.diagnostics

        def hook(module, args):
            x = args[0]  # (B_eff, H, W, 1280)
            B_eff, H, W, C = x.shape
            S = H * W
            reshaped = x.view(B_eff, S, NUM_HEADS, HEAD_DIM).clone()

            for h, mean_t in head_means.items():
                S_mean = mean_t.shape[0]

                if collect:
                    computed = reshaped[:, :, h, :].detach()
                    if S_mean == S:
                        mean_expanded = mean_t.unsqueeze(0).expand(B_eff, -1, -1)
                    else:
                        mean_pooled = mean_t.mean(dim=0, keepdim=True).expand(S, -1)
                        mean_expanded = mean_pooled.unsqueeze(0).expand(B_eff, -1, -1)
                    cos = F.cosine_similarity(
                        computed.flatten(1), mean_expanded.to(computed.dtype).flatten(1), dim=1
                    ).mean().item()
                    diag[(block_idx, h)].append(cos)

                if S_mean == S:
                    reshaped[:, :, h, :] = mean_t.to(x.dtype).unsqueeze(0)
                else:
                    mean_pooled = mean_t.mean(dim=0, keepdim=True).expand(S, -1)
                    reshaped[:, :, h, :] = mean_pooled.to(x.dtype).unsqueeze(0)

            return (reshaped.view(B_eff, H, W, C),)

        return hook

    def get_diagnostics_summary(self) -> Dict[str, float]:
        summary = {}
        for (b, h), vals in sorted(self.diagnostics.items()):
            if vals:
                summary[f"block{b}_head{h}"] = {
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
