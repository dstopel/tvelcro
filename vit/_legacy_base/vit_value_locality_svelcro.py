#!/usr/bin/env python3
"""
vit_value_locality_svelcro.py

Value locality analysis for Vision Transformers (timm ViT) using S-VELCRO
channel-rank method for head freezing.

Three-phase pipeline:

Phase 1 (profile):  Welford variance + mean + entropy per head and block
Phase 2 (freeze):   S-VELCRO channel-rank: rank heads by variance sum,
                     freeze the T_k lowest-ranked heads per block.
                     No reference group needed -- only the specialized group's
                     variance profiles are used.
Phase 3 (eval):     Mean-head replacement + accuracy measurement

Architecture:
  vit_base_patch16_224: 12 blocks x 12 heads x 64 head_dim x 197 seq_len
  Hook targets:
    - Block output: after attn + FFN + residuals (block-level variance)
    - Pre-projection heads: input to self.proj (per-head variance, BEFORE mixing)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, "..")
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from utils.models import get_net
import utils.datasets as datasets


# ==============================================================================
# Constants
# ==============================================================================

NUM_BLOCKS = 12
NUM_HEADS = 12
HEAD_DIM = 64
SEQ_LEN = 197       # 196 patches + 1 CLS token
HIDDEN_DIM = 768    # NUM_HEADS * HEAD_DIM

AVAILABLE_MODES = ["profile", "freeze", "eval"]


# ==============================================================================
# Welford Accumulator
# ==============================================================================

class WelfordAccumulator:
    """Numerically stable online variance (float64)."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.M2 = None
        self.entropy_sum = 0.0
        self.entropy_count = 0

    def update(self, x: torch.Tensor):
        arr = x.double()
        if self.mean is None:
            self.mean = torch.zeros_like(arr)
            self.M2 = torch.zeros_like(arr)

        self.count += 1
        delta = arr - self.mean
        self.mean += delta / self.count
        delta2 = arr - self.mean
        self.M2 += delta * delta2

    def update_entropy(self, x: torch.Tensor):
        """Activation entropy: energy distribution across sequence positions."""
        magnitudes = torch.sqrt(torch.sum(x.float() ** 2, dim=-1) + 1e-10)
        probs = magnitudes / (torch.sum(magnitudes) + 1e-10)
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
        self.entropy_sum += entropy
        self.entropy_count += 1

    def variance(self) -> Optional[np.ndarray]:
        if self.count < 2 or self.M2 is None:
            return None
        return (self.M2 / (self.count - 1)).numpy().astype(np.float32)

    def get_mean(self) -> Optional[np.ndarray]:
        if self.mean is None:
            return None
        return self.mean.numpy().astype(np.float32)

    def get_entropy(self) -> Optional[float]:
        if self.entropy_count == 0:
            return None
        return self.entropy_sum / self.entropy_count


# ==============================================================================
# Hook Manager for ViT
# ==============================================================================

class ViTHookManager:
    """
    Registers hooks on ViT blocks and attention projection layers.

    Two hook types:
      1. Block output hook: captures output of full block (attn + FFN + residuals)
         Shape: (B, 197, 768) -- block-level variance
      2. Pre-projection head hook: captures INPUT to self.proj
         Shape: (B, 197, 768) -- split into (B, 197, 12, 64) for per-head variance
         This is BEFORE self.proj mixes heads, so heads are still separable.
    """

    def __init__(self, model):
        self.model = model
        self.handles: List = []

        # Block-level accumulators
        self.block_acc: Dict[int, WelfordAccumulator] = {}
        # Head-level accumulators: (block_idx, head_idx)
        self.head_acc: Dict[Tuple[int, int], WelfordAccumulator] = {}

    def register(self):
        for i, block in enumerate(self.model.blocks):
            # Block output hook
            self.handles.append(
                block.register_forward_hook(self._make_block_hook(i))
            )
            # Pre-projection head hook (input to self.proj)
            if hasattr(block, "attn") and hasattr(block.attn, "proj"):
                self.handles.append(
                    block.attn.proj.register_forward_hook(self._make_head_hook(i))
                )

    def _make_block_hook(self, block_idx: int):
        def hook(module, inputs, output):
            x = output.detach().cpu()  # (B, 197, 768)
            if block_idx not in self.block_acc:
                self.block_acc[block_idx] = WelfordAccumulator()
            for bi in range(x.shape[0]):
                self.block_acc[block_idx].update(x[bi])
        return hook

    def _make_head_hook(self, block_idx: int):
        def hook(module, inputs, output):
            # inputs[0] is the pre-projection tensor: (B, 197, 768)
            # This is the concatenated head outputs BEFORE self.proj mixes them
            x = inputs[0].detach().cpu()
            B, N, C = x.shape
            x = x.view(B, N, NUM_HEADS, HEAD_DIM)  # (B, 197, 12, 64)

            for bi in range(B):
                for h in range(NUM_HEADS):
                    key = (block_idx, h)
                    if key not in self.head_acc:
                        self.head_acc[key] = WelfordAccumulator()
                    head_data = x[bi, :, h, :]  # (197, 64)
                    self.head_acc[key].update(head_data)
                    self.head_acc[key].update_entropy(head_data)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ==============================================================================
# Mean Head Patcher for ViT
# ==============================================================================

class ViTMeanHeadPatcher:
    """
    Replaces frozen heads with constant mean heads in a ViT.

    For each frozen head:
      1. Zeros out Q/K/V rows in the fused self.qkv weight
      2. Registers pre-hook on self.proj to overwrite frozen slots with mean

    The network keeps all 12 heads. Frozen heads always output the mean.
    """

    def __init__(self, model, frozen_heads: Dict[Tuple[int, int], np.ndarray]):
        self.model = model
        self.frozen_heads = frozen_heads
        self.originals: Dict[int, dict] = {}
        self.hooks: List = []

        # Group frozen heads by block
        self.block_frozen: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        for (block_idx, head_idx), mean_arr in frozen_heads.items():
            self.block_frozen[block_idx][head_idx] = mean_arr

    def apply(self):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        for block_idx, head_means in self.block_frozen.items():
            block = self.model.blocks[block_idx]
            attn = block.attn
            qkv = attn.qkv

            # Save originals
            self.originals[block_idx] = {
                "qkv_weight": qkv.weight.data.clone(),
                "qkv_bias": qkv.bias.data.clone() if qkv.bias is not None else None,
            }

            # Zero out Q/K/V for each frozen head
            # qkv weight shape: (3 * 768, 768) = (2304, 768)
            # Rows [0:768] = Q, [768:1536] = K, [1536:2304] = V
            for h in head_means:
                start = h * HEAD_DIM
                end = (h + 1) * HEAD_DIM
                # Zero Q
                qkv.weight.data[start:end, :] = 0
                # Zero K
                qkv.weight.data[HIDDEN_DIM + start:HIDDEN_DIM + end, :] = 0
                # Zero V
                qkv.weight.data[2 * HIDDEN_DIM + start:2 * HIDDEN_DIM + end, :] = 0

                if qkv.bias is not None:
                    qkv.bias.data[start:end] = 0
                    qkv.bias.data[HIDDEN_DIM + start:HIDDEN_DIM + end] = 0
                    qkv.bias.data[2 * HIDDEN_DIM + start:2 * HIDDEN_DIM + end] = 0

            # Pre-compute mean tensors on GPU
            mean_tensors = {}
            for h, mean_arr in head_means.items():
                mean_tensors[h] = torch.from_numpy(
                    mean_arr.astype(np.float32)
                ).to(device=device, dtype=dtype)

            # Register pre-hook on self.proj to overwrite frozen head slots
            frozen_set = set(head_means.keys())
            self._add_hook(attn.proj, frozen_set, mean_tensors)

            print(f"  [mean-head] block {block_idx}: heads {sorted(frozen_set)} "
                  f"-> mean heads ({NUM_HEADS - len(frozen_set)} active + {len(frozen_set)} mean)")

    def _add_hook(self, proj_module, frozen_set, mean_tensors):
        frozen_sorted = sorted(frozen_set)

        def hook(module, inputs):
            x = inputs[0]  # (B, 197, 768)
            B, N, C = x.shape
            reshaped = x.view(B, N, NUM_HEADS, HEAD_DIM).clone()

            for h in frozen_sorted:
                if h in mean_tensors:
                    mean_h = mean_tensors[h]  # (197, 64)
                    if mean_h.shape[0] != N:
                        mean_h = mean_h.mean(dim=0, keepdim=True).expand(N, -1)
                    reshaped[:, :, h, :] = mean_h.unsqueeze(0)

            return (reshaped.view(B, N, C),)

        self.hooks.append(proj_module.register_forward_pre_hook(hook))

    def restore(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

        for block_idx, orig in self.originals.items():
            qkv = self.model.blocks[block_idx].attn.qkv
            qkv.weight.data.copy_(orig["qkv_weight"])
            if orig["qkv_bias"] is not None:
                qkv.bias.data.copy_(orig["qkv_bias"])

        self.originals.clear()


def build_imagenet_label_mapping(synset_words_path: str, class_names: List[str]) -> Dict[int, int]:
    """
    Build {local_ImageFolder_idx -> ImageNet_class_idx} from synset_words.txt.

    synset_words.txt: 1000 lines, line N (0-indexed) = ImageNet class N.
    Format: "n02093991 Irish terrier[, synonym2, ...]"

    Folder names in ILSVRC2012_val_by_name use ALL synonyms joined with underscores:
      "Dandie Dinmont, Dandie Dinmont terrier" -> "Dandie_Dinmont_Dandie_Dinmont_terrier"
      "Samoyed, Samoyede"                      -> "Samoyed_Samoyede"
      "bakery, bakeshop, bakehouse"            -> "bakery_bakeshop_bakehouse"
    Also tries first-synonym-only as fallback:
      "Irish terrier"                          -> "Irish_terrier"
    """
    folder_to_imagenet: Dict[str, int] = {}
    with open(synset_words_path) as f:
        for imagenet_idx, line in enumerate(f):
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            synonyms = [s.strip() for s in parts[1].split(',')]
            # Key 1: all synonyms joined by underscore (spaces->underscores within each)
            joined = '_'.join(s.replace(' ', '_') for s in synonyms)
            folder_to_imagenet[joined] = imagenet_idx
            folder_to_imagenet[joined.lower()] = imagenet_idx
            # Key 2: first synonym only (fallback)
            first = synonyms[0].replace(' ', '_')
            folder_to_imagenet[first] = imagenet_idx
            folder_to_imagenet[first.lower()] = imagenet_idx

    mapping: Dict[int, int] = {}
    missing = []
    for local_idx, cls_name in enumerate(class_names):
        imagenet_idx = folder_to_imagenet.get(cls_name,
                       folder_to_imagenet.get(cls_name.lower()))
        if imagenet_idx is None:
            missing.append(cls_name)
            mapping[local_idx] = local_idx  # fallback: leave unchanged
        else:
            mapping[local_idx] = imagenet_idx

    if missing:
        print(f"  [WARN] {len(missing)} class(es) not in synset_words.txt "
              f"(using local idx): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return mapping


def apply_imagenet_label_mapping(data_gen, synset_words_path: Optional[str]) -> None:
    """
    Patch data_gen.test_set.target_transform in-place so targets are ImageNet
    class indices (0-999) instead of local alphabetical indices from ImageFolder.

    Must be called after the data_gen object is created but before test_loader() is called.
    """
    if not synset_words_path:
        print("  [WARN] --synset_words not set: accuracy will be wrong "
              "(ImageFolder local indices != ImageNet class indices)")
        return
    dataset = data_gen.test_set          # triggers lazy ImageFolder creation
    mapping = build_imagenet_label_mapping(synset_words_path, dataset.classes)
    dataset.target_transform = lambda y, m=mapping: m.get(y, y)
    print(f"  [label mapping] {len(dataset.classes)} classes mapped via {synset_words_path}")


# ==============================================================================
# Data Loading
# ==============================================================================

def load_data(img_path: str, model, batch_size: int = 32, num_workers: int = 4):
    """Load ImageNet-style data using ViTImageNetLoaderGenerator."""
    return datasets.ViTImageNetLoaderGenerator(
        img_path, 'imagenet',
        batch_size, batch_size, num_workers,
        kwargs={"model": model}
    )


# ==============================================================================
# Inference
# ==============================================================================

def run_inference(model, data_gen, max_batches: int, hook_mgr: Optional[ViTHookManager] = None):
    """Run inference, return (top1_correct, top5_correct, total)."""
    device = next(model.parameters()).device
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.inference_mode():
        loader = data_gen.test_loader()
        for i, (inp, target) in enumerate(loader):
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(inp)

            # Top-1
            _, pred = output.max(1)
            top1_correct += pred.eq(target).sum().item()

            # Top-5
            _, pred5 = output.topk(5, 1, True, True)
            top5_correct += pred5.eq(target.unsqueeze(1).expand_as(pred5)).sum().item()

            total += target.size(0)

            if (i + 1) % 5 == 0:
                print(f"    batch {i+1}/{max_batches}  "
                      f"top1={top1_correct/total*100:.1f}%  "
                      f"top5={top5_correct/total*100:.1f}%")

            if i + 1 >= max_batches:
                break

    return top1_correct, top5_correct, total


# ==============================================================================
# Save Functions
# ==============================================================================

def save_variances(group_name: str, hook_mgr: ViTHookManager, out_dir: Path, bins: int):
    """Save block and head variance/mean/entropy to disk."""
    # Block variance
    block_dir = out_dir / "block_variance" / group_name
    block_dir.mkdir(parents=True, exist_ok=True)

    for block_idx, acc in hook_mgr.block_acc.items():
        var = acc.variance()
        if var is None:
            continue
        arr = var.ravel()
        np.save(block_dir / f"{group_name}_block{block_idx}_variance.npy", arr)

        p99 = float(np.percentile(arr, 99))
        if p99 <= 0:
            p99 = float(np.max(arr)) + 1e-8

        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=bins, range=(0, p99), color='skyblue', edgecolor='black')
        plt.title(f"Block Variance - block {block_idx} ({group_name})")
        plt.xlabel("Variance")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(block_dir / f"{group_name}_block{block_idx}_variance.png", dpi=150)
        plt.close()

    print(f"  [block] Saved {len(hook_mgr.block_acc)} block variance files for '{group_name}'")

    # Head variance + mean + entropy
    head_dir = out_dir / "head_variance" / group_name
    head_dir.mkdir(parents=True, exist_ok=True)
    entropy_data = {}

    for (block_idx, head_idx), acc in hook_mgr.head_acc.items():
        var = acc.variance()
        mean = acc.get_mean()
        if var is None:
            continue

        arr = var.ravel()
        np.save(head_dir / f"{group_name}_block{block_idx}_head{head_idx}_variance.npy", arr)

        if mean is not None:
            np.save(head_dir / f"{group_name}_block{block_idx}_head{head_idx}_mean.npy", mean)

        ent = acc.get_entropy()
        if ent is not None:
            entropy_data[f"block{block_idx}_head{head_idx}"] = round(ent, 6)

        p99 = float(np.percentile(arr, 99))
        if p99 <= 0:
            p99 = float(np.max(arr)) + 1e-8

        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=bins, range=(0, p99), color='skyblue', edgecolor='black')
        plt.title(f"Head Variance - block {block_idx} head {head_idx} ({group_name})")
        plt.xlabel("Variance")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(head_dir / f"{group_name}_block{block_idx}_head{head_idx}_variance.png", dpi=150)
        plt.close()

    # Save entropy
    if entropy_data:
        ent_dir = out_dir / "entropy" / group_name
        ent_dir.mkdir(parents=True, exist_ok=True)
        with open(ent_dir / f"{group_name}_entropy.json", "w") as f:
            json.dump(entropy_data, f, indent=2)

    print(f"  [head] Saved {len(hook_mgr.head_acc)} head variance/mean/entropy files for '{group_name}'")


def save_comparison_plots(group_names: List[str], out_dir: Path, bins: int):
    """Cross-group comparison plots for blocks and heads."""
    # Block comparison
    comp_dir = out_dir / "comparison" / "blocks"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for b in range(NUM_BLOCKS):
        data = {}
        for g in group_names:
            p = out_dir / "block_variance" / g / f"{g}_block{b}_variance.npy"
            if p.exists():
                arr = np.load(p)
                if arr.size > 0:
                    data[g] = arr

        if not data:
            continue

        all_vals = np.concatenate(list(data.values()))
        p99 = float(np.percentile(all_vals, 99))

        plt.figure(figsize=(7, 5))
        for g, arr in data.items():
            plt.hist(arr, bins=bins, range=(0, p99), histtype='step', label=g, linewidth=1.5)
        plt.title(f"Block Variance Comparison - block {b}")
        plt.xlabel("Variance")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(comp_dir / f"compare_block{b}.png", dpi=150)
        plt.close()

    # Head comparison
    comp_dir = out_dir / "comparison" / "heads"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            data = {}
            for g in group_names:
                p = out_dir / "head_variance" / g / f"{g}_block{b}_head{h}_variance.npy"
                if p.exists():
                    arr = np.load(p)
                    if arr.size > 0:
                        data[g] = arr

            if not data:
                continue

            all_vals = np.concatenate(list(data.values()))
            p99 = float(np.percentile(all_vals, 99))
            if p99 <= 0:
                p99 = float(np.max(all_vals)) + 1e-8

            plt.figure(figsize=(7, 5))
            for g, arr in data.items():
                plt.hist(arr, bins=bins, range=(0, p99), histtype='step', label=g, linewidth=1.5)
            plt.title(f"Head Variance Comparison - block {b} head {h}")
            plt.xlabel("Variance")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(comp_dir / f"compare_block{b}_head{h}.png", dpi=150)
            plt.close()

    print(f"  [comparison] Saved comparison plots")


def save_before_after_plots(
    group_names: List[str], out_dir: Path, baseline_dir: Path,
    frozen_entries: List[dict], bins: int,
):
    """Before/after block and head plots with KS test."""
    ba_dir = out_dir / "before_after"
    colors = plt.cm.tab10(np.linspace(0, 0.3, len(group_names)))

    # Block before/after
    blk_dir = ba_dir / "blocks"
    blk_dir.mkdir(parents=True, exist_ok=True)

    for b in range(NUM_BLOCKS):
        all_arrs = []
        for src in [baseline_dir, out_dir]:
            for g in group_names:
                p = src / "block_variance" / g / f"{g}_block{b}_variance.npy"
                if p.exists():
                    arr = np.load(p)
                    if arr.size > 0:
                        all_arrs.append(arr)

        if not all_arrs:
            continue

        p99 = float(np.percentile(np.concatenate(all_arrs), 99))
        if p99 <= 0:
            p99 = float(np.max(np.concatenate(all_arrs))) + 1e-8

        fig, axes = plt.subplots(2, 1, figsize=(8, 8), squeeze=False)
        fig.suptitle(f"Block {b} -- Before/After Freezing", fontsize=14, fontweight="bold")

        for row, (label, src) in enumerate([("Before", baseline_dir), ("After", out_dir)]):
            ax = axes[row][0]
            for gi, g in enumerate(group_names):
                p = src / "block_variance" / g / f"{g}_block{b}_variance.npy"
                if p.exists():
                    arr = np.load(p)
                    if arr.size > 0:
                        ax.hist(arr, bins=bins, range=(0, p99), histtype='step',
                                label=g, color=colors[gi], linewidth=1.5)

            ax.set_ylabel(f"{label}\nFrequency", fontsize=10)
            ax.set_xlabel("Variance")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(blk_dir / f"before_after_block{b}.png", dpi=150)
        plt.close()

    # Head before/after
    frozen_set = set()
    for e in frozen_entries:
        frozen_set.add((e["block"], e["head"]))

    head_ba_dir = ba_dir / "heads"
    head_ba_dir.mkdir(parents=True, exist_ok=True)

    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            all_arrs = []
            for src in [baseline_dir, out_dir]:
                for g in group_names:
                    p = src / "head_variance" / g / f"{g}_block{b}_head{h}_variance.npy"
                    if p.exists():
                        arr = np.load(p)
                        if arr.size > 0:
                            all_arrs.append(arr)

            if not all_arrs:
                continue

            p99 = float(np.percentile(np.concatenate(all_arrs), 99))
            if p99 <= 0:
                p99 = float(np.max(np.concatenate(all_arrs))) + 1e-8

            is_frozen = (b, h) in frozen_set
            tag = " [FROZEN]" if is_frozen else ""

            fig, axes = plt.subplots(2, 1, figsize=(8, 8), squeeze=False)
            fig.suptitle(f"Block {b} Head {h}{tag} -- Before/After",
                         fontsize=14, fontweight="bold",
                         color="red" if is_frozen else "black")

            for row, (label, src) in enumerate([("Before", baseline_dir), ("After", out_dir)]):
                ax = axes[row][0]
                for gi, g in enumerate(group_names):
                    p = src / "head_variance" / g / f"{g}_block{b}_head{h}_variance.npy"
                    if p.exists():
                        arr = np.load(p)
                        if arr.size > 0:
                            ax.hist(arr, bins=bins, range=(0, p99), histtype='step',
                                    label=g, color=colors[gi], linewidth=1.5)

                # KS test
                if row == 0 and SCIPY_AVAILABLE:
                    ks_texts = []
                    for g in group_names:
                        bp = baseline_dir / "head_variance" / g / f"{g}_block{b}_head{h}_variance.npy"
                        ep = out_dir / "head_variance" / g / f"{g}_block{b}_head{h}_variance.npy"
                        if bp.exists() and ep.exists():
                            ba, ea = np.load(bp), np.load(ep)
                            if ba.size > 0 and ea.size > 0:
                                ks_stat, ks_p = scipy_stats.ks_2samp(ba, ea)
                                sig = "*" if ks_p < 0.05 else ""
                                ks_texts.append(f"{g}:p={ks_p:.3f}{sig}")
                    if ks_texts:
                        ax.set_title(f"KS: {', '.join(ks_texts)}", fontsize=8)

                ax.set_ylabel(f"{label}\nFrequency", fontsize=10)
                ax.set_xlabel("Variance")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

                if is_frozen and row == 1:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(2)

            plt.tight_layout()
            plt.savefig(head_ba_dir / f"before_after_block{b}_head{h}.png", dpi=150)
            plt.close()

    print(f"  [before/after] Saved before/after plots")


# ==============================================================================
# S-VELCRO Freeze Logic (channel-rank method)
# ==============================================================================

def load_frozen_heads_svelcro(
    profile_dir: Path, profile_group: str,
    num_freeze_per_block: Optional[int] = None,
    freeze_fraction: float = 0.25,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[dict], int, int]:
    """
    S-VELCRO channel-rank head selection.

    For each block, rank heads by their variance sum (channel rank).
    Freeze the T_k lowest-ranked heads per block (lowest variance sum =
    most value locality = most compressible).

    No reference group is needed -- only the specialized group's variance
    profiles are used.

    Args:
        profile_dir: Directory containing head_variance/<profile_group>/ files.
        profile_group: Name of the specialized group to load variances from.
        num_freeze_per_block: Exact number of heads to freeze per block.
            If None, uses freeze_fraction * NUM_HEADS (rounded down).
        freeze_fraction: Fraction of heads to freeze per block when
            num_freeze_per_block is not specified. Default 0.25.

    Returns:
        frozen_heads: Dict mapping (block_idx, head_idx) -> mean array.
        frozen_entries: List of dicts with per-head metadata.
        n_frozen: Total number of frozen heads.
        n_total: Total number of heads checked.
    """
    spec_dir = profile_dir / "head_variance" / profile_group

    # Determine how many heads to freeze per block
    if num_freeze_per_block is not None:
        t_k = num_freeze_per_block
    else:
        t_k = int(freeze_fraction * NUM_HEADS)

    t_k = max(0, min(t_k, NUM_HEADS))

    frozen_heads = {}
    frozen_entries = []
    n_total = 0
    n_frozen = 0

    for b in range(NUM_BLOCKS):
        # Compute channel rank for each head in this block
        head_ranks = []  # list of (head_idx, rank_value, variance_path)
        for h in range(NUM_HEADS):
            var_path = spec_dir / f"{profile_group}_block{b}_head{h}_variance.npy"
            if not var_path.exists():
                continue
            n_total += 1
            var_arr = np.load(var_path)
            rank_value = float(np.sum(var_arr))
            head_ranks.append((h, rank_value))

        # Sort by rank ascending (lowest variance sum = most value locality)
        head_ranks.sort(key=lambda x: x[1])

        # Freeze the T_k lowest-ranked heads
        heads_to_freeze = head_ranks[:t_k]

        for h, rank_value in heads_to_freeze:
            mean_path = spec_dir / f"{profile_group}_block{b}_head{h}_mean.npy"
            if not mean_path.exists():
                print(f"  [WARN] Mean missing: block {b} head {h}")
                continue

            mean_arr = np.load(mean_path)
            frozen_heads[(b, h)] = mean_arr
            frozen_entries.append({
                "block": b, "head": h,
                "channel_rank": rank_value,
                "mean_path": str(mean_path),
            })
            n_frozen += 1

    return frozen_heads, frozen_entries, n_frozen, n_total


# ==============================================================================
# Freeze Manifest
# ==============================================================================

def save_freeze_manifest(
    out_dir: Path, profile_dir: str, profile_group: str,
    num_freeze_per_block: Optional[int], freeze_fraction: float,
    frozen_entries: List[dict], n_frozen: int, n_total: int,
) -> Path:
    manifest = {
        "version": "1.5-vit-svelcro",
        "model": "vit_base_patch16_224",
        "profile_dir": str(profile_dir),
        "profile_group": profile_group,
        "num_freeze_per_block": num_freeze_per_block,
        "freeze_fraction": freeze_fraction,
        "freeze_criterion": "S-VELCRO channel rank: freeze T_k lowest-ranked heads per block",
        "frozen_heads": frozen_entries,
        "summary": {
            "total_heads_checked": n_total,
            "total_frozen": n_frozen,
            "freeze_rate": round(n_frozen / n_total, 4) if n_total > 0 else 0.0,
        }
    }
    manifest_path = out_dir / "freeze_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def load_freeze_manifest(path: str):
    with open(path) as f:
        manifest = json.load(f)
    frozen_heads = {}
    for e in manifest["frozen_heads"]:
        mean_path = e["mean_path"]
        if not Path(mean_path).exists():
            print(f"  [WARN] Missing: {mean_path}")
            continue
        frozen_heads[(e["block"], e["head"])] = np.load(mean_path)
    return frozen_heads, manifest


def save_freeze_map(frozen_entries: List[dict], out_dir: Path,
                    num_freeze_per_block: Optional[int], freeze_fraction: float):
    """Save 12x12 freeze map heatmap."""
    maps_dir = out_dir / "freeze_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.zeros((NUM_BLOCKS, NUM_HEADS), dtype=np.float32)
    for e in frozen_entries:
        matrix[e["block"], e["head"]] = 1.0

    if num_freeze_per_block is not None:
        title_param = f"T_k={num_freeze_per_block}/block"
    else:
        title_param = f"frac={freeze_fraction}"

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = matplotlib.colors.ListedColormap(["#2196F3", "#FF5722"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(NUM_HEADS))
    ax.set_xticklabels([f"H{h}" for h in range(NUM_HEADS)], fontsize=8)
    ax.set_yticks(range(NUM_BLOCKS))
    ax.set_yticklabels([f"Block {b}" for b in range(NUM_BLOCKS)], fontsize=9)
    ax.set_xlabel("Head")
    ax.set_ylabel("Block")
    ax.set_title(f"S-VELCRO Freeze Map ({title_param})\nBlue=active, Red=frozen")

    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            txt = "F" if matrix[b, h] > 0 else ""
            ax.text(h, b, txt, ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white" if matrix[b, h] > 0 else "gray")

    plt.tight_layout()
    plt.savefig(maps_dir / "freeze_map.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary text
    with open(maps_dir / "freeze_summary.txt", "w") as f:
        if num_freeze_per_block is not None:
            f.write(f"S-VELCRO channel rank: T_k = {num_freeze_per_block} heads/block\n")
        else:
            f.write(f"S-VELCRO channel rank: freeze_fraction = {freeze_fraction}\n")
        f.write(f"Total frozen: {len(frozen_entries)} / {NUM_BLOCKS * NUM_HEADS}\n\n")
        f.write(f"{'Block':<8} {'Head':<6} {'Channel Rank':<18}\n")
        f.write("-" * 35 + "\n")
        for e in sorted(frozen_entries, key=lambda x: (x["block"], x["head"])):
            f.write(f"{e['block']:<8} {e['head']:<6} {e['channel_rank']:<18.6f}\n")

    print(f"  [freeze map] Saved to {maps_dir}")


# ==============================================================================
# FLOP Calculation
# ==============================================================================

def compute_attention_flops_per_head():
    """FLOPs for one head in ViT: Q*K^T + softmax + attn*V."""
    return 4 * SEQ_LEN * SEQ_LEN * HEAD_DIM + 5 * SEQ_LEN * SEQ_LEN


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="ViT Value Locality Analysis — S-VELCRO channel-rank (profile / freeze / eval)"
    )
    ap.add_argument("--mode", required=True, choices=AVAILABLE_MODES)
    ap.add_argument("--image_dirs", nargs="+", default=None,
                    help="Directories of images (profile/eval). Each dir has val/ subfolder.")
    ap.add_argument("--group_names", default=None, help="Comma-separated group names")
    ap.add_argument("--model_name", default="vit_base_patch16_224")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_batches", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--synset_words", default=None,
                    help="Path to synset_words.txt for correct ImageNet label mapping. "
                         "Without this, accuracy is computed against local alphabetical indices "
                         "and will be wrong.")

    # Freeze args (S-VELCRO: no reference group or variance threshold needed)
    ap.add_argument("--profile_dir", default=None)
    ap.add_argument("--profile_group", default=None)
    ap.add_argument("--num_freeze_per_block", type=int, default=None,
                    help="Exact number of heads to freeze per block (S-VELCRO).")
    ap.add_argument("--freeze_fraction", type=float, default=0.25,
                    help="Fraction of heads to freeze per block if --num_freeze_per_block "
                         "not given. Default 0.25.")

    # Eval args
    ap.add_argument("--freeze_manifest", default=None)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)

    # ======================================================================
    # FREEZE MODE
    # ======================================================================
    if args.mode == "freeze":
        if not args.profile_dir or not args.profile_group:
            raise ValueError("--profile_dir, --profile_group required")

        profile_dir = Path(args.profile_dir)

        # Resolve effective T_k for display
        if args.num_freeze_per_block is not None:
            effective_tk = args.num_freeze_per_block
        else:
            effective_tk = int(args.freeze_fraction * NUM_HEADS)

        print("\n" + "=" * 70)
        print("FREEZE MODE — S-VELCRO Channel Rank (ViT)")
        print("=" * 70)
        print(f"Profile dir           : {args.profile_dir}")
        print(f"Specialized group     : {args.profile_group}")
        print(f"Freeze per block (T_k): {effective_tk}")
        if args.num_freeze_per_block is not None:
            print(f"  (from --num_freeze_per_block {args.num_freeze_per_block})")
        else:
            print(f"  (from --freeze_fraction {args.freeze_fraction})")
        print("=" * 70)

        frozen_heads, frozen_entries, n_frozen, n_total = load_frozen_heads_svelcro(
            profile_dir, args.profile_group,
            num_freeze_per_block=args.num_freeze_per_block,
            freeze_fraction=args.freeze_fraction,
        )

        print(f"\n  Checked: {n_total}  Frozen: {n_frozen} "
              f"({n_frozen/n_total*100:.1f}%)" if n_total > 0 else "  No heads found")

        manifest_path = save_freeze_manifest(
            out_dir, args.profile_dir, args.profile_group,
            args.num_freeze_per_block, args.freeze_fraction,
            frozen_entries, n_frozen, n_total,
        )
        save_freeze_map(frozen_entries, out_dir,
                        args.num_freeze_per_block, args.freeze_fraction)

        print(f"\n  Manifest: {manifest_path}")
        print("=" * 70)
        return

    # ======================================================================
    # PROFILE / EVAL MODE (both need model + data)
    # ======================================================================
    if not args.image_dirs:
        raise ValueError("--image_dirs required for profile/eval modes")

    group_names = (
        [Path(d).name for d in args.image_dirs]
        if args.group_names is None
        else [x.strip() for x in args.group_names.split(",")]
    )
    if len(group_names) != len(args.image_dirs):
        raise ValueError("group_names count must match image_dirs count")

    mode_label = "PROFILE" if args.mode == "profile" else "EVAL (MEAN-HEAD)"
    print("\n" + "=" * 70)
    print(f"ViT VALUE LOCALITY — {mode_label}")
    print("=" * 70)
    print(f"Model         : {args.model_name}")
    print(f"Blocks        : {NUM_BLOCKS}")
    print(f"Heads/block   : {NUM_HEADS}")
    print(f"Head dim      : {HEAD_DIM}")
    print(f"Seq len       : {SEQ_LEN}")
    print(f"Max batches   : {args.max_batches}")
    print(f"Batch size    : {args.batch_size}")
    for g, d in zip(group_names, args.image_dirs):
        print(f"  {g:15s}: {d}")
    print("=" * 70)

    # Load model
    print("\n[INFO] Loading model...")
    net = get_net(args.model_name)
    device = next(net.parameters()).device

    # Eval mode: load manifest and apply mean heads
    patcher = None
    if args.mode == "eval":
        if not args.freeze_manifest:
            raise ValueError("--freeze_manifest required for eval mode")
        frozen_heads, manifest = load_freeze_manifest(args.freeze_manifest)
        print(f"\n  Freeze manifest: {args.freeze_manifest}")
        print(f"  Frozen heads   : {len(frozen_heads)}")
        patcher = ViTMeanHeadPatcher(net, frozen_heads)
        patcher.apply()

    # Per-group inference
    all_accuracy = {}

    for g, img_dir in zip(group_names, args.image_dirs):
        print(f"\n  Group: {g} ({img_dir})")

        # Fresh hook manager per group
        hook_mgr = ViTHookManager(net)
        hook_mgr.register()

        data_gen = load_data(img_dir, net, args.batch_size, args.num_workers)
        apply_imagenet_label_mapping(data_gen, args.synset_words)

        t0 = time.time()
        top1, top5, total = run_inference(net, data_gen, args.max_batches, hook_mgr)
        elapsed = time.time() - t0

        top1_acc = top1 / total * 100 if total > 0 else 0
        top5_acc = top5 / total * 100 if total > 0 else 0
        all_accuracy[g] = {
            "top1": round(top1_acc, 2), "top5": round(top5_acc, 2),
            "correct_top1": top1, "correct_top5": top5, "total": total,
            "time_sec": round(elapsed, 2),
        }

        print(f"    Top-1: {top1_acc:.2f}%  Top-5: {top5_acc:.2f}%  "
              f"({total} images, {elapsed:.1f}s)")

        save_variances(g, hook_mgr, out_dir, args.bins)
        hook_mgr.remove()

    # Comparison plots
    if len(group_names) > 1:
        print("\n[INFO] Saving comparison plots...")
        save_comparison_plots(group_names, out_dir, args.bins)

    # Before/after plots (eval mode)
    if args.mode == "eval" and args.profile_dir:
        frozen_entries = manifest.get("frozen_heads", [])
        print("\n[INFO] Saving before/after plots...")
        save_before_after_plots(
            group_names, out_dir, Path(args.profile_dir),
            frozen_entries, args.bins,
        )

    # Restore model
    if patcher is not None:
        patcher.restore()

    # Save metadata
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        "mode": args.mode,
        "model": args.model_name,
        "groups": group_names,
        "accuracy": all_accuracy,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
    }
    if args.mode == "eval":
        run_info["freeze_manifest"] = args.freeze_manifest

        # Compute report
        flops_per_head = compute_attention_flops_per_head()
        frozen_blocks = defaultdict(set)
        for e in manifest.get("frozen_heads", []):
            frozen_blocks[e["block"]].add(e["head"])

        total_flops = NUM_BLOCKS * NUM_HEADS * flops_per_head
        saved_flops = sum(len(heads) for heads in frozen_blocks.values()) * flops_per_head
        run_info["flop_savings_pct"] = round(saved_flops / total_flops * 100, 2) if total_flops > 0 else 0

    with open(meta_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DONE ({mode_label})")
    print(f"Output: {args.out_dir}")

    if args.mode == "eval":
        # Print accuracy comparison
        print("\nACCURACY SUMMARY:")
        print(f"{'Group':<15} {'Top-1':<10} {'Top-5':<10} {'Images':<10}")
        print("-" * 45)
        for g, acc in all_accuracy.items():
            print(f"{g:<15} {acc['top1']:<10.2f} {acc['top5']:<10.2f} {acc['total']:<10}")

    print("=" * 70)


if __name__ == "__main__":
    main()
