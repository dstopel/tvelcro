#!/usr/bin/env python3
"""
vit_giant_svelcro.py

Value locality analysis for DINOv2 ViT-Giant using S-VELCRO
channel-rank method for head freezing.

Three-phase pipeline:

Phase 1 (profile):  Welford variance + mean + entropy per head and block
Phase 2 (freeze):   S-VELCRO channel-rank: rank heads by variance sum,
                     freeze the T_k lowest-ranked heads per block.
                     No reference group needed.
Phase 3 (eval):     Mean-head replacement + accuracy measurement

Architecture:
  facebook/dinov2-giant-imagenet1k-1-layer
  40 blocks x 24 heads x 64 head_dim x 1370 seq_len (at 518x518)
  Separate Q/K/V projections (not fused)
  Classifier: Linear(3072 -> 1000) using CLS + mean patch concat

Freezing approach:
  Output-only — pre-hook on attention.output.dense replaces frozen
  head slots with precomputed mean. No Q/K/V weight modification.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


# ==============================================================================
# Constants
# ==============================================================================

NUM_BLOCKS = 40
NUM_HEADS = 24
HEAD_DIM = 64
HIDDEN_DIM = 1536    # NUM_HEADS * HEAD_DIM
MODEL_NAME = "facebook/dinov2-giant-imagenet1k-1-layer"

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
# Hook Manager for DINOv2 ViT-Giant
# ==============================================================================

class ViTHookManager:
    """
    Registers hooks on DINOv2 ViT-Giant encoder layers.

    Two hook types:
      1. Block output hook: captures output of full encoder layer
         Shape: (B, seq_len, 1536) -- block-level variance
      2. Pre-projection head hook: captures INPUT to attention.output.dense
         Shape: (B, seq_len, 1536) -- split into (B, seq_len, 24, 64)
         This is BEFORE output.dense mixes heads, so heads are still separable.

    HuggingFace DINOv2 structure:
      model.dinov2.encoder.layer[i]                          -- encoder layer
      model.dinov2.encoder.layer[i].attention.output.dense   -- output projection
    """

    def __init__(self, model):
        self.model = model
        self.handles: List = []
        self.block_acc: Dict[int, WelfordAccumulator] = {}
        self.head_acc: Dict[Tuple[int, int], WelfordAccumulator] = {}

    def _get_encoder_layers(self):
        return self.model.dinov2.encoder.layer

    def register(self):
        layers = self._get_encoder_layers()
        for i, layer in enumerate(layers):
            self.handles.append(
                layer.register_forward_hook(self._make_block_hook(i))
            )
            if hasattr(layer, "attention") and hasattr(layer.attention, "output"):
                self.handles.append(
                    layer.attention.output.dense.register_forward_hook(
                        self._make_head_hook(i)
                    )
                )

    def _make_block_hook(self, block_idx: int):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                x = output[0].detach().cpu()
            else:
                x = output.detach().cpu()
            if block_idx not in self.block_acc:
                self.block_acc[block_idx] = WelfordAccumulator()
            for bi in range(x.shape[0]):
                self.block_acc[block_idx].update(x[bi])
        return hook

    def _make_head_hook(self, block_idx: int):
        def hook(module, inputs, output):
            x = inputs[0].detach().cpu()
            B, N, C = x.shape
            x = x.view(B, N, NUM_HEADS, HEAD_DIM)

            for bi in range(B):
                for h in range(NUM_HEADS):
                    key = (block_idx, h)
                    if key not in self.head_acc:
                        self.head_acc[key] = WelfordAccumulator()
                    head_data = x[bi, :, h, :]
                    self.head_acc[key].update(head_data)
                    self.head_acc[key].update_entropy(head_data)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ==============================================================================
# Mean Head Patcher for DINOv2 ViT-Giant
# ==============================================================================

class ViTMeanHeadPatcher:
    """
    Output-only head freezing for DINOv2 ViT-Giant.

    Registers pre-hook on attention.output.dense to overwrite frozen head
    slots with precomputed mean. No Q/K/V weights are modified.

    Restore is trivial: just remove hooks.
    """

    def __init__(self, model, frozen_heads: Dict[Tuple[int, int], np.ndarray]):
        self.model = model
        self.frozen_heads = frozen_heads
        self.hooks: List = []

        self.block_frozen: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        for (block_idx, head_idx), mean_arr in frozen_heads.items():
            self.block_frozen[block_idx][head_idx] = mean_arr

    def apply(self):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        layers = self.model.dinov2.encoder.layer

        for block_idx, head_means in self.block_frozen.items():
            layer = layers[block_idx]
            proj = layer.attention.output.dense

            mean_tensors = {}
            for h, mean_arr in head_means.items():
                mean_tensors[h] = torch.from_numpy(
                    mean_arr.astype(np.float32)
                ).to(device=device, dtype=dtype)

            frozen_sorted = sorted(head_means.keys())
            self._add_hook(proj, frozen_sorted, mean_tensors)

            print(f"  [mean-head] block {block_idx}: heads {frozen_sorted} "
                  f"-> {NUM_HEADS - len(frozen_sorted)} active + {len(frozen_sorted)} mean")

    def _add_hook(self, proj_module, frozen_sorted, mean_tensors):
        def hook(module, inputs):
            x = inputs[0]
            B, N, C = x.shape
            reshaped = x.view(B, N, NUM_HEADS, HEAD_DIM).clone()

            for h in frozen_sorted:
                if h in mean_tensors:
                    mean_h = mean_tensors[h]
                    if mean_h.shape[0] != N:
                        mean_h = mean_h.mean(dim=0, keepdim=True).expand(N, -1)
                    reshaped[:, :, h, :] = mean_h.unsqueeze(0)

            return (reshaped.view(B, N, C),)

        self.hooks.append(proj_module.register_forward_pre_hook(hook))

    def restore(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ==============================================================================
# ImageNet Label Mapping
# ==============================================================================

def build_imagenet_label_mapping(synset_words_path: str, class_names: List[str]) -> Dict[int, int]:
    folder_to_imagenet: Dict[str, int] = {}
    with open(synset_words_path) as f:
        for imagenet_idx, line in enumerate(f):
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            synonyms = [s.strip() for s in parts[1].split(',')]
            joined = '_'.join(s.replace(' ', '_') for s in synonyms)
            folder_to_imagenet[joined] = imagenet_idx
            folder_to_imagenet[joined.lower()] = imagenet_idx
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
            mapping[local_idx] = local_idx
        else:
            mapping[local_idx] = imagenet_idx

    if missing:
        print(f"  [WARN] {len(missing)} class(es) not in synset_words.txt "
              f"(using local idx): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return mapping


# ==============================================================================
# Data Loading
# ==============================================================================

class ProcessorTransform:
    """Wraps HuggingFace AutoImageProcessor as a torchvision transform."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)


def load_data(img_path: str, processor, max_images: Optional[int] = None):
    transform = ProcessorTransform(processor)
    val_dir = os.path.join(img_path, "val")
    if not os.path.isdir(val_dir):
        val_dir = img_path
    dataset = ImageFolder(val_dir, transform=transform)
    available = len(dataset)
    if max_images is not None and max_images < available:
        dataset = Subset(dataset, list(range(max_images)))
        print(f"    Using {max_images}/{available} images")
    else:
        print(f"    Using all {available} images")
    return dataset


def apply_imagenet_label_mapping(dataset, synset_words_path: Optional[str]) -> None:
    if not synset_words_path:
        print("  [WARN] --synset_words not set: accuracy will use local indices")
        return
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    mapping = build_imagenet_label_mapping(synset_words_path, base_dataset.classes)
    base_dataset.target_transform = lambda y, m=mapping: m.get(y, y)
    print(f"  [label mapping] {len(base_dataset.classes)} classes mapped via {synset_words_path}")


# ==============================================================================
# Inference
# ==============================================================================

def run_inference(model, dataset, batch_size, num_workers,
                  hook_mgr: Optional[ViTHookManager] = None):
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    total_batches = len(loader)
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.inference_mode():
        for i, (inp, target) in enumerate(loader):
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(inp)
            logits = output.logits if hasattr(output, 'logits') else output

            _, pred = logits.max(1)
            top1_correct += pred.eq(target).sum().item()

            _, pred5 = logits.topk(5, 1, True, True)
            top5_correct += pred5.eq(target.unsqueeze(1).expand_as(pred5)).sum().item()

            total += target.size(0)

            if (i + 1) % 5 == 0:
                print(f"    batch {i+1}/{total_batches}  "
                      f"top1={top1_correct/total*100:.1f}%  "
                      f"top5={top5_correct/total*100:.1f}%")

    return top1_correct, top5_correct, total


# ==============================================================================
# Save Functions
# ==============================================================================

def save_variances(group_name: str, hook_mgr: ViTHookManager, out_dir: Path, bins: int):
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

    if entropy_data:
        ent_dir = out_dir / "entropy" / group_name
        ent_dir.mkdir(parents=True, exist_ok=True)
        with open(ent_dir / f"{group_name}_entropy.json", "w") as f:
            json.dump(entropy_data, f, indent=2)

    print(f"  [head] Saved {len(hook_mgr.head_acc)} head variance/mean/entropy files for '{group_name}'")


def save_comparison_plots(group_names: List[str], out_dir: Path, bins: int):
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
    ba_dir = out_dir / "before_after"
    colors = plt.cm.tab10(np.linspace(0, 0.3, len(group_names)))

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
# S-VELCRO Freeze Logic
# ==============================================================================

def load_frozen_heads_svelcro(
    profile_dir: Path, profile_group: str,
    num_freeze_per_block: Optional[int] = None,
    freeze_fraction: float = 0.25,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[dict], int, int]:
    spec_dir = profile_dir / "head_variance" / profile_group

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
        head_ranks = []
        for h in range(NUM_HEADS):
            var_path = spec_dir / f"{profile_group}_block{b}_head{h}_variance.npy"
            if not var_path.exists():
                continue
            n_total += 1
            var_arr = np.load(var_path)
            rank_value = float(np.sum(var_arr))
            head_ranks.append((h, rank_value))

        head_ranks.sort(key=lambda x: x[1])
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
        "version": "2.0-dinov2-giant-svelcro",
        "model": MODEL_NAME,
        "profile_dir": str(profile_dir),
        "profile_group": profile_group,
        "num_freeze_per_block": num_freeze_per_block,
        "freeze_fraction": freeze_fraction,
        "freeze_criterion": "S-VELCRO channel rank: freeze T_k lowest-ranked heads per block",
        "freeze_method": "output-only (pre-hook on attention.output.dense, no weight modification)",
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
    maps_dir = out_dir / "freeze_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.zeros((NUM_BLOCKS, NUM_HEADS), dtype=np.float32)
    for e in frozen_entries:
        matrix[e["block"], e["head"]] = 1.0

    if num_freeze_per_block is not None:
        title_param = f"T_k={num_freeze_per_block}/block"
    else:
        title_param = f"frac={freeze_fraction}"

    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = matplotlib.colors.ListedColormap(["#2196F3", "#FF5722"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(NUM_HEADS))
    ax.set_xticklabels([f"H{h}" for h in range(NUM_HEADS)], fontsize=7)
    ax.set_yticks(range(NUM_BLOCKS))
    ax.set_yticklabels([f"B{b}" for b in range(NUM_BLOCKS)], fontsize=7)
    ax.set_xlabel("Head")
    ax.set_ylabel("Block")
    ax.set_title(f"S-VELCRO Freeze Map — DINOv2 ViT-Giant ({title_param})\nBlue=active, Red=frozen")

    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            txt = "F" if matrix[b, h] > 0 else ""
            ax.text(h, b, txt, ha="center", va="center", fontsize=5,
                    fontweight="bold", color="white" if matrix[b, h] > 0 else "gray")

    plt.tight_layout()
    plt.savefig(maps_dir / "freeze_map.png", dpi=150, bbox_inches="tight")
    plt.close()

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

def compute_attention_flops_per_head(seq_len: int):
    return 4 * seq_len * seq_len * HEAD_DIM + 5 * seq_len * seq_len


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="DINOv2 ViT-Giant Value Locality — S-VELCRO (profile / freeze / eval)"
    )
    ap.add_argument("--mode", required=True, choices=AVAILABLE_MODES)
    ap.add_argument("--image_dirs", nargs="+", default=None)
    ap.add_argument("--group_names", default=None, help="Comma-separated group names")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=None,
                    help="Max images to use per group. If more than available, uses all. "
                         "If not set, uses all images in the directory.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--synset_words", default=None,
                    help="Path to synset_words.txt for correct ImageNet label mapping.")

    # Freeze args
    ap.add_argument("--profile_dir", default=None)
    ap.add_argument("--profile_group", default=None)
    ap.add_argument("--num_freeze_per_block", type=int, default=None)
    ap.add_argument("--freeze_fraction", type=float, default=0.25)

    # Eval args
    ap.add_argument("--freeze_manifest", default=None)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)

    # ==================================================================
    # FREEZE MODE
    # ==================================================================
    if args.mode == "freeze":
        if not args.profile_dir or not args.profile_group:
            raise ValueError("--profile_dir, --profile_group required")

        profile_dir = Path(args.profile_dir)

        if args.num_freeze_per_block is not None:
            effective_tk = args.num_freeze_per_block
        else:
            effective_tk = int(args.freeze_fraction * NUM_HEADS)

        print("\n" + "=" * 70)
        print("FREEZE MODE — S-VELCRO Channel Rank (DINOv2 ViT-Giant)")
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

    # ==================================================================
    # PROFILE / EVAL MODE
    # ==================================================================
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
    print(f"DINOv2 ViT-Giant VALUE LOCALITY — {mode_label}")
    print("=" * 70)
    print(f"Model         : {MODEL_NAME}")
    print(f"Blocks        : {NUM_BLOCKS}")
    print(f"Heads/block   : {NUM_HEADS}")
    print(f"Head dim      : {HEAD_DIM}")
    print(f"Hidden dim    : {HIDDEN_DIM}")
    print(f"Max images    : {args.max_images or 'all'}")
    print(f"Batch size    : {args.batch_size}")
    for g, d in zip(group_names, args.image_dirs):
        print(f"  {g:15s}: {d}")
    print("=" * 70)

    # Load model
    print("\n[INFO] Loading DINOv2 ViT-Giant with ImageNet-1k linear head...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print(f"  Device: {torch.cuda.get_device_name()}")
    else:
        print("  Device: CPU (WARNING: will be very slow)")

    # Determine seq_len from model config
    img_size = processor.size.get("height", processor.size.get("shortest_edge", 518))
    patch_size = model.config.patch_size
    seq_len = (img_size // patch_size) ** 2 + 1
    print(f"  Image size  : {img_size}")
    print(f"  Patch size  : {patch_size}")
    print(f"  Seq len     : {seq_len}")

    # Eval mode: load manifest and apply mean heads
    patcher = None
    if args.mode == "eval":
        if not args.freeze_manifest:
            raise ValueError("--freeze_manifest required for eval mode")
        frozen_heads, manifest = load_freeze_manifest(args.freeze_manifest)
        print(f"\n  Freeze manifest: {args.freeze_manifest}")
        print(f"  Frozen heads   : {len(frozen_heads)}")
        patcher = ViTMeanHeadPatcher(model, frozen_heads)
        patcher.apply()

    # Per-group inference
    all_accuracy = {}

    for g, img_dir in zip(group_names, args.image_dirs):
        print(f"\n  Group: {g} ({img_dir})")

        hook_mgr = ViTHookManager(model)
        hook_mgr.register()

        dataset = load_data(img_dir, processor, args.max_images)
        apply_imagenet_label_mapping(dataset, args.synset_words)

        t0 = time.time()
        top1, top5, total = run_inference(
            model, dataset, args.batch_size, args.num_workers, hook_mgr,
        )
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

    # Comparison plots (per-head, all groups overlaid)
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
        "model": MODEL_NAME,
        "groups": group_names,
        "accuracy": all_accuracy,
        "max_images": args.max_images,
        "batch_size": args.batch_size,
        "image_size": img_size,
        "seq_len": seq_len,
    }
    if args.mode == "eval":
        run_info["freeze_manifest"] = args.freeze_manifest

        flops_per_head = compute_attention_flops_per_head(seq_len)
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
        print("\nACCURACY SUMMARY:")
        print(f"{'Group':<15} {'Top-1':<10} {'Top-5':<10} {'Images':<10}")
        print("-" * 45)
        for g, acc in all_accuracy.items():
            print(f"{g:<15} {acc['top1']:<10.2f} {acc['top5']:<10.2f} {acc['total']:<10}")

    print("=" * 70)


if __name__ == "__main__":
    main()
