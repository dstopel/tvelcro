"""S-VELCRO channel ranking and freeze manifest management."""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    GLOBAL_ATTN_INDEXES,
    NUM_BLOCKS,
    NUM_HEADS,
)
from .flops import compute_flop_reduction


def portable_profile_root(profile_dir) -> str:
    """Return profile_dir relative to cwd (repo root) when possible.

    Relative roots keep manifests valid after the repo moves to another
    machine (e.g. a RunPod pod), as long as scripts run from repo root.
    """
    rel = os.path.relpath(str(profile_dir))
    return rel if not rel.startswith("..") else str(profile_dir)


def resolve_profile_path(manifest: dict, stored_path: str) -> Path:
    """Resolve a manifest-stored profile path.

    New manifests store paths relative to manifest["profile_root"]; legacy
    manifests stored absolute paths, which pathlib joining passes through
    unchanged (Path(root) / "/abs/x" == "/abs/x").
    """
    return Path(manifest.get("profile_root", ".")) / stored_path


def compute_channel_ranks(profile_dir: Path, group_name: str) -> List[dict]:
    group_dir = profile_dir / group_name
    if not group_dir.exists():
        raise FileNotFoundError(f"Profile directory not found: {group_dir}")

    # Entropy scores were computed at profile time (see hooks.py save()).
    entropies = {}
    summary_path = group_dir / "profile_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            for key, head in json.load(f).get("heads", {}).items():
                entropies[key] = head.get("entropy")

    entries = []
    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            var_path = group_dir / f"block{b}_head{h}_variance.npy"
            if not var_path.exists():
                continue
            var_arr = np.load(var_path)
            rank_score = float(np.sum(var_arr))
            block_type = "global" if b in GLOBAL_ATTN_INDEXES else "windowed"
            entry = {
                "block": b,
                "head": h,
                "block_type": block_type,
                "rank_score": rank_score,
                "mean_shape": list(var_arr.shape),
                # Paths relative to profile_root (recorded in the manifest)
                "mean_path": f"{group_name}/block{b}_head{h}_mean.npy",
                "variance_path": f"{group_name}/block{b}_head{h}_variance.npy",
            }
            entropy = entropies.get(f"{b}_{h}")
            if entropy is not None:
                entry["entropy"] = entropy
            entries.append(entry)
    return entries


def select_heads_to_freeze(
    entries: List[dict],
    freeze_fraction: Optional[float] = None,
    num_freeze_per_block: Optional[int] = None,
    ranking_method: str = "variance",
    seed: int = 0,
) -> List[dict]:
    if num_freeze_per_block is not None:
        t_k = num_freeze_per_block
    elif freeze_fraction is not None:
        t_k = max(0, min(int(freeze_fraction * NUM_HEADS), NUM_HEADS - 1))
    else:
        raise ValueError("Must specify freeze_fraction or num_freeze_per_block")

    frozen = []
    for b in range(NUM_BLOCKS):
        block_entries = [e for e in entries if e["block"] == b]
        if ranking_method == "random":
            # Seeded shuffle, independent per block
            random.Random(seed * NUM_BLOCKS + b).shuffle(block_entries)
        elif ranking_method == "entropy":
            if any("entropy" not in e for e in block_entries):
                raise ValueError(
                    "entropy ranking requires 'entropy' values in profile_summary.json "
                    "(re-run profiling with the current code)"
                )
            block_entries.sort(key=lambda e: e["entropy"])
        elif ranking_method == "variance":
            block_entries.sort(key=lambda e: e["rank_score"])
        else:
            raise ValueError(f"Unknown ranking_method: {ranking_method}")
        for e in block_entries[:t_k]:
            frozen.append(e)

    return frozen


def select_heads_by_ratio(
    profile_dir: Path,
    domain_group: str,
    general_group: str,
    ratio_threshold: float,
) -> List[dict]:
    """Select heads to freeze based on variance reduction ratio between domain and general profiles."""
    domain_dir = profile_dir / domain_group
    general_dir = profile_dir / general_group

    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain profile directory not found: {domain_dir}")
    if not general_dir.exists():
        raise FileNotFoundError(f"General profile directory not found: {general_dir}")

    frozen = []
    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            domain_var_path = domain_dir / f"block{b}_head{h}_variance.npy"
            general_var_path = general_dir / f"block{b}_head{h}_variance.npy"

            if not domain_var_path.exists() or not general_var_path.exists():
                continue

            domain_var = np.load(domain_var_path)
            general_var = np.load(general_var_path)

            domain_score = float(np.sum(domain_var))
            general_score = float(np.sum(general_var))

            reduction = 1.0 - (domain_score / (general_score + 1e-10))

            if reduction > ratio_threshold:
                block_type = "global" if b in GLOBAL_ATTN_INDEXES else "windowed"
                frozen.append({
                    "block": b,
                    "head": h,
                    "block_type": block_type,
                    "rank_score": domain_score,
                    "mean_shape": list(domain_var.shape),
                    "mean_path": f"{domain_group}/block{b}_head{h}_mean.npy",
                    "variance_path": f"{domain_group}/block{b}_head{h}_variance.npy",
                    "reduction": reduction,
                    "general_score": general_score,
                })

    frozen.sort(key=lambda e: e["reduction"], reverse=True)
    return frozen


def build_manifest(
    frozen_entries: List[dict],
    profile_group: str,
    freeze_fraction: Optional[float],
    num_freeze_per_block: Optional[int],
    ratio_threshold: Optional[float] = None,
    profile_root: Optional[str] = None,
    ranking_method: str = "variance",
    seed: int = 0,
    mlp: Optional[dict] = None,
) -> dict:
    total_heads = NUM_BLOCKS * NUM_HEADS
    n_frozen = len(frozen_entries)
    n_windowed = sum(1 for e in frozen_entries if e["block_type"] == "windowed")
    n_global = n_frozen - n_windowed
    flop_info = compute_flop_reduction(frozen_entries)

    manifest = {
        "version": "5.0-sam-vit-h-svelcro",
        "model": "SAM ViT-H (vit_h)",
        "profile_group": profile_group,
        "profile_root": portable_profile_root(profile_root) if profile_root else ".",
        "freeze_fraction": freeze_fraction,
        "num_freeze_per_block": num_freeze_per_block,
        "freeze_method": "output-only pre-hook (replace frozen head slots with mean)",
        "ranking_method": ranking_method,
        "seed": seed,
        "frozen_heads": frozen_entries,
        "summary": {
            "total_heads": total_heads,
            "frozen": n_frozen,
            "frozen_windowed": n_windowed,
            "frozen_global": n_global,
            "freeze_rate": n_frozen / total_heads if total_heads > 0 else 0,
        },
        "flops": {
            "attention_flop_reduction_pct": round(flop_info["attention_flop_reduction_pct"], 2),
            "encoder_flop_reduction_pct": round(flop_info["encoder_flop_reduction_pct"], 2),
            "saved_gflops": round(flop_info["saved_gflops"], 1),
            "total_attn_gflops": round(flop_info["total_attn_gflops"], 1),
            "total_encoder_gflops": round(flop_info["total_encoder_gflops"], 1),
            "per_head_windowed_gflops": round(flop_info["per_head_windowed_gflops"], 2),
            "per_head_global_gflops": round(flop_info["per_head_global_gflops"], 2),
        },
    }

    if ratio_threshold is not None:
        manifest["ratio_threshold"] = ratio_threshold
        manifest["freeze_method_type"] = "ratio"
        manifest["ranking_method"] = "ratio"

    if mlp is not None:
        manifest["mlp"] = mlp

    return manifest


def save_manifest(manifest: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "freeze_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_frozen_heads_from_manifest(
    manifest: dict,
) -> Dict[Tuple[int, int], np.ndarray]:
    frozen = {}
    for entry in manifest["frozen_heads"]:
        b, h = entry["block"], entry["head"]
        mean_path = resolve_profile_path(manifest, entry["mean_path"])
        mean_arr = np.load(mean_path)
        frozen[(b, h)] = mean_arr
    return frozen
