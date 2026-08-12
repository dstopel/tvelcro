#!/usr/bin/env python3
"""
exp_combined_freeze.py — Combined Attention + MLP Freeze Experiment

This script:
  1. Profiles MLP neurons for a domain (Phase 1)
  2. Creates combined freeze config (25% attention heads + X% MLP neurons)
  3. Runs evaluation with both frozen (Phase 2)
  4. Performs miss rate decomposition (domain vs background)
  5. Checks domain coverage comparison (ref vs frozen)

Usage:
  python scripts/exp_combined_freeze.py \
      --checkpoint checkpoints/sam_vit_h_4b8939.pth \
      --domain kvasir \
      --image_dir ./data \
      --data_root ./data \
      --attn_manifest results_cross_domain/kvasir/freeze_kvasir_f0.375/freeze_manifest.json \
      --mlp_fractions 0.125,0.25,0.375

Skip profiling if already done:
  python scripts/exp_combined_freeze.py \
      --checkpoint checkpoints/sam_vit_h_4b8939.pth \
      --domain kvasir \
      --image_dir ./data \
      --attn_manifest results_cross_domain/kvasir/freeze_kvasir_f0.375/freeze_manifest.json \
      --skip_profile
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from sam_velcro_pkg.dataset import ImageDirDataset
from sam_velcro_pkg.evaluation import (
    DOMAIN_THRESHOLDS,
    evaluate_masks,
    match_masks,
)
from sam_velcro_pkg.freeze import load_frozen_heads_from_manifest, load_manifest
from sam_velcro_pkg.gt_masks import load_gt_masks
from sam_velcro_pkg.hooks import FreezeHookManager
from sam_velcro_pkg.mlp_hooks import (
    MLPFreezeHookManager,
    MLPProfileHookManager,
    select_mlp_neurons_to_freeze,
)


# ============================================================================
# Constants
# ============================================================================

DOMAINS = ["kvasir", "isic", "busi", "pcb"]

DOMAIN_OBJECTS = {
    "kvasir": "polyps",
    "isic": "lesions",
    "busi": "tumors",
    "deepglobe": "road segments",
    "pcb": "defects",
}

IMG_TIMEOUT = 180  # seconds per image

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
_global_start = time.time()


def dbg(msg: str):
    """Print debug message with timestamp if DEBUG is enabled."""
    if DEBUG:
        elapsed = time.time() - _global_start
        print(f"  [DBG {elapsed:7.1f}s] {msg}", flush=True)


# ============================================================================
# GT Coverage Computation
# ============================================================================

def compute_gt_coverage(
    gt_masks: List[np.ndarray], sam_masks: List[dict]
) -> Optional[float]:
    """Fraction of GT pixels covered by ANY SAM mask.

    Returns None if no valid GT masks or GT area is zero.
    """
    if not gt_masks:
        return None

    gt_union = np.zeros_like(gt_masks[0], dtype=bool)
    for gt in gt_masks:
        gt_union |= gt.astype(bool)

    if gt_union.sum() == 0:
        return None

    model_union = np.zeros_like(gt_union, dtype=bool)
    for m in sam_masks:
        seg = m["segmentation"]
        if seg.shape == gt_union.shape:
            model_union |= seg.astype(bool)

    return float((gt_union & model_union).sum() / gt_union.sum())


# ============================================================================
# Miss Classification (simplified inline version)
# ============================================================================

def classify_misses_simple(
    masks_orig: List[dict],
    masks_patch: List[dict],
    gt_masks: List[np.ndarray],
    domain: str,
    matched_pairs: Optional[List[Tuple[int, int, float]]] = None,
) -> dict:
    """Classify missed masks into domain-relevant vs background.

    Simplified version for combined freeze experiment.
    """
    n_orig = len(masks_orig)

    # Get domain thresholds
    thresh_entry = DOMAIN_THRESHOLDS.get(domain, {"iou": 0.3, "containment": 0.5})
    gt_overlap_thresh = thresh_entry["iou"] if isinstance(thresh_entry, dict) else thresh_entry

    # Determine unmatched original masks (the "misses")
    if matched_pairs is not None:
        matched_orig_indices = {pair[0] for pair in matched_pairs}
    else:
        _, unmatched_orig, _ = match_masks(masks_orig, masks_patch, iou_threshold=0.5)
        matched_orig_indices = set(range(n_orig)) - unmatched_orig

    unmatched_orig_indices = set(range(n_orig)) - matched_orig_indices
    n_misses = len(unmatched_orig_indices)

    has_gt = gt_masks is not None and len(gt_masks) > 0

    if not has_gt or n_misses == 0:
        return {
            "total_misses": n_misses,
            "domain_misses": 0,
            "background_misses": n_misses,
            "has_gt": has_gt,
        }

    # Classify each miss
    domain_misses = 0
    background_misses = 0

    for idx in unmatched_orig_indices:
        missed_seg = masks_orig[idx]["segmentation"]

        # Resize if shape mismatch
        if missed_seg.shape != gt_masks[0].shape:
            missed_seg = cv2.resize(
                missed_seg.astype(np.uint8),
                (gt_masks[0].shape[1], gt_masks[0].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Check overlap with any GT mask
        max_gt_iou = 0.0
        max_containment = 0.0
        for gt in gt_masks:
            intersection = np.logical_and(missed_seg, gt).sum()
            union = np.logical_or(missed_seg, gt).sum()
            seg_area = missed_seg.sum()

            iou = float(intersection / union) if union > 0 else 0.0
            containment = float(intersection / seg_area) if seg_area > 0 else 0.0
            max_gt_iou = max(max_gt_iou, iou)
            max_containment = max(max_containment, containment)

        # Domain-relevant if either threshold is met
        containment_thresh = thresh_entry.get("containment", 0.5) if isinstance(thresh_entry, dict) else 0.5
        if max_gt_iou >= gt_overlap_thresh or max_containment >= containment_thresh:
            domain_misses += 1
        else:
            background_misses += 1

    return {
        "total_misses": n_misses,
        "domain_misses": domain_misses,
        "background_misses": background_misses,
        "has_gt": True,
    }


# ============================================================================
# Phase 1: MLP Profiling
# ============================================================================

def run_mlp_profiling(
    sam,
    domain: str,
    image_dir: str,
    out_dir: Path,
    profile_images: int,
    device: torch.device,
):
    """Profile MLP neurons on domain images."""

    profile_dir_images = f"{image_dir}/{domain}_profile"
    print(f"  Image source: {profile_dir_images}", flush=True)

    dataset = ImageDirDataset(profile_dir_images, domain, max_images=profile_images)
    n_images = len(dataset)
    print(f"  Found {n_images} profile images", flush=True)

    if n_images == 0:
        print("  ERROR: No profile images found. Cannot profile MLP.", flush=True)
        return False

    # Create profile hook manager
    profile_mgr = MLPProfileHookManager(sam, device)
    profile_mgr.register()

    t_start = time.time()
    processed = 0

    for i in range(n_images):
        try:
            image, _, filename = dataset[i]
        except Exception as e:
            print(f"  [WARN] Failed to load image {i}: {e}", flush=True)
            continue

        dbg(f"Profile image {i}/{n_images}: {filename}")

        try:
            with torch.inference_mode():
                # Run image through encoder only (for profiling)
                from segment_anything.utils.transforms import ResizeLongestSide
                transform = ResizeLongestSide(1024)
                input_image = transform.apply_image(image)
                input_tensor = (
                    torch.as_tensor(input_image, device=device)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                )
                input_tensor = sam.preprocess(input_tensor)
                _ = sam.image_encoder(input_tensor)

            profile_mgr.increment_sample_count()
            processed += 1

        except torch.cuda.OutOfMemoryError:
            print(f"  [WARN] OOM on profile image {i}, skipping...", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [WARN] OOM on profile image {i}, skipping...", flush=True)
            else:
                print(f"  [WARN] RuntimeError on image {i}: {e}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Progress
        if (i + 1) % 20 == 0 or (i + 1) == n_images:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"    [{i+1:4d}/{n_images}] processed={processed} [{rate:.1f} img/s]",
                flush=True,
            )

        # Memory cleanup
        del image
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    profile_mgr.remove()

    # Save profiles
    save_dir = out_dir / domain / "mlp_profile"
    save_dir.mkdir(parents=True, exist_ok=True)
    profile_mgr.save(save_dir, domain)

    elapsed_total = time.time() - t_start
    print(
        f"  Profiling complete: {processed} images in {elapsed_total:.1f}s",
        flush=True,
    )
    print(f"  Saved to: {save_dir}", flush=True)
    return True


# ============================================================================
# Phase 2: Combined Freeze Evaluation
# ============================================================================

def run_combined_eval(
    sam,
    gen: SamAutomaticMaskGenerator,
    domain: str,
    image_dir: str,
    data_root: str,
    out_dir: Path,
    attn_manifest_path: str,
    mlp_frac: float,
    eval_images: int,
    device: torch.device,
) -> Optional[dict]:
    """Run evaluation with combined attention + MLP freeze."""

    frac_label = f"combined_f{mlp_frac}"
    result_dir = out_dir / domain / frac_label
    result_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load attention frozen heads ----
    dbg("Loading attention manifest...")
    manifest = load_manifest(Path(attn_manifest_path))
    attn_frozen_heads = load_frozen_heads_from_manifest(manifest)
    n_attn_frozen = manifest.get("summary", {}).get("frozen", len(attn_frozen_heads))
    total_attn_heads = manifest.get("summary", {}).get("total_heads", 512)

    # ---- Select MLP neurons to freeze ----
    dbg(f"Selecting MLP neurons (fraction={mlp_frac})...")
    profile_dir = out_dir / domain / "mlp_profile"
    mlp_frozen = select_mlp_neurons_to_freeze(
        profile_dir=profile_dir,
        group_name=domain,
        freeze_fraction=mlp_frac,
    )

    # Compute MLP neurons frozen per block for display
    # mlp_frozen is whatever select_mlp_neurons_to_freeze returns
    # We need to figure out the per-block count from the MLP_DIM constant
    from sam_velcro_pkg.constants import MLP_DIM, NUM_BLOCKS
    mlp_neurons_per_block = int(mlp_frac * MLP_DIM)

    print(f"  Attention: {n_attn_frozen}/{total_attn_heads} heads frozen ({n_attn_frozen/total_attn_heads*100:.1f}%)", flush=True)
    print(f"  MLP: {mlp_neurons_per_block}/{MLP_DIM} neurons frozen per block ({mlp_frac*100:.1f}%)", flush=True)

    # ---- Create hook managers ----
    attn_hook_mgr = FreezeHookManager(sam, attn_frozen_heads, device)
    mlp_hook_mgr = MLPFreezeHookManager(sam, mlp_frozen, device)

    # ---- Load evaluation dataset ----
    eval_dir = f"{image_dir}/{domain}_eval"
    max_eval = eval_images if eval_images > 0 else None
    dataset = ImageDirDataset(eval_dir, domain, max_images=max_eval)
    n_images = len(dataset)
    print(f"  Eval images: {n_images} from {eval_dir}", flush=True)

    if n_images == 0:
        print("  ERROR: No eval images found.", flush=True)
        return None

    # ---- Run evaluation ----
    t_start = time.time()

    # Accumulators
    all_ious = []
    total_orig_masks = 0
    total_combined_masks = 0
    total_misses = 0
    total_domain_misses = 0
    total_background_misses = 0
    total_false_pos = 0

    ref_coverages = []
    frozen_coverages = []
    images_full_loss = 0
    images_any_drop = 0
    images_with_gt = 0
    images_processed = 0

    for i in range(n_images):
        try:
            image, _, filename = dataset[i]
        except Exception as e:
            print(f"  [WARN] Failed to load image {i}: {e}", flush=True)
            continue

        dbg(f"Eval image {i}/{n_images}: {filename}")
        img_start = time.time()

        try:
            with torch.inference_mode():
                # ---- Original SAM (no hooks) ----
                masks_orig = gen.generate(image)

                elapsed_orig = time.time() - img_start
                if elapsed_orig > IMG_TIMEOUT:
                    print(
                        f"  [WARN] Image {i} ({filename}) took {elapsed_orig:.0f}s "
                        f"for orig pass - skipping",
                        flush=True,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                # ---- Combined frozen SAM (both hooks) ----
                attn_hook_mgr.register()
                mlp_hook_mgr.register()
                masks_combined = gen.generate(image)
                mlp_hook_mgr.remove()
                attn_hook_mgr.remove()

        except torch.cuda.OutOfMemoryError:
            print(f"  [WARN] OOM on image {i} ({filename}), skipping...", flush=True)
            try:
                mlp_hook_mgr.remove()
            except Exception:
                pass
            try:
                attn_hook_mgr.remove()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [WARN] OOM on image {i} ({filename}), skipping...", flush=True)
            else:
                print(f"  [WARN] RuntimeError on image {i} ({filename}): {e}", flush=True)
            try:
                mlp_hook_mgr.remove()
            except Exception:
                pass
            try:
                attn_hook_mgr.remove()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  [WARN] Error on image {i} ({filename}): {e}", flush=True)
            try:
                mlp_hook_mgr.remove()
            except Exception:
                pass
            try:
                attn_hook_mgr.remove()
            except Exception:
                pass
            continue

        # ---- Hungarian matching ----
        eval_result = evaluate_masks(masks_orig, masks_combined, iou_threshold=0.5)
        matched_pairs = eval_result["matched_pairs"]

        ious = [m[2] for m in matched_pairs]
        all_ious.extend(ious)
        total_orig_masks += eval_result["num_masks_original"]
        total_combined_masks += eval_result["num_masks_patched"]

        n_miss = len(masks_orig) - eval_result["num_matched"]
        n_fp = len(masks_combined) - eval_result["num_matched"]
        total_misses += n_miss
        total_false_pos += n_fp

        # ---- Load GT & classify misses ----
        gt_masks = load_gt_masks(filename, domain, data_root)

        miss_info = classify_misses_simple(
            masks_orig, masks_combined, gt_masks, domain, matched_pairs=matched_pairs
        )
        total_domain_misses += miss_info["domain_misses"]
        total_background_misses += miss_info["background_misses"]

        # ---- GT coverage comparison ----
        if gt_masks:
            images_with_gt += 1
            ref_cov = compute_gt_coverage(gt_masks, masks_orig)
            frozen_cov = compute_gt_coverage(gt_masks, masks_combined)

            if ref_cov is not None:
                ref_coverages.append(ref_cov)
                frozen_cov_val = frozen_cov if frozen_cov is not None else 0.0
                frozen_coverages.append(frozen_cov_val)

                if frozen_cov_val == 0.0 and ref_cov > 0.0:
                    images_full_loss += 1
                if frozen_cov_val < ref_cov - 1e-6:
                    images_any_drop += 1

        images_processed += 1

        # Progress
        if (i + 1) % 20 == 0 or (i + 1) == n_images:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            mean_iou_so_far = float(np.mean(all_ious)) if all_ious else 0.0
            print(
                f"    [{i+1:4d}/{n_images}] "
                f"orig={len(masks_orig):3d} comb={len(masks_combined):3d} "
                f"miss={n_miss:2d} IoU={mean_iou_so_far:.4f} "
                f"[{rate:.1f} img/s]",
                flush=True,
            )

        # Memory cleanup
        del masks_orig, masks_combined, image, gt_masks
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed_total = time.time() - t_start

    # ---- Aggregate results ----
    if images_processed == 0:
        print("  ERROR: No images were successfully processed.", flush=True)
        return None

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    miss_rate = total_misses / total_orig_masks if total_orig_masks > 0 else 0.0
    false_pos_rate = total_false_pos / total_combined_masks if total_combined_masks > 0 else 0.0
    domain_miss_rate = total_domain_misses / total_orig_masks if total_orig_masks > 0 else 0.0
    pct_misses_domain = (
        total_domain_misses / total_misses * 100 if total_misses > 0 else 0.0
    )

    # Coverage stats
    ref_mean_cov = float(np.mean(ref_coverages)) if ref_coverages else 0.0
    frozen_mean_cov = float(np.mean(frozen_coverages)) if frozen_coverages else 0.0
    mean_drop = ref_mean_cov - frozen_mean_cov

    drops = [r - f for r, f in zip(ref_coverages, frozen_coverages)]
    max_drop = float(max(drops)) if drops else 0.0

    # ---- Print summary ----
    objects_name = DOMAIN_OBJECTS.get(domain, "objects")
    print(f"\n{'='*64}", flush=True)
    print(f"Combined Freeze Results: {domain.upper()}", flush=True)
    print(
        f"  Attention: {n_attn_frozen/total_attn_heads*100:.1f}% heads frozen ({n_attn_frozen}/{total_attn_heads})",
        flush=True,
    )
    print(
        f"  MLP: {mlp_frac*100:.1f}% neurons frozen "
        f"({mlp_neurons_per_block}/{MLP_DIM} per block)",
        flush=True,
    )
    print(f"{'='*64}", flush=True)
    print(flush=True)
    print(f"Mask Comparison:", flush=True)
    print(f"  Mean IoU:              {mean_iou:.4f}", flush=True)
    print(f"  Miss rate:             {miss_rate*100:.1f}%", flush=True)
    print(flush=True)
    print(f"Miss Decomposition:", flush=True)
    print(
        f"  Domain misses:         {total_domain_misses} ({domain_miss_rate*100:.2f}%)",
        flush=True,
    )
    print(
        f"  Background misses:     {total_background_misses} "
        f"({total_background_misses/total_orig_masks*100:.1f}%)"
        if total_orig_masks > 0
        else f"  Background misses:     {total_background_misses} (0.0%)",
        flush=True,
    )
    print(flush=True)
    print(f"Coverage Comparison:", flush=True)
    print(f"  Ref GT coverage:       {ref_mean_cov*100:.1f}%", flush=True)
    print(f"  Frozen GT coverage:    {frozen_mean_cov*100:.1f}%", flush=True)
    print(f"  Drop:                  {mean_drop*100:.1f} pp", flush=True)
    print(f"  Images with full loss: {images_full_loss}", flush=True)
    print(f"{'='*64}\n", flush=True)
    print(
        f"  Time: {elapsed_total:.1f}s ({images_processed/elapsed_total:.1f} img/s)",
        flush=True,
    )

    # ---- Build results dict ----
    results = {
        "domain": domain,
        "attn_freeze_fraction": round(n_attn_frozen / total_attn_heads, 4),
        "mlp_freeze_fraction": mlp_frac,
        "attn_heads_frozen": n_attn_frozen,
        "mlp_neurons_frozen_per_block": mlp_neurons_per_block,
        "num_eval_images": images_processed,
        "mask_comparison": {
            "mean_iou": round(mean_iou, 4),
            "miss_rate": round(miss_rate, 4),
            "false_pos_rate": round(false_pos_rate, 4),
        },
        "miss_decomposition": {
            "total_misses": total_misses,
            "domain_misses": total_domain_misses,
            "background_misses": total_background_misses,
            "domain_miss_rate": round(domain_miss_rate, 4),
            "pct_misses_domain": round(pct_misses_domain, 1),
        },
        "coverage_comparison": {
            "ref_mean_coverage": round(ref_mean_cov, 4),
            "frozen_mean_coverage": round(frozen_mean_cov, 4),
            "mean_drop": round(mean_drop, 4),
            "max_drop": round(max_drop, 4),
            "images_full_loss": images_full_loss,
            "images_any_drop": images_any_drop,
        },
    }

    # ---- Save JSON ----
    json_path = result_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}", flush=True)

    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined Attention + MLP Freeze Experiment for SAM S-VELCRO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/exp_combined_freeze.py \\
      --checkpoint checkpoints/sam_vit_h_4b8939.pth \\
      --domain kvasir \\
      --image_dir ./data \\
      --attn_manifest results_cross_domain/kvasir/freeze_kvasir_f0.375/freeze_manifest.json

  # Skip profiling (reuse existing):
  python scripts/exp_combined_freeze.py \\
      --checkpoint checkpoints/sam_vit_h_4b8939.pth \\
      --domain kvasir \\
      --image_dir ./data \\
      --attn_manifest results_cross_domain/kvasir/freeze_kvasir_f0.375/freeze_manifest.json \\
      --skip_profile
        """,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to SAM ViT-H checkpoint (.pth)",
    )
    parser.add_argument(
        "--domain", type=str, required=True, choices=DOMAINS,
        help="Domain name: kvasir|isic|busi|pcb",
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing {domain}_profile/ and {domain}_eval/",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data",
        help="Root for GT masks (default: ./data)",
    )
    parser.add_argument(
        "--attn_manifest", type=str, required=True,
        help="Attention freeze manifest (25% heads)",
    )
    parser.add_argument(
        "--mlp_fractions", type=str, default="0.125,0.25,0.375",
        help="MLP freeze fractions to test (comma-separated, default: 0.125,0.25,0.375)",
    )
    parser.add_argument(
        "--profile_images", type=int, default=300,
        help="Number of images for MLP profiling (default: 300)",
    )
    parser.add_argument(
        "--eval_images", type=int, default=300,
        help="Number of images for evaluation (default: 300, 0=all)",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results_combined_freeze",
        help="Output directory (default: ./results_combined_freeze)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda|cpu (default: cuda)",
    )
    parser.add_argument(
        "--skip_profile", action="store_true",
        help="Skip profiling if MLP profiles already exist",
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Parse MLP fractions
    mlp_fractions = [float(x.strip()) for x in args.mlp_fractions.split(",")]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}", flush=True)
    print(f"  Combined Freeze Experiment: {args.domain.upper()}", flush=True)
    print(f"{'='*64}", flush=True)
    print(f"  Checkpoint:     {args.checkpoint}", flush=True)
    print(f"  Domain:         {args.domain}", flush=True)
    print(f"  Image dir:      {args.image_dir}", flush=True)
    print(f"  Data root:      {args.data_root}", flush=True)
    print(f"  Attn manifest:  {args.attn_manifest}", flush=True)
    print(f"  MLP fractions:  {mlp_fractions}", flush=True)
    print(f"  Profile images: {args.profile_images}", flush=True)
    print(f"  Eval images:    {args.eval_images} {'(all)' if args.eval_images == 0 else ''}", flush=True)
    print(f"  Output dir:     {out_dir}", flush=True)
    print(f"  Device:         {device}", flush=True)
    print(f"  Skip profile:   {args.skip_profile}", flush=True)
    print(f"{'='*64}\n", flush=True)

    # ---- Load SAM model ----
    print("[Loading] SAM ViT-H...", flush=True)
    sam = sam_model_registry["vit_h"](checkpoint=args.checkpoint)
    sam.to(device).eval()
    dbg("Model loaded and on device")

    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
    )
    dbg("MaskGenerator created")

    # ==================================================================
    # Phase 1: Profile MLP neurons
    # ==================================================================
    profile_dir = out_dir / args.domain / "mlp_profile"

    if args.skip_profile:
        # Verify profile exists
        expected_profile = profile_dir / args.domain
        if not expected_profile.exists():
            print(
                f"  [WARN] --skip_profile set but profile not found at {expected_profile}",
                flush=True,
            )
            print("  Running profiling anyway...", flush=True)
            args.skip_profile = False

    if not args.skip_profile:
        print(f"\n[Phase 1] Profiling MLP neurons...", flush=True)
        success = run_mlp_profiling(
            sam=sam,
            domain=args.domain,
            image_dir=args.image_dir,
            out_dir=out_dir,
            profile_images=args.profile_images,
            device=device,
        )
        if not success:
            print("  FATAL: MLP profiling failed. Exiting.", flush=True)
            sys.exit(1)
    else:
        print(f"\n[Phase 1] Skipping MLP profiling (using existing at {profile_dir})", flush=True)

    # ==================================================================
    # Phase 2: Combined freeze evaluation for each MLP fraction
    # ==================================================================
    all_results = []

    for mlp_frac in mlp_fractions:
        print(f"\n[Phase 2] Combined freeze: attn=25% + mlp={mlp_frac*100:.1f}%", flush=True)
        print(f"{'-'*64}", flush=True)

        try:
            result = run_combined_eval(
                sam=sam,
                gen=gen,
                domain=args.domain,
                image_dir=args.image_dir,
                data_root=args.data_root,
                out_dir=out_dir,
                attn_manifest_path=args.attn_manifest,
                mlp_frac=mlp_frac,
                eval_images=args.eval_images,
                device=device,
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] Failed for mlp_frac={mlp_frac}: {e}", flush=True)
            traceback.print_exc()
            continue

        # Force cleanup between fractions
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ==================================================================
    # Phase 3: Final Summary
    # ==================================================================
    if all_results:
        print(f"\n\n{'='*64}", flush=True)
        print(f"  FINAL SUMMARY: {args.domain.upper()}", flush=True)
        print(f"{'='*64}", flush=True)
        print(
            f"  {'MLP%':>6} {'IoU':>7} {'Miss%':>7} {'Dom':>5} {'BG':>5} "
            f"{'RefCov':>7} {'FrzCov':>7} {'Drop':>6}",
            flush=True,
        )
        print(f"  {'-'*56}", flush=True)

        for r in all_results:
            mc = r["mask_comparison"]
            md = r["miss_decomposition"]
            cc = r["coverage_comparison"]
            print(
                f"  {r['mlp_freeze_fraction']*100:>5.1f}% "
                f"{mc['mean_iou']:>7.4f} "
                f"{mc['miss_rate']*100:>6.1f}% "
                f"{md['domain_misses']:>5} "
                f"{md['background_misses']:>5} "
                f"{cc['ref_mean_coverage']*100:>6.1f}% "
                f"{cc['frozen_mean_coverage']*100:>6.1f}% "
                f"{cc['mean_drop']*100:>5.1f}pp",
                flush=True,
            )

        print(f"{'='*64}\n", flush=True)

        # Save combined summary
        summary_path = out_dir / args.domain / "combined_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Combined summary saved: {summary_path}", flush=True)

    else:
        print("\n  No results to summarize.", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
