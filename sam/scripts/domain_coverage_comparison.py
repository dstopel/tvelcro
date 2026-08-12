#!/usr/bin/env python3
"""
domain_coverage_comparison.py — GT Coverage Comparison: Original SAM vs Frozen SAM

Compares ground-truth coverage between original SAM and frozen SAM for each image.
The key metric is the COVERAGE DROP (ref - frozen), not absolute coverage.

Coverage = fraction of GT pixels covered by ANY mask from the model.
Answers: "Can the model see the domain object at all?"

Usage:
  python scripts/domain_coverage_comparison.py \
      --domain kvasir \
      --checkpoint checkpoints/sam_vit_h_4b8939.pth \
      --freeze_manifest results_cross_domain/kvasir/freeze_kvasir_f0.375/freeze_manifest.json \
      --image_dir ./data --data_root ./data
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from sam_velcro_pkg.dataset import ImageDirDataset
from sam_velcro_pkg.freeze import load_frozen_heads_from_manifest, load_manifest
from sam_velcro_pkg.gt_masks import load_gt_masks
from sam_velcro_pkg.hooks import FreezeHookManager


# ============================================================================
# Constants
# ============================================================================

DOMAINS = ["kvasir", "isic", "busi", "deepglobe", "pcb"]

DOMAIN_OBJECT_NAMES = {
    "kvasir": "polyps",
    "isic": "lesions",
    "busi": "tumors",
    "deepglobe": "road segments",
    "pcb": "defects",
}

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
_global_start = time.time()


def dbg(msg: str):
    """Print debug message with timestamp if DEBUG is enabled."""
    if DEBUG:
        elapsed = time.time() - _global_start
        print(f"  [DBG {elapsed:7.1f}s] {msg}", flush=True)


# ============================================================================
# Core Metric
# ============================================================================

def compute_gt_coverage(gt_masks: List[np.ndarray], sam_masks: List[dict]) -> Optional[float]:
    """What fraction of GT pixels are covered by ANY mask from this model?

    This uses ALL masks from the model (not just matched ones).
    Answers: "Can the model see the domain object at all?"

    Args:
        gt_masks: List of binary (H,W) uint8 GT mask arrays.
        sam_masks: List of SAM mask dicts (each with 'segmentation' key).

    Returns:
        Float in [0, 1] or None if no valid GT.
    """
    if not gt_masks:
        return None  # No GT annotation

    # Union all GT masks
    gt_union = np.zeros_like(gt_masks[0], dtype=bool)
    for gt in gt_masks:
        gt_union |= gt.astype(bool)
    gt_area = gt_union.sum()
    if gt_area == 0:
        return None

    # Union all SAM masks
    model_union = np.zeros_like(gt_masks[0], dtype=bool)
    for mask_dict in sam_masks:
        seg = mask_dict["segmentation"]
        if seg.shape == gt_union.shape:
            model_union |= seg.astype(bool)

    # Coverage = intersection / GT area
    covered = (gt_union & model_union).sum()
    return float(covered / gt_area)


# ============================================================================
# Main Analysis
# ============================================================================

def run_analysis(
    checkpoint: str,
    freeze_manifest_path: str,
    domain: str,
    image_dir: str,
    data_root: str,
    max_images: Optional[int] = None,
    out_dir: Optional[str] = None,
    device_str: str = "cuda",
    save_details: bool = False,
) -> dict:
    """Run GT coverage comparison for a single domain + manifest."""

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Output directory
    if out_dir is None:
        out_dir = f"./results_coverage_comparison/{domain}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- Load freeze manifest (do this first for display) ----
    dbg(f"Reading manifest: {freeze_manifest_path}")
    manifest = load_manifest(Path(freeze_manifest_path))
    frozen_heads = load_frozen_heads_from_manifest(manifest)
    freeze_fraction = manifest.get("freeze_fraction", 0.0)
    n_frozen = manifest.get("summary", {}).get("frozen", len(frozen_heads))

    objects_name = DOMAIN_OBJECT_NAMES.get(domain, "objects")

    print(f"\n{'='*64}", flush=True)
    print(f"Domain Coverage Comparison: {domain.upper()} @ {freeze_fraction*100:.1f}% freeze", flush=True)
    print(f"{'='*64}", flush=True)
    print(f"  Checkpoint:  {checkpoint}", flush=True)
    print(f"  Manifest:    {freeze_manifest_path}", flush=True)
    print(f"  Image dir:   {image_dir}/{domain}_eval/", flush=True)
    print(f"  Data root:   {data_root}", flush=True)
    print(f"  Device:      {device}", flush=True)
    print(f"  Frozen heads: {n_frozen} (fraction={freeze_fraction})", flush=True)
    print(f"{'='*64}\n", flush=True)

    # ---- Load SAM model ----
    print("[1/4] Loading SAM ViT-H...", flush=True)
    dbg("Loading checkpoint from disk...")
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    dbg("Moving model to device...")
    sam.to(device)
    sam.eval()
    dbg("Model loaded and on device")

    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
    )
    dbg("MaskGenerator created")

    # ---- Create FreezeHookManager ----
    print("[2/4] Preparing freeze hooks...", flush=True)
    dbg("Creating FreezeHookManager...")
    hook_mgr = FreezeHookManager(sam, frozen_heads, device)
    dbg("FreezeHookManager ready")

    # ---- Load dataset ----
    print("[3/4] Loading evaluation images...", flush=True)
    eval_dir = f"{image_dir}/{domain}_eval"
    dbg(f"Looking for eval images in: {eval_dir}")
    dataset = ImageDirDataset(eval_dir, domain, max_images=max_images)
    n_images = len(dataset)
    print(f"       Found {n_images} images in {eval_dir}", flush=True)

    if n_images == 0:
        print("ERROR: No images found. Exiting.", flush=True)
        return {}

    # ---- Run per-image comparison ----
    print(f"[4/4] Running coverage comparison on {n_images} images...\n", flush=True)

    # Per-image accumulators
    per_image_results = []
    ref_coverages = []
    frozen_coverages = []
    drops = []
    images_with_gt = 0
    images_without_gt = 0
    images_ref_zero = 0  # ref coverage == 0 (excluded from drop analysis)

    IMG_TIMEOUT = 180  # seconds — skip if original pass takes longer

    t_start = time.time()

    for i in range(n_images):
        try:
            image, _, filename = dataset[i]
        except Exception as e:
            print(f"  [WARN] Failed to load image {i}: {e}", flush=True)
            continue

        dbg(f"Image {i}/{n_images}: {filename} (shape={image.shape})")
        img_start = time.time()

        try:
            with torch.inference_mode():
                # Original SAM (no hooks)
                dbg(f"  gen.generate() [original] starting...")
                masks_orig = gen.generate(image)
                dbg(f"  gen.generate() [original] done: {len(masks_orig)} masks in {time.time()-img_start:.1f}s")

                elapsed_orig = time.time() - img_start
                if elapsed_orig > IMG_TIMEOUT:
                    print(f"  [WARN] Image {i} ({filename}) took {elapsed_orig:.0f}s for orig pass — skipping", flush=True)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                # Frozen SAM (with hooks)
                dbg(f"  gen.generate() [frozen] starting...")
                hook_mgr.register()
                masks_patch = gen.generate(image)
                hook_mgr.remove()
                dbg(f"  gen.generate() [frozen] done: {len(masks_patch)} masks in {time.time()-img_start:.1f}s")

        except torch.cuda.OutOfMemoryError:
            print(f"  [WARN] OOM on image {i} ({filename}), skipping...", flush=True)
            hook_mgr.remove()
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [WARN] OOM on image {i} ({filename}), skipping...", flush=True)
            else:
                print(f"  [WARN] RuntimeError on image {i} ({filename}): {e}", flush=True)
            hook_mgr.remove()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  [WARN] Error on image {i} ({filename}): {e}", flush=True)
            hook_mgr.remove()
            continue

        # Load GT masks
        dbg(f"  Loading GT masks...")
        gt_masks = load_gt_masks(filename, domain, data_root)
        dbg(f"  GT: {len(gt_masks)} masks loaded")

        if not gt_masks:
            images_without_gt += 1
            dbg(f"  No GT for {filename}, skipping coverage computation")
            del masks_orig, masks_patch, image
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        images_with_gt += 1

        # Compute coverage for both
        ref_coverage = compute_gt_coverage(gt_masks, masks_orig)
        frozen_coverage = compute_gt_coverage(gt_masks, masks_patch)

        # Handle edge case: GT exists but GT union area is 0
        if ref_coverage is None or frozen_coverage is None:
            images_without_gt += 1
            images_with_gt -= 1
            del masks_orig, masks_patch, image, gt_masks
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # Exclude from drop analysis if ref coverage is 0
        # (SAM can't see it either way — not a freeze problem)
        if ref_coverage == 0.0:
            images_ref_zero += 1
            dbg(f"  ref_coverage=0, excluding from drop analysis")
        else:
            drop = ref_coverage - frozen_coverage
            ref_coverages.append(ref_coverage)
            frozen_coverages.append(frozen_coverage)
            drops.append(drop)

            per_image_results.append({
                "filename": filename,
                "ref_coverage": round(ref_coverage, 6),
                "frozen_coverage": round(frozen_coverage, 6),
                "drop": round(drop, 6),
            })

        # Progress logging
        if (i + 1) % 20 == 0 or (i + 1) == n_images:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(
                f"  [{i+1:4d}/{n_images}] "
                f"ref={ref_coverage:.3f} frozen={frozen_coverage:.3f} "
                f"drop={ref_coverage - frozen_coverage:+.3f}pp "
                f"[{rate:.1f} img/s]",
                flush=True,
            )

        # Memory cleanup
        del masks_orig, masks_patch, image, gt_masks
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed_total = time.time() - t_start

    # ============================================================================
    # Aggregate Results
    # ============================================================================

    n_evaluated = images_with_gt + images_without_gt
    n_in_drop_analysis = len(drops)

    if n_in_drop_analysis == 0:
        print("ERROR: No images with valid GT coverage were processed.", flush=True)
        return {}

    ref_arr = np.array(ref_coverages)
    frozen_arr = np.array(frozen_coverages)
    drop_arr = np.array(drops)

    ref_mean = float(np.mean(ref_arr))
    ref_median = float(np.median(ref_arr))
    frozen_mean = float(np.mean(frozen_arr))
    frozen_median = float(np.median(frozen_arr))
    mean_drop = float(np.mean(drop_arr))
    median_drop = float(np.median(drop_arr))
    max_drop = float(np.max(drop_arr))
    std_drop = float(np.std(drop_arr))

    # Domain loss categories
    images_no_drop = int(np.sum(drop_arr <= 0))
    images_any_drop = int(np.sum(drop_arr > 0))
    images_drop_gt_10pp = int(np.sum(drop_arr > 0.10))
    images_drop_gt_25pp = int(np.sum(drop_arr > 0.25))
    images_drop_gt_50pp = int(np.sum(drop_arr > 0.50))
    # Full loss: frozen=0 while ref>0
    images_full_loss = int(np.sum((frozen_arr == 0) & (ref_arr > 0)))

    # Top 10 worst images (highest drop)
    sorted_indices = np.argsort(drop_arr)[::-1]
    worst_10 = []
    for rank, idx in enumerate(sorted_indices[:10]):
        worst_10.append(per_image_results[idx])

    # ============================================================================
    # Print Summary
    # ============================================================================

    print(f"\n{'='*64}", flush=True)
    print(f"Domain Coverage Comparison: {domain.upper()} @ {freeze_fraction*100:.1f}% freeze", flush=True)
    print(f"{'='*64}\n", flush=True)

    print(f"Images evaluated:         {n_evaluated}", flush=True)
    print(f"Images with GT:           {images_with_gt}", flush=True)
    print(f"Images without GT:        {images_without_gt} (skipped)", flush=True)
    if images_ref_zero > 0:
        print(f"Images ref_coverage=0:    {images_ref_zero} (excluded from drop)", flush=True)
    print(f"Images in drop analysis:  {n_in_drop_analysis}", flush=True)
    print(flush=True)

    print(f"Reference (original SAM):", flush=True)
    print(f"  Mean GT coverage:       {ref_mean*100:.1f}%", flush=True)
    print(f"  Median GT coverage:     {ref_median*100:.1f}%", flush=True)
    print(flush=True)

    print(f"Frozen SAM:", flush=True)
    print(f"  Mean GT coverage:       {frozen_mean*100:.1f}%", flush=True)
    print(f"  Median GT coverage:     {frozen_median*100:.1f}%", flush=True)
    print(flush=True)

    print(f"Coverage DROP (ref - frozen):", flush=True)
    print(f"  Mean drop:              {mean_drop*100:.1f} pp", flush=True)
    print(f"  Median drop:            {median_drop*100:.1f} pp", flush=True)
    print(f"  Max drop:               {max_drop*100:.1f} pp", flush=True)
    print(f"  Std drop:               {std_drop*100:.1f} pp", flush=True)
    print(flush=True)

    print(f"Domain loss summary:", flush=True)
    print(
        f"  Images with NO drop:           {images_no_drop} / {n_in_drop_analysis} "
        f"({images_no_drop/n_in_drop_analysis*100:.1f}%)  <- identical coverage",
        flush=True,
    )
    print(
        f"  Images with ANY drop (>0):     {images_any_drop} / {n_in_drop_analysis} "
        f"({images_any_drop/n_in_drop_analysis*100:.1f}%)",
        flush=True,
    )
    print(
        f"  Images with >10pp drop:        {images_drop_gt_10pp} / {n_in_drop_analysis} "
        f"({images_drop_gt_10pp/n_in_drop_analysis*100:.1f}%)",
        flush=True,
    )
    print(
        f"  Images with >25pp drop:        {images_drop_gt_25pp} / {n_in_drop_analysis} "
        f"({images_drop_gt_25pp/n_in_drop_analysis*100:.1f}%)",
        flush=True,
    )
    print(
        f"  Images with >50pp drop:        {images_drop_gt_50pp} / {n_in_drop_analysis} "
        f"({images_drop_gt_50pp/n_in_drop_analysis*100:.1f}%)",
        flush=True,
    )
    print(
        f"  Images with FULL loss:         {images_full_loss} / {n_in_drop_analysis} "
        f"({images_full_loss/n_in_drop_analysis*100:.1f}%)  <- frozen=0 while ref>0",
        flush=True,
    )
    print(flush=True)

    print(f"Top 10 worst images (highest coverage drop):", flush=True)
    for rank, entry in enumerate(worst_10):
        print(
            f"  #{rank+1:<2d} {entry['filename']}:  "
            f"ref={entry['ref_coverage']*100:.1f}% -> frozen={entry['frozen_coverage']*100:.1f}%  "
            f"(drop={entry['drop']*100:.1f}pp)",
            flush=True,
        )
    print(f"{'='*64}", flush=True)
    print(f"\n  Time: {elapsed_total:.1f}s ({n_evaluated/elapsed_total:.1f} img/s)\n", flush=True)

    # ============================================================================
    # Build Results Dict
    # ============================================================================

    results = {
        "domain": domain,
        "freeze_fraction": freeze_fraction,
        "num_images": n_evaluated,
        "num_with_gt": images_with_gt,
        "num_without_gt": images_without_gt,
        "num_ref_zero": images_ref_zero,
        "num_in_drop_analysis": n_in_drop_analysis,
        "summary": {
            "ref_mean_coverage": round(ref_mean, 4),
            "ref_median_coverage": round(ref_median, 4),
            "frozen_mean_coverage": round(frozen_mean, 4),
            "frozen_median_coverage": round(frozen_median, 4),
            "mean_drop": round(mean_drop, 4),
            "median_drop": round(median_drop, 4),
            "max_drop": round(max_drop, 4),
            "std_drop": round(std_drop, 4),
            "images_no_drop": images_no_drop,
            "images_any_drop": images_any_drop,
            "images_drop_gt_10pp": images_drop_gt_10pp,
            "images_drop_gt_25pp": images_drop_gt_25pp,
            "images_drop_gt_50pp": images_drop_gt_50pp,
            "images_full_loss": images_full_loss,
        },
        "worst_images": worst_10,
    }

    if save_details:
        results["per_image"] = per_image_results

    # ============================================================================
    # Save JSON
    # ============================================================================

    json_path = out_path / "coverage_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}", flush=True)

    # ============================================================================
    # Save CSV Summary Line
    # ============================================================================

    csv_path = out_path / "coverage_summary.csv"
    csv_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "domain", "freeze_fraction", "num_with_gt",
                "ref_mean", "frozen_mean", "mean_drop", "max_drop",
                "images_any_drop", "images_full_loss",
            ])
        writer.writerow([
            domain,
            freeze_fraction,
            n_in_drop_analysis,
            f"{ref_mean:.4f}",
            f"{frozen_mean:.4f}",
            f"{mean_drop:.4f}",
            f"{max_drop:.4f}",
            images_any_drop,
            images_full_loss,
        ])
    print(f"  Saved: {csv_path}", flush=True)

    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="S-VELCRO Domain Coverage Comparison: Original SAM vs Frozen SAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/domain_coverage_comparison.py \\
      --domain kvasir \\
      --checkpoint checkpoints/sam_vit_h_4b8939.pth \\
      --freeze_manifest results_cross_domain/kvasir/freeze_kvasir_f0.375/freeze_manifest.json \\
      --image_dir ./data --data_root ./data

  # Limit to 50 images for quick test:
  python scripts/domain_coverage_comparison.py \\
      --domain isic --max_images 50 \\
      --checkpoint checkpoints/sam_vit_h_4b8939.pth \\
      --freeze_manifest results_cross_domain/isic/freeze_isic_f0.25/freeze_manifest.json \\
      --image_dir ./data --data_root ./data
        """,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to SAM ViT-H checkpoint (.pth)",
    )
    parser.add_argument(
        "--freeze_manifest", type=str, required=True,
        help="Path to freeze_manifest.json",
    )
    parser.add_argument(
        "--domain", type=str, required=True, choices=DOMAINS,
        help="Domain name: kvasir|isic|busi|deepglobe|pcb",
    )
    parser.add_argument(
        "--image_dir", type=str, default="./data",
        help="Directory containing {domain}_eval/ image folders",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data",
        help="Root data directory for GT masks",
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Limit number of eval images (default: all)",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: ./results_coverage_comparison/{domain})",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--save_details", action="store_true",
        help="Include per-image results in JSON output",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_analysis(
        checkpoint=args.checkpoint,
        freeze_manifest_path=args.freeze_manifest,
        domain=args.domain,
        image_dir=args.image_dir,
        data_root=args.data_root,
        max_images=args.max_images,
        out_dir=args.out_dir,
        device_str=args.device,
        save_details=args.save_details,
    )


if __name__ == "__main__":
    main()
