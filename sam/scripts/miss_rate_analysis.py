#!/usr/bin/env python3
"""
miss_rate_analysis.py — Miss Rate Decomposition for S-VELCRO

Decomposes the total miss rate into:
  - Domain misses:     frozen model lost a domain-relevant object (overlaps GT)
  - Redundant misses:  sub-masks of GT objects already covered by other matched masks
  - Background misses: frozen model lost a shadow/texture/clutter mask (no GT overlap)

Standalone script. Usage:
  python scripts/miss_rate_analysis.py --domain kvasir \
      --checkpoint checkpoints/sam_vit_h_4b8939.pth \
      --freeze_manifest results_cross_domain/kvasir/freeze_kvasir_f0.125/freeze_manifest.json \
      --image_dir ./data --data_root ./data

Batch mode (all domains x fractions):
  python scripts/miss_rate_analysis.py --batch --checkpoint checkpoints/sam_vit_h_4b8939.pth
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from sam_velcro_pkg.dataset import ImageDirDataset
from sam_velcro_pkg.evaluation import (
    DOMAIN_THRESHOLDS,
    compute_mask_iou,
    evaluate_masks,
    match_masks,
)
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

FREEZE_FRACTIONS = [0.125, 0.25, 0.375]

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


def dbg(msg: str):
    """Print debug message with timestamp if DEBUG is enabled."""
    if DEBUG:
        elapsed = time.time() - _global_start
        print(f"  [DBG {elapsed:7.1f}s] {msg}", flush=True)


_global_start = time.time()


# GT mask loading is handled by sam_velcro_pkg.gt_masks.load_gt_masks (imported above)


# ============================================================================
# Miss Classification
# ============================================================================

def classify_misses(
    masks_orig: List[dict],
    masks_patch: List[dict],
    gt_masks: List[np.ndarray],
    domain: str,
    matched_pairs: Optional[List[Tuple[int, int, float]]] = None,
    gt_overlap_thresh: Optional[float] = None,
) -> dict:
    """
    Classify missed masks into domain-relevant, redundant, and background.

    Args:
        masks_orig: original SAM mask dicts
        masks_patch: frozen SAM mask dicts
        gt_masks: list of binary GT mask arrays
        domain: domain name for threshold lookup
        matched_pairs: list of (orig_idx, patch_idx, iou) from Hungarian matching
        gt_overlap_thresh: IoU threshold override (default: domain-specific)

    Returns:
        dict with classification results
    """
    if gt_overlap_thresh is None:
        thresh_entry = DOMAIN_THRESHOLDS.get(domain, {"iou": 0.3, "containment": 0.5})
        if isinstance(thresh_entry, dict):
            gt_overlap_thresh = thresh_entry["iou"]
        else:
            gt_overlap_thresh = thresh_entry

    # Determine unmatched original masks (the "misses")
    if matched_pairs is not None:
        matched_orig_indices = {pair[0] for pair in matched_pairs}
    else:
        # Fall back to computing match
        _, unmatched_orig, _ = match_masks(masks_orig, masks_patch, iou_threshold=0.5)
        matched_orig_indices = set(range(len(masks_orig))) - unmatched_orig

    unmatched_orig_indices = set(range(len(masks_orig))) - matched_orig_indices
    n_misses = len(unmatched_orig_indices)

    domain_misses = []
    redundant_misses = []
    background_misses = []

    if not gt_masks or n_misses == 0:
        # No GT available or no misses — all misses are unclassified background
        background_misses = list(unmatched_orig_indices)
        return {
            "n_misses": n_misses,
            "domain_misses": domain_misses,
            "redundant_misses": redundant_misses,
            "background_misses": background_misses,
            "has_gt": len(gt_masks) > 0,
            "gt_coverage": 0.0,
        }

    # Compute GT coverage: what fraction of GT pixels are covered by matched masks
    gt_union = np.zeros_like(gt_masks[0], dtype=bool)
    for gt in gt_masks:
        gt_union |= gt.astype(bool)

    matched_coverage = np.zeros_like(gt_masks[0], dtype=bool)
    for orig_idx in matched_orig_indices:
        seg = masks_orig[orig_idx]["segmentation"]
        if seg.shape == gt_union.shape:
            matched_coverage |= seg.astype(bool)

    gt_total = gt_union.sum()
    gt_covered = (gt_union & matched_coverage).sum()
    gt_coverage = float(gt_covered / gt_total) if gt_total > 0 else 1.0

    # Classify each miss
    for idx in sorted(unmatched_orig_indices):
        missed_seg = masks_orig[idx]["segmentation"]

        # Resize if shape mismatch (SAM output might differ from GT resolution)
        if missed_seg.shape != gt_masks[0].shape:
            missed_seg = cv2.resize(
                missed_seg.astype(np.uint8),
                (gt_masks[0].shape[1], gt_masks[0].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Compute max IoU against all GT masks
        max_gt_iou = 0.0
        for gt in gt_masks:
            iou = compute_mask_iou(missed_seg, gt)
            max_gt_iou = max(max_gt_iou, iou)

        if max_gt_iou >= gt_overlap_thresh:
            # Check if this GT region is already covered by matched masks
            # (redundant = sub-mask of an already-covered GT object)
            missed_bool = missed_seg.astype(bool)
            missed_in_coverage = (missed_bool & matched_coverage).sum()
            missed_total = missed_bool.sum()
            coverage_ratio = missed_in_coverage / missed_total if missed_total > 0 else 0

            if coverage_ratio > 0.8:
                redundant_misses.append(idx)
            else:
                domain_misses.append(idx)
        else:
            background_misses.append(idx)

    return {
        "n_misses": n_misses,
        "domain_misses": domain_misses,
        "redundant_misses": redundant_misses,
        "background_misses": background_misses,
        "has_gt": True,
        "gt_coverage": gt_coverage,
    }


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
    """Run miss rate decomposition for a single domain + manifest."""

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Output directory
    if out_dir is None:
        out_dir = f"./results_miss_analysis/{domain}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Miss Rate Decomposition: {domain.upper()}")
    print(f"{'='*64}")
    print(f"  Checkpoint:  {checkpoint}")
    print(f"  Manifest:    {freeze_manifest_path}")
    print(f"  Image dir:   {image_dir}/{domain}_eval/")
    print(f"  Data root:   {data_root}")
    print(f"  Device:      {device}")
    print(f"{'='*64}\n")

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

    # ---- Load freeze manifest ----
    print("[2/4] Loading freeze manifest...", flush=True)
    dbg(f"Reading manifest: {freeze_manifest_path}")
    manifest = load_manifest(Path(freeze_manifest_path))
    frozen_heads = load_frozen_heads_from_manifest(manifest)
    freeze_fraction = manifest.get("freeze_fraction", 0.0)
    n_frozen = manifest.get("summary", {}).get("frozen", len(frozen_heads))
    print(f"       Frozen heads: {n_frozen} (fraction={freeze_fraction})", flush=True)

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
        print("ERROR: No images found. Exiting.")
        return {}

    # ---- Run analysis ----
    print(f"[4/4] Running analysis on {n_images} images...\n")

    # Accumulators
    total_masks_orig = 0
    total_misses = 0
    total_domain_misses = 0
    total_redundant_misses = 0
    total_background_misses = 0
    gt_coverages = []
    images_with_gt = 0
    images_without_gt = 0
    per_image_results = []
    images_with_domain_misses = 0
    per_image_domain_miss_rates = []
    per_image_bg_miss_rates = []
    max_domain_misses_single_img = 0

    t_start = time.time()

    IMG_TIMEOUT = 180  # seconds — skip image if it takes longer than 3 min

    for i in range(n_images):
        try:
            image, _, filename = dataset[i]
        except Exception as e:
            print(f"  [WARN] Failed to load image {i}: {e}")
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
                    print(f"  [WARN] Image {i} ({filename}) took {elapsed_orig:.0f}s for orig pass — skipping")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                # Frozen SAM (with hooks)
                dbg(f"  gen.generate() [frozen] starting...")
                hook_mgr.register()
                masks_patch = gen.generate(image)
                hook_mgr.remove()
                dbg(f"  gen.generate() [frozen] done: {len(masks_patch)} masks in {time.time()-img_start:.1f}s")

        except torch.cuda.OutOfMemoryError:
            print(f"  [WARN] OOM on image {i} ({filename}), skipping...")
            hook_mgr.remove()
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [WARN] OOM on image {i} ({filename}), skipping...")
            else:
                print(f"  [WARN] RuntimeError on image {i} ({filename}): {e}")
            hook_mgr.remove()
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  [WARN] Error on image {i} ({filename}): {e}")
            hook_mgr.remove()
            continue

        # Evaluate masks (Hungarian matching)
        dbg(f"  Hungarian matching ({len(masks_orig)} vs {len(masks_patch)} masks)...")
        eval_result = evaluate_masks(masks_orig, masks_patch, iou_threshold=0.5)
        matched_pairs = eval_result["matched_pairs"]
        dbg(f"  Matching done: {len(matched_pairs)} pairs")

        # Load GT masks
        dbg(f"  Loading GT masks...")
        gt_masks = load_gt_masks(filename, domain, data_root)
        dbg(f"  GT: {len(gt_masks)} masks loaded")

        # Classify misses
        dbg(f"  Classifying misses...")
        miss_info = classify_misses(
            masks_orig, masks_patch, gt_masks, domain, matched_pairs=matched_pairs
        )
        dbg(f"  Done: dom={len(miss_info['domain_misses'])} bg={len(miss_info['background_misses'])} [{time.time()-img_start:.1f}s total]")

        # Accumulate
        n_orig = len(masks_orig)
        n_miss = miss_info["n_misses"]
        n_dom = len(miss_info["domain_misses"])
        n_red = len(miss_info["redundant_misses"])
        n_bg = len(miss_info["background_misses"])

        total_masks_orig += n_orig
        total_misses += n_miss
        total_domain_misses += n_dom
        total_redundant_misses += n_red
        total_background_misses += n_bg

        if miss_info["has_gt"]:
            images_with_gt += 1
            gt_coverages.append(miss_info["gt_coverage"])
        else:
            images_without_gt += 1

        if n_dom > 0:
            images_with_domain_misses += 1
        max_domain_misses_single_img = max(max_domain_misses_single_img, n_dom)

        dm_rate = n_dom / n_orig if n_orig > 0 else 0.0
        bg_rate = n_bg / n_orig if n_orig > 0 else 0.0
        per_image_domain_miss_rates.append(dm_rate)
        per_image_bg_miss_rates.append(bg_rate)

        if save_details:
            per_image_results.append({
                "filename": filename,
                "n_masks_orig": n_orig,
                "n_masks_patch": len(masks_patch),
                "n_misses": n_miss,
                "n_domain_misses": n_dom,
                "n_redundant_misses": n_red,
                "n_background_misses": n_bg,
                "gt_coverage": miss_info["gt_coverage"],
                "has_gt": miss_info["has_gt"],
            })

        # Progress logging
        if (i + 1) % 20 == 0 or (i + 1) == n_images:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(
                f"  [{i+1:4d}/{n_images}] "
                f"masks={n_orig:3d} miss={n_miss:2d} "
                f"(dom={n_dom} red={n_red} bg={n_bg}) "
                f"[{rate:.1f} img/s]"
            )

        # Memory cleanup
        del masks_orig, masks_patch, image, gt_masks
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed_total = time.time() - t_start

    # ---- Aggregate results ----
    n_evaluated = len(per_image_domain_miss_rates)
    if n_evaluated == 0:
        print("ERROR: No images were successfully processed.")
        return {}

    total_miss_rate = total_misses / total_masks_orig if total_masks_orig > 0 else 0
    domain_miss_rate = total_domain_misses / total_masks_orig if total_masks_orig > 0 else 0
    redundant_miss_rate = total_redundant_misses / total_masks_orig if total_masks_orig > 0 else 0
    background_miss_rate = total_background_misses / total_masks_orig if total_masks_orig > 0 else 0
    pct_misses_domain = (
        total_domain_misses / total_misses * 100 if total_misses > 0 else 0
    )
    mean_gt_coverage = float(np.mean(gt_coverages)) if gt_coverages else 0.0
    mean_dm_rate = float(np.mean(per_image_domain_miss_rates))
    mean_bg_rate = float(np.mean(per_image_bg_miss_rates))

    objects_name = DOMAIN_OBJECT_NAMES.get(domain, "objects")

    # ---- Print summary ----
    print(f"\n{'='*64}")
    print(f"Miss Rate Decomposition: {domain.upper()} @ {freeze_fraction}")
    print(f"{'='*64}")
    print()
    print(f"  Images evaluated:     {n_evaluated:,}")
    print(f"  Total original masks: {total_masks_orig:,}")
    print(f"  Total misses:         {total_misses:,} ({total_miss_rate:.1%})")
    print()
    print(
        f"    Domain misses:      {total_domain_misses:,} ({domain_miss_rate:.1%})"
        f"  <- missed actual {objects_name}"
    )
    print(
        f"    Redundant misses:   {total_redundant_misses:,} ({redundant_miss_rate:.1%})"
        f"  <- sub-masks of covered GT"
    )
    print(
        f"    Background misses:  {total_background_misses:,} ({background_miss_rate:.1%})"
        f"  <- missed shadows/texture"
    )
    print()
    print(f"  GT coverage by matched masks: {mean_gt_coverage:.1%}")
    print()
    print(f"  Per-image breakdown:")
    print(
        f"    Images with domain misses:    "
        f"{images_with_domain_misses} / {n_evaluated} "
        f"({images_with_domain_misses/n_evaluated:.1%})"
    )
    print(f"    Mean domain miss rate:        {mean_dm_rate:.1%}")
    print(f"    Mean background miss rate:    {mean_bg_rate:.1%}")
    print(f"    Max domain misses in one img: {max_domain_misses_single_img}")
    print()
    print(f"  Time: {elapsed_total:.1f}s ({n_evaluated/elapsed_total:.1f} img/s)")
    print(f"{'='*64}\n")

    # ---- Build results dict ----
    results = {
        "domain": domain,
        "freeze_manifest": str(freeze_manifest_path),
        "freeze_fraction": freeze_fraction,
        "num_images": n_evaluated,
        "summary": {
            "total_masks_original": total_masks_orig,
            "total_misses": total_misses,
            "total_miss_rate": round(total_miss_rate, 4),
            "domain_misses": total_domain_misses,
            "domain_miss_rate": round(domain_miss_rate, 4),
            "redundant_misses": total_redundant_misses,
            "redundant_miss_rate": round(redundant_miss_rate, 4),
            "background_misses": total_background_misses,
            "background_miss_rate": round(background_miss_rate, 4),
            "pct_misses_domain": round(pct_misses_domain, 1),
            "mean_gt_coverage": round(mean_gt_coverage, 4),
            "images_with_gt": images_with_gt,
            "images_without_gt": images_without_gt,
            "mean_domain_miss_rate_per_image": round(mean_dm_rate, 4),
            "mean_background_miss_rate_per_image": round(mean_bg_rate, 4),
            "images_with_domain_misses": images_with_domain_misses,
            "max_domain_misses_single_image": max_domain_misses_single_img,
        },
    }

    if save_details:
        results["per_image"] = per_image_results

    # ---- Save JSON ----
    json_path = out_path / "miss_analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # ---- Save one-line CSV summary ----
    csv_path = out_path / "miss_analysis_summary.csv"
    csv_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "domain", "freeze_fraction", "num_images",
                "total_masks", "total_misses", "total_miss_rate",
                "domain_misses", "domain_miss_rate",
                "redundant_misses", "redundant_miss_rate",
                "background_misses", "background_miss_rate",
                "pct_misses_domain", "mean_gt_coverage",
            ])
        writer.writerow([
            domain, freeze_fraction, n_evaluated,
            total_masks_orig, total_misses, f"{total_miss_rate:.4f}",
            total_domain_misses, f"{domain_miss_rate:.4f}",
            total_redundant_misses, f"{redundant_miss_rate:.4f}",
            total_background_misses, f"{background_miss_rate:.4f}",
            f"{pct_misses_domain:.1f}", f"{mean_gt_coverage:.4f}",
        ])
    print(f"  Saved: {csv_path}")

    return results


# ============================================================================
# Batch Mode
# ============================================================================

def run_batch(
    checkpoint: str,
    image_dir: str,
    data_root: str,
    max_images: Optional[int] = None,
    out_dir: Optional[str] = None,
    device_str: str = "cuda",
    manifest_pattern: Optional[str] = None,
    save_details: bool = False,
) -> List[dict]:
    """Run analysis across all domains and freeze fractions."""

    if out_dir is None:
        out_dir = "./results_miss_analysis"
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Combined CSV for batch
    batch_csv_path = out_base / "batch_summary.csv"
    with open(batch_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain", "freeze_fraction", "num_images",
            "total_masks", "total_misses", "total_miss_rate",
            "domain_misses", "domain_miss_rate",
            "redundant_misses", "redundant_miss_rate",
            "background_misses", "background_miss_rate",
            "pct_misses_domain", "mean_gt_coverage",
        ])

    for domain in DOMAINS:
        for frac in FREEZE_FRACTIONS:
            frac_str = str(frac).replace(".", "")  # e.g. "0125"

            # Resolve manifest path
            if manifest_pattern:
                manifest_path = manifest_pattern.format(
                    domain=domain, fraction=frac, frac_str=frac_str
                )
            else:
                manifest_path = (
                    f"./results_cross_domain/{domain}/"
                    f"freeze_{domain}_f{frac}/freeze_manifest.json"
                )

            if not Path(manifest_path).exists():
                print(
                    f"\n[SKIP] Manifest not found: {manifest_path}"
                )
                continue

            domain_out = str(out_base / domain / f"f{frac}")

            try:
                result = run_analysis(
                    checkpoint=checkpoint,
                    freeze_manifest_path=manifest_path,
                    domain=domain,
                    image_dir=image_dir,
                    data_root=data_root,
                    max_images=max_images,
                    out_dir=domain_out,
                    device_str=device_str,
                    save_details=save_details,
                )

                if result:
                    all_results.append(result)

                    # Append to batch CSV
                    s = result["summary"]
                    with open(batch_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            domain, frac, result["num_images"],
                            s["total_masks_original"], s["total_misses"],
                            f"{s['total_miss_rate']:.4f}",
                            s["domain_misses"], f"{s['domain_miss_rate']:.4f}",
                            s["redundant_misses"], f"{s['redundant_miss_rate']:.4f}",
                            s["background_misses"], f"{s['background_miss_rate']:.4f}",
                            f"{s['pct_misses_domain']:.1f}",
                            f"{s['mean_gt_coverage']:.4f}",
                        ])

            except Exception as e:
                print(f"\n[ERROR] {domain} @ f={frac}: {e}")
                traceback.print_exc()
                continue

    # ---- Print combined summary table ----
    if all_results:
        print(f"\n{'='*80}")
        print("BATCH SUMMARY: Miss Rate Decomposition Across All Domains")
        print(f"{'='*80}")
        print(
            f"{'Domain':<12} {'Frac':>5} {'Imgs':>5} "
            f"{'Miss%':>7} {'Dom%':>7} {'Red%':>7} {'BG%':>7} "
            f"{'%Dom':>6} {'GTcov':>6}"
        )
        print("-" * 80)
        for r in all_results:
            s = r["summary"]
            print(
                f"{r['domain']:<12} {r['freeze_fraction']:>5.3f} "
                f"{r['num_images']:>5} "
                f"{s['total_miss_rate']*100:>6.1f}% "
                f"{s['domain_miss_rate']*100:>6.1f}% "
                f"{s['redundant_miss_rate']*100:>6.1f}% "
                f"{s['background_miss_rate']*100:>6.1f}% "
                f"{s['pct_misses_domain']:>5.1f}% "
                f"{s['mean_gt_coverage']*100:>5.1f}%"
            )
        print(f"{'='*80}")
        print(f"\n  Batch CSV saved: {batch_csv_path}")

    return all_results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="S-VELCRO Miss Rate Decomposition Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single domain:
  python scripts/miss_rate_analysis.py \\
      --domain kvasir \\
      --checkpoint checkpoints/sam_vit_h_4b8939.pth \\
      --freeze_manifest results_cross_domain/kvasir/freeze_kvasir_f0.125/freeze_manifest.json \\
      --image_dir ./data --data_root ./data

  # Batch mode (all domains x fractions):
  python scripts/miss_rate_analysis.py --batch \\
      --checkpoint checkpoints/sam_vit_h_4b8939.pth \\
      --image_dir ./data --data_root ./data
        """,
    )

    # Mode
    parser.add_argument(
        "--batch", action="store_true",
        help="Run all 5 domains x 3 freeze fractions",
    )

    # Required (single mode)
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to SAM ViT-H checkpoint (.pth)",
    )
    parser.add_argument(
        "--freeze_manifest", type=str, default=None,
        help="Path to freeze_manifest.json (required for single-domain mode)",
    )
    parser.add_argument(
        "--domain", type=str, default=None, choices=DOMAINS,
        help="Domain name (required for single-domain mode)",
    )

    # Paths
    parser.add_argument(
        "--image_dir", type=str, default="./data",
        help="Directory containing {domain}_eval/ image folders",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data",
        help="Root data directory for GT masks",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: ./results_miss_analysis/{domain})",
    )

    # Options
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Limit number of eval images (default: all)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--save_details", action="store_true",
        help="Save per-image details in JSON output",
    )
    parser.add_argument(
        "--manifest_pattern", type=str, default=None,
        help=(
            "Pattern for manifest paths in batch mode. "
            "Use {domain}, {fraction}, {frac_str} placeholders. "
            "Default: ./results_cross_domain/{domain}/freeze_{domain}_f{fraction}/freeze_manifest.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch:
        run_batch(
            checkpoint=args.checkpoint,
            image_dir=args.image_dir,
            data_root=args.data_root,
            max_images=args.max_images,
            out_dir=args.out_dir,
            device_str=args.device,
            manifest_pattern=args.manifest_pattern,
            save_details=args.save_details,
        )
    else:
        # Single domain mode
        if args.domain is None:
            print("ERROR: --domain is required in single-domain mode.")
            sys.exit(1)
        if args.freeze_manifest is None:
            print("ERROR: --freeze_manifest is required in single-domain mode.")
            sys.exit(1)

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
