#!/usr/bin/env python3
"""
check_full_domain_loss.py — Find images where domain objects are completely lost.

Checks: is there any image where NO matched mask covers the GT annotation?
This is the worst-case scenario (entire tumor/road/polyp invisible to frozen model).

Usage:
  python3 scripts/check_full_domain_loss.py \
      --domain deepglobe \
      --checkpoint ./data/sam_vit_h_4b8939.pth \
      --freeze_manifest ./results_cross_domain/freeze_deepglobe_f0.375/freeze_manifest.json \
      --image_dir ./image_dirs_wave2 \
      --data_root ./data
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from sam_velcro_pkg.dataset import ImageDirDataset
from sam_velcro_pkg.evaluation import match_masks
from sam_velcro_pkg.freeze import load_frozen_heads_from_manifest, load_manifest
from sam_velcro_pkg.gt_masks import load_gt_masks
from sam_velcro_pkg.hooks import FreezeHookManager


def compute_gt_coverage(gt_masks, masks_orig, matched_orig_indices):
    """What fraction of GT pixels are covered by matched (non-missed) masks?"""
    if not gt_masks:
        return None  # No GT for this image

    gt_union = np.zeros_like(gt_masks[0], dtype=bool)
    for gt in gt_masks:
        gt_union |= gt.astype(bool)

    gt_area = gt_union.sum()
    if gt_area == 0:
        return None

    matched_union = np.zeros_like(gt_masks[0], dtype=bool)
    for idx in matched_orig_indices:
        seg = masks_orig[idx]["segmentation"]
        if seg.shape == gt_union.shape:
            matched_union |= seg.astype(bool)

    covered = (gt_union & matched_union).sum()
    return float(covered / gt_area)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--freeze_manifest", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading SAM...", flush=True)
    sam = sam_model_registry["vit_h"](checkpoint=args.checkpoint)
    sam.to(device).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32,
                                     pred_iou_thresh=0.88,
                                     stability_score_thresh=0.95)

    # Load freeze config
    manifest = load_manifest(Path(args.freeze_manifest))
    frozen_heads = load_frozen_heads_from_manifest(manifest)
    frac = manifest.get("freeze_fraction", 0.0)
    hook_mgr = FreezeHookManager(sam, frozen_heads, device)

    # Load dataset
    eval_dir = f"{args.image_dir}/{args.domain}_eval"
    dataset = ImageDirDataset(eval_dir, args.domain, max_images=args.max_images)
    print(f"Domain: {args.domain}, Freeze: {frac}, Images: {len(dataset)}\n")

    # Track results
    full_losses = []  # Images where GT coverage = 0
    low_coverage = []  # Images where GT coverage < 0.5
    all_coverages = []

    for i in range(len(dataset)):
        image, _, filename = dataset[i]

        try:
            with torch.inference_mode():
                masks_orig = gen.generate(image)
                hook_mgr.register()
                masks_patch = gen.generate(image)
                hook_mgr.remove()
        except Exception as e:
            print(f"  [SKIP] {filename}: {e}")
            hook_mgr.remove()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Match
        matched, unmatched_orig, _ = match_masks(masks_orig, masks_patch, 0.5)
        matched_indices = {m[0] for m in matched}

        # Load GT
        gt_masks = load_gt_masks(filename, args.domain, args.data_root)

        # Compute coverage
        coverage = compute_gt_coverage(gt_masks, masks_orig, matched_indices)

        if coverage is None:
            continue  # No GT for this image

        all_coverages.append((filename, coverage))

        if coverage == 0.0:
            full_losses.append(filename)
            print(f"  *** FULL LOSS: {filename} (GT coverage = 0%)")
        elif coverage < 0.5:
            low_coverage.append((filename, coverage))
            print(f"  ! LOW: {filename} (GT coverage = {coverage:.1%})")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(dataset)}] processed", flush=True)

        del masks_orig, masks_patch, image
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.domain} @ {frac} freeze")
    print(f"{'='*60}")
    print(f"  Images with GT:         {len(all_coverages)}")
    print(f"  Full domain loss (0%):  {len(full_losses)}")
    print(f"  Low coverage (<50%):    {len(low_coverage)}")

    if all_coverages:
        covs = [c for _, c in all_coverages]
        print(f"  Mean GT coverage:       {np.mean(covs):.1%}")
        print(f"  Median GT coverage:     {np.median(covs):.1%}")
        print(f"  Min GT coverage:        {np.min(covs):.1%}")

    if full_losses:
        print(f"\n  Images with COMPLETE domain loss:")
        for f in full_losses:
            print(f"    - {f}")

    if low_coverage:
        print(f"\n  Images with low coverage (<50%):")
        for f, c in sorted(low_coverage, key=lambda x: x[1]):
            print(f"    - {f} ({c:.1%})")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
