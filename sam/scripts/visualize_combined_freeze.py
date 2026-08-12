#!/usr/bin/env python3
"""
visualize_combined_freeze.py — Generate side-by-side mask comparison images.

For each dataset, produces one image showing:
  LEFT:  Original SAM masks (colored overlay)
  RIGHT: Combined frozen SAM masks (attn 25% + MLP X%)

Usage:
  python3 scripts/visualize_combined_freeze.py \
      --checkpoint ./data/sam_vit_h_4b8939.pth \
      --domain kvasir \
      --attn_manifest ./results_cross_domain/freeze_kvasir_f0.25/freeze_manifest.json \
      --mlp_profile_dir ./results_combined_freeze/kvasir/mlp_profile \
      --mlp_fraction 0.25 \
      --image_dir ./image_dirs_cross_domain \
      --data_root ./data \
      --out_dir ./results_combined_freeze/visualizations
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from sam_velcro_pkg.dataset import ImageDirDataset
from sam_velcro_pkg.freeze import load_frozen_heads_from_manifest, load_manifest
from sam_velcro_pkg.gt_masks import load_gt_masks
from sam_velcro_pkg.hooks import FreezeHookManager
from sam_velcro_pkg.mlp_hooks import MLPFreezeHookManager, select_mlp_neurons_to_freeze


DOMAIN_OBJECTS = {
    "kvasir": "polyps",
    "isic": "lesions",
    "busi": "tumors",
    "pcb": "defects",
}


def overlay_masks(image: np.ndarray, masks: List[dict], alpha: float = 0.4) -> np.ndarray:
    """Overlay colored masks on an image."""
    overlay = image.copy().astype(np.float32)
    np.random.seed(42)

    # Sort by area descending (draw large masks first, small on top)
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    for mask in sorted_masks:
        seg = mask["segmentation"]
        color = np.random.randint(60, 255, size=3).astype(np.float32)
        overlay[seg] = overlay[seg] * (1 - alpha) + color * alpha

    return overlay.astype(np.uint8)


def overlay_gt_contour(image: np.ndarray, gt_masks: List[np.ndarray]) -> np.ndarray:
    """Draw GT mask contour in green on the image."""
    result = image.copy()
    for gt in gt_masks:
        contours, _ = cv2.findContours(
            gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result


def generate_comparison(
    image: np.ndarray,
    masks_orig: List[dict],
    masks_frozen: List[dict],
    gt_masks: List[np.ndarray],
    filename: str,
    domain: str,
    attn_frac: float,
    mlp_frac: float,
    out_path: Path,
):
    """Generate a side-by-side comparison figure."""
    # Create overlays
    orig_overlay = overlay_masks(image, masks_orig)
    frozen_overlay = overlay_masks(image, masks_frozen)

    # Add GT contour
    if gt_masks:
        orig_overlay = overlay_gt_contour(orig_overlay, gt_masks)
        frozen_overlay = overlay_gt_contour(frozen_overlay, gt_masks)

    # Compute stats
    from sam_velcro_pkg.evaluation import match_masks
    matched, unmatched_orig, unmatched_frozen = match_masks(masks_orig, masks_frozen, 0.5)
    n_orig = len(masks_orig)
    n_frozen = len(masks_frozen)
    n_matched = len(matched)
    mean_iou = np.mean([m[2] for m in matched]) if matched else 0.0
    miss_rate = len(unmatched_orig) / n_orig if n_orig > 0 else 0.0

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(orig_overlay)
    axes[0].set_title(f"Original SAM\n{n_orig} masks", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(frozen_overlay)
    axes[1].set_title(
        f"Frozen (attn {attn_frac*100:.0f}% + MLP {mlp_frac*100:.0f}%)\n"
        f"{n_frozen} masks | IoU={mean_iou:.3f} | miss={len(unmatched_orig)}",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].axis("off")

    objects_name = DOMAIN_OBJECTS.get(domain, "objects")
    gt_info = f"GT: {len(gt_masks)} {objects_name}" if gt_masks else "No GT"
    fig.suptitle(
        f"{domain.upper()} — {filename}\n{gt_info} (green contour)",
        fontsize=12,
        y=0.02,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def run_visualization(
    checkpoint: str,
    domain: str,
    attn_manifest: str,
    mlp_profile_dir: str,
    mlp_fraction: float,
    image_dir: str,
    data_root: str,
    out_dir: str,
    device_str: str = "cuda",
    image_index: int = 0,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"[1/5] Loading SAM ViT-H...", flush=True)
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to(device).eval()
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95
    )

    # Load attention manifest
    print(f"[2/5] Loading attention manifest...", flush=True)
    manifest = load_manifest(Path(attn_manifest))
    attn_frozen = load_frozen_heads_from_manifest(manifest)
    attn_frac = manifest.get("freeze_fraction", 0.25)
    attn_hook_mgr = FreezeHookManager(sam, attn_frozen, device)

    # Load MLP neurons
    print(f"[3/5] Selecting MLP neurons to freeze ({mlp_fraction*100:.1f}%)...", flush=True)
    mlp_frozen = select_mlp_neurons_to_freeze(
        Path(mlp_profile_dir), domain, mlp_fraction
    )
    mlp_hook_mgr = MLPFreezeHookManager(sam, mlp_frozen, device)
    print(f"       Frozen: {len(mlp_frozen)} MLP neurons", flush=True)

    # Load dataset
    print(f"[4/5] Loading dataset...", flush=True)
    eval_dir = f"{image_dir}/{domain}_eval"
    dataset = ImageDirDataset(eval_dir, domain)
    print(f"       Found {len(dataset)} images", flush=True)

    # Pick image
    idx = min(image_index, len(dataset) - 1)
    image, _, filename = dataset[idx]
    print(f"[5/5] Processing image {idx}: {filename}", flush=True)

    # Generate original masks
    with torch.inference_mode():
        masks_orig = gen.generate(image)

    # Generate frozen masks
    with torch.inference_mode():
        attn_hook_mgr.register()
        mlp_hook_mgr.register()
        masks_frozen = gen.generate(image)
        attn_hook_mgr.remove()
        mlp_hook_mgr.remove()

    # Load GT
    gt_masks = load_gt_masks(filename, domain, data_root)

    # Generate comparison
    out_path = Path(out_dir) / f"{domain}_comparison.png"
    generate_comparison(
        image, masks_orig, masks_frozen, gt_masks,
        filename, domain, attn_frac, mlp_fraction, out_path,
    )

    # Cleanup
    del sam, gen, attn_hook_mgr, mlp_hook_mgr
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side mask comparison for combined freeze"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_OBJECTS.keys()))
    parser.add_argument("--attn_manifest", required=True)
    parser.add_argument("--mlp_profile_dir", required=True)
    parser.add_argument("--mlp_fraction", type=float, default=0.25)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default="./results_combined_freeze/visualizations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_index", type=int, default=5,
                        help="Which image to visualize (default: 5)")
    args = parser.parse_args()

    run_visualization(
        checkpoint=args.checkpoint,
        domain=args.domain,
        attn_manifest=args.attn_manifest,
        mlp_profile_dir=args.mlp_profile_dir,
        mlp_fraction=args.mlp_fraction,
        image_dir=args.image_dir,
        data_root=args.data_root,
        out_dir=args.out_dir,
        device_str=args.device,
        image_index=args.image_index,
    )


if __name__ == "__main__":
    main()
