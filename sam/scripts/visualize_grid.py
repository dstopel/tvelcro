#!/usr/bin/env python3
"""
visualize_grid.py — Generate 3x4 grid: best/median/worst × freeze levels.

For each dataset, produces a 3-row × 4-column figure:
  Row 1 (BEST):   image where freezing has minimal effect
  Row 2 (MEDIAN): image with typical behavior
  Row 3 (WORST):  image where freezing causes most damage

  Col 0: Original | Col 1: 12.5% | Col 2: 25% | Col 3: 37.5%

The script first does a quick ranking pass (original vs 37.5% only),
picks the 3 representative images, then generates the full grid.

Usage:
  python3 scripts/visualize_grid.py \
      --checkpoint ./data/sam_vit_h_4b8939.pth \
      --domain kvasir \
      --image_dir ./image_dirs_cross_domain \
      --data_root ./data \
      --out_dir ./visualizations \
      --mode attention   # or "combined"
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from sam_velcro_pkg.dataset import ImageDirDataset
from sam_velcro_pkg.evaluation import match_masks
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
    overlay = image.copy().astype(np.float32)
    np.random.seed(42)
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    for mask in sorted_masks:
        seg = mask["segmentation"]
        color = np.random.randint(60, 255, size=3).astype(np.float32)
        overlay[seg] = overlay[seg] * (1 - alpha) + color * alpha
    return overlay.astype(np.uint8)


def overlay_gt_contour(image: np.ndarray, gt_masks: List[np.ndarray]) -> np.ndarray:
    result = image.copy()
    for gt in gt_masks:
        contours, _ = cv2.findContours(
            gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result


def find_attn_manifest(domain: str, frac: float) -> Optional[str]:
    candidates = [
        f"./results_cross_domain/{domain}/freeze_{domain}_f{frac}/freeze_manifest.json",
        f"./results_cross_domain/freeze_{domain}_f{frac}/freeze_manifest.json",
        f"./results_wave2/{domain}/freeze_{domain}_f{frac}/freeze_manifest.json",
        f"./results_wave2/freeze_{domain}_f{frac}/freeze_manifest.json",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def generate_frozen_masks(gen, sam, image, device, attn_manifest_path=None,
                          mlp_profile_dir=None, mlp_fraction=None, domain=None):
    attn_mgr = None
    mlp_mgr = None

    if attn_manifest_path:
        manifest = load_manifest(Path(attn_manifest_path))
        attn_frozen = load_frozen_heads_from_manifest(manifest)
        attn_mgr = FreezeHookManager(sam, attn_frozen, device)

    if mlp_profile_dir and mlp_fraction and domain:
        mlp_frozen = select_mlp_neurons_to_freeze(
            Path(mlp_profile_dir), domain, mlp_fraction
        )
        if mlp_frozen:
            mlp_mgr = MLPFreezeHookManager(sam, mlp_frozen, device)

    with torch.inference_mode():
        if attn_mgr:
            attn_mgr.register()
        if mlp_mgr:
            mlp_mgr.register()
        masks = gen.generate(image)
        if attn_mgr:
            attn_mgr.remove()
        if mlp_mgr:
            mlp_mgr.remove()

    return masks


def compute_quality_score(masks_orig, masks_frozen):
    """Score = mean IoU of matched pairs. Higher = better preservation."""
    if not masks_orig or not masks_frozen:
        return 0.0
    matched, unmatched_orig, _ = match_masks(masks_orig, masks_frozen, 0.5)
    if not matched:
        return 0.0
    return float(np.mean([m[2] for m in matched]))


def rank_images(gen, sam, dataset, device, domain, mode, max_rank_images=50):
    """Quick ranking pass: run orig vs most aggressive freeze, rank by quality."""
    print(f"  Ranking images (orig vs aggressive freeze on {min(max_rank_images, len(dataset))} images)...", flush=True)

    if mode == "attention":
        manifest_path = find_attn_manifest(domain, 0.375)
        if not manifest_path:
            manifest_path = find_attn_manifest(domain, 0.25)
    else:  # combined
        manifest_path = find_attn_manifest(domain, 0.25)

    mlp_profile_dir = f"./results_combined_freeze/{domain}/mlp_profile"
    use_mlp = mode == "combined" and Path(mlp_profile_dir).exists()

    scores = []
    n = min(max_rank_images, len(dataset))

    for i in range(n):
        try:
            image, _, filename = dataset[i]
        except Exception:
            continue

        with torch.inference_mode():
            masks_orig = gen.generate(image)

        masks_frozen = generate_frozen_masks(
            gen, sam, image, device,
            attn_manifest_path=manifest_path,
            mlp_profile_dir=mlp_profile_dir if use_mlp else None,
            mlp_fraction=0.375 if use_mlp else None,
            domain=domain if use_mlp else None,
        )

        score = compute_quality_score(masks_orig, masks_frozen)
        scores.append((i, score, filename))

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{n}] ranked", flush=True)

        del masks_orig, masks_frozen, image
        gc.collect()
        torch.cuda.empty_cache()

    # Sort by score
    scores.sort(key=lambda x: x[1])

    worst_idx = scores[0][0]
    median_idx = scores[len(scores) // 2][0]
    best_idx = scores[-1][0]

    print(f"  Selected: best=img[{best_idx}] (IoU={scores[-1][1]:.3f}), "
          f"median=img[{median_idx}] (IoU={scores[len(scores)//2][1]:.3f}), "
          f"worst=img[{worst_idx}] (IoU={scores[0][1]:.3f})", flush=True)

    return best_idx, median_idx, worst_idx


def generate_grid(gen, sam, dataset, device, domain, mode,
                  best_idx, median_idx, worst_idx, data_root, out_dir):
    """Generate the 3×4 grid figure."""

    if mode == "attention":
        fracs = [0.125, 0.25, 0.375]
        col_titles = ["Original", "Attn 12.5%", "Attn 25%", "Attn 37.5%"]
    else:
        fracs = [0.125, 0.25, 0.375]
        col_titles = ["Original", "Attn25+MLP12.5%", "Attn25+MLP25%", "Attn25+MLP37.5%"]

    row_labels = ["BEST", "MEDIAN", "WORST"]
    row_indices = [best_idx, median_idx, worst_idx]

    mlp_profile_dir = f"./results_combined_freeze/{domain}/mlp_profile"

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    for row, (label, img_idx) in enumerate(zip(row_labels, row_indices)):
        image, _, filename = dataset[img_idx]
        gt_masks = load_gt_masks(filename, domain, data_root)

        # Column 0: Original
        with torch.inference_mode():
            masks_orig = gen.generate(image)

        overlay = overlay_masks(image, masks_orig)
        if gt_masks:
            overlay = overlay_gt_contour(overlay, gt_masks)
        axes[row, 0].imshow(overlay)
        axes[row, 0].set_title(f"{col_titles[0]}\n{len(masks_orig)} masks", fontsize=11)
        axes[row, 0].set_ylabel(f"{label}\n{filename[:20]}...", fontsize=11, fontweight="bold")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # Columns 1-3: frozen at each level
        for col, frac in enumerate(fracs, start=1):
            if mode == "attention":
                manifest_path = find_attn_manifest(domain, frac)
                masks_frozen = generate_frozen_masks(
                    gen, sam, image, device,
                    attn_manifest_path=manifest_path,
                )
            else:
                manifest_path = find_attn_manifest(domain, 0.25)
                masks_frozen = generate_frozen_masks(
                    gen, sam, image, device,
                    attn_manifest_path=manifest_path,
                    mlp_profile_dir=mlp_profile_dir,
                    mlp_fraction=frac,
                    domain=domain,
                )

            # Compute stats
            matched, unmatched_orig, _ = match_masks(masks_orig, masks_frozen, 0.5)
            iou = np.mean([m[2] for m in matched]) if matched else 0.0
            n_miss = len(unmatched_orig)

            overlay = overlay_masks(image, masks_frozen)
            if gt_masks:
                overlay = overlay_gt_contour(overlay, gt_masks)

            axes[row, col].imshow(overlay)
            axes[row, col].set_title(
                f"{col_titles[col]}\n{len(masks_frozen)} masks | IoU={iou:.3f} | miss={n_miss}",
                fontsize=10,
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

            del masks_frozen
            gc.collect()
            torch.cuda.empty_cache()

        del masks_orig, image
        gc.collect()
        torch.cuda.empty_cache()

    objects_name = DOMAIN_OBJECTS.get(domain, "objects")
    mode_label = "Attention Head Freeze" if mode == "attention" else "Combined Attn 25% + MLP Freeze"
    fig.suptitle(
        f"{domain.upper()} — {mode_label}\n"
        f"Green contour = GT {objects_name} | Best/Median/Worst selected by IoU at max freeze",
        fontsize=14, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(out_dir) / f"{domain}_{mode}_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    domain = args.domain
    mode = args.mode

    print(f"\n{'='*60}", flush=True)
    print(f"  {domain.upper()} — {mode} grid visualization", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load model
    print("Loading SAM ViT-H...", flush=True)
    sam = sam_model_registry["vit_h"](checkpoint=args.checkpoint)
    sam.to(device).eval()
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95
    )

    # Load dataset
    eval_dir = f"{args.image_dir}/{domain}_eval"
    dataset = ImageDirDataset(eval_dir, domain)
    print(f"  Dataset: {len(dataset)} images\n", flush=True)

    # Build filename→index map
    name_to_idx = {}
    for i in range(len(dataset)):
        _, _, fname = dataset[i]
        name_to_idx[fname] = i

    # Determine best/median/worst indices
    if args.from_json:
        # Pull from existing coverage_comparison.json
        import json
        print(f"  Loading image selection from: {args.from_json}", flush=True)
        with open(args.from_json) as f:
            cov_data = json.load(f)

        worst_images = cov_data.get("worst_images", [])
        if not worst_images:
            print("  ERROR: no worst_images in JSON", flush=True)
            return

        # Worst = first entry (highest drop)
        worst_name = worst_images[0]["filename"]
        # Median = middle entry
        median_name = worst_images[len(worst_images) // 2]["filename"]
        # Best = find an image NOT in worst list (low/no drop)
        worst_names = {w["filename"] for w in worst_images}
        best_name = None
        for i in range(len(dataset)):
            _, _, fname = dataset[i]
            if fname not in worst_names:
                best_name = fname
                break
        if best_name is None:
            best_name = worst_images[-1]["filename"]

        best_idx = name_to_idx.get(best_name, 0)
        median_idx = name_to_idx.get(median_name, len(dataset) // 2)
        worst_idx = name_to_idx.get(worst_name, 0)

        print(f"  From JSON: best={best_name} (idx={best_idx})", flush=True)
        print(f"             median={median_name} (idx={median_idx})", flush=True)
        print(f"             worst={worst_name} (idx={worst_idx})", flush=True)

    elif args.best_index is not None and args.worst_index is not None:
        # Use explicit indices
        best_idx = args.best_index
        median_idx = args.median_index if args.median_index is not None else len(dataset) // 2
        worst_idx = args.worst_index
        print(f"  Explicit indices: best={best_idx}, median={median_idx}, worst={worst_idx}", flush=True)

    else:
        # Run ranking pass
        best_idx, median_idx, worst_idx = rank_images(
            gen, sam, dataset, device, domain, mode,
            max_rank_images=min(args.rank_images, len(dataset)),
        )

    # Generate grid
    print(f"\n  Generating 3×4 grid...", flush=True)
    generate_grid(gen, sam, dataset, device, domain, mode,
                  best_idx, median_idx, worst_idx, args.data_root, args.out_dir)

    del sam, gen
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3×4 grid: best/median/worst × freeze levels"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_OBJECTS.keys()))
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default="./visualizations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["attention", "combined"], default="attention",
                        help="'attention' = attn sweep only, 'combined' = attn25 + MLP sweep")
    parser.add_argument("--rank_images", type=int, default=50,
                        help="Number of images to rank for best/median/worst selection")
    parser.add_argument("--from_json", type=str, default=None,
                        help="Path to coverage_comparison.json to pick best/median/worst from existing results")
    parser.add_argument("--best_index", type=int, default=None,
                        help="Explicit image index for best case")
    parser.add_argument("--median_index", type=int, default=None,
                        help="Explicit image index for median case")
    parser.add_argument("--worst_index", type=int, default=None,
                        help="Explicit image index for worst case")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
